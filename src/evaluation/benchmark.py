"""Stage 6: classification-only accuracy / latency / cost benchmark.

Four configurations, all classification, no extraction:

* **baseline_only** — TF-IDF + logistic regression (full held-out test set)
* **transformer_only** — DistilBERT (full held-out test set)
* **llm_only** — Gemini classification on a stratified subsample
* **router** — cheapest-first cascade (baseline → transformer → LLM)

Extraction (`route_extraction`) is still wired into the live API; it is not
measured here. Classical and transformer inference are treated as $0 after
training. LLM cost uses published Gemini 3.1 Flash-Lite paid list prices and
the same chars/4 token heuristic as `models.llm_fallback`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.metrics import compute_classification_metrics  # noqa: E402
from models.llm_fallback import (  # noqa: E402
    GEMINI_MODEL,
    LLMCallFailed,
    LLMQuotaExceeded,
    classify_with_llm,
    gemini_configured,
    quota_exhausted,
    reset_client,
    reset_quota_gate,
)
from models import llm_fallback as llm_mod  # noqa: E402
from router import router as router_mod  # noqa: E402
from router.router import (  # noqa: E402
    THRESHOLD_1,
    THRESHOLD_2,
    _predict_baseline,
    candidate_labels,
    finish_routing_trace,
    load_classifiers,
    route_document,
    start_routing_trace,
)
from models import transformer_classifier  # noqa: E402

CSV_PATH = REPO_ROOT / "data" / "processed" / "rvl_cdip_text.csv"
RESULTS_DIR = REPO_ROOT / "results"
TABLE_PATH = RESULTS_DIR / "benchmark_table.csv"
DETAILS_PATH = RESULTS_DIR / "benchmark_details.json"
CHART_PATH = RESULTS_DIR / "benchmark_chart.png"
ESCALATION_CSV_PATH = RESULTS_DIR / "escalation_breakdown.csv"
ESCALATION_CHART_PATH = RESULTS_DIR / "escalation_chart.png"
PREDICTIONS_PATH = RESULTS_DIR / "benchmark_predictions.jsonl"
CACHE_PATH = RESULTS_DIR / "benchmark_llm_cache.jsonl"
LOG_PATH = REPO_ROOT / "logs" / "routing_log.jsonl"
KNOWN_SPLITS = {"train", "validation", "val", "test"}

LLM_SUBSAMPLE_N = 64
LLM_SUBSAMPLE_SEED = 42

# Single model for every *new* classification LLM call, and the only cache
# prefix this run will reuse. Other prefixes in the cache file (bare
# `classify:` / `gemini-2.5-flash:classify:`) are ignored so reported numbers
# stay single-model and comparable to the existing LLM-only arm.
BENCHMARK_GEMINI_MODEL = "gemini-3.1-flash-lite"
USD_PER_MILLION_INPUT = 0.25
USD_PER_MILLION_OUTPUT = 1.50

# Free-tier RPM is tight. Pacing applies only to genuine API calls.
# Free-tier Flash-Lite is 15 RPM. 2s pacing burst the per-minute cap and
# the old latch treated that 429 like a daily exhaustion. Stay under 15/min.
MIN_SECONDS_BETWEEN_LLM_CALLS = 4.5

_last_llm_call = 0.0
_token_log: list[dict] = []


def _split_from_doc_id(doc_id: str) -> str | None:
    prefix = str(doc_id).split("_", 1)[0]
    return prefix if prefix in KNOWN_SPLITS else None


def load_test_split(csv_path: Path = CSV_PATH) -> tuple[list[str], list[str], list[str]]:
    """Same test split as `scripts/train_baseline.py` / `compare_baseline_vs_transformer.py`."""
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}.")
    frame = pd.read_csv(csv_path)
    frame["split"] = frame["doc_id"].map(_split_from_doc_id)
    if frame["split"].notna().all():
        test_frame = frame[frame["split"] == "test"]
        if test_frame.empty:
            test_frame = frame[frame["split"].isin(["validation", "val"])]
    else:
        from sklearn.model_selection import train_test_split

        _, test_frame = train_test_split(
            frame, test_size=0.2, random_state=42, stratify=frame["label"]
        )
    if test_frame.empty:
        raise SystemExit("Test split is empty.")
    texts = test_frame["text"].fillna("").astype(str).tolist()
    labels = test_frame["label"].astype(str).tolist()
    doc_ids = test_frame["doc_id"].astype(str).tolist()
    return texts, labels, doc_ids


def stratified_subsample(
    texts: list[str],
    labels: list[str],
    n: int,
    seed: int = LLM_SUBSAMPLE_SEED,
) -> tuple[list[str], list[str], list[int]]:
    """Take roughly equal counts from each class. Returns texts, labels, indices."""
    if n >= len(texts):
        return list(texts), list(labels), list(range(len(texts)))
    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    rng = random.Random(seed)
    classes = sorted(by_label)
    per = n // len(classes)
    remainder = n % len(classes)
    chosen: list[int] = []
    for j, lab in enumerate(classes):
        take = per + (1 if j < remainder else 0)
        idxs = by_label[lab][:]
        rng.shuffle(idxs)
        chosen.extend(idxs[:take])
    chosen.sort()
    return [texts[i] for i in chosen], [labels[i] for i in chosen], chosen


def install_token_probe() -> None:
    """Count prompt/response chars with Stage 5's chars/4 heuristic, current prices."""
    original = llm_mod._add_cost

    def wrapped(prompt: str, response_text: str) -> None:
        in_tokens = max(len(prompt), 1) / 4.0
        out_tokens = max(len(response_text), 1) / 4.0
        _token_log.append(
            {
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
            }
        )
        original(prompt, response_text)

    llm_mod._add_cost = wrapped


def tokens_to_usd(input_tokens: float, output_tokens: float) -> float:
    return (input_tokens / 1_000_000.0) * USD_PER_MILLION_INPUT + (
        output_tokens / 1_000_000.0
    ) * USD_PER_MILLION_OUTPUT


def doc_hash(text: str) -> str:
    return hashlib.sha256(("" if text is None else str(text)).encode("utf-8")).hexdigest()


def _classify_key(text: str) -> str:
    return f"{BENCHMARK_GEMINI_MODEL}:classify:{doc_hash(text)}"


def load_cache(path: Path = CACHE_PATH) -> dict:
    cache: dict = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["key"]] = row
    return cache


def append_cache(row: dict, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _pace_llm() -> None:
    """Keep sequential *new* calls under the free-tier RPM cap."""
    global _last_llm_call
    if _last_llm_call:
        wait = MIN_SECONDS_BETWEEN_LLM_CALLS - (time.perf_counter() - _last_llm_call)
        if wait > 0:
            time.sleep(wait)
    _last_llm_call = time.perf_counter()


def install_benchmark_llm() -> None:
    """Use Flash-Lite for this run. Timeout/retry policy lives in llm_fallback."""
    llm_mod.GEMINI_MODEL = BENCHMARK_GEMINI_MODEL
    reset_client()
    reset_quota_gate()


def cached_classify(text: str, labels: list[str], cache: dict) -> dict:
    """Classify via the Flash-Lite cache, or a new API call on miss.

    Returns a dict with label/confidence/latency/tokens/cache_hit.
    Raises LLMQuotaExceeded or LLMCallFailed; never mixes other-model cache rows.
    """
    key = _classify_key(text)
    if key in cache:
        row = cache[key]
        return {
            "label": row["label"],
            "confidence": float(row["confidence"]),
            "latency_ms": float(row["latency_ms"]),
            "input_tokens": float(row["input_tokens"]),
            "output_tokens": float(row["output_tokens"]),
            "cache_hit": True,
        }

    if quota_exhausted():
        raise LLMQuotaExceeded(
            f"Gemini quota already exhausted; skipping API call. ({llm_mod.quota_reason()})"
        )

    _pace_llm()
    before = len(_token_log)
    started = time.perf_counter()
    pred, confidence = classify_with_llm(text, labels)
    latency_ms = (time.perf_counter() - started) * 1000.0
    new = _token_log[before:]
    in_tok = sum(item["input_tokens"] for item in new)
    out_tok = sum(item["output_tokens"] for item in new)
    row = {
        "key": key,
        "label": pred,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }
    cache[key] = row
    append_cache(row)
    return {
        "label": pred,
        "confidence": float(confidence),
        "latency_ms": latency_ms,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_hit": False,
    }


def run_baseline_only(texts: list[str], gold: list[str]) -> dict:
    predictions = []
    latencies = []
    for i, text in enumerate(texts):
        started = time.perf_counter()
        pred, _conf = _predict_baseline(text)
        latencies.append((time.perf_counter() - started) * 1000.0)
        predictions.append(pred)
        if (i + 1) % 128 == 0:
            print(f"  baseline_only {i + 1}/{len(texts)}")
    metrics = compute_classification_metrics(gold, predictions)
    return _pack("baseline_only", metrics, latencies, cost_per_1000=0.0, n=len(texts))


def run_transformer_only(texts: list[str], gold: list[str]) -> dict:
    predictions = []
    latencies = []
    for i, text in enumerate(texts):
        started = time.perf_counter()
        pred, _conf = transformer_classifier.predict(text)
        latencies.append((time.perf_counter() - started) * 1000.0)
        predictions.append(pred)
        if (i + 1) % 64 == 0:
            print(f"  transformer_only {i + 1}/{len(texts)}")
    metrics = compute_classification_metrics(gold, predictions)
    return _pack("transformer_only", metrics, latencies, cost_per_1000=0.0, n=len(texts))


def run_llm_only(
    texts: list[str],
    gold: list[str],
    doc_ids: list[str],
    labels: list[str],
    cache: dict,
    pred_rows: list[dict],
) -> dict:
    if not gemini_configured():
        raise SystemExit(
            "GEMINI_API_KEY is not set. LLM-only and router arms need it. "
            "See .env.example."
        )
    predictions = []
    gold_kept = []
    latencies = []
    in_tokens = []
    out_tokens = []
    n_attempted = len(texts)
    n_failed = 0
    n_quota = 0
    n_cache_hits = 0
    print(
        f"  llm_only subsample: {n_attempted} documents "
        f"(NOT the full test set; stratified, seed={LLM_SUBSAMPLE_SEED})"
    )
    for i, (text, true_label, doc_id) in enumerate(zip(texts, gold, doc_ids)):
        try:
            hit = cached_classify(text, labels, cache)
        except LLMQuotaExceeded as err:
            n_quota += 1
            print(
                f"  llm_only SKIP quota doc_hash={doc_hash(text)} "
                f"doc_id={doc_id} ({err})",
                flush=True,
            )
            continue
        except LLMCallFailed as err:
            n_failed += 1
            print(
                f"  llm_only SKIP failed doc_hash={doc_hash(text)} "
                f"doc_id={doc_id} ({err})",
                flush=True,
            )
            continue

        if hit["cache_hit"]:
            n_cache_hits += 1
        latencies.append(hit["latency_ms"])
        predictions.append(hit["label"])
        gold_kept.append(true_label)
        in_tokens.append(hit["input_tokens"])
        out_tokens.append(hit["output_tokens"])
        pred_rows.append(
            {
                "configuration": "llm_only",
                "doc_id": doc_id,
                "doc_hash": doc_hash(text),
                "true_label": true_label,
                "predicted_label": hit["label"],
                "confidence": hit["confidence"],
                "tier": "llm",
                "latency_ms": hit["latency_ms"],
                "input_tokens": hit["input_tokens"],
                "output_tokens": hit["output_tokens"],
                "cost_usd": tokens_to_usd(hit["input_tokens"], hit["output_tokens"]),
                "cache_hit": hit["cache_hit"],
            }
        )
        print(
            f"  llm_only {i + 1}/{n_attempted}  "
            f"cls={hit['latency_ms']:.0f}ms  cache={'hit' if hit['cache_hit'] else 'miss'}",
            flush=True,
        )

    n = len(predictions)
    if n == 0:
        raise SystemExit("llm_only completed 0 documents (all skipped).")
    metrics = compute_classification_metrics(gold_kept, predictions)
    avg_in = sum(in_tokens) / n
    avg_out = sum(out_tokens) / n
    cost_per_doc = tokens_to_usd(avg_in, avg_out)
    result = _pack(
        "llm_only",
        metrics,
        latencies,
        cost_per_1000=cost_per_doc * 1000.0,
        n=n,
    )
    result["avg_input_tokens"] = avg_in
    result["avg_output_tokens"] = avg_out
    result["cost_per_doc"] = cost_per_doc
    result["subsample_n"] = n_attempted
    result["n_attempted"] = n_attempted
    result["n_cache_hits"] = n_cache_hits
    result["n_failed"] = n_failed
    result["n_quota_skipped"] = n_quota
    if n < n_attempted:
        print(
            f"  llm_only scored n={n}/{n_attempted} "
            f"(failed={n_failed}, quota_skipped={n_quota}, cache_hits={n_cache_hits})",
            flush=True,
        )
    return result


_last_llm_meta: dict = {"cache_hit": False, "input_tokens": 0.0, "output_tokens": 0.0, "latency_ms": 0.0}


def _patch_router_llm(cache: dict) -> None:
    """Point the Stage 5 router at the cached Gemini classifier (resume-safe)."""

    def _classify(text: str, labels: list[str]):
        hit = cached_classify(text, list(labels), cache)
        _last_llm_meta.update(
            {
                "cache_hit": hit["cache_hit"],
                "input_tokens": hit["input_tokens"],
                "output_tokens": hit["output_tokens"],
                "latency_ms": hit["latency_ms"],
            }
        )
        return hit["label"], hit["confidence"]

    router_mod.classify_with_llm = _classify


def run_router(
    texts: list[str],
    gold: list[str],
    doc_ids: list[str],
    cache: dict,
    pred_rows: list[dict],
) -> tuple[dict, dict]:
    """Classification-only Stage 5 path: `route_document`, no extraction."""
    if not gemini_configured():
        raise SystemExit(
            "GEMINI_API_KEY is not set. Router arm needs it for escalations."
        )
    _patch_router_llm(cache)

    predictions = []
    gold_kept = []
    latencies = []
    in_tokens = []
    out_tokens = []
    classify_tier = Counter()
    classify_attempted = Counter()
    n_attempted = len(texts)
    n_failed = 0
    n_quota = 0
    n_cache_hits = 0

    for i, (text, true_label, doc_id) in enumerate(zip(texts, gold, doc_ids)):
        start_routing_trace()
        started = time.perf_counter()
        try:
            classification = route_document(text)
        except LLMQuotaExceeded as err:
            n_quota += 1
            classify_attempted["llm"] += 1
            print(
                f"  router SKIP quota {i + 1}/{n_attempted} "
                f"doc_hash={doc_hash(text)} doc_id={doc_id} ({err})",
                flush=True,
            )
            continue
        except LLMCallFailed as err:
            n_failed += 1
            classify_attempted["llm"] += 1
            print(
                f"  router SKIP failed {i + 1}/{n_attempted} "
                f"doc_hash={doc_hash(text)} doc_id={doc_id} ({err})",
                flush=True,
            )
            continue

        wall_ms = (time.perf_counter() - started) * 1000.0
        trace = finish_routing_trace()
        tier = classification.tier
        classify_tier[tier] += 1
        classify_attempted[tier] += 1

        doc_in = doc_out = api_ms = 0.0
        cache_hit = False
        if "llm" in trace.tiers_used:
            doc_in = float(_last_llm_meta["input_tokens"])
            doc_out = float(_last_llm_meta["output_tokens"])
            api_ms = float(_last_llm_meta["latency_ms"])
            cache_hit = bool(_last_llm_meta["cache_hit"])
            if cache_hit:
                n_cache_hits += 1

        latency_ms = max(wall_ms, api_ms)
        latencies.append(latency_ms)
        predictions.append(classification.label)
        gold_kept.append(true_label)
        in_tokens.append(doc_in)
        out_tokens.append(doc_out)
        pred_rows.append(
            {
                "configuration": "router",
                "doc_id": doc_id,
                "doc_hash": doc_hash(text),
                "true_label": true_label,
                "predicted_label": classification.label,
                "confidence": classification.confidence,
                "tier": tier,
                "latency_ms": latency_ms,
                "input_tokens": doc_in,
                "output_tokens": doc_out,
                "cost_usd": tokens_to_usd(doc_in, doc_out),
                "cache_hit": cache_hit if tier == "llm" else False,
            }
        )

        if (i + 1) % 16 == 0 or i == 0:
            print(
                f"  router {i + 1}/{n_attempted}  tier={tier}  "
                f"completed={len(predictions)}  llm={classify_tier['llm']}  "
                f"failed={n_failed}  quota_skip={n_quota}",
                flush=True,
            )

    n = len(predictions)
    if n == 0:
        raise SystemExit("router completed 0 documents (all skipped).")
    metrics = compute_classification_metrics(gold_kept, predictions)
    avg_in = sum(in_tokens) / n
    avg_out = sum(out_tokens) / n
    cost_per_doc = tokens_to_usd(avg_in, avg_out)
    result = _pack(
        "router",
        metrics,
        latencies,
        cost_per_1000=cost_per_doc * 1000.0,
        n=n,
    )
    result["avg_input_tokens"] = avg_in
    result["avg_output_tokens"] = avg_out
    result["cost_per_doc"] = cost_per_doc
    result["n_attempted"] = n_attempted
    result["n_cache_hits"] = n_cache_hits
    result["n_failed"] = n_failed
    result["n_quota_skipped"] = n_quota
    if n < n_attempted:
        print(
            f"  router scored n={n}/{n_attempted} "
            f"(failed={n_failed}, quota_skipped={n_quota})",
            flush=True,
        )
    attempted_n = n_attempted or 1
    breakdown = {
        "n": n,
        "n_attempted": n_attempted,
        "baseline": classify_tier["baseline"],
        "transformer": classify_tier["transformer"],
        "llm": classify_tier["llm"],
        "pct_baseline": classify_tier["baseline"] / n,
        "pct_transformer": classify_tier["transformer"] / n,
        "pct_llm": classify_tier["llm"] / n,
        "n_failed": n_failed,
        "n_quota_skipped": n_quota,
        "attempted": {
            "baseline": classify_attempted["baseline"],
            "transformer": classify_attempted["transformer"],
            "llm": classify_attempted["llm"],
            "pct_baseline": classify_attempted["baseline"] / attempted_n,
            "pct_transformer": classify_attempted["transformer"] / attempted_n,
            "pct_llm": classify_attempted["llm"] / attempted_n,
        },
    }
    return result, breakdown


def _pack(name: str, metrics: dict, latencies: list[float], cost_per_1000: float, n: int) -> dict:
    return {
        "configuration": name,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "avg_latency_ms": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "cost_per_1000_docs": float(cost_per_1000),
        "n_docs": n,
    }


def write_table(rows: list[dict], path: Path = TABLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "configuration",
        "accuracy",
        "macro_f1",
        "avg_latency_ms",
        "cost_per_1000_docs",
        "n_docs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def write_escalation_csv(br: dict, path: Path = ESCALATION_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "tier", "count", "pct"])
        writer.writeheader()
        scored_n = br.get("n", 0) or 1
        for tier in ("baseline", "transformer", "llm"):
            count = int(br.get(tier, 0))
            writer.writerow(
                {"scope": "scored", "tier": tier, "count": count, "pct": count / scored_n}
            )
        attempted_n = br.get("n_attempted", scored_n) or 1
        attempted = br.get("attempted") or {
            "baseline": br.get("baseline", 0),
            "transformer": br.get("transformer", 0),
            "llm": br.get("llm", 0) + br.get("n_failed", 0) + br.get("n_quota_skipped", 0),
        }
        for tier in ("baseline", "transformer", "llm"):
            count = int(attempted.get(tier, 0))
            writer.writerow(
                {
                    "scope": "attempted",
                    "tier": tier,
                    "count": count,
                    "pct": count / attempted_n,
                }
            )


def write_predictions(rows: list[dict], path: Path = PREDICTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def plot_accuracy_vs_cost(rows: list[dict], path: Path = CHART_PATH) -> None:
    names = [row["configuration"] for row in rows]
    acc = [row["accuracy"] for row in rows]
    cost = [row["cost_per_1000_docs"] for row in rows]
    x = list(range(len(names)))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    bars1 = ax1.bar([i - width / 2 for i in x], acc, width, color="#3b6d9a", label="Accuracy")
    bars2 = ax2.bar(
        [i + width / 2 for i in x], cost, width, color="#c47b3b", label="Cost / 1k docs (USD)"
    )
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Estimated USD per 1,000 documents")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("DocRouter classification benchmark: accuracy vs. estimated cost")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _ = bars1, bars2


def plot_escalation(br: dict, path: Path = ESCALATION_CHART_PATH) -> None:
    labels = ["baseline", "transformer", "llm"]
    attempted = br.get("attempted")
    if attempted:
        pcts = [float(attempted.get(f"pct_{name}", 0.0)) * 100.0 for name in labels]
        title_n = br.get("n_attempted", br.get("n", 0))
        title = f"Router classification escalation, attempted (n={title_n})"
    else:
        pcts = [br.get(f"pct_{name}", 0.0) * 100.0 for name in labels]
        title = f"Router classification escalation (n={br.get('n', 0)})"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4c7c5f", "#3b6d9a", "#c47b3b"]
    ax.bar(labels, pcts, color=colors)
    ax.set_ylabel("% of documents")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    for i, pct in enumerate(pcts):
        ax.text(i, pct + 1.5, f"{pct:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def escalation_from_stage5_logs(path: Path = LOG_PATH) -> dict:
    """Classification-tier mix from Stage 5 `logs/routing_log.jsonl`."""
    empty = {
        "n": 0,
        "baseline": 0,
        "transformer": 0,
        "llm": 0,
        "pct_baseline": 0.0,
        "pct_transformer": 0.0,
        "pct_llm": 0.0,
    }
    if not path.exists():
        return empty
    n = base = trans = llm = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tiers = row.get("tiers_used") or []
            n += 1
            if "llm" in tiers:
                llm += 1
            elif "transformer" in tiers:
                trans += 1
            elif "baseline" in tiers:
                base += 1
    if n == 0:
        return empty
    return {
        "n": n,
        "baseline": base,
        "transformer": trans,
        "llm": llm,
        "pct_baseline": base / n,
        "pct_transformer": trans / n,
        "pct_llm": llm / n,
    }


def print_cost_arithmetic(llm_row: dict | None, router_row: dict | None) -> None:
    print()
    print("=== Cost arithmetic (Gemini 3.1 Flash-Lite, paid list price) ===")
    print(f"Model: {BENCHMARK_GEMINI_MODEL} (runtime patch; Stage 5 default is {GEMINI_MODEL})")
    print("Source: Google Gemini API paid list price, ai.google.dev/gemini-api/docs/pricing")
    print(f"  USD_PER_MILLION_INPUT  = ${USD_PER_MILLION_INPUT:.2f}")
    print(f"  USD_PER_MILLION_OUTPUT = ${USD_PER_MILLION_OUTPUT:.2f} (includes thinking tokens if billed)")
    print("Token estimate (same as Stage 5): tokens = chars / 4")
    print(
        f"cost_per_doc = (avg_input_tokens / 1e6)*{USD_PER_MILLION_INPUT} "
        f"+ (avg_output_tokens / 1e6)*{USD_PER_MILLION_OUTPUT}"
    )
    print("cost_per_1000 = cost_per_doc * 1000")
    print("baseline_only / transformer_only: $0.00 (inference after training treated as free).")
    print("Classification tokens only; extraction is not included in any cost figure.")
    if llm_row and "avg_input_tokens" in llm_row:
        print()
        print(
            f"llm_only (scored n={llm_row['n_docs']}, attempted={llm_row.get('subsample_n', llm_row['n_docs'])}): "
            f"avg_in={llm_row['avg_input_tokens']:.1f} tok/doc, "
            f"avg_out={llm_row['avg_output_tokens']:.1f} tok/doc"
        )
        print(
            f"  cost/doc = ({llm_row['avg_input_tokens']:.1f}/1e6)*{USD_PER_MILLION_INPUT} "
            f"+ ({llm_row['avg_output_tokens']:.1f}/1e6)*{USD_PER_MILLION_OUTPUT} "
            f"= ${llm_row['cost_per_doc']:.6f}"
        )
        print(f"  cost/1000 = ${llm_row['cost_per_1000_docs']:.4f}")
    if router_row and "avg_input_tokens" in router_row:
        print()
        print(
            f"router (scored n={router_row['n_docs']}, attempted={router_row.get('n_attempted', router_row['n_docs'])}): "
            f"avg_in={router_row['avg_input_tokens']:.1f} tok/doc, "
            f"avg_out={router_row['avg_output_tokens']:.1f} tok/doc"
        )
        print(
            f"  cost/doc = ({router_row['avg_input_tokens']:.1f}/1e6)*{USD_PER_MILLION_INPUT} "
            f"+ ({router_row['avg_output_tokens']:.1f}/1e6)*{USD_PER_MILLION_OUTPUT} "
            f"= ${router_row['cost_per_doc']:.6f}"
        )
        print(f"  cost/1000 = ${router_row['cost_per_1000_docs']:.4f}")


def print_escalation(title: str, br: dict) -> None:
    print()
    print(f"=== {title} ===")
    n = br.get("n", 0)
    if n == 0:
        print("  (no records)")
        return
    attempted = br.get("n_attempted", n)
    print(f"  scored n = {n}" + (f"  (attempted {attempted})" if attempted != n else ""))
    print("  scored mix (denominator = completed classifications):")
    print(
        f"    resolved at baseline:     {br['baseline']:4d}  {br['pct_baseline']:6.1%}"
    )
    print(
        f"    escalated to transformer: {br['transformer']:4d}  {br['pct_transformer']:6.1%}"
    )
    print(f"    escalated to LLM:         {br['llm']:4d}  {br['pct_llm']:6.1%}")
    attempted_mix = br.get("attempted")
    if attempted_mix and attempted:
        print("  attempted mix (denominator = all docs the router saw):")
        print(
            f"    resolved at baseline:     {attempted_mix['baseline']:4d}  "
            f"{attempted_mix['baseline'] / attempted:6.1%}"
        )
        print(
            f"    escalated to transformer: {attempted_mix['transformer']:4d}  "
            f"{attempted_mix['transformer'] / attempted:6.1%}"
        )
        print(
            f"    reached LLM gate:         {attempted_mix['llm']:4d}  "
            f"{attempted_mix['llm'] / attempted:6.1%}"
        )
    if br.get("n_failed") or br.get("n_quota_skipped"):
        print(
            f"  skipped (failed / quota): {br.get('n_failed', 0):4d} / "
            f"{br.get('n_quota_skipped', 0):4d}  (not in accuracy/cost)"
        )


def print_table(rows: list[dict]) -> None:
    print()
    print("=== benchmark_table.csv ===")
    print(
        f"{'configuration':<18} {'accuracy':>10} {'macro_f1':>10} "
        f"{'avg_ms':>12} {'usd_per_1k':>14} {'n':>6}"
    )
    print("-" * 76)
    for row in rows:
        print(
            f"{row['configuration']:<18} {row['accuracy']:10.4f} {row['macro_f1']:10.4f} "
            f"{row['avg_latency_ms']:12.1f} {row['cost_per_1000_docs']:14.4f} {row['n_docs']:6d}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6 classification-only routing benchmark")
    parser.add_argument(
        "--llm-n",
        type=int,
        default=LLM_SUBSAMPLE_N,
        help=f"LLM-only subsample size (default {LLM_SUBSAMPLE_N})",
    )
    parser.add_argument(
        "--router-n",
        type=int,
        default=None,
        help="Cap the router arm at N documents (default: full test set).",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip llm_only and router (local arms only). Use if you do not want API spend.",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Reuse baseline_only / transformer_only rows already in the CSV.",
    )
    parser.add_argument(
        "--rerun-local",
        action="store_true",
        help="Recompute baseline_only / transformer_only even if CSV rows exist.",
    )
    parser.add_argument(
        "--router-only",
        action="store_true",
        help=(
            "Reuse baseline_only, transformer_only, and llm_only rows; re-run "
            "only the router arm (threshold retunes)."
        ),
    )
    return parser.parse_args(argv)


def _local_rows_from_csv(n_docs: int) -> list[dict]:
    if not TABLE_PATH.exists():
        raise SystemExit(f"--skip-local requires {TABLE_PATH}")
    wanted = {"baseline_only", "transformer_only"}
    found = []
    with TABLE_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("configuration") in wanted:
                csv_n = int(float(row["n_docs"])) if row.get("n_docs") else n_docs
                found.append(
                    {
                        "configuration": row["configuration"],
                        "accuracy": float(row["accuracy"]),
                        "macro_f1": float(row["macro_f1"]),
                        "avg_latency_ms": float(row["avg_latency_ms"]),
                        "cost_per_1000_docs": float(row["cost_per_1000_docs"]),
                        "n_docs": csv_n,
                    }
                )
    names = {row["configuration"] for row in found}
    if wanted - names:
        raise SystemExit(f"--skip-local missing rows {wanted - names} in {TABLE_PATH}")
    order = {"baseline_only": 0, "transformer_only": 1}
    return sorted(found, key=lambda row: order[row["configuration"]])


def _csv_has_local_rows() -> bool:
    if not TABLE_PATH.exists():
        return False
    try:
        with TABLE_PATH.open(encoding="utf-8", newline="") as handle:
            names = {row.get("configuration") for row in csv.DictReader(handle)}
        return {"baseline_only", "transformer_only"} <= names
    except Exception:
        return False


def _row_from_details(name: str) -> dict | None:
    if not DETAILS_PATH.exists():
        return None
    try:
        payload = json.loads(DETAILS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    for row in payload.get("rows") or []:
        if row.get("configuration") == name:
            return dict(row)
    return None


def _csv_row(name: str, n_docs: int) -> dict | None:
    if not TABLE_PATH.exists():
        return None
    with TABLE_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("configuration") == name:
                csv_n = int(float(row["n_docs"])) if row.get("n_docs") else n_docs
                return {
                    "configuration": name,
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "avg_latency_ms": float(row["avg_latency_ms"]),
                    "cost_per_1000_docs": float(row["cost_per_1000_docs"]),
                    "n_docs": csv_n,
                }
    return None


def _llm_only_row(n_docs: int) -> dict:
    row = _row_from_details("llm_only") or _csv_row("llm_only", n_docs)
    if row is None:
        raise SystemExit(
            "--router-only requires an existing llm_only row in "
            f"{TABLE_PATH} or {DETAILS_PATH}"
        )
    return row


def _predictions_for(configuration: str, path: Path = PREDICTIONS_PATH) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("configuration") == configuration:
                rows.append(row)
    return rows


def run_benchmark(
    llm_n: int = LLM_SUBSAMPLE_N,
    skip_llm: bool = False,
    skip_local: bool = False,
    rerun_local: bool = False,
    router_n: int | None = None,
    router_only: bool = False,
) -> list[dict]:
    texts, gold, doc_ids = load_test_split()
    print(f"Held-out test set: {len(texts)} documents from {CSV_PATH}")
    print(f"Classification gates: THRESHOLD_1={THRESHOLD_1}, THRESHOLD_2={THRESHOLD_2}")
    print("This benchmark is classification-only. Extraction is not measured.")
    print("Loading classifiers...")
    load_classifiers()
    labels = candidate_labels()
    install_token_probe()
    install_benchmark_llm()
    cache = load_cache()
    n_lite_cls = sum(
        1 for key in cache if key.startswith(f"{BENCHMARK_GEMINI_MODEL}:classify:")
    )
    print(f"LLM cache entries: {len(cache)} ({n_lite_cls} {BENCHMARK_GEMINI_MODEL} classify)")
    print(
        f"Benchmark Gemini model: {BENCHMARK_GEMINI_MODEL} "
        f"(Stage 5 file still says {GEMINI_MODEL}; patched at runtime)"
    )
    print(
        "Cache policy: reuse only "
        f"`{BENCHMARK_GEMINI_MODEL}:classify:<hash>`; other model prefixes are ignored."
    )

    if skip_llm and router_only:
        raise SystemExit("Use --router-only or --skip-llm, not both.")

    reuse_local = (
        skip_local or router_only or _csv_has_local_rows()
    ) and not rerun_local
    rows: list[dict] = []
    pred_rows: list[dict] = []

    if reuse_local:
        print("\n--- baseline_only / transformer_only (reused from CSV; not recomputed) ---")
        rows.extend(_local_rows_from_csv(len(texts)))
        for row in rows:
            print(
                f"  {row['configuration']}: acc={row['accuracy']:.4f}  "
                f"macro-F1={row['macro_f1']:.4f}  avg={row['avg_latency_ms']:.1f} ms  "
                f"n={row['n_docs']}"
            )
    else:
        print("\n--- baseline_only ---")
        base_row = run_baseline_only(texts, gold)
        rows.append(base_row)
        write_table(rows)
        print(
            f"  acc={base_row['accuracy']:.4f}  macro-F1={base_row['macro_f1']:.4f}  "
            f"avg={base_row['avg_latency_ms']:.1f} ms"
        )

        print("\n--- transformer_only ---")
        trans_row = run_transformer_only(texts, gold)
        rows.append(trans_row)
        write_table(rows)
        print(
            f"  acc={trans_row['accuracy']:.4f}  macro-F1={trans_row['macro_f1']:.4f}  "
            f"avg={trans_row['avg_latency_ms']:.1f} ms"
        )

    llm_row = None
    router_row = None
    router_breakdown = None
    if skip_llm:
        print("\n--skip-llm: not calling Gemini. llm_only and router omitted.")
    else:
        if router_only:
            print("\n--- llm_only (reused from previous run; not recomputed) ---")
            llm_row = _llm_only_row(len(texts))
            rows.append(llm_row)
            write_table(rows)
            pred_rows.extend(_predictions_for("llm_only"))
            print(
                f"  {llm_row['configuration']}: acc={llm_row['accuracy']:.4f}  "
                f"macro-F1={llm_row['macro_f1']:.4f}  "
                f"avg={llm_row['avg_latency_ms']:.1f} ms  "
                f"n={llm_row['n_docs']}"
            )
        else:
            sub_texts, sub_gold, sub_idx = stratified_subsample(texts, gold, n=llm_n)
            sub_ids = [doc_ids[i] for i in sub_idx]
            print("\n--- llm_only (classification only) ---")
            llm_row = run_llm_only(sub_texts, sub_gold, sub_ids, labels, cache, pred_rows)
            rows.append(llm_row)
            write_table(rows)
            print(
                f"  acc={llm_row['accuracy']:.4f}  macro-F1={llm_row['macro_f1']:.4f}  "
                f"avg={llm_row['avg_latency_ms']:.1f} ms  "
                f"cost/1k=${llm_row['cost_per_1000_docs']:.4f}  n={llm_row['n_docs']}"
            )

        router_texts, router_gold, router_ids = texts, gold, doc_ids
        if router_n is not None and router_n < len(texts):
            router_texts = texts[:router_n]
            router_gold = gold[:router_n]
            router_ids = doc_ids[:router_n]
            print(
                f"\n--- router (classification only, first {router_n} of {len(texts)} test docs) ---"
            )
        else:
            print(f"\n--- router (classification only, full test set n={len(texts)}) ---")
        router_row, router_breakdown = run_router(
            router_texts, router_gold, router_ids, cache, pred_rows
        )
        rows.append(router_row)
        print(
            f"  acc={router_row['accuracy']:.4f}  macro-F1={router_row['macro_f1']:.4f}  "
            f"avg={router_row['avg_latency_ms']:.1f} ms  "
            f"cost/1k=${router_row['cost_per_1000_docs']:.4f}  n={router_row['n_docs']}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_table(rows)
    plot_accuracy_vs_cost(rows)
    write_predictions(pred_rows)
    print(f"\nWrote {TABLE_PATH}")
    print(f"Wrote {CHART_PATH}")
    if pred_rows:
        print(f"Wrote {PREDICTIONS_PATH}")

    print_table(rows)
    print_cost_arithmetic(llm_row, router_row)

    log_br = escalation_from_stage5_logs()
    print_escalation("Escalation breakdown (Stage 5 logs/routing_log.jsonl)", log_br)
    if log_br["n"] < 20:
        print(
            "  Note: Stage 5 logs are a smoke-test sample, not the held-out set. "
            "The router arm of this benchmark is the statistically meaningful mix."
        )
    if router_breakdown:
        print_escalation(
            "Escalation breakdown (benchmark router arm, classification only)",
            router_breakdown,
        )
        write_escalation_csv(router_breakdown)
        plot_escalation(router_breakdown)
        print(f"Wrote {ESCALATION_CSV_PATH}")
        print(f"Wrote {ESCALATION_CHART_PATH}")

    details = {
        "model": BENCHMARK_GEMINI_MODEL,
        "stage5_model": GEMINI_MODEL,
        "task": "classification_only",
        "usd_per_million_input": USD_PER_MILLION_INPUT,
        "usd_per_million_output": USD_PER_MILLION_OUTPUT,
        "llm_subsample_n": None if llm_row is None else llm_row.get("subsample_n"),
        "router_n_attempted": None if router_row is None else router_row.get("n_attempted"),
        "rows": [
            {k: row[k] for k in (
                "configuration",
                "accuracy",
                "macro_f1",
                "avg_latency_ms",
                "cost_per_1000_docs",
                "n_docs",
            ) if k in row}
            | {k: row[k] for k in (
                "avg_input_tokens",
                "avg_output_tokens",
                "cost_per_doc",
                "subsample_n",
                "n_attempted",
                "n_cache_hits",
                "n_failed",
                "n_quota_skipped",
            ) if k in row}
            for row in rows
        ],
        "stage5_log_escalation": log_br,
        "router_escalation": router_breakdown,
    }
    DETAILS_PATH.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"Wrote {DETAILS_PATH}")

    return rows


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    if args.llm_n < 1:
        raise SystemExit("--llm-n must be positive")
    if args.router_n is not None and args.router_n < 1:
        raise SystemExit("--router-n must be positive")
    if args.skip_llm and args.router_only:
        raise SystemExit("Use --router-only or --skip-llm, not both.")
    run_benchmark(
        llm_n=args.llm_n,
        skip_llm=args.skip_llm,
        skip_local=args.skip_local,
        rerun_local=args.rerun_local,
        router_n=args.router_n,
        router_only=args.router_only,
    )


if __name__ == "__main__":
    main()
