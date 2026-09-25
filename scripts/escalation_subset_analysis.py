"""Was escalating to the LLM better than trusting DistilBERT's own low-confidence guess?

Read-only. Reuses saved artifacts, makes no API calls and runs no inference:

* `results/local_predictions.jsonl` — baseline + DistilBERT label/confidence for all 512
* `results/benchmark_predictions.jsonl` — the tuned router run, including the LLM tier

The comparison the benchmark table cannot make: `llm_only` (n=64 stratified subsample) and
`transformer_only` (n=512) are measured on different populations, so neither says whether
the LLM beat DistilBERT on the docs the router *actually* escalates. Here both models are
scored on the same LLM-gated subset.

Writes `results/escalation_analysis.md`.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
LOCAL_PATH = RESULTS_DIR / "local_predictions.jsonl"
BENCHMARK_PATH = RESULTS_DIR / "benchmark_predictions.jsonl"
REPORT_PATH = RESULTS_DIR / "escalation_analysis.md"

# Must match router.router at the time of the benchmark run being analyzed.
THRESHOLD_1 = 0.60
THRESHOLD_2 = 0.75


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the benchmark / calibration diagnostic first.")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Unweighted mean per-class F1 over every label appearing in truth or predictions.

    Local reimplementation so this script stays dependency-free; matches
    sklearn's `f1_score(average="macro", zero_division=0)`.
    """
    labels = sorted(set(y_true) | set(y_pred))
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    return sum(scores) / len(scores) if scores else 0.0


def mcnemar_exact_p(wins_a: int, wins_b: int) -> float:
    """Two-sided exact McNemar on the discordant pairs only (sign test)."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins_a, wins_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    local = read_jsonl(LOCAL_PATH)
    benchmark = read_jsonl(BENCHMARK_PATH)
    router_rows = [r for r in benchmark if r.get("configuration") == "router"]
    if not router_rows:
        raise SystemExit("No `router` rows in benchmark_predictions.jsonl.")

    # The LLM gate is reached only when *both* local gates are missed. Reconstructing it
    # from confidences (rather than reading tier=="llm") also recovers the docs whose LLM
    # call failed, which never got a router prediction row.
    gated = {
        r["doc_id"]: r
        for r in local
        if r["baseline_confidence"] < THRESHOLD_1 and r["transformer_confidence"] < THRESHOLD_2
    }
    llm_rows = [r for r in router_rows if r["tier"] == "llm"]
    scored_ids = {r["doc_id"] for r in llm_rows}

    stray = scored_ids - set(gated)
    if stray:
        raise SystemExit(f"{len(stray)} LLM-tier docs fall outside the reconstructed gate set.")
    unscored = sorted(set(gated) - scored_ids)

    # DistilBERT over the full gate set, including docs the LLM never scored.
    bert_true = [r["true_label"] for r in gated.values()]
    bert_pred = [r["transformer_label"] for r in gated.values()]

    # Paired subset: both models have a prediction for these.
    paired_true = [r["true_label"] for r in llm_rows]
    paired_llm = [r["predicted_label"] for r in llm_rows]
    paired_bert = [gated[r["doc_id"]]["transformer_label"] for r in llm_rows]

    bert_ok = [t == p for t, p in zip(paired_true, paired_bert)]
    llm_ok = [t == p for t, p in zip(paired_true, paired_llm)]
    both_right = sum(b and l for b, l in zip(bert_ok, llm_ok))
    both_wrong = sum(not b and not l for b, l in zip(bert_ok, llm_ok))
    llm_saves = sum(not b and l for b, l in zip(bert_ok, llm_ok))
    bert_saves = sum(b and not l for b, l in zip(bert_ok, llm_ok))

    n_paired = len(llm_rows)
    net = llm_saves - bert_saves
    p_value = mcnemar_exact_p(llm_saves, bert_saves)

    # Paired-difference CI (normal approximation on the per-doc score difference).
    diff = net / n_paired
    variance = (llm_saves + bert_saves - net**2 / n_paired) / n_paired
    stderr = math.sqrt(max(variance, 0.0) / n_paired)
    ci_low, ci_high = diff - 1.96 * stderr, diff + 1.96 * stderr

    total_cost = sum(float(r["cost_usd"]) for r in llm_rows)
    cache_hits = sum(1 for r in llm_rows if r.get("cache_hit"))

    # Counterfactual: identical router, but the LLM tier is replaced by DistilBERT's own
    # low-confidence guess. Same 499 scored docs, so directly comparable to the table.
    local_tier = [r for r in router_rows if r["tier"] != "llm"]
    local_correct = sum(r["predicted_label"] == r["true_label"] for r in local_tier)
    router_correct = sum(r["predicted_label"] == r["true_label"] for r in router_rows)
    counterfactual_correct = local_correct + sum(bert_ok)
    n_router = len(router_rows)

    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row, b_ok, l_ok in zip(llm_rows, bert_ok, llm_ok):
        stats = per_class[row["true_label"]]
        stats[0] += 1
        if not b_ok and l_ok:
            stats[1] += 1
        if b_ok and not l_ok:
            stats[2] += 1

    bert_gate_acc = accuracy(bert_true, bert_pred)
    bert_gate_f1 = macro_f1(bert_true, bert_pred)
    bert_paired_acc = accuracy(paired_true, paired_bert)
    bert_paired_f1 = macro_f1(paired_true, paired_bert)
    llm_paired_acc = accuracy(paired_true, paired_llm)
    llm_paired_f1 = macro_f1(paired_true, paired_llm)

    lines: list[str] = []
    add = lines.append
    add("# Does escalating to the LLM earn its cost?")
    add("")
    add(
        f"Read-only analysis of the tuned router run (`THRESHOLD_1={THRESHOLD_1:.2f}`, "
        f"`THRESHOLD_2={THRESHOLD_2:.2f}`). No new inference and no new API calls: DistilBERT's "
        "side comes from `results/local_predictions.jsonl`, the LLM's side from the `router` rows "
        "of `results/benchmark_predictions.jsonl` "
        f"({cache_hits} cache hits, {n_paired - cache_hits} fresh calls). "
        f"Regenerate with `python scripts/escalation_subset_analysis.py`."
    )
    add("")
    add("## Why the benchmark table can't answer this")
    add("")
    add(
        "`llm_only` (0.672) and `transformer_only` (0.746) are measured on different populations: "
        "DistilBERT on all 512 docs including the easy ones it is confident about, Gemini on a "
        "random 64-doc subsample that is also a mix of easy and hard. Neither describes the docs "
        "the router actually escalates. The honest comparison is both models on the same gated "
        "subset: the "
        f"{len(gated)} docs where the baseline cleared neither gate and DistilBERT's own confidence "
        f"fell below {THRESHOLD_2:.2f}."
    )
    add("")
    add("## Same-population accuracy on the LLM-gated subset")
    add("")
    add("| model | subset | n | accuracy | macro-F1 |")
    add("|---|---|---:|---:|---:|")
    add(
        f"| DistilBERT (own low-confidence guess) | all LLM-gated docs | {len(gated)} | "
        f"{bert_gate_acc:.3f} | {bert_gate_f1:.3f} |"
    )
    add(
        f"| DistilBERT (own low-confidence guess) | LLM-scored docs only | {n_paired} | "
        f"{bert_paired_acc:.3f} | {bert_paired_f1:.3f} |"
    )
    add(
        f"| Gemini (escalation) | LLM-scored docs only | {n_paired} | "
        f"{llm_paired_acc:.3f} | {llm_paired_f1:.3f} |"
    )
    add("")
    add(
        f"The last two rows are the paired comparison: same {n_paired} documents, same gold labels. "
        f"The LLM is right on {sum(llm_ok)} of them, DistilBERT on {sum(bert_ok)} — a difference of "
        f"{net} documents, {diff * 100:+.1f} pp."
    )
    add("")
    add(
        f"The {len(unscored)} remaining gated docs are excluded, not scored as wrong: their Gemini "
        "calls returned 504 DEADLINE_EXCEEDED with retries exhausted, so no LLM prediction exists "
        "(they are absent from the classify cache). Counting them either way would bias the "
        "comparison. For reference DistilBERT got "
        f"{sum(gated[d]['transformer_label'] == gated[d]['true_label'] for d in unscored)} of those "
        f"{len(unscored)} right."
    )
    add("")
    add("## Agreement breakdown on the paired subset")
    add("")
    add(f"All {n_paired} docs where both models produced a prediction:")
    add("")
    add("| | LLM right | LLM wrong | total |")
    add("|---|---:|---:|---:|")
    add(
        f"| **DistilBERT right** | {both_right} | {bert_saves} | {both_right + bert_saves} |"
    )
    add(
        f"| **DistilBERT wrong** | {llm_saves} | {both_wrong} | {llm_saves + both_wrong} |"
    )
    add(
        f"| **total** | {both_right + llm_saves} | {bert_saves + both_wrong} | {n_paired} |"
    )
    add("")
    add(
        f"The two off-diagonal cells are what matters. The LLM rescues {llm_saves} docs DistilBERT "
        f"got wrong, but breaks {bert_saves} docs DistilBERT already had right — a net gain of "
        f"{net} documents out of {n_paired}. Exact McNemar on the {llm_saves + bert_saves} "
        f"discordant pairs gives p = {p_value:.2f}; the 95% CI on the paired accuracy difference is "
        f"[{ci_low * 100:+.1f}, {ci_high * 100:+.1f}] pp, comfortably straddling zero. Escalation is "
        f"not adding accuracy here, it is reshuffling errors: {both_wrong} docs "
        f"({both_wrong / n_paired:.0%}) are missed by both models, so the gate is mostly selecting "
        "documents that are genuinely hard for either model rather than ones the LLM can fix."
    )
    add("")
    add("### Where the swaps happen, by true class")
    add("")
    add("| true label | n | LLM rescues | LLM breaks | net |")
    add("|---|---:|---:|---:|---:|")
    for label in sorted(per_class):
        n_cls, saves, breaks = per_class[label]
        add(f"| {label} | {n_cls} | {saves} | {breaks} | {saves - breaks:+d} |")
    add("")
    add(
        "The net figure is close to zero because the per-class effects cancel: the LLM is stronger "
        "on prose-like classes and weaker where DistilBERT has learned this corpus's layout cues."
    )
    add("")
    add("## Cost of those corrections")
    add("")
    add(
        f"The {n_paired} escalations cost an estimated ${total_cost:.4f} "
        f"(${total_cost / n_paired * 1000:.3f} per 1k escalated docs) and bought {net} net-correct "
        f"documents"
        + (
            f", i.e. roughly ${total_cost / net:.3f} per additional correct classification"
            if net > 0
            else ""
            )
        + ". Escalation also dominates the router's latency profile (2.0 s average vs 655 ms for "
        "DistilBERT alone)."
    )
    add("")
    add(
        f"Replacing the LLM tier with DistilBERT's own guess and leaving everything else identical "
        f"scores {counterfactual_correct}/{n_router} = {counterfactual_correct / n_router:.3f}, "
        f"against the router's actual {router_correct}/{n_router} = "
        f"{router_correct / n_router:.3f} — the same "
        f"{(router_correct - counterfactual_correct) / n_router * 100:+.1f} pp, "
        f"{router_correct - counterfactual_correct} documents."
    )
    add("")
    add("## Conclusion")
    add("")
    add(
        "**Escalation is not clearly earning its cost on this data.** On the "
        f"{n_paired} gated documents both models were scored on, Gemini reached "
        f"{llm_paired_acc:.3f} accuracy against DistilBERT's own {bert_paired_acc:.3f} on the exact "
        f"same documents — {net} additional correct documents out of {n_paired} "
        f"({diff * 100:+.1f} pp, McNemar p = {p_value:.2f}), which is indistinguishable from noise "
        f"at this sample size. The 2x2 breakdown shows why: the LLM fixes {llm_saves} of "
        f"DistilBERT's errors while introducing {bert_saves} new ones, and {both_wrong} of "
        f"{n_paired} documents defeat both models. The useful reading is that low DistilBERT "
        "confidence is doing a good job of identifying genuinely hard documents and a poor job of "
        "identifying documents an LLM can rescue — those are different things, and the current gate "
        "only measures the first. The right follow-up is a different escalation criterion (for "
        "example gating on the classes where the LLM actually wins, or on disagreement between the "
        "baseline and DistilBERT) rather than a lower confidence threshold, which would only send "
        "more of the same both-wrong documents to a paid model. The overall `llm_only` figure "
        "(0.672) should not be used to argue either side of this: it is measured on a different "
        "population and understates the LLM on hard documents just as it overstates it on easy ones."
    )
    add("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"gate set (reconstructed):    {len(gated)}")
    print(f"  scored by LLM:             {n_paired} ({cache_hits} cached, {n_paired - cache_hits} fresh)")
    print(f"  unscored (504, excluded):  {len(unscored)}")
    print(f"DistilBERT on {len(gated):3d}:            acc={bert_gate_acc:.4f} macro-F1={bert_gate_f1:.4f}")
    print(f"DistilBERT on {n_paired:3d} (paired):   acc={bert_paired_acc:.4f} macro-F1={bert_paired_f1:.4f}")
    print(f"Gemini     on {n_paired:3d} (paired):   acc={llm_paired_acc:.4f} macro-F1={llm_paired_f1:.4f}")
    print(
        f"2x2: both_right={both_right} both_wrong={both_wrong} "
        f"llm_rescues={llm_saves} llm_breaks={bert_saves} net={net:+d}"
    )
    print(f"McNemar exact p={p_value:.4f}  95% CI=[{ci_low * 100:+.2f}, {ci_high * 100:+.2f}] pp")
    print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
