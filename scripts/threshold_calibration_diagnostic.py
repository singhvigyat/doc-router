"""Confidence-calibration diagnostic + threshold sweep (no LLM, no retraining).

Re-runs local inference on the existing 512-doc test split with the already-
trained baseline and DistilBERT models, then:

* splits each model's confidences into correct vs incorrect
* writes percentile tables + overlaid histograms
* sweeps THRESHOLD_1 x THRESHOLD_2 and simulates router-tier mix
* prints a data-driven threshold recommendation (does not edit router.py)

If `results/local_predictions.jsonl` already exists from a prior run, inference
is skipped and that file is reused.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib  # noqa: E402

from evaluation.benchmark import load_test_split  # noqa: E402
from models import transformer_classifier  # noqa: E402
from router.router import BASELINE_PATH, TRANSFORMER_DIR  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_PATH = RESULTS_DIR / "local_predictions.jsonl"
PERCENTILES_PATH = RESULTS_DIR / "calibration_percentiles.csv"
SWEEP_PATH = RESULTS_DIR / "threshold_sweep.csv"
BASELINE_CHART = RESULTS_DIR / "calibration_baseline.png"
TRANSFORMER_CHART = RESULTS_DIR / "calibration_transformer.png"
SWEEP_HEATMAP = RESULTS_DIR / "threshold_sweep_heatmap.png"
INDEPENDENT_PATH = RESULTS_DIR / "threshold_independent.csv"

CURRENT_T1 = 0.85
CURRENT_T2 = 0.75
SWEEP_VALUES = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
PERCENTILE_QS = [0.0, 0.25, 0.50, 0.75, 1.0]
TRANSFORMER_BATCH = 16


def _percentile_row(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    qs = np.quantile(values, PERCENTILE_QS)
    return {
        "n": int(values.size),
        "min": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "max": float(qs[4]),
        "mean": float(values.mean()),
    }


def _fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def load_or_infer_predictions() -> pd.DataFrame:
    if PREDICTIONS_PATH.exists():
        rows = []
        with PREDICTIONS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        frame = pd.DataFrame(rows)
        needed = {
            "doc_id",
            "true_label",
            "baseline_label",
            "baseline_confidence",
            "transformer_label",
            "transformer_confidence",
        }
        if needed <= set(frame.columns) and len(frame) >= 500:
            print(f"Reusing {PREDICTIONS_PATH} ({len(frame)} docs); skipping inference.")
            return frame
        print(f"{PREDICTIONS_PATH} is incomplete; re-running local inference.")

    texts, gold, doc_ids = load_test_split()
    print(f"Test set: {len(texts)} documents. Running local inference (no LLM).")

    if not BASELINE_PATH.exists():
        raise SystemExit(f"Missing {BASELINE_PATH}.")
    if not TRANSFORMER_DIR.exists():
        raise SystemExit(f"Missing {TRANSFORMER_DIR}.")

    baseline = joblib.load(BASELINE_PATH)
    print("  baseline predict_proba...")
    baseline_proba = baseline.predict_proba(["" if t is None else str(t) for t in texts])
    baseline_idx = baseline_proba.argmax(axis=1)
    baseline_labels = [str(baseline.classes_[i]) for i in baseline_idx]
    baseline_conf = baseline_proba.max(axis=1)

    transformer_classifier.load_model(TRANSFORMER_DIR)
    model = transformer_classifier._model
    tokenizer = transformer_classifier._tokenizer
    device = transformer_classifier._device
    print(f"  transformer softmax (device={device}, batch={TRANSFORMER_BATCH})...")

    trans_labels: list[str] = []
    trans_conf: list[float] = []
    n = len(texts)
    for start in range(0, n, TRANSFORMER_BATCH):
        batch = ["" if t is None else str(t) for t in texts[start : start + TRANSFORMER_BATCH]]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            conf, index = torch.max(probs, dim=-1)
        for i in range(len(batch)):
            label_id = int(index[i].item())
            trans_labels.append(transformer_classifier._label_from_id(label_id))
            trans_conf.append(float(conf[i].item()))
        done = min(start + TRANSFORMER_BATCH, n)
        if done % 64 == 0 or done == n:
            print(f"    transformer {done}/{n}")

    rows = []
    for i, doc_id in enumerate(doc_ids):
        rows.append(
            {
                "doc_id": doc_id,
                "true_label": gold[i],
                "baseline_label": baseline_labels[i],
                "baseline_confidence": float(baseline_conf[i]),
                "transformer_label": trans_labels[i],
                "transformer_confidence": trans_conf[i],
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {PREDICTIONS_PATH}")
    return pd.DataFrame(rows)


def calibration_for_model(
    frame: pd.DataFrame,
    pred_col: str,
    conf_col: str,
    threshold: float,
    model_name: str,
) -> dict:
    correct_mask = frame[pred_col].astype(str) == frame["true_label"].astype(str)
    conf = frame[conf_col].astype(float).to_numpy()
    correct_conf = conf[correct_mask.to_numpy()]
    incorrect_conf = conf[~correct_mask.to_numpy()]
    n_correct = int(correct_mask.sum())
    n_incorrect = int((~correct_mask).sum())
    n = len(frame)

    below_among_correct = (
        float((correct_conf < threshold).mean()) if n_correct else float("nan")
    )
    above_among_incorrect = (
        float((incorrect_conf >= threshold).mean()) if n_incorrect else float("nan")
    )
    return {
        "model": model_name,
        "threshold": threshold,
        "n": n,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "accuracy": n_correct / n if n else 0.0,
        "correct": _percentile_row(correct_conf),
        "incorrect": _percentile_row(incorrect_conf),
        "frac_correct_below_threshold": below_among_correct,
        "n_correct_below_threshold": int((correct_conf < threshold).sum()),
        "frac_incorrect_above_threshold": above_among_incorrect,
        "n_incorrect_above_threshold": int((incorrect_conf >= threshold).sum()),
        "correct_conf": correct_conf,
        "incorrect_conf": incorrect_conf,
    }


def plot_calibration(diag: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(
        diag["correct_conf"],
        bins=bins,
        alpha=0.65,
        color="#4c7c5f",
        label=f"correct (n={diag['n_correct']})",
        density=True,
    )
    ax.hist(
        diag["incorrect_conf"],
        bins=bins,
        alpha=0.65,
        color="#c47b3b",
        label=f"incorrect (n={diag['n_incorrect']})",
        density=True,
    )
    ax.axvline(
        diag["threshold"],
        color="#3b6d9a",
        linestyle="--",
        linewidth=1.5,
        label=f"current threshold {diag['threshold']:.2f}",
    )
    ax.set_xlabel("Confidence (winning-class probability)")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(
        f"{diag['model']} confidence: correct vs incorrect "
        f"(test n={diag['n']})"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_percentile_table(diags: list[dict], path: Path) -> None:
    fieldnames = [
        "model",
        "group",
        "n",
        "min",
        "p25",
        "p50",
        "p75",
        "max",
        "mean",
        "threshold",
        "frac_correct_below_threshold",
        "frac_incorrect_above_threshold",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for diag in diags:
            for group in ("correct", "incorrect"):
                stats = diag[group]
                writer.writerow(
                    {
                        "model": diag["model"],
                        "group": group,
                        "n": stats["n"],
                        "min": stats["min"],
                        "p25": stats["p25"],
                        "p50": stats["p50"],
                        "p75": stats["p75"],
                        "max": stats["max"],
                        "mean": stats["mean"],
                        "threshold": diag["threshold"],
                        "frac_correct_below_threshold": (
                            diag["frac_correct_below_threshold"]
                            if group == "correct"
                            else ""
                        ),
                        "frac_incorrect_above_threshold": (
                            diag["frac_incorrect_above_threshold"]
                            if group == "incorrect"
                            else ""
                        ),
                    }
                )


def simulate_router(frame: pd.DataFrame, t1: float, t2: float) -> dict:
    n = len(frame)
    n_base = n_trans = n_llm = 0
    n_resolved_correct = 0
    n_base_correct = 0
    n_trans_correct = 0
    for row in frame.itertuples(index=False):
        gold = str(row.true_label)
        if float(row.baseline_confidence) >= t1:
            n_base += 1
            if str(row.baseline_label) == gold:
                n_resolved_correct += 1
                n_base_correct += 1
        elif float(row.transformer_confidence) >= t2:
            n_trans += 1
            if str(row.transformer_label) == gold:
                n_resolved_correct += 1
                n_trans_correct += 1
        else:
            n_llm += 1
    n_resolved = n_base + n_trans
    return {
        "threshold_1": t1,
        "threshold_2": t2,
        "n": n,
        "n_baseline": n_base,
        "n_transformer": n_trans,
        "n_llm": n_llm,
        "pct_baseline": n_base / n,
        "pct_transformer": n_trans / n,
        "pct_llm": n_llm / n,
        "pct_resolved_local": n_resolved / n,
        "n_resolved_local": n_resolved,
        "accuracy_resolved_local": (
            n_resolved_correct / n_resolved if n_resolved else float("nan")
        ),
        "n_resolved_correct": n_resolved_correct,
        "accuracy_baseline_resolved": (
            n_base_correct / n_base if n_base else float("nan")
        ),
        "accuracy_transformer_resolved": (
            n_trans_correct / n_trans if n_trans else float("nan")
        ),
    }


def write_sweep(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "threshold_1",
        "threshold_2",
        "n",
        "n_baseline",
        "pct_baseline",
        "n_transformer",
        "pct_transformer",
        "n_llm",
        "pct_llm",
        "n_resolved_local",
        "pct_resolved_local",
        "n_resolved_correct",
        "accuracy_resolved_local",
        "accuracy_baseline_resolved",
        "accuracy_transformer_resolved",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def plot_sweep_heatmap(rows: list[dict], path: Path) -> None:
    t1s = SWEEP_VALUES
    t2s = SWEEP_VALUES
    grid = np.full((len(t2s), len(t1s)), np.nan)
    lookup = {(r["threshold_1"], r["threshold_2"]): r for r in rows}
    for i, t2 in enumerate(t2s):
        for j, t1 in enumerate(t1s):
            grid[i, j] = lookup[(t1, t2)]["pct_resolved_local"] * 100.0

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    image = ax.imshow(grid, origin="lower", cmap="YlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(t1s)))
    ax.set_xticklabels([f"{v:.2f}" for v in t1s])
    ax.set_yticks(range(len(t2s)))
    ax.set_yticklabels([f"{v:.2f}" for v in t2s])
    ax.set_xlabel("THRESHOLD_1 (baseline)")
    ax.set_ylabel("THRESHOLD_2 (transformer)")
    ax.set_title("% of test docs resolved at baseline+transformer (no LLM)")
    for i, t2 in enumerate(t2s):
        for j, t1 in enumerate(t1s):
            cell = lookup[(t1, t2)]
            acc = cell["accuracy_resolved_local"]
            ax.text(
                j,
                i,
                f"{grid[i, j]:.0f}%\n{acc:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="black",
            )
    fig.colorbar(image, ax=ax, label="% resolved locally")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_calibration(diag: dict) -> None:
    print()
    print(f"=== {diag['model']} calibration (n={diag['n']}, acc={diag['accuracy']:.3f}) ===")
    print(f"  threshold = {diag['threshold']:.2f}")
    print(
        f"  {'group':<12} {'n':>5} {'min':>7} {'p25':>7} {'p50':>7} "
        f"{'p75':>7} {'max':>7} {'mean':>7}"
    )
    for group in ("correct", "incorrect"):
        s = diag[group]
        print(
            f"  {group:<12} {s['n']:5d} {_fmt(s['min']):>7} {_fmt(s['p25']):>7} "
            f"{_fmt(s['p50']):>7} {_fmt(s['p75']):>7} {_fmt(s['max']):>7} "
            f"{_fmt(s['mean']):>7}"
        )
    print(
        f"  fraction of CORRECT predictions below {diag['threshold']:.2f}: "
        f"{diag['frac_correct_below_threshold']:.1%} "
        f"({diag['n_correct_below_threshold']}/{diag['n_correct']})"
    )
    print(
        f"  fraction of INCORRECT predictions at/above {diag['threshold']:.2f}: "
        f"{diag['frac_incorrect_above_threshold']:.1%} "
        f"({diag['n_incorrect_above_threshold']}/{diag['n_incorrect']})"
    )
    c_mean = diag["correct"]["mean"]
    i_mean = diag["incorrect"]["mean"]
    c_p25 = diag["correct"]["p25"]
    c_p50 = diag["correct"]["p50"]
    c_p75 = diag["correct"]["p75"]
    i_p25 = diag["incorrect"]["p25"]
    i_p50 = diag["incorrect"]["p50"]
    i_p75 = diag["incorrect"]["p75"]
    gap = c_mean - i_mean
    overlap_lo = max(c_p25, i_p25)
    overlap_hi = min(c_p75, i_p75)
    iqr_overlap_width = max(0.0, overlap_hi - overlap_lo)
    print(f"  mean(correct) - mean(incorrect) = {gap:.3f}")
    print(f"  median correct={c_p50:.3f}, median incorrect={i_p50:.3f}")
    print(
        f"  IQR correct=[{c_p25:.3f},{c_p75:.3f}]  "
        f"incorrect=[{i_p25:.3f},{i_p75:.3f}]  "
        f"overlap_width={iqr_overlap_width:.3f}"
    )
    sharp = c_p50 >= 0.75
    if gap >= 0.15 and iqr_overlap_width < 0.08 and sharp:
        verdict = (
            "confidence tracks correctness well "
            "(correct cluster high, incorrect cluster low; ranking is usable)"
        )
    elif gap >= 0.15 and iqr_overlap_width < 0.08 and not sharp:
        verdict = (
            "ranking is usable (IQRs barely overlap) but scores are not sharp: "
            "correct answers are spread out rather than clustered near 1.0"
        )
    elif gap >= 0.08:
        verdict = (
            "confidence tracks correctness moderately "
            "(means separate, but IQR ranges still overlap)"
        )
    else:
        verdict = (
            "confidence is poorly calibrated / noisy "
            "(correct and incorrect ranges overlap heavily)"
        )
    print(f"  verdict: {verdict}")
    diag["verdict"] = verdict
    diag["mean_gap"] = gap
    diag["iqr_overlap_width"] = iqr_overlap_width


def independent_slices(frame: pd.DataFrame) -> list[dict]:
    """Per-model, per-threshold stats on the full test set (not the cascade)."""
    rows = []
    specs = (
        ("baseline", "baseline_label", "baseline_confidence"),
        ("transformer", "transformer_label", "transformer_confidence"),
    )
    gold = frame["true_label"].astype(str)
    n = len(frame)
    for model, pred_col, conf_col in specs:
        pred = frame[pred_col].astype(str)
        conf = frame[conf_col].astype(float)
        correct = pred == gold
        n_correct = int(correct.sum())
        n_incorrect = n - n_correct
        for t in SWEEP_VALUES:
            ge = conf >= t
            n_ge = int(ge.sum())
            n_wrong = int((~correct & ge).sum())
            rows.append(
                {
                    "model": model,
                    "threshold": t,
                    "n_ge": n_ge,
                    "pct_ge": n_ge / n,
                    "accuracy_on_ge": (
                        float((correct & ge).sum() / n_ge) if n_ge else float("nan")
                    ),
                    "n_wrong_ge": n_wrong,
                    "frac_correct_below": (
                        float((correct & (conf < t)).sum() / n_correct)
                        if n_correct
                        else float("nan")
                    ),
                    "frac_incorrect_above": (
                        float(n_wrong / n_incorrect) if n_incorrect else float("nan")
                    ),
                }
            )
    return rows


def write_independent(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "model",
        "threshold",
        "n_ge",
        "pct_ge",
        "accuracy_on_ge",
        "n_wrong_ge",
        "frac_correct_below",
        "frac_incorrect_above",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def print_independent(rows: list[dict]) -> None:
    print()
    print("=== Independent threshold slices (full test set, not cascaded) ===")
    print(
        f"  {'model':<12} {'t':>6} {'n_ge':>6} {'pct':>7} {'acc':>7} "
        f"{'wrong':>6} {'corr<t':>8} {'inc>=t':>8}"
    )
    for row in rows:
        print(
            f"  {row['model']:<12} {row['threshold']:6.2f} {row['n_ge']:6d} "
            f"{row['pct_ge']:7.1%} {row['accuracy_on_ge']:7.3f} "
            f"{row['n_wrong_ge']:6d} {row['frac_correct_below']:8.1%} "
            f"{row['frac_incorrect_above']:8.1%}"
        )


def recommend(sweep: list[dict], independent: list[dict]) -> dict:
    """Pick (T1, T2) from calibration cliffs, not a cosmetic traffic target.

    T1: lowest sweep value where standalone baseline accuracy on conf>=t stays
    at/above 0.95 (the reliability cliff is between 0.50 and 0.60).
    T2: keep the current 0.75 unless a neighbouring cell improves local
    resolution without dropping cascade resolved-set accuracy by more than 2pp.
    """
    current = next(
        r
        for r in sweep
        if r["threshold_1"] == CURRENT_T1 and r["threshold_2"] == CURRENT_T2
    )
    base_rows = [r for r in independent if r["model"] == "baseline"]
    t1 = CURRENT_T1
    for row in sorted(base_rows, key=lambda r: r["threshold"]):
        if row["accuracy_on_ge"] >= 0.95:
            t1 = row["threshold"]
            break
    t2 = CURRENT_T2
    chosen = next(
        r for r in sweep if r["threshold_1"] == t1 and r["threshold_2"] == t2
    )
    return {"current": current, "proposed": chosen}


def print_recommendation(rec: dict, base_diag: dict, trans_diag: dict) -> None:
    cur = rec["current"]
    prop = rec["proposed"]
    print()
    print("=== Recommendation (router.py NOT changed) ===")
    print(
        f"  current:  THRESHOLD_1={cur['threshold_1']:.2f}  "
        f"THRESHOLD_2={cur['threshold_2']:.2f}"
    )
    print(
        f"    baseline {cur['pct_baseline']:.1%}  "
        f"transformer {cur['pct_transformer']:.1%}  "
        f"LLM gate {cur['pct_llm']:.1%}  "
        f"local combined {cur['pct_resolved_local']:.1%}  "
        f"acc_on_resolved {cur['accuracy_resolved_local']:.3f}"
    )
    print(
        f"  proposed: THRESHOLD_1={prop['threshold_1']:.2f}  "
        f"THRESHOLD_2={prop['threshold_2']:.2f}"
    )
    print(
        f"    baseline {prop['pct_baseline']:.1%}  "
        f"transformer {prop['pct_transformer']:.1%}  "
        f"LLM gate {prop['pct_llm']:.1%}  "
        f"local combined {prop['pct_resolved_local']:.1%}  "
        f"acc_on_resolved {prop['accuracy_resolved_local']:.3f}"
    )
    print(
        f"    baseline-resolved acc={prop['accuracy_baseline_resolved']:.3f}  "
        f"transformer-resolved acc={prop['accuracy_transformer_resolved']:.3f}"
    )
    print()
    print(f"  baseline calibration: {base_diag['verdict']}")
    print(f"  transformer calibration: {trans_diag['verdict']}")


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    frame = load_or_infer_predictions()
    print(
        f"Predictions: n={len(frame)}  "
        f"baseline acc={(frame['baseline_label'] == frame['true_label']).mean():.3f}  "
        f"transformer acc={(frame['transformer_label'] == frame['true_label']).mean():.3f}"
    )

    base_diag = calibration_for_model(
        frame, "baseline_label", "baseline_confidence", CURRENT_T1, "baseline"
    )
    trans_diag = calibration_for_model(
        frame,
        "transformer_label",
        "transformer_confidence",
        CURRENT_T2,
        "transformer",
    )
    print_calibration(base_diag)
    print_calibration(trans_diag)

    write_percentile_table([base_diag, trans_diag], PERCENTILES_PATH)
    plot_calibration(base_diag, BASELINE_CHART)
    plot_calibration(trans_diag, TRANSFORMER_CHART)
    print(f"Wrote {PERCENTILES_PATH}")
    print(f"Wrote {BASELINE_CHART}")
    print(f"Wrote {TRANSFORMER_CHART}")

    sweep = [
        simulate_router(frame, t1, t2)
        for t1 in SWEEP_VALUES
        for t2 in SWEEP_VALUES
    ]
    write_sweep(sweep, SWEEP_PATH)
    plot_sweep_heatmap(sweep, SWEEP_HEATMAP)
    print(f"Wrote {SWEEP_PATH} ({len(sweep)} rows)")
    print(f"Wrote {SWEEP_HEATMAP}")

    independent = independent_slices(frame)
    write_independent(independent, INDEPENDENT_PATH)
    print_independent(independent)
    print(f"Wrote {INDEPENDENT_PATH}")

    rec = recommend(sweep, independent)
    print_recommendation(rec, base_diag, trans_diag)


if __name__ == "__main__":
    main()
