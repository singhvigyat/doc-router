"""Score the Stage 2 baseline and Stage 3 DistilBERT on the same test split.

Prints accuracy, macro-F1, and average sequential latency (100 predictions).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.metrics import compute_classification_metrics  # noqa: E402
from models.transformer_classifier import load_model, predict  # noqa: E402

CSV_PATH = REPO_ROOT / "data" / "processed" / "rvl_cdip_text.csv"
BASELINE_PATH = REPO_ROOT / "models_store" / "baseline_classifier.joblib"
TRANSFORMER_DIR = REPO_ROOT / "models_store" / "distilbert_classifier"
KNOWN_SPLITS = {"train", "validation", "val", "test"}
LATENCY_N = 100


def _split_from_doc_id(doc_id: str) -> str | None:
    """Same prefix rule as scripts/train_baseline.py (Stage 2)."""
    prefix = str(doc_id).split("_", 1)[0]
    return prefix if prefix in KNOWN_SPLITS else None


def load_test_split(csv_path: Path) -> tuple[list[str], list[str]]:
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
    return texts, labels


def average_latency(predict_one, texts: list[str], n: int = LATENCY_N) -> float:
    """Seconds per prediction, sequential, after a short warmup."""
    if not texts:
        raise SystemExit("No texts to time.")
    sample = (texts * ((n // len(texts)) + 1))[:n]
    for text in sample[:3]:
        predict_one(text)
    start = time.perf_counter()
    for text in sample:
        predict_one(text)
    return (time.perf_counter() - start) / n


def _fmt(value, digits=4) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    if not BASELINE_PATH.exists():
        raise SystemExit(f"Missing {BASELINE_PATH}. Run scripts/train_baseline.py.")
    if not TRANSFORMER_DIR.exists():
        raise SystemExit(
            f"Missing {TRANSFORMER_DIR}.\n"
            "Fine-tune in Colab (notebooks/train_transformer.ipynb or "
            "scripts/train_transformer_colab.py), download the saved model "
            "folder, and place its files directly in "
            "models_store/distilbert_classifier/ "
            "(you should see config.json in that folder)."
        )

    texts, labels = load_test_split(CSV_PATH)
    print(f"Test documents: {len(texts)}")

    baseline = joblib.load(BASELINE_PATH)
    load_model(TRANSFORMER_DIR)

    baseline_pred = baseline.predict(texts).tolist()
    transformer_pred = [predict(text)[0] for text in texts]

    baseline_metrics = compute_classification_metrics(labels, baseline_pred)
    transformer_metrics = compute_classification_metrics(labels, transformer_pred)

    baseline_ms = average_latency(lambda t: baseline.predict([t])[0], texts) * 1000
    transformer_ms = average_latency(lambda t: predict(t), texts) * 1000

    rows = [
        ("accuracy", baseline_metrics["accuracy"], transformer_metrics["accuracy"]),
        ("macro-F1", baseline_metrics["macro_f1"], transformer_metrics["macro_f1"]),
        ("avg latency (ms/pred)", baseline_ms, transformer_ms),
    ]

    print()
    print("Baseline vs DistilBERT  (same test split, sequential latency n=100)")
    print(f"{'Metric':<24} {'Baseline':>12} {'DistilBERT':>12}")
    print("-" * 50)
    for name, left, right in rows:
        digits = 2 if "latency" in name else 4
        print(f"{name:<24} {_fmt(left, digits):>12} {_fmt(right, digits):>12}")
    print()
    print(
        "Accuracy / macro-F1: higher is better. Latency: lower is better. "
        "Latency is one document at a time (how the future router will call "
        "these models), so batched GPU throughput would look faster than this."
    )


if __name__ == "__main__":
    main()
