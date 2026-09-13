"""Load OCR CSV, train the TF-IDF baseline, evaluate, save the model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.baseline_classifier import (  # noqa: E402
    evaluate_baseline,
    save_model,
    train_baseline,
)

CSV_PATH = REPO_ROOT / "data" / "processed" / "rvl_cdip_text.csv"
MODEL_PATH = REPO_ROOT / "models_store" / "baseline_classifier.joblib"
RESULTS_DIR = REPO_ROOT / "results"
KNOWN_SPLITS = {"train", "validation", "val", "test"}


def _split_from_doc_id(doc_id: str) -> str | None:
    prefix = str(doc_id).split("_", 1)[0]
    return prefix if prefix in KNOWN_SPLITS else None


def _texts_and_labels(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    texts = frame["text"].fillna("").astype(str).tolist()
    labels = frame["label"].astype(str).tolist()
    return texts, labels


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"Missing {CSV_PATH}. Run scripts/ocr_dataset_to_text.py first."
        )

    frame = pd.read_csv(CSV_PATH)
    if frame.empty:
        raise SystemExit(f"{CSV_PATH} is empty.")

    frame["split"] = frame["doc_id"].map(_split_from_doc_id)
    if frame["split"].notna().all():
        train_frame = frame[frame["split"] == "train"]
        test_frame = frame[frame["split"] == "test"]
        if test_frame.empty:
            test_frame = frame[frame["split"].isin(["validation", "val"])]
        print(
            f"Using CSV split prefixes: train={len(train_frame)} "
            f"test={len(test_frame)}"
        )
    else:
        print("doc_id prefixes missing; falling back to a stratified 80/20 split.")
        train_frame, test_frame = train_test_split(
            frame, test_size=0.2, random_state=42, stratify=frame["label"]
        )

    train_texts, train_labels = _texts_and_labels(train_frame)
    test_texts, test_labels = _texts_and_labels(test_frame)
    if not train_texts or not test_texts:
        raise SystemExit("Need non-empty train and test sets.")

    print("Training baseline...")
    pipeline = train_baseline(train_texts, train_labels)
    print("Evaluating...")
    metrics = evaluate_baseline(
        pipeline, test_texts, test_labels, results_dir=RESULTS_DIR
    )
    save_model(pipeline, MODEL_PATH)

    print(json.dumps(
        {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "model_path": str(MODEL_PATH),
            "report_path": str(RESULTS_DIR / "baseline_classification_report.json"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
