"""Classical TF-IDF + logistic regression baseline for document type."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluation.metrics import compute_classification_metrics, plot_confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def train_baseline(train_texts, train_labels) -> Pipeline:
    """Fit a TF-IDF (1–2 grams, 5k features) + logistic regression pipeline."""
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    return pipeline


def evaluate_baseline(pipeline, test_texts, test_labels, results_dir=None) -> dict:
    """Score the pipeline and write the report JSON plus confusion-matrix PNG."""
    results_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    y_pred = pipeline.predict(test_texts)
    metrics = compute_classification_metrics(test_labels, y_pred)

    report_path = results_dir / "baseline_classification_report.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    labels = list(pipeline.classes_)
    plot_confusion_matrix(
        test_labels,
        y_pred,
        labels=labels,
        save_path=results_dir / "baseline_confusion_matrix.png",
    )
    return metrics


def save_model(pipeline, path) -> None:
    """Serialize a fitted sklearn pipeline with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
