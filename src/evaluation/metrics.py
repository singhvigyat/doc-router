"""Generic classification metrics used by every training stage."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Return accuracy, macro-F1, and the full sklearn classification report.

    The report is a dict (`output_dict=True`) so later stages can dump it to
    JSON without scraping a formatted string. `zero_division=0` keeps the
    functions defined when a class is missing from y_pred.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def plot_confusion_matrix(y_true, y_pred, labels, save_path) -> None:
    """Save a labeled confusion-matrix heatmap to `save_path`.

    Rows are true classes, columns are predicted classes. `labels` sets both
    the axis order and which classes are included, so it should be the full
    label inventory (not just classes that happened to appear in this batch).
    """
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        square=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
