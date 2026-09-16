"""Inference wrapper for a fine-tuned DistilBERT document classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "models_store" / "distilbert_classifier"

# Populated by load_model(); predict() reads these.
_tokenizer = None
_model = None
_device = None


def load_model(path=None):
    """Load a `save_pretrained` DistilBERT classifier from disk.

    Call this once before `predict`. Re-calling it replaces the in-memory model.
    Returns the HuggingFace model (already in eval mode, on CPU or GPU).
    """
    global _tokenizer, _model, _device

    model_dir = Path(path) if path is not None else DEFAULT_MODEL_DIR
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No transformer weights at {model_dir}. Fine-tune in Colab "
            "(see notebooks/train_transformer.ipynb), then place the saved "
            "folder at models_store/distilbert_classifier/."
        )

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    _model.eval()
    return _model


def _label_from_id(label_id: int) -> str:
    """Resolve HuggingFace config id2label keys (int or str)."""
    id2label = _model.config.id2label
    if label_id in id2label:
        return id2label[label_id]
    return id2label[str(label_id)]


def predict(text: str) -> tuple[str, float]:
    """Return `(predicted_label, confidence)` for one document.

    Tokenizes the same way as training (truncate at 512), runs a single
    forward pass, then softmax + argmax. Confidence is the softmax
    probability of the winning class, not a calibrated probability.
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Call load_model(path) before predict().")

    encoded = _tokenizer(
        "" if text is None else str(text),
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {key: value.to(_device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = _model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)
        confidence, index = torch.max(probabilities, dim=-1)

    label_id = int(index.item())
    return _label_from_id(label_id), float(confidence.item())
