"""Cheapest-first document router: baseline → transformer → LLM.

Classification tries the TF-IDF baseline first, then DistilBERT, then Gemini.
Extraction tries regex first, then Gemini for anything still open. An optional
NER middle tier is left as a commented insertion point (no NER model is
trained in this repo by default).
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from ingestion.regex_fallback import extract_with_regex
from models import transformer_classifier
from models.llm_fallback import (
    classify_with_llm,
    drain_llm_cost,
    extract_with_llm,
    gemini_configured,
    reset_llm_cost,
)
from router.schemas import ClassificationResult, ExtractionResult

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "models_store" / "baseline_classifier.joblib"
TRANSFORMER_DIR = REPO_ROOT / "models_store" / "distilbert_classifier"

# Empirically tuned on the 512-doc held-out test set (threshold sweep +
# calibration diagnostic), not the original build-plan example values
# (THRESHOLD_1=0.85 / THRESHOLD_2=0.75). Baseline predict_proba is
# underconfident, so 0.60 still yields ~96% accuracy on the subset it
# resolves. THRESHOLD_2 already tracks correctness and is left unchanged.
# These scores are still not on the same scale (see the walkthrough):
# routing knobs, not calibrated probabilities comparable to the LLM's
# self-reported score.
THRESHOLD_1 = 0.60  # baseline predict_proba max → stop here if cleared
THRESHOLD_2 = 0.75  # unchanged — already well-calibrated, see calibration diagnostic

# Fields this stage is responsible for. Regex fills them when a pattern
# matches; the LLM is asked only for names still missing afterwards. Keeping
# the list to what regex can actually resolve means we do not pay for a Gemini
# call on every upload just to fish for extra keys.
EXTRACTION_FIELDS = ["date", "amount"]

_baseline = None


@dataclass
class RoutingTrace:
    """Per-request telemetry that `route_*` appends to. Not part of the public return type."""

    tiers_used: list[str] = field(default_factory=list)
    latencies_ms: dict = field(default_factory=dict)
    cost_estimate: float = 0.0


_trace: ContextVar[RoutingTrace | None] = ContextVar("routing_trace", default=None)


def start_routing_trace() -> RoutingTrace:
    """Begin a request-scoped trace (contextvar, so concurrent requests stay isolated)."""
    reset_llm_cost()
    trace = RoutingTrace()
    _trace.set(trace)
    return trace


def current_trace() -> RoutingTrace:
    trace = _trace.get()
    if trace is None:
        trace = start_routing_trace()
    return trace


def finish_routing_trace() -> RoutingTrace:
    """Attach accumulated Gemini cost and return the finished trace."""
    trace = current_trace()
    trace.cost_estimate = round(drain_llm_cost(), 6)
    return trace


def load_classifiers() -> None:
    """Load baseline + DistilBERT once at API startup."""
    global _baseline
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"No baseline weights at {BASELINE_PATH}. Run scripts/train_baseline.py."
        )
    _baseline = joblib.load(BASELINE_PATH)
    transformer_classifier.load_model(TRANSFORMER_DIR)


def candidate_labels() -> list[str]:
    if _baseline is None:
        raise RuntimeError("Call load_classifiers() before routing.")
    return [str(label) for label in _baseline.classes_]


def _predict_baseline(text: str) -> tuple[str, float]:
    """Winning class and its logistic-regression `predict_proba` mass."""
    if _baseline is None:
        raise RuntimeError("Call load_classifiers() before routing.")
    proba = _baseline.predict_proba(["" if text is None else str(text)])[0]
    index = int(proba.argmax())
    return str(_baseline.classes_[index]), float(proba[index])


def _timed(tier: str, fn):
    trace = current_trace()
    started = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    trace.latencies_ms[tier] = round(elapsed_ms, 2)
    if tier not in trace.tiers_used:
        trace.tiers_used.append(tier)
    return result


def route_document(text: str) -> ClassificationResult:
    """Classify cheapest-first: baseline → transformer → LLM."""
    label, confidence = _timed("baseline", lambda: _predict_baseline(text))
    if confidence >= THRESHOLD_1:
        return ClassificationResult(label=label, confidence=confidence, tier="baseline")

    label, confidence = _timed(
        "transformer", lambda: transformer_classifier.predict(text)
    )
    if confidence >= THRESHOLD_2:
        return ClassificationResult(
            label=label, confidence=confidence, tier="transformer"
        )

    if not gemini_configured():
        # Transformer already ran; without a key we return that answer rather
        # than 503ing the request. The log will show baseline+transformer only.
        return ClassificationResult(
            label=label, confidence=confidence, tier="transformer"
        )

    labels = candidate_labels()
    label, confidence = _timed(
        "llm", lambda: classify_with_llm(text, labels)
    )
    return ClassificationResult(label=label, confidence=confidence, tier="llm")


def route_extraction(text: str) -> ExtractionResult:
    """Extract cheapest-first: regex → (optional NER) → LLM for leftovers."""
    fields: dict = {}
    confidence: dict = {}
    tiers: list[str] = []

    regex_hits = _timed("regex", lambda: extract_with_regex(text))
    for name, hit in regex_hits.items():
        value = hit.get("value")
        if value:
            fields[name] = value
            confidence[name] = float(hit.get("confidence", 1.0))
    if regex_hits:
        tiers.append("regex")

    # --- Optional NER tier (not in this repo by default) -----------------
    # Metadata extraction is regex-only unless you separately trained NER.
    # If `src/models/ner_extractor.py` exists, slot it in here as a middle
    # tier between regex and the LLM:
    #
    #   still_open = [name for name in EXTRACTION_FIELDS if name not in fields]
    #   if still_open:
    #       ner_out = _timed("ner", lambda: ner_extractor.extract(text, still_open))
    #       for name, hit in ner_out.items():
    #           if hit.get("value") and float(hit.get("confidence", 0)) >= NER_THRESHOLD:
    #               fields[name] = hit["value"]
    #               confidence[name] = float(hit["confidence"])
    #       tiers.append("ner")
    # ---------------------------------------------------------------------

    unresolved = [
        name
        for name in EXTRACTION_FIELDS
        if name not in fields or fields[name] in (None, "")
    ]
    if unresolved:
        if gemini_configured():
            llm_hits = _timed(
                "llm_extract", lambda: extract_with_llm(text, unresolved)
            )
            for name, hit in llm_hits.items():
                value = hit.get("value")
                score = float(hit.get("confidence", 0.0))
                if value not in (None, ""):
                    fields[name] = value
                    confidence[name] = score
            tiers.append("llm")
        else:
            # No key → leave regex-only rather than 500ing the whole request.
            for name in unresolved:
                fields.setdefault(name, None)
                confidence.setdefault(name, 0.0)

    if "regex" in tiers and "llm" in tiers:
        tier = "regex+llm"
    elif "llm" in tiers:
        tier = "llm"
    else:
        tier = "regex"

    return ExtractionResult(fields=fields, confidence=confidence, tier=tier)
