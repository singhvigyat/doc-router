"""Pydantic contracts for router outputs and the per-request routing log."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Predicted document type plus the cheapest tier that produced it."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    tier: str


class ExtractionResult(BaseModel):
    """Structured fields pulled from the document text.

    `fields` maps field name -> extracted value (string or null).
    `confidence` maps the same names -> a 0–1 score from the tier that filled them.
    `tier` is the highest (most expensive) extraction tier that actually ran:
    `regex`, `llm`, or `regex+llm` when both contributed.
    """

    fields: dict
    confidence: dict
    tier: str


class RoutingLog(BaseModel):
    """One JSONL record per `/process-document` request."""

    doc_id: str
    tiers_used: list[str]
    latencies_ms: dict
    cost_estimate: float = Field(ge=0.0)
