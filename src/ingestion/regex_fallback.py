"""Deterministic extractors for dates and monetary amounts.

Regex is the cheapest extraction tier: no model load, no API call. Confidence
is 1.0 when a pattern matches because the match *is* the evidence — there is
no probabilistic classifier behind it. The trade-off is recall: amounts
written as "42.00 USD" with no currency symbol will be missed, and that is
exactly when `route_extraction` escalates to the LLM.
"""

from __future__ import annotations

import re

# Prompt-specified shapes, plus ISO dates and a few extra currency symbols.
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)
AMOUNT_RE = re.compile(
    r"(?:[$₹€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|[$₹€£]\s?\d+[,.]?\d*)"
)

# Only fields this module knows how to fill. Anything else is unresolved.
REGEX_FIELDS = ("date", "amount")


def extract_with_regex(text: str) -> dict:
    """Return `{field: {"value": str, "confidence": 1.0}}` for confident matches.

    Fields with no match are omitted so the router can tell what is still open.
    """
    haystack = "" if text is None else str(text)
    found: dict = {}

    date_match = DATE_RE.search(haystack)
    if date_match:
        found["date"] = {"value": date_match.group(0), "confidence": 1.0}

    amount_match = AMOUNT_RE.search(haystack)
    if amount_match:
        found["amount"] = {
            "value": amount_match.group(0).strip(),
            "confidence": 1.0,
        }

    return found
