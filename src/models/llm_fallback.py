"""Gemini fallback for classification and field extraction.

Uses the current Google Gen AI SDK (`google-genai`, already in requirements.txt)
rather than the older `google-generativeai` package. Both talk to the same
Gemini API; the new client is what this repo already depends on.

Pattern: JSON-schema prompting + parse/validate + one retry on failure.
The model is asked for a specific JSON shape, the response is parsed, and if
validation fails we send the error back once instead of trusting free-form text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from contextvars import ContextVar
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

GEMINI_MODEL = "gemini-3.6-flash"
# Keep prompts inside a cheap window so a long OCR dump cannot blow the quota.
MAX_DOC_CHARS = 8000

# Fail-fast HTTP policy. The google-genai SDK defaults to 5 attempts with
# exponential backoff up to 60s, which turned a single 504 into a ~9-minute
# stall. We disable SDK retries and handle a short retry ourselves.
HTTP_TIMEOUT_MS = 18_000
TRANSIENT_RETRIES = 2  # original call + this many retries (covers one RPM wait)
TRANSIENT_BACKOFF_S = (1.0, 2.0, 4.0)

# Published Gemini 2.0 Flash list prices (USD per 1M tokens). Used only to
# populate RoutingLog.cost_estimate — not billed by us, just an approximation.
USD_PER_MILLION_INPUT = 0.10
USD_PER_MILLION_OUTPUT = 0.40

_session_cost: ContextVar[float] = ContextVar("llm_session_cost", default=0.0)
_client = None
_quota_exhausted = False
_quota_reason = ""


class LLMError(RuntimeError):
    """Gemini failed in a way the caller should not hang on."""


class LLMQuotaExceeded(LLMError):
    """Hard quota / 429. No further Gemini calls this process."""


class LLMCallFailed(LLMError):
    """Retries exhausted or a non-retryable error. Skip this document."""


def quota_exhausted() -> bool:
    return _quota_exhausted


def quota_reason() -> str:
    return _quota_reason


def reset_quota_gate() -> None:
    """Clear the process-wide quota latch (tests / a fresh benchmark run)."""
    global _quota_exhausted, _quota_reason
    _quota_exhausted = False
    _quota_reason = ""


def reset_client() -> None:
    global _client
    _client = None


class LLMClassification(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class LLMFieldValue(BaseModel):
    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


def reset_llm_cost() -> None:
    """Zero the per-request cost accumulator (call at the start of a request)."""
    _session_cost.set(0.0)


def drain_llm_cost() -> float:
    """Return accumulated LLM cost for this request and reset it."""
    value = float(_session_cost.get())
    _session_cost.set(0.0)
    return value


def gemini_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def _add_cost(prompt: str, response_text: str) -> None:
    in_tokens = max(len(prompt), 1) / 4.0
    out_tokens = max(len(response_text), 1) / 4.0
    cost = (in_tokens / 1_000_000) * USD_PER_MILLION_INPUT + (
        out_tokens / 1_000_000
    ) * USD_PER_MILLION_OUTPUT
    _session_cost.set(_session_cost.get() + cost)


def _client_or_raise():
    global _client
    if not gemini_configured():
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the repo root "
            "(see .env.example) and restart the server."
        )
    if _client is None:
        from google import genai
        from google.genai import types

        _client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"].strip(),
            http_options=types.HttpOptions(
                timeout=HTTP_TIMEOUT_MS,
                # attempts=1 => no SDK retries. We retry 503/504 ourselves.
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    return _client


def _doc_hash(text: str) -> str:
    return hashlib.sha256(("" if text is None else str(text)).encode("utf-8")).hexdigest()


def _log_skip(kind: str, text: str, reason: str) -> None:
    print(
        f"  [llm] {kind} skipped doc_hash={_doc_hash(text)} reason={reason}",
        file=sys.stderr,
        flush=True,
    )


def _error_code(err: BaseException) -> int | None:
    code = getattr(err, "code", None)
    if isinstance(code, int):
        return code
    return None


def _is_daily_quota_error(err: BaseException) -> bool:
    """Hard stop: free-tier *daily* cap. RPM 429s are not this."""
    message = str(err)
    lowered = message.lower()
    if "GenerateRequestsPerDay" in message or "PerDayPerProjectPerModel" in message:
        return True
    if "daily quota" in lowered:
        return True
    return bool(re.search(r"limit:\s*500", message) and "free_tier" in lowered)


def _is_quota_error(err: BaseException) -> bool:
    """Latch the process only on daily exhaustion, not per-minute 429s."""
    return _is_daily_quota_error(err)


def _is_rate_limit_error(err: BaseException) -> bool:
    """Per-minute / short 429. Wait and retry; do not skip the rest of the run."""
    if _is_daily_quota_error(err):
        return False
    if _error_code(err) == 429:
        return True
    message = str(err).lower()
    return any(
        needle in message
        for needle in (
            "resource_exhausted",
            "exceeded your current quota",
            "quota exceeded",
            "perminute",
        )
    )


def _retry_wait_seconds(err: BaseException, attempt_index: int) -> float:
    match = re.search(r"retry in ([0-9.]+)\s*s", str(err), re.I)
    if match:
        return min(max(float(match.group(1)) + 0.5, 1.0), 60.0)
    if _is_rate_limit_error(err):
        return 30.0
    return TRANSIENT_BACKOFF_S[min(attempt_index, len(TRANSIENT_BACKOFF_S) - 1)]


def _is_timeout_error(err: BaseException) -> bool:
    try:
        import httpx
    except ImportError:
        httpx = None  # type: ignore[assignment]
    timeout_types: tuple = (TimeoutError,)
    if httpx is not None:
        timeout_types = timeout_types + (
            httpx.TimeoutException,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
        )
    if isinstance(err, timeout_types):
        return True
    message = str(err).lower()
    return any(
        needle in message
        for needle in ("timeout", "timed out", "deadline exceeded", "readtimeout")
    )


def _is_transient_error(err: BaseException) -> bool:
    if _is_quota_error(err):
        return False
    if _is_rate_limit_error(err):
        return True
    if _is_timeout_error(err):
        return True
    code = _error_code(err)
    if code in {408, 500, 502, 503, 504}:
        return True
    message = str(err).lower()
    return any(
        needle in message
        for needle in ("unavailable", "503", "504", "502", "connection reset")
    )


def _mark_quota(err: BaseException) -> LLMQuotaExceeded:
    global _quota_exhausted, _quota_reason
    _quota_exhausted = True
    _quota_reason = str(err)
    wrapped = LLMQuotaExceeded(
        f"Gemini quota exceeded; stopping new API calls. ({err})"
    )
    return wrapped


def _generate_once(prompt: str) -> str:
    """One Gemini HTTP round-trip. Cost is charged to the request trace."""
    from google.genai import types

    client = _client_or_raise()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    text = getattr(response, "text", None) or ""
    _add_cost(prompt, text)
    return text


def _generate(prompt: str) -> str:
    """Call Gemini with a short timeout and at most one brief transient retry."""
    if _quota_exhausted:
        raise LLMQuotaExceeded(
            f"Gemini quota already exhausted; skipping API call. ({_quota_reason})"
        )

    attempts = TRANSIENT_RETRIES + 1
    last_err: BaseException | None = None
    for i in range(attempts):
        try:
            return _generate_once(prompt)
        except LLMQuotaExceeded:
            raise
        except Exception as err:  # noqa: BLE001 — must classify every SDK failure
            if _is_quota_error(err):
                raise _mark_quota(err) from err
            last_err = err
            if _is_transient_error(err) and i < attempts - 1:
                wait = _retry_wait_seconds(err, i)
                print(
                    f"  [llm] transient {err!r}; retry {i + 1}/{TRANSIENT_RETRIES} "
                    f"in {wait:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                reset_client()
                time.sleep(wait)
                continue
            raise LLMCallFailed(f"Gemini call failed: {err}") from err
    raise LLMCallFailed(f"Gemini call failed: {last_err}") from last_err


def _clip(text: str) -> str:
    raw = "" if text is None else str(text)
    if len(raw) <= MAX_DOC_CHARS:
        return raw
    return raw[:MAX_DOC_CHARS] + "\n...[truncated]..."


def _parse_json_object(raw: str) -> dict:
    """Parse a JSON object, including replies wrapped in markdown fences."""
    blob = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", blob)
    if fenced:
        blob = fenced.group(1).strip()
    else:
        start, end = blob.find("{"), blob.rfind("}")
        if start != -1 and end > start:
            blob = blob[start : end + 1]
    parsed = json.loads(blob)
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed


def _generate_with_retry(prompt: str, validate) -> dict:
    """Call Gemini, validate, retry once with the parser error in the prompt."""
    first = _generate(prompt)
    try:
        return validate(_parse_json_object(first))
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as err:
        retry_prompt = (
            f"{prompt}\n\n"
            "Your previous reply was not valid JSON matching the required schema.\n"
            f"Parser error: {err}\n"
            f"Previous reply:\n{first}\n\n"
            "Respond again with JSON only — no markdown, no commentary."
        )
        second = _generate(retry_prompt)
        return validate(_parse_json_object(second))


def classify_with_llm(text: str, candidate_labels: list[str]) -> tuple[str, float]:
    """Ask Gemini to pick one label. Returns `(label, self-reported confidence)`."""
    labels = [str(label) for label in candidate_labels]
    if not labels:
        raise ValueError("candidate_labels must be non-empty")

    schema = '{"label": "<one of the candidate labels>", "confidence": <float 0-1>}'
    prompt = (
        "You are a document-type classifier.\n"
        "Choose exactly one label from this list:\n"
        f"{json.dumps(labels)}\n\n"
        "Return JSON only, matching this schema:\n"
        f"{schema}\n"
        "`confidence` is your estimated probability that the chosen label is "
        "correct (between 0 and 1).\n\n"
        "Document:\n\"\"\"\n"
        f"{_clip(text)}\n"
        "\"\"\"\n"
    )

    def validate(payload: dict) -> dict:
        parsed = LLMClassification.model_validate(payload)
        if parsed.label not in labels:
            raise ValueError(
                f"label {parsed.label!r} is not in candidate_labels {labels}"
            )
        return parsed.model_dump()

    try:
        result = _generate_with_retry(prompt, validate)
    except LLMQuotaExceeded as err:
        _log_skip("classify", text, f"quota_exceeded: {err}")
        raise
    except LLMCallFailed as err:
        _log_skip("classify", text, str(err))
        raise
    return result["label"], float(result["confidence"])


def extract_with_llm(text: str, fields: list[str]) -> dict:
    """Extract named fields. Returns `{field: {"value": ..., "confidence": ...}}`."""
    names = [str(name) for name in fields]
    if not names:
        return {}

    field_schema = (
        "{ "
        + ", ".join(
            f'"{name}": {{"value": "<string or null>", "confidence": <float 0-1>}}'
            for name in names
        )
        + " }"
    )
    prompt = (
        "Extract the requested fields from the document. Use null for value "
        "when the field is not present. Return JSON only, matching this schema:\n"
        f"{field_schema}\n"
        "`confidence` is your estimated probability that the value is correct.\n\n"
        "Document:\n\"\"\"\n"
        f"{_clip(text)}\n"
        "\"\"\"\n"
    )

    def validate(payload: dict) -> dict:
        out = {}
        for name in names:
            raw_field = payload.get(name, {"value": None, "confidence": 0.0})
            if not isinstance(raw_field, dict):
                raw_field = {"value": raw_field, "confidence": 0.5}
            parsed = LLMFieldValue.model_validate(raw_field)
            out[name] = parsed.model_dump()
        return out

    try:
        return _generate_with_retry(prompt, validate)
    except LLMQuotaExceeded as err:
        _log_skip("extract", text, f"quota_exceeded: {err}")
        raise
    except LLMCallFailed as err:
        _log_skip("extract", text, str(err))
        raise
