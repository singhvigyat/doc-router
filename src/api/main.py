"""FastAPI app: upload a document, parse it, run the tiered router, log the path.

Run from the repo root:

    uvicorn src.api.main:app --reload

Then POST a file to `/process-document` (or use the interactive docs at `/docs`).
"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(REPO_ROOT / ".env")

from ingestion.email_parser import parse_email  # noqa: E402
from ingestion.ocr_parser import extract_text_from_image  # noqa: E402
from ingestion.pdf_parser import extract_text_from_pdf  # noqa: E402
from router.router import (  # noqa: E402
    finish_routing_trace,
    load_classifiers,
    route_document,
    route_extraction,
    start_routing_trace,
)
from router.schemas import RoutingLog  # noqa: E402

LOG_PATH = REPO_ROOT / "logs" / "routing_log.jsonl"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
PDF_EXTENSIONS = {".pdf"}
EMAIL_EXTENSIONS = {".eml"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_classifiers()
    yield


app = FastAPI(
    title="DocRouter",
    description="Cheapest-first document classifier and field extractor.",
    lifespan=lifespan,
)


def detect_file_kind(filename: str | None, content: bytes) -> str:
    """Prefer the file extension; fall back to magic bytes / email headers."""
    suffix = Path(filename or "").suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in EMAIL_EXTENSIONS:
        return "email"

    if content.startswith(b"%PDF"):
        return "pdf"
    if content.startswith(b"\x89PNG") or content[:3] == b"\xff\xd8\xff":
        return "image"
    if content.startswith(b"II*\x00") or content.startswith(b"MM\x00*"):
        return "image"  # TIFF
    head = content[:4096]
    if b"Content-Type:" in head or b"MIME-Version:" in head or b"From:" in head:
        return "email"

    raise HTTPException(
        status_code=400,
        detail=(
            "Unsupported file type. Upload a PDF, an image "
            "(png/jpg/tiff/webp), or an .eml file."
        ),
    )


def _pdf_text(path: str) -> str:
    """Native PDF text, with a first-pages OCR fallback for scanned files."""
    text = extract_text_from_pdf(path)
    if text.strip():
        return text

    # Scanned PDF: the text layer is empty, so rasterize a few pages and OCR.
    import io

    import pymupdf as fitz
    import pytesseract
    from PIL import Image

    chunks: list[str] = []
    with fitz.open(path) as document:
        for page in document:
            if len(chunks) >= 3:
                break
            pixmap = page.get_pixmap()
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            chunks.append(pytesseract.image_to_string(image))
    return "\n".join(chunks)


def extract_text(kind: str, path: str) -> str:
    if kind == "pdf":
        return _pdf_text(path)
    if kind == "image":
        return extract_text_from_image(path)
    parsed = parse_email(path)
    return (
        f"From: {parsed.get('sender', '')}\n"
        f"To: {parsed.get('recipient', '')}\n"
        f"Subject: {parsed.get('subject', '')}\n\n"
        f"{parsed.get('body', '')}"
    )


def append_routing_log(record: RoutingLog) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(), ensure_ascii=True) + "\n")


@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    """Parse an uploaded file, route it, and return classification + extraction."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    kind = detect_file_kind(file.filename, content)
    suffix = Path(file.filename or "").suffix or {
        "pdf": ".pdf",
        "image": ".png",
        "email": ".eml",
    }[kind]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        text = extract_text(kind, tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Failed to parse {kind} file: {exc}"
        ) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    doc_id = str(uuid.uuid4())
    start_routing_trace()
    try:
        classification = route_document(text)
        extraction = route_extraction(text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    trace = finish_routing_trace()
    log = RoutingLog(
        doc_id=doc_id,
        tiers_used=trace.tiers_used,
        latencies_ms=trace.latencies_ms,
        cost_estimate=trace.cost_estimate,
    )
    append_routing_log(log)

    return JSONResponse(
        {
            "doc_id": doc_id,
            "source_type": kind,
            "classification": classification.model_dump(),
            "extraction": extraction.model_dump(),
            "routing": log.model_dump(),
        }
    )
