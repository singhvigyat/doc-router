"""Extract native text from digitally generated PDFs using PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz


def extract_text_from_pdf(path: str) -> str:
    """Return concatenated text from every page of a PDF.

    PyMuPDF reads the PDF's text layer (the actual glyphs/unicode already
    embedded in the file). That is fast and accurate for native PDFs, but
    it returns empty or near-empty strings for scanned pages with no text
    layer — those belong to the OCR parser, not this function.
    """
    text = ""
    with fitz.open(path) as document:
        for page in document:
            text += page.get_text()
    return text


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    sample_path = repo_root / "data" / "raw" / "samples" / "sample.pdf"
    print(extract_text_from_pdf(str(sample_path)))
