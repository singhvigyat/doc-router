"""Extract text from scanned images via Tesseract OCR."""

from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image


def extract_text_from_image(path: str) -> str:
    """Return OCR text from a raster image (JPG/PNG/TIFF, etc.).

    PIL(pillow) only opens the pixels; pytesseract shells out to the Tesseract
    binary to actually recognize characters. This is the right tool for
    scans and screenshots that have no selectable text layer.
    """
    with Image.open(path) as image:
        return pytesseract.image_to_string(image)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    sample_path = repo_root / "data" / "raw" / "samples" / "sample.png"
    print(extract_text_from_image(str(sample_path)))
