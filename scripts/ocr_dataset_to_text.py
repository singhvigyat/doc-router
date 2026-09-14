"""OCR every image in the RVL-CDIP subset into a cached CSV.

Columns: doc_id, text, label. `doc_id` is `{split}_{index:05d}` so
train/validation/test membership survives without an extra column.
Re-runs skip doc_ids already present in the CSV.
"""

from __future__ import annotations

import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingestion.ocr_parser import extract_text_from_image  # noqa: E402

SUBSET_DIR = REPO_ROOT / "data" / "processed" / "rvl_cdip_subset"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "rvl_cdip_text.csv"
FLUSH_EVERY = 32
OCR_WORKERS = 4


def _ocr_pil_image(image) -> str:
    """Write a PIL image to a temp PNG and call Stage 1's path-based OCR."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        image.convert("RGB").save(tmp_path)
        return extract_text_from_image(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def _ocr_job(doc_id: str, label: str, image) -> dict:
    try:
        text = _ocr_pil_image(image)
    except Exception as exc:  # one bad image shouldn't abort the run
        print(f"  OCR failed for {doc_id}: {exc}")
        text = ""
    return {"doc_id": doc_id, "text": text, "label": label}


def _load_cached_rows() -> tuple[list[dict], set[str]]:
    if not OUTPUT_CSV.exists():
        return [], set()
    frame = pd.read_csv(OUTPUT_CSV)
    rows = frame.to_dict(orient="records")
    processed = {str(row["doc_id"]) for row in rows}
    print(f"Cache hit: {len(processed)} rows already in {OUTPUT_CSV}")
    return rows, processed


def _flush(rows: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["doc_id", "text", "label"]).to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8"
    )


def main() -> None:
    if not (SUBSET_DIR / "dataset_dict.json").exists():
        raise SystemExit(
            f"Missing {SUBSET_DIR}. Run scripts/download_data.py first."
        )

    dataset = load_from_disk(str(SUBSET_DIR))
    rows, processed = _load_cached_rows()
    pending = 0
    new_rows = 0

    for split_name in dataset.keys():
        split = dataset[split_name]
        label_names = split.features["label"].names
        print(f"OCR split={split_name} n={len(split)} workers={OCR_WORKERS}")
        batch: list[tuple[str, str, object]] = []

        def _run_batch(jobs: list[tuple[str, str, object]]) -> None:
            nonlocal pending, new_rows
            with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
                futures = [
                    pool.submit(_ocr_job, doc_id, label, image)
                    for doc_id, label, image in jobs
                ]
                for future in as_completed(futures):
                    result = future.result()
                    rows.append(result)
                    processed.add(result["doc_id"])
                    pending += 1
                    new_rows += 1
            if pending >= FLUSH_EVERY:
                _flush(rows)
                pending = 0
                print(f"  flushed {len(rows)} rows ({new_rows} new this run)")

        for index, example in enumerate(split):
            doc_id = f"{split_name}_{index:05d}"
            if doc_id in processed:
                continue
            label = label_names[int(example["label"])]
            batch.append((doc_id, label, example["image"]))
            if len(batch) >= FLUSH_EVERY:
                _run_batch(batch)
                batch = []
        if batch:
            _run_batch(batch)

    rows.sort(key=lambda row: str(row["doc_id"]))
    _flush(rows)
    print(f"Wrote {len(rows)} rows ({new_rows} new) to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
