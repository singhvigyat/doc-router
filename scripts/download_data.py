"""Download RVL-CDIP, keep 8 enterprise classes, subsample, save_to_disk.

`datasets>=4` dropped dataset-loading scripts, so `load_dataset("rvl_cdip")`
fails against the official Hub repo (`aharley/rvl_cdip` still ships
`rvl_cdip.py`). This script still uses the Hub copy of RVL-CDIP and the
`datasets` library (`Features`, `Dataset`, `save_to_disk`): it streams the
~38 GB tar.gz, keeps the first N images per class per split, and never
writes the full archive to disk.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import requests
import tarfile
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "rvl_cdip_subset"

HF_DATASET_ID = "aharley/rvl_cdip"
ARCHIVE_FILENAME = "data/rvl-cdip.tar.gz"
IMAGES_PREFIX = "images/"

# Official 16-class inventory (printed before filtering).
ALL_CLASSES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]

# Enterprise-relevant subset requested for this stage.
TARGET_CLASSES = [
    "letter",
    "form",
    "email",
    "scientific report",
    "budget",
    "invoice",
    "resume",
    "memo",
]

TRAIN_PER_CLASS = 400
VAL_PER_CLASS = 64
TEST_PER_CLASS = 64

SPLIT_FILES = {
    "train": ("data/train.txt", TRAIN_PER_CLASS),
    "validation": ("data/val.txt", VAL_PER_CLASS),
    "test": ("data/test.txt", TEST_PER_CLASS),
}


class _IterStream:
    """File-like wrapper so tarfile can read a streamed HTTP body."""

    def __init__(self, iterator):
        self._iterator = iterator
        self._buffer = bytearray()

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            self._buffer.extend(b"".join(self._iterator))
            data = bytes(self._buffer)
            self._buffer.clear()
            return data
        while len(self._buffer) < size:
            try:
                chunk = next(self._iterator)
            except StopIteration:
                break
            if not chunk:
                break
            self._buffer.extend(chunk)
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data


def _canonical_features() -> Features:
    return Features(
        {
            "image": Image(),
            "label": ClassLabel(names=list(ALL_CLASSES)),
        }
    )


def _print_schema_from_features(features: Features) -> None:
    print("dataset['train'].features:")
    print(features)
    print("label names:")
    for index, name in enumerate(features["label"].names):
        print(f"  {index}: {name}")


def _try_print_hub_schema() -> bool:
    """Print schema via load_dataset when the Hub copy is script-free."""
    try:
        dataset = load_dataset(HF_DATASET_ID, streaming=True)
    except TypeError:
        try:
            dataset = load_dataset(HF_DATASET_ID, streaming=True, trust_remote_code=True)
        except Exception as exc:
            print(f"load_dataset({HF_DATASET_ID!r}) failed: {exc}")
            return False
    except Exception as exc:
        print(f"load_dataset({HF_DATASET_ID!r}) failed: {exc}")
        return False
    _print_schema_from_features(dataset["train"].features)
    return True


def _load_split_lookup() -> dict[str, tuple[str, str]]:
    """Map tar member path -> (split_name, class_name) for target classes."""
    lookup: dict[str, tuple[str, str]] = {}
    for split_name, (filename, _quota) in SPLIT_FILES.items():
        local = hf_hub_download(
            repo_id=HF_DATASET_ID, filename=filename, repo_type="dataset"
        )
        with open(local, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rel_path, class_id_str = line.rsplit(" ", 1)
                class_name = ALL_CLASSES[int(class_id_str)]
                if class_name not in TARGET_CLASSES:
                    continue
                member_name = IMAGES_PREFIX + rel_path.replace("\\", "/")
                lookup[member_name] = (split_name, class_name)
    return lookup


def _quotas_filled(buckets: dict[str, dict[str, list]]) -> bool:
    for split_name, (_filename, quota) in SPLIT_FILES.items():
        for class_name in TARGET_CLASSES:
            if len(buckets[split_name][class_name]) < quota:
                return False
    return True


def _stream_archive(lookup: dict[str, tuple[str, str]]) -> dict[str, dict[str, list]]:
    buckets: dict[str, dict[str, list]] = {
        split_name: {name: [] for name in TARGET_CLASSES}
        for split_name in SPLIT_FILES
    }
    url = hf_hub_url(
        HF_DATASET_ID, filename=ARCHIVE_FILENAME, repo_type="dataset"
    )
    headers = build_hf_headers()
    print(f"Streaming {ARCHIVE_FILENAME} from {HF_DATASET_ID} ...")
    response = requests.get(
        url, stream=True, headers=headers, timeout=120, allow_redirects=True
    )
    response.raise_for_status()
    scanned = 0
    kept = 0
    try:
        stream = _IterStream(response.iter_content(1024 * 1024))
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                scanned += 1
                name = member.name.replace("\\", "/").lstrip("./")
                meta = lookup.get(name)
                if meta is None:
                    if scanned % 2000 == 0:
                        filled = {
                            split: {cls: len(rows) for cls, rows in classes.items()}
                            for split, classes in buckets.items()
                        }
                        print(
                            f"  scanned_files={scanned} kept={kept} filled={filled}"
                        )
                    continue
                split_name, class_name = meta
                quota = SPLIT_FILES[split_name][1]
                if len(buckets[split_name][class_name]) >= quota:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                image = (
                    PILImage.open(io.BytesIO(extracted.read()))
                    .convert("L")
                    .copy()
                )
                buckets[split_name][class_name].append(
                    {"image": image, "label": class_name}
                )
                kept += 1
                if _quotas_filled(buckets):
                    print(
                        f"  quotas filled after scanning {scanned} image files "
                        f"({kept} kept)."
                    )
                    break
                if scanned % 2000 == 0:
                    filled = {
                        split: {cls: len(rows) for cls, rows in classes.items()}
                        for split, classes in buckets.items()
                    }
                    print(
                        f"  scanned_files={scanned} kept={kept} filled={filled}"
                    )
    finally:
        response.close()
    return buckets


def _to_dataset(rows: list[dict]) -> Dataset:
    features = Features(
        {
            "image": Image(),
            "label": ClassLabel(names=list(TARGET_CLASSES)),
        }
    )
    return Dataset.from_list(rows, features=features)


def main() -> None:
    if (OUTPUT_DIR / "dataset_dict.json").exists():
        print(f"Subset already exists at {OUTPUT_DIR}; skipping download.")
        print("Delete that directory to force a fresh subsample.")
        return

    print("=== schema (before filtering) ===")
    if not _try_print_hub_schema():
        print("Using the canonical RVL-CDIP feature schema:")
        _print_schema_from_features(_canonical_features())

    missing = [name for name in TARGET_CLASSES if name not in ALL_CLASSES]
    if missing:
        raise SystemExit(f"Target classes not in RVL-CDIP: {missing}")

    print("Loading split label maps (target classes only)...")
    lookup = _load_split_lookup()
    print(f"  {len(lookup)} target-class paths across train/val/test")

    buckets = _stream_archive(lookup)

    splits = {}
    for split_name, (_filename, quota) in SPLIT_FILES.items():
        rows: list[dict] = []
        print(f"=== {split_name} ===")
        for class_name in TARGET_CLASSES:
            got = len(buckets[split_name][class_name])
            print(f"  {class_name}: {got}/{quota}")
            if got < quota:
                print(
                    f"  WARNING: {split_name}/{class_name} is short "
                    f"({got} < {quota}); using what was found."
                )
            rows.extend(buckets[split_name][class_name])
        splits[split_name] = _to_dataset(rows)

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving subset to {OUTPUT_DIR} ...")
    DatasetDict(splits).save_to_disk(str(OUTPUT_DIR))
    print("Done.")
    for split_name, split in splits.items():
        print(f"  {split_name}: {len(split)} rows")


if __name__ == "__main__":
    main()
