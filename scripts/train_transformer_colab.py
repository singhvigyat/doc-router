"""Fine-tune DistilBERT on the same OCR CSV the Stage 2 baseline used.

This file is written to be copied into Google Colab / Kaggle as well as run
from this repo. The coding-agent environment has no GPU (and no PyTorch), so
training is expected to happen on a free T4, not here.

Colab (typical):
  1. Runtime → Change runtime type → T4 GPU
  2. Upload data/processed/rvl_cdip_text.csv and src/evaluation/metrics.py
  3. Run this script (or the notebook that wraps it)
  4. Download the saved folder into models_store/distilbert_classifier/

Repo (if you have a local GPU):
  python scripts/train_transformer_colab.py
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Same 8 enterprise classes, same order as scripts/download_data.py TARGET_CLASSES
# and the HuggingFace ClassLabel on data/processed/rvl_cdip_subset.
LABELS = [
    "letter",
    "form",
    "email",
    "scientific report",
    "budget",
    "invoice",
    "resume",
    "memo",
]
LABEL2ID = {name: index for index, name in enumerate(LABELS)}
ID2LABEL = {index: name for index, name in enumerate(LABELS)}
KNOWN_SPLITS = {"train", "validation", "val", "test"}
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _repo_root() -> Path:
    """Prefer the git repo; fall back to Colab's /content when pasted elsewhere."""
    try:
        here = Path(__file__).resolve()
        candidate = here.parents[1]
        if (candidate / "src" / "evaluation" / "metrics.py").exists():
            return candidate
    except NameError:
        pass
    return Path("/content") if _in_colab() else Path.cwd()


def _split_from_doc_id(doc_id: str) -> str | None:
    """Same prefix rule as scripts/train_baseline.py (Stage 2)."""
    prefix = str(doc_id).split("_", 1)[0]
    return prefix if prefix in KNOWN_SPLITS else None


def import_stage2_metrics(metrics_file: Path | None):
    """Import Stage 2's helpers. Do not reimplement them."""
    try:
        from evaluation.metrics import (  # noqa: WPS433
            compute_classification_metrics,
            plot_confusion_matrix,
        )
        return compute_classification_metrics, plot_confusion_matrix
    except ImportError:
        pass

    search_files = []
    if metrics_file is not None:
        search_files.append(Path(metrics_file))
    root = _repo_root()
    search_files.extend(
        [
            root / "src" / "evaluation" / "metrics.py",
            Path("/content/src/evaluation/metrics.py"),
            Path("/content/metrics.py"),
            Path.cwd() / "metrics.py",
        ]
    )

    src_dir = root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        try:
            from evaluation.metrics import (  # noqa: WPS433
                compute_classification_metrics,
                plot_confusion_matrix,
            )
            return compute_classification_metrics, plot_confusion_matrix
        except ImportError:
            pass

    for path in search_files:
        if path is None or not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("stage2_metrics", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.compute_classification_metrics, module.plot_confusion_matrix

    raise SystemExit(
        "Could not import src/evaluation/metrics.py. In Colab, upload that "
        "file (or the whole src/ folder) — this script reuses it, it does "
        "not copy the metric code."
    )


def load_frames(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild Stage 2's train / validation / test membership from doc_id."""
    if not csv_path.exists():
        raise SystemExit(
            f"Missing {csv_path}. Upload data/processed/rvl_cdip_text.csv "
            "or pass --csv."
        )
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise SystemExit(f"{csv_path} is empty.")

    unknown = sorted(set(frame["label"].astype(str)) - set(LABELS))
    if unknown:
        raise SystemExit(
            f"CSV has labels that are not in the Stage 2 class list: {unknown}"
        )

    frame["split"] = frame["doc_id"].map(_split_from_doc_id)
    if frame["split"].notna().all():
        train_frame = frame[frame["split"] == "train"].copy()
        val_frame = frame[frame["split"].isin(["validation", "val"])].copy()
        test_frame = frame[frame["split"] == "test"].copy()
        if test_frame.empty:
            test_frame = val_frame
            print("No test_* rows; using validation as the held-out set.")
        print(
            f"Using CSV split prefixes: train={len(train_frame)} "
            f"validation={len(val_frame)} test={len(test_frame)}"
        )
        return train_frame, val_frame, test_frame

    from sklearn.model_selection import train_test_split

    print("doc_id prefixes missing; falling back to stratified 80/10/10.")
    train_frame, holdout = train_test_split(
        frame, test_size=0.2, random_state=SEED, stratify=frame["label"]
    )
    val_frame, test_frame = train_test_split(
        holdout, test_size=0.5, random_state=SEED, stratify=holdout["label"]
    )
    return train_frame, val_frame, test_frame


def frame_to_dataset(frame: pd.DataFrame) -> Dataset:
    work = pd.DataFrame(
        {
            "text": frame["text"].fillna("").astype(str),
            "labels": frame["label"].astype(str).map(LABEL2ID).astype(int),
        }
    )
    if work["labels"].isna().any():
        raise SystemExit("A row could not be mapped onto the Stage 2 label ids.")
    return Dataset.from_pandas(work, preserve_index=False)


def tokenize_batch(batch, tokenizer):
    """Subword-tokenize a batch of raw OCR strings.

    truncation=True + max_length=512 drops tokens past DistilBERT's window.
    Padding is left to DataCollatorWithPadding so each *batch* pads to its
    longest sequence instead of wasting compute on 512-token pads everywhere.
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def make_training_arguments(output_dir: Path, batch_size: int) -> TrainingArguments:
    """Build TrainingArguments with a transformers-version-safe eval key."""
    use_gpu = torch.cuda.is_available()
    kwargs = dict(
        output_dir=str(output_dir / "training_run"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=use_gpu,  # mixed precision only when a GPU is actually present
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        seed=SEED,
        report_to="none",  # skip wandb / hub prompts in Colab
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def compute_metrics_factory(accuracy_metric, f1_metric):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(
                predictions=predictions, references=labels
            )["accuracy"],
            "f1": f1_metric.compute(
                predictions=predictions, references=labels, average="macro"
            )["f1"],
        }

    return compute_metrics


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    default_csv = root / "data" / "processed" / "rvl_cdip_text.csv"
    if _in_colab() and not default_csv.exists():
        default_csv = Path("/content/rvl_cdip_text.csv")
    default_out = root / "models_store" / "distilbert_classifier"
    if _in_colab():
        default_out = Path("/content/distilbert_classifier")
    default_results = root / "results"
    if _in_colab():
        default_results = Path("/content/results")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16 if torch.cuda.is_available() else 4,
        help="Prompt asks for ~16; drops to 4 on CPU to keep RAM in check.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    print("torch", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    else:
        print(
            "WARNING: no GPU. DistilBERT on this 3.2k-doc set takes hours on "
            "CPU. In Colab: Runtime → Change runtime type → T4 GPU."
        )

    compute_classification_metrics, plot_confusion_matrix = import_stage2_metrics(
        args.metrics_file
    )

    train_frame, val_frame, test_frame = load_frames(args.csv)
    if train_frame.empty or test_frame.empty:
        raise SystemExit("Need non-empty train and test sets.")
    if val_frame.empty:
        print("No validation split; scoring Trainer eval on a 10% slice of train.")
        val_frame = train_frame.sample(frac=0.1, random_state=SEED)

    raw_datasets = DatasetDict(
        {
            "train": frame_to_dataset(train_frame),
            "validation": frame_to_dataset(val_frame),
            "test": frame_to_dataset(test_frame),
        }
    )
    print(raw_datasets)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def _tokenize(batch):
        return tokenize_batch(batch, tokenizer)

    tokenized = raw_datasets.map(_tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    import evaluate

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    training_args = make_training_arguments(args.output_dir, args.batch_size)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_factory(accuracy_metric, f1_metric),
    )
    # transformers 4.46+ renamed tokenizer → processing_class
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    # Held-out test split — same rows Stage 2 scored, for a fair comparison.
    test_output = trainer.predict(tokenized["test"])
    y_pred_ids = np.argmax(test_output.predictions, axis=-1)
    y_true_ids = np.array(tokenized["test"]["labels"])
    y_pred = [ID2LABEL[int(i)] for i in y_pred_ids]
    y_true = [ID2LABEL[int(i)] for i in y_true_ids]

    metrics = compute_classification_metrics(y_true, y_pred)
    report_path = args.results_dir / "transformer_classification_report.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
        save_path=args.results_dir / "transformer_confusion_matrix.png",
    )

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(
        json.dumps(
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "model_dir": str(args.output_dir),
                "report_path": str(report_path),
            },
            indent=2,
        )
    )
    if _in_colab():
        print(
            "\nDownload the folder "
            f"{args.output_dir} and place it at "
            "models_store/distilbert_classifier/ in the repo "
            "(config.json, model.safetensors / pytorch_model.bin, and the "
            "tokenizer files should sit *directly* in that folder, not one "
            "level nested)."
        )


if __name__ == "__main__":
    main()
