"""
bert_finetune.py
===================
BERT Fine-Tuning on FPB and cross-evaluation on FiQA

Designed to run on Google Colab with Google Drive integration.

Setup (run these in a Colab cell before this script):
    from google.colab import drive
    drive.mount('/content/drive')

    !pip install transformers datasets scikit-learn accelerate evaluate -q
    !pip install "datasets>=2.19.0,<3.0.0" -q

    # Upload your data/processed/ CSVs to Drive and set DATA_DIR below

Run:
    python bert_finetune.py --run 1
    python bert_finetune.py --run 2
    python bert_finetune.py --run 3

Hyperparameter runs:
    Run 1: lr=2e-5, epochs=3, batch=16
    Run 2: lr=3e-5, epochs=4, batch=32
    Run 3: lr=2e-5, epochs=5, batch=16

Outputs (saved to RESULTS_DIR):
    bert_run{N}_fpb_metrics.txt          — classification report on FPB test
    bert_run{N}_fiqa_metrics.txt         — classification report on FiQA test
    bert_run{N}_fpb_confusion.png        — confusion matrix on FPB
    bert_run{N}_fiqa_confusion.png       — confusion matrix on FiQA
    bert_run{N}_fpb_predictions.csv      — predictions on FPB test
    bert_run{N}_fiqa_predictions.csv     — predictions on FiQA test
    bert_run{N}_training_loss.png        — training + validation loss curve
    bert_cross_eval_summary.csv          — updated after each run
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths — update DATA_DIR if running on Colab with Drive ───────────────────
DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("/content/drive/MyDrive/NLP_Project/results")
MODELS_DIR  = Path("/content/drive/MyDrive/NLP_Project/models/bert")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME  = "bert-base-uncased"
LABEL_NAMES = ["negative", "neutral", "positive"]
MAX_LENGTH  = 128

# ── Hyperparameter runs ───────────────────────────────────────────────────────
RUNS = {
    1: {"lr": 2e-5, "epochs": 3, "batch": 16},
    2: {"lr": 3e-5, "epochs": 4, "batch": 32},
    3: {"lr": 2e-5, "epochs": 5, "batch": 16},
}


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        log.info("Using Apple MPS backend")
    else:
        device = "cpu"
        log.warning("No GPU found — training on CPU will be very slow")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class FinancialSentimentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH):
        self.labels = df["label"].tolist()
        self.encodings = tokenizer(
            df["sentence"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="macro"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / f"{dataset}_train.csv")
    val   = pd.read_csv(DATA_DIR / f"{dataset}_validation.csv")
    test  = pd.read_csv(DATA_DIR / f"{dataset}_test.csv")
    log.info(f"Loaded {dataset.upper()}: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    run: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    hparams: dict,
    device: str,
) -> tuple:
    """Fine-tune BERT on the given training data. Returns (model, history)."""

    log.info(f"\nRun {run} — lr={hparams['lr']}, epochs={hparams['epochs']}, batch={hparams['batch']}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id={l: i for i, l in enumerate(LABEL_NAMES)},
    )

    train_dataset = FinancialSentimentDataset(train_df, tokenizer)
    val_dataset   = FinancialSentimentDataset(val_df,   tokenizer)

    output_dir = str(MODELS_DIR / f"run{run}")

    # fp16 only on CUDA; bf16 not used for compatibility
    use_fp16 = (device == "cuda")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hparams["epochs"],
        per_device_train_batch_size=hparams["batch"],
        per_device_eval_batch_size=32,
        learning_rate=hparams["lr"],
        warmup_steps=200,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=str(RESULTS_DIR / f"logs_run{run}"),
        logging_steps=10,
        fp16=use_fp16,
        report_to="none",           # disable wandb / tensorboard
        save_total_limit=2,         # keep only the 2 best checkpoints
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    log.info("Starting training...")
    train_result = trainer.train()
    log.info(f"Training complete. Best checkpoint: {trainer.state.best_model_checkpoint}")

    return trainer.model, trainer.state.log_history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    run: int,
    eval_name: str,
    device: str,
) -> dict:
    """Run inference and compute metrics on a test split."""

    model.eval()
    model.to(device)

    dataset = FinancialSentimentDataset(test_df, tokenizer)
    all_preds = []

    with torch.no_grad():
        for i in range(len(dataset)):
            item = {k: v.unsqueeze(0).to(device) for k, v in dataset[i].items() if k != "labels"}
            outputs = model(**item)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            all_preds.append(pred)

    true_labels = test_df["label"].tolist()
    acc    = accuracy_score(true_labels, all_preds)
    f1     = f1_score(true_labels, all_preds, average="macro")
    report = classification_report(
        true_labels, all_preds,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    print(f"\n{'='*55}")
    print(f"BERT Run {run}  →  evaluated on {eval_name.upper()}")
    print(f"{'='*55}")
    print(report)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")

    # Save report
    report_path = RESULTS_DIR / f"bert_run{run}_{eval_name}_metrics.txt"
    report_path.write_text(
        f"BERT Run {run} | eval={eval_name.upper()}\n"
        f"lr={RUNS[run]['lr']}  epochs={RUNS[run]['epochs']}  batch={RUNS[run]['batch']}\n"
        f"{'='*55}\n{report}\n"
        f"Accuracy : {acc:.4f}\nMacro-F1 : {f1:.4f}\n"
    )
    log.info(f"  Saved {report_path}")

    # Confusion matrix
    cm   = confusion_matrix(true_labels, all_preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"BERT Run {run} → {eval_name.upper()}\n"
        f"Acc={acc:.3f}  Macro-F1={f1:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"bert_run{run}_{eval_name}_confusion.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {cm_path}")

    # Predictions CSV
    pred_df = test_df[["sentence", "label"]].copy()
    pred_df["predicted"] = all_preds
    pred_df["correct"]   = (pred_df["label"] == pred_df["predicted"]).astype(int)
    pred_path = RESULTS_DIR / f"bert_run{run}_{eval_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info(f"  Saved {pred_path}")

    return {
        "run":      run,
        "lr":       RUNS[run]["lr"],
        "epochs":   RUNS[run]["epochs"],
        "batch":    RUNS[run]["batch"],
        "eval_on":  eval_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loss curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(history: list, run: int) -> None:
    """Plot training and validation loss over epochs."""

    train_losses = [(h["step"], h["loss"])       for h in history if "loss"      in h and "eval_loss" not in h]
    eval_losses  = [(h["epoch"], h["eval_loss"])  for h in history if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(8, 4))

    if train_losses:
        steps, losses = zip(*train_losses)
        ax.plot(steps, losses, label="Train loss", color="steelblue", linewidth=1.2)

    if eval_losses:
        epochs, losses = zip(*eval_losses)
        ax.plot(epochs, losses, label="Val loss", color="darkorange",
                linewidth=2, marker="o")

    ax.set_xlabel("Step / Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"BERT Run {run} — Training Curve\n"
                 f"lr={RUNS[run]['lr']}  epochs={RUNS[run]['epochs']}  batch={RUNS[run]['batch']}")
    ax.legend()
    plt.tight_layout()

    out = RESULTS_DIR / f"bert_run{run}_training_loss.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Update cross-eval summary
# ─────────────────────────────────────────────────────────────────────────────

def update_summary(new_rows: list[dict]) -> None:
    summary_path = RESULTS_DIR / "bert_cross_eval_summary.csv"

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        updated  = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        updated = pd.DataFrame(new_rows)

    updated.to_csv(summary_path, index=False)

    print("\n" + "=" * 55)
    print("BERT CROSS-EVAL SUMMARY (all runs so far)")
    print("=" * 55)
    print(updated.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=int, choices=[1, 2, 3], required=True,
        help="Which hyperparameter run to execute (1, 2, or 3)"
    )
    args = parser.parse_args()
    run  = args.run

    print("=" * 55)
    print(f"Phase 3 — BERT Fine-Tuning  |  Run {run}/3")
    print(f"  lr={RUNS[run]['lr']}  epochs={RUNS[run]['epochs']}  batch={RUNS[run]['batch']}")
    print("=" * 55)

    device = get_device()

    # Load tokenizer once
    log.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    fpb_train,  fpb_val,  fpb_test  = load_splits("fpb")
    fiqa_train, fiqa_val, fiqa_test = load_splits("fiqa")

    # Combine train + val for this run
    fpb_trainval = pd.concat([fpb_train, fpb_val], ignore_index=True)

    # Train on FPB
    model, history = train(run, fpb_trainval, fpb_val, tokenizer, RUNS[run], device)

    # Plot loss curve
    plot_loss_curve(history, run)

    # Evaluate on FPB test and FiQA test
    results = []
    results.append(evaluate(model, tokenizer, fpb_test,  run, "fpb",  device))
    results.append(evaluate(model, tokenizer, fiqa_test, run, "fiqa", device))

    # Save model checkpoint
    model_save_path = MODELS_DIR / f"run{run}_final"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    log.info(f"Model saved to {model_save_path}")

    # Update running summary
    update_summary(results)

    print(f"\n  Run {run} complete.")
    if run < 3:
        print(f"    Next: python bert_finetune.py --run {run + 1}")
    else:
        print("    All 3 runs complete. Check bert_cross_eval_summary.csv for results.")
