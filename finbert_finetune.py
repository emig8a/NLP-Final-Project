"""
finbert_finetune.py
======================
Phase 4: FinBERT Fine-Tuning and Zero-Shot Evaluation

FinBERT (ProsusAI/finbert) is BERT already fine-tuned on Financial PhraseBank.
We run three experiments:

  Experiment A — Zero-shot evaluation
      Use FinBERT out-of-the-box (no additional training) on both FPB and FiQA.
      This tells us how well domain pre-training alone performs.

  Experiment B — Fine-tune FinBERT on FiQA
      Re-initialize the classification head and fine-tune on FiQA.
      This tests whether domain knowledge + task-specific fine-tuning helps.

  Experiment C — Fine-tune FinBERT on FPB
      Fine-tune on FPB to see if we can squeeze more out of an already
      strong model on its native dataset.

Run:
    python finbert_finetune.py --exp A
    python finbert_finetune.py --exp B
    python finbert_finetune.py --exp C

Outputs (saved to results/):
    finbert_expA_fpb_metrics.txt / fiqa_metrics.txt
    finbert_expB_fpb_metrics.txt / fiqa_metrics.txt
    finbert_expC_fpb_metrics.txt / fiqa_metrics.txt
    finbert_confusion_*.png
    finbert_predictions_*.csv
    finbert_summary.csv             — full cross-eval table updated after each exp
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
    pipeline,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths — update if running on Colab with Drive ─────────────────────────────
DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("/content/drive/MyDrive/NLP_Project/results")
MODELS_DIR  = Path("/content/drive/MyDrive/NLP_Project/models/finbert")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"
LABEL_NAMES   = ["negative", "neutral", "positive"]
MAX_LENGTH    = 128

# FinBERT uses a different label order than our convention
# ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
FINBERT_LABEL_MAP = {"positive": 2, "negative": 0, "neutral": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    elif torch.backends.mps.is_available():
        log.info("Using Apple MPS backend")
        return "mps"
    else:
        log.warning("No GPU found — running on CPU")
        return "cpu"


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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / f"{dataset}_train.csv")
    val   = pd.read_csv(DATA_DIR / f"{dataset}_validation.csv")
    test  = pd.read_csv(DATA_DIR / f"{dataset}_test.csv")
    log.info(f"Loaded {dataset.upper()}: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


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
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    exp: str,
    eval_name: str,
    device: str,
    label_names: list = LABEL_NAMES,
) -> dict:
    """Run inference and compute metrics. Save report, confusion matrix, predictions."""

    model.eval()
    model.to(device)

    dataset  = FinancialSentimentDataset(test_df, tokenizer)
    all_preds = []

    with torch.no_grad():
        for i in range(len(dataset)):
            item = {k: v.unsqueeze(0).to(device)
                    for k, v in dataset[i].items() if k != "labels"}
            outputs = model(**item)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            all_preds.append(pred)

    true_labels = test_df["label"].tolist()
    acc    = accuracy_score(true_labels, all_preds)
    f1     = f1_score(true_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        true_labels, all_preds,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )

    print(f"\n{'='*55}")
    print(f"FinBERT Exp {exp}  →  evaluated on {eval_name.upper()}")
    print(f"{'='*55}")
    print(report)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")

    # Save report
    report_path = RESULTS_DIR / f"finbert_exp{exp}_{eval_name}_metrics.txt"
    report_path.write_text(
        f"FinBERT Exp {exp} | eval={eval_name.upper()}\n"
        f"{'='*55}\n{report}\n"
        f"Accuracy : {acc:.4f}\nMacro-F1 : {f1:.4f}\n"
    )
    log.info(f"  Saved {report_path}")

    # Confusion matrix
    cm   = confusion_matrix(true_labels, all_preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(
        f"FinBERT Exp {exp} → {eval_name.upper()}\n"
        f"Acc={acc:.3f}  Macro-F1={f1:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"finbert_exp{exp}_{eval_name}_confusion.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {cm_path}")

    # Predictions CSV
    pred_df = test_df[["sentence", "label"]].copy()
    pred_df["predicted"] = all_preds
    pred_df["correct"]   = (pred_df["label"] == pred_df["predicted"]).astype(int)
    pred_path = RESULTS_DIR / f"finbert_exp{exp}_{eval_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info(f"  Saved {pred_path}")

    return {
        "experiment": f"Exp {exp}",
        "eval_on":    eval_name,
        "accuracy":   round(acc, 4),
        "macro_f1":   round(f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning helper
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    exp: str,
    device: str,
    lr: float = 2e-5,
    epochs: int = 4,
    batch: int = 16,
):
    """Fine-tune FinBERT on the given dataset. Returns the trained model."""

    log.info(f"\nFine-tuning FinBERT — Exp {exp} | lr={lr}, epochs={epochs}, batch={batch}")

    # Re-initialize classification head for the new task
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL,
        num_labels=3,
        ignore_mismatched_sizes=True,
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id={l: i for i, l in enumerate(LABEL_NAMES)},
    )

    train_dataset = FinancialSentimentDataset(train_df, tokenizer)
    val_dataset   = FinancialSentimentDataset(val_df,   tokenizer)

    output_dir = str(MODELS_DIR / f"exp{exp}")
    use_fp16   = (device == "cuda")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=32,
        learning_rate=lr,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_steps=10,
        fp16=use_fp16,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    log.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

    # Save final model
    save_path = MODELS_DIR / f"exp{exp}_final"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    log.info(f"Model saved to {save_path}")

    return trainer.model


# ─────────────────────────────────────────────────────────────────────────────
# Update summary
# ─────────────────────────────────────────────────────────────────────────────

def update_summary(new_rows: list[dict]) -> None:
    summary_path = RESULTS_DIR / "finbert_summary.csv"

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        updated  = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        updated = pd.DataFrame(new_rows)

    updated.to_csv(summary_path, index=False)

    print("\n" + "=" * 55)
    print("FINBERT SUMMARY (all experiments so far)")
    print("=" * 55)
    print(updated.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A — Zero-shot evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_exp_A(fpb_test, fiqa_test, device) -> list[dict]:
    """
    Use FinBERT out-of-the-box — no additional fine-tuning.
    FinBERT was already trained on FPB so this is essentially its native task.
    """
    log.info("\n" + "="*55)
    log.info("Experiment A — FinBERT Zero-Shot (no additional training)")
    log.info("="*55)

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

    # FinBERT's native label order is different: positive=0, negative=1, neutral=2
    # We remap predictions to our convention: negative=0, neutral=1, positive=2
    finbert_to_ours = {0: 2, 1: 0, 2: 1}  # finbert_id -> our_id

    def remap_predictions(model, tokenizer, test_df, device):
        model.eval()
        model.to(device)
        dataset   = FinancialSentimentDataset(test_df, tokenizer)
        all_preds = []
        with torch.no_grad():
            for i in range(len(dataset)):
                item = {k: v.unsqueeze(0).to(device)
                        for k, v in dataset[i].items() if k != "labels"}
                outputs = model(**item)
                raw_pred     = torch.argmax(outputs.logits, dim=-1).item()
                mapped_pred  = finbert_to_ours[raw_pred]
                all_preds.append(mapped_pred)
        return all_preds

    results = []
    for test_df, eval_name in [(fpb_test, "fpb"), (fiqa_test, "fiqa")]:
        preds = remap_predictions(model, tokenizer, test_df, device)
        true  = test_df["label"].tolist()
        acc   = accuracy_score(true, preds)
        f1    = f1_score(true, preds, average="macro", zero_division=0)
        report = classification_report(true, preds, target_names=LABEL_NAMES,
                                       digits=4, zero_division=0)

        print(f"\n{'='*55}")
        print(f"FinBERT Exp A (zero-shot)  →  {eval_name.upper()}")
        print(f"{'='*55}")
        print(report)
        print(f"Accuracy : {acc:.4f}")
        print(f"Macro-F1 : {f1:.4f}")

        # Save
        rp = RESULTS_DIR / f"finbert_expA_{eval_name}_metrics.txt"
        rp.write_text(f"FinBERT Exp A (zero-shot) | eval={eval_name.upper()}\n"
                      f"{'='*55}\n{report}\nAccuracy : {acc:.4f}\nMacro-F1 : {f1:.4f}\n")

        cm   = confusion_matrix(true, preds, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False, cmap="Greens")
        ax.set_title(f"FinBERT Exp A (zero-shot) → {eval_name.upper()}\n"
                     f"Acc={acc:.3f}  Macro-F1={f1:.3f}", fontsize=11)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"finbert_expA_{eval_name}_confusion.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        pred_df = test_df[["sentence", "label"]].copy()
        pred_df["predicted"] = preds
        pred_df["correct"]   = (pred_df["label"] == pred_df["predicted"]).astype(int)
        pred_df.to_csv(RESULTS_DIR / f"finbert_expA_{eval_name}_predictions.csv", index=False)

        results.append({"experiment": "Exp A (zero-shot)", "eval_on": eval_name,
                         "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", type=str, choices=["A", "B", "C"], required=True,
        help="A=zero-shot, B=fine-tune on FiQA, C=fine-tune on FPB"
    )
    args = parser.parse_args()

    print("=" * 55)
    print(f"Phase 4 — FinBERT  |  Experiment {args.exp}")
    desc = {"A": "Zero-Shot Evaluation",
            "B": "Fine-Tune on FiQA",
            "C": "Fine-Tune on FPB"}
    print(f"  {desc[args.exp]}")
    print("=" * 55)

    device = get_device()

    fpb_train,  fpb_val,  fpb_test  = load_splits("fpb")
    fiqa_train, fiqa_val, fiqa_test = load_splits("fiqa")

    if args.exp == "A":
        results = run_exp_A(fpb_test, fiqa_test, device)

    elif args.exp == "B":
        # Fine-tune FinBERT on FiQA
        tokenizer    = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        fiqa_trainval = pd.concat([fiqa_train, fiqa_val], ignore_index=True)
        model        = fine_tune(fiqa_trainval, fiqa_val, tokenizer,
                                 exp="B", device=device, lr=2e-5, epochs=4, batch=16)
        results = []
        results.append(evaluate(model, tokenizer, fpb_test,  "B", "fpb",  device))
        results.append(evaluate(model, tokenizer, fiqa_test, "B", "fiqa", device))

    elif args.exp == "C":
        # Fine-tune FinBERT on FPB
        tokenizer   = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        fpb_trainval = pd.concat([fpb_train, fpb_val], ignore_index=True)
        model       = fine_tune(fpb_trainval, fpb_val, tokenizer,
                                exp="C", device=device, lr=2e-5, epochs=4, batch=16)
        results = []
        results.append(evaluate(model, tokenizer, fpb_test,  "C", "fpb",  device))
        results.append(evaluate(model, tokenizer, fiqa_test, "C", "fiqa", device))

    update_summary(results)
