"""
baseline.py
==============
TF-IDF + Logistic Regression Baseline

Run:
    python baseline.py

Inputs (from data/processed/):
    fpb_train.csv, fpb_validation.csv, fpb_test.csv
    fiqa_train.csv, fiqa_validation.csv, fiqa_test.csv

Outputs (saved to results/):
    baseline_fpb_metrics.txt        — classification report for FPB
    baseline_fiqa_metrics.txt       — classification report for FiQA
    baseline_cross_eval.csv         — accuracy + macro-F1 for all 4 eval combos
    baseline_confusion_fpb.png      — confusion matrix on FPB test
    baseline_confusion_fiqa.png     — confusion matrix on FiQA test
    baseline_predictions_fpb.csv    — sentence + true label + predicted label
    baseline_predictions_fiqa.csv   — same for FiQA
"""

import logging
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR  = Path("models/tfidf_lr")
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["negative", "neutral", "positive"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load processed splits
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation/test CSVs for a given dataset (fpb or fiqa)."""
    train = pd.read_csv(DATA_DIR / f"{dataset}_train.csv")
    val   = pd.read_csv(DATA_DIR / f"{dataset}_validation.csv")
    test  = pd.read_csv(DATA_DIR / f"{dataset}_test.csv")
    log.info(f"Loaded {dataset.upper()}: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build and train the pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    TF-IDF (unigrams + bigrams) + Logistic Regression.

    Key decisions:
      - max_features=20000 : cap vocabulary to avoid overfitting on rare tokens
      - ngram_range=(1,2)  : capture common two-word financial phrases
      - class_weight='balanced' : compensate for FPB neutral dominance
      - C=1.0              : default regularisation; tuned via validation below
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,      # apply 1 + log(tf) scaling
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]*\b",  # letters only
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
        )),
    ])


def tune_regularisation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    c_values: list[float] = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
) -> float:
    """
    Grid search over C (regularisation strength) using macro-F1 on the
    validation set. Returns the best C value.
    """
    log.info("Tuning regularisation (C) on validation set...")
    best_c, best_f1 = 1.0, 0.0
    rows = []

    for c in c_values:
        pipe = build_pipeline()
        pipe.set_params(clf__C=c)
        pipe.fit(train_df["sentence"], train_df["label"])
        preds = pipe.predict(val_df["sentence"])
        f1 = f1_score(val_df["label"], preds, average="macro")
        rows.append({"C": c, "macro_f1": round(f1, 4)})
        log.info(f"  C={c:<6}  macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_c = f1, c

    log.info(f"Best C={best_c}  (macro-F1={best_f1:.4f})")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "baseline_c_tuning.csv", index=False)
    return best_c


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    pipeline: Pipeline,
    df: pd.DataFrame,
    train_name: str,
    eval_name: str,
) -> dict:
    """
    Run predictions and compute accuracy + macro-F1.
    Saves a classification report text file and confusion matrix PNG.
    Returns a dict of metrics.
    """
    preds = pipeline.predict(df["sentence"])
    acc   = accuracy_score(df["label"], preds)
    f1    = f1_score(df["label"], preds, average="macro")
    report = classification_report(
        df["label"], preds,
        target_names=LABEL_NAMES,
        digits=4,
    )

    label = f"trained_on_{train_name}_eval_on_{eval_name}"
    print(f"\n{'='*55}")
    print(f"TF-IDF+LR  |  trained on {train_name.upper()}  →  evaluated on {eval_name.upper()}")
    print(f"{'='*55}")
    print(report)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")

    # Save report
    report_path = RESULTS_DIR / f"baseline_{label}_metrics.txt"
    report_path.write_text(
        f"TF-IDF + LR | trained={train_name.upper()} | eval={eval_name.upper()}\n"
        f"{'='*55}\n{report}\n"
        f"Accuracy : {acc:.4f}\nMacro-F1 : {f1:.4f}\n"
    )
    log.info(f"  Saved {report_path}")

    # Confusion matrix
    cm   = confusion_matrix(df["label"], preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"TF-IDF+LR  |  train={train_name.upper()}  eval={eval_name.upper()}\n"
        f"Acc={acc:.3f}  Macro-F1={f1:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"baseline_{label}_confusion.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {cm_path}")

    # Predictions CSV
    pred_df = df[["sentence", "label"]].copy()
    pred_df["predicted"] = preds
    pred_df["correct"]   = (pred_df["label"] == pred_df["predicted"]).astype(int)
    pred_path = RESULTS_DIR / f"baseline_{label}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info(f"  Saved {pred_path}")

    return {
        "train_on": train_name,
        "eval_on":  eval_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Phase 2 — TF-IDF + Logistic Regression Baseline")
    print("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    fpb_train,  fpb_val,  fpb_test  = load_splits("fpb")
    fiqa_train, fiqa_val, fiqa_test = load_splits("fiqa")

    # Combine train + validation for final model training
    fpb_trainval  = pd.concat([fpb_train,  fpb_val],  ignore_index=True)
    fiqa_trainval = pd.concat([fiqa_train, fiqa_val], ignore_index=True)

    # ── Tune and train on FPB ─────────────────────────────────────────────────
    log.info("\nTraining on FPB...")
    best_c_fpb = tune_regularisation(fpb_train, fpb_val)

    fpb_pipeline = build_pipeline()
    fpb_pipeline.set_params(clf__C=best_c_fpb)
    fpb_pipeline.fit(fpb_trainval["sentence"], fpb_trainval["label"])

    # Save model
    with open(MODELS_DIR / "tfidf_lr_fpb.pkl", "wb") as f:
        pickle.dump(fpb_pipeline, f)
    log.info("  Model saved to models/tfidf_lr/tfidf_lr_fpb.pkl")

    # ── Tune and train on FiQA ────────────────────────────────────────────────
    log.info("\nTraining on FiQA...")
    best_c_fiqa = tune_regularisation(fiqa_train, fiqa_val)

    fiqa_pipeline = build_pipeline()
    fiqa_pipeline.set_params(clf__C=best_c_fiqa)
    fiqa_pipeline.fit(fiqa_trainval["sentence"], fiqa_trainval["label"])

    with open(MODELS_DIR / "tfidf_lr_fiqa.pkl", "wb") as f:
        pickle.dump(fiqa_pipeline, f)
    log.info("  Model saved to models/tfidf_lr/tfidf_lr_fiqa.pkl")

    # ── Cross-dataset evaluation ──────────────────────────────────────────────
    # Evaluate all 4 combos: (train_fpb, eval_fpb), (train_fpb, eval_fiqa),
    #                         (train_fiqa, eval_fpb), (train_fiqa, eval_fiqa)
    log.info("\nRunning cross-dataset evaluation...")
    results = []
    results.append(evaluate(fpb_pipeline,  fpb_test,  "fpb",  "fpb"))
    results.append(evaluate(fpb_pipeline,  fiqa_test, "fpb",  "fiqa"))
    results.append(evaluate(fiqa_pipeline, fpb_test,  "fiqa", "fpb"))
    results.append(evaluate(fiqa_pipeline, fiqa_test, "fiqa", "fiqa"))

    # ── Cross-eval summary table ──────────────────────────────────────────────
    cross_df = pd.DataFrame(results)
    cross_df.to_csv(RESULTS_DIR / "baseline_cross_eval.csv", index=False)

    print("\n" + "=" * 55)
    print("CROSS-DATASET EVALUATION SUMMARY")
    print("=" * 55)
    print(cross_df.to_string(index=False))
    print("\nNote: off-diagonal rows measure generalisation across datasets.")
    print("      These numbers will be compared against BERT and FinBERT later.")
