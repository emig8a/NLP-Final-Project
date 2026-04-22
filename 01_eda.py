"""
01_eda.py
=========
Phase 1: Data Collection, Inspection & EDA

Run:
    python 01_eda.py

Outputs (saved to results/):
    class_distribution.png
    class_distribution_summary.csv
    sentence_lengths.png
    vocab_overlap.png
    baseline_confusion.png
    spy_overview.png
    eda_summary.txt

Processed splits (saved to data/processed/):
    fpb_train.csv, fpb_validation.csv, fpb_test.csv
    fiqa_train.csv, fiqa_validation.csv, fiqa_test.csv
"""

import logging
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on all machines
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from data_utils import (
    ID2LABEL,
    LABEL_NAMES,
    dataset_to_df,
    fetch_market_data,
    load_fiqa,
    load_fpb,
    save_processed,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Directories ───────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
DATA_DIR    = Path("data")
RESULTS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "market").mkdir(parents=True, exist_ok=True)

# ── Plotting style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
PALETTE = {"negative": "#d62728", "neutral": "#aec7e8", "positive": "#2ca02c"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load datasets
# ─────────────────────────────────────────────────────────────────────────────

def load_all() -> tuple[dict, dict]:
    log.info("=" * 55)
    log.info("Step 1 — Loading datasets")
    log.info("=" * 55)

    fpb  = load_fpb(config="sentences_allagree")
    fiqa = load_fiqa()

    fpb_dfs  = dataset_to_df(fpb)
    fiqa_dfs = dataset_to_df(fiqa)

    for name, dfs in [("FPB", fpb_dfs), ("FiQA", fiqa_dfs)]:
        total = sum(len(d) for d in dfs.values())
        log.info(f"{name}: {total:,} total examples")
        for split, df in dfs.items():
            log.info(f"  {split:12s}: {len(df):,} rows")

    return fpb_dfs, fiqa_dfs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Class distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(fpb_dfs: dict, fiqa_dfs: dict) -> None:
    log.info("Step 2 — Class distribution")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Class Distribution by Split", fontsize=16, fontweight="bold")

    rows = []
    for row_i, (ds_name, dfs) in enumerate([("FPB", fpb_dfs), ("FiQA", fiqa_dfs)]):
        for col_i, split in enumerate(["train", "validation", "test"]):
            ax = axes[row_i][col_i]
            df = dfs[split]
            counts = df["label_str"].value_counts().reindex(LABEL_NAMES, fill_value=0)

            bars = ax.bar(
                counts.index, counts.values,
                color=[PALETTE[l] for l in counts.index],
                edgecolor="white", linewidth=0.8,
            )
            ax.set_title(f"{ds_name} — {split}")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")

            for bar, val in zip(bars, counts.values):
                pct = val / len(df) * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3,
                    f"{val}\n({pct:.0f}%)",
                    ha="center", va="bottom", fontsize=9,
                )

            pcts = (counts / len(df) * 100).round(1)
            rows.append({
                "Dataset": ds_name, "Split": split, "Total": len(df),
                **{f"{l} (%)": pcts[l] for l in LABEL_NAMES},
            })

    plt.tight_layout()
    out = RESULTS_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out}")

    summary = pd.DataFrame(rows).set_index(["Dataset", "Split"])
    csv_out = RESULTS_DIR / "class_distribution_summary.csv"
    summary.to_csv(csv_out)
    log.info(f"  Saved {csv_out}")
    print("\nClass distribution summary:")
    print(summary.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sentence length analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_sentence_lengths(fpb_dfs: dict, fiqa_dfs: dict) -> None:
    log.info("Step 3 — Sentence length analysis")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sentence Length (word count) by Class — Train Split",
                 fontsize=15, fontweight="bold")

    stats_rows = []
    for ax, (ds_name, dfs) in zip(axes, [("FPB", fpb_dfs), ("FiQA", fiqa_dfs)]):
        df = dfs["train"].copy()
        df["word_count"] = df["sentence"].str.split().str.len()

        sns.violinplot(
            data=df, x="label_str", y="word_count",
            order=LABEL_NAMES, palette=PALETTE, ax=ax, inner="quartile",
        )
        ax.set_title(f"{ds_name} (train)")
        ax.set_xlabel("Sentiment Class")
        ax.set_ylabel("Word Count")

        for label in LABEL_NAMES:
            wc = df.loc[df["label_str"] == label, "word_count"]
            stats_rows.append({
                "Dataset": ds_name, "Label": label,
                "Mean": round(wc.mean(), 1), "Median": round(wc.median(), 1),
                "Std": round(wc.std(), 1), "Max": int(wc.max()),
            })

    plt.tight_layout()
    out = RESULTS_DIR / "sentence_lengths.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out}")

    stats_df = pd.DataFrame(stats_rows).set_index(["Dataset", "Label"])
    print("\nSentence length stats (train split):")
    print(stats_df.to_string())

    # BERT token length check
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("\nBERT token length (max_length=128 check):")
        for ds_name, dfs in [("FPB", fpb_dfs), ("FiQA", fiqa_dfs)]:
            sentences = dfs["train"]["sentence"].tolist()
            lengths = [
                len(tokenizer.encode(s, add_special_tokens=True))
                for s in sentences
            ]
            over = sum(1 for l in lengths if l > 128)
            print(
                f"  {ds_name}: mean={np.mean(lengths):.1f} tokens | "
                f"max={max(lengths)} | >128: {over} ({over/len(lengths)*100:.1f}%)"
            )
    except Exception as e:
        log.warning(f"  Skipped token length check: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Vocabulary overlap
# ─────────────────────────────────────────────────────────────────────────────

def plot_vocab_overlap(fpb_dfs: dict, fiqa_dfs: dict) -> None:
    log.info("Step 4 — Vocabulary overlap")

    def get_vocab(df: pd.DataFrame, min_freq: int = 2) -> set:
        words: Counter = Counter()
        for sent in df["sentence"]:
            words.update(sent.lower().split())
        return {w for w, c in words.items() if c >= min_freq}

    fpb_vocab  = get_vocab(fpb_dfs["train"])
    fiqa_vocab = get_vocab(fiqa_dfs["train"])
    shared     = fpb_vocab & fiqa_vocab
    union      = fpb_vocab | fiqa_vocab

    jaccard = len(shared) / len(union)
    print(f"\nVocabulary overlap (min freq=2):")
    print(f"  FPB  vocab : {len(fpb_vocab):,} tokens")
    print(f"  FiQA vocab : {len(fiqa_vocab):,} tokens")
    print(f"  Shared     : {len(shared):,} tokens")
    print(f"  Jaccard    : {jaccard:.3f}")

    only_fpb  = len(fpb_vocab  - fiqa_vocab)
    only_fiqa = len(fiqa_vocab - fpb_vocab)

    fig, ax = plt.subplots(figsize=(8, 4))
    vals   = [only_fpb, len(shared), only_fiqa]
    labels = ["FPB only", "Shared", "FiQA only"]
    colors = ["#1f77b4", "#9467bd", "#ff7f0e"]

    bars = ax.barh(labels, vals, color=colors, edgecolor="white", height=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=11)
    ax.set_xlabel("Token count")
    ax.set_title("Vocabulary Overlap: FPB vs FiQA (min freq=2)")
    ax.set_xlim(0, max(vals) * 1.15)

    plt.tight_layout()
    out = RESULTS_DIR / "vocab_overlap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Majority-class baseline confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_baseline_confusion(fpb_dfs: dict, fiqa_dfs: dict) -> None:
    log.info("Step 5 — Majority-class baseline")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Majority-Class Baseline — Test Split", fontsize=15, fontweight="bold")

    baseline_rows = []
    for ax, (ds_name, dfs) in zip(axes, [("FPB", fpb_dfs), ("FiQA", fiqa_dfs)]):
        train_y = dfs["train"]["label"].tolist()
        test_y  = dfs["test"]["label"].tolist()

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit([[0]] * len(train_y), train_y)
        preds = dummy.predict([[0]] * len(test_y))

        acc = accuracy_score(test_y, preds)
        f1  = f1_score(test_y, preds, average="macro", zero_division=0)

        cm   = confusion_matrix(test_y, preds, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{ds_name}  |  Acc={acc:.3f}  Macro-F1={f1:.3f}")

        baseline_rows.append({"Dataset": ds_name, "Accuracy": round(acc, 3), "Macro-F1": round(f1, 3)})

    plt.tight_layout()
    out = RESULTS_DIR / "baseline_confusion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out}")

    df = pd.DataFrame(baseline_rows)
    print("\nMajority-class baseline (floor all models must beat):")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Market data (SPY)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_plot_market() -> pd.DataFrame:
    log.info("Step 6 — Fetching SPY market data")

    spy = fetch_market_data(
        "SPY",
        start="2019-01-01",
        end="2023-12-31",
        save_path=DATA_DIR / "market" / "spy_daily.csv",
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("SPY — Price & Daily Return (2019–2023)", fontsize=15, fontweight="bold")

    axes[0].plot(spy.index, spy["close"], color="steelblue", linewidth=1)
    axes[0].set_ylabel("Close Price (USD)")
    axes[0].set_title("Adjusted Close")

    colors = np.where(spy["daily_return"] >= 0, "#2ca02c", "#d62728")
    axes[1].bar(spy.index, spy["daily_return"] * 100, color=colors, width=1.5, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Daily Return (%)")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    out = RESULTS_DIR / "spy_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out}")

    up_days = int(spy["direction"].sum())
    print(f"\nSPY stats (2019–2023):")
    print(f"  Trading days : {len(spy)}")
    print(f"  Up days      : {up_days} ({up_days/len(spy)*100:.1f}%)")
    print(f"  Mean return  : {spy['daily_return'].mean()*100:.3f}%")
    print(f"  Std  return  : {spy['daily_return'].std()*100:.3f}%")

    return spy


# ─────────────────────────────────────────────────────────────────────────────
# 7. Save processed splits
# ─────────────────────────────────────────────────────────────────────────────

def save_splits(fpb_dfs: dict, fiqa_dfs: dict) -> None:
    log.info("Step 7 — Saving processed splits")

    for ds_name, dfs in [("fpb", fpb_dfs), ("fiqa", fiqa_dfs)]:
        for split, df in dfs.items():
            path = DATA_DIR / "processed" / f"{ds_name}_{split}.csv"
            save_processed(df, path)

    print(f"\nAll splits saved to {DATA_DIR / 'processed'}/")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Write EDA summary text file
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(fpb_dfs: dict, fiqa_dfs: dict, spy: pd.DataFrame) -> None:
    lines = [
        "EDA SUMMARY",
        "=" * 55,
        "",
        "DATASETS",
    ]
    for ds_name, dfs in [("FPB", fpb_dfs), ("FiQA", fiqa_dfs)]:
        total = sum(len(d) for d in dfs.values())
        lines.append(f"  {ds_name}: {total:,} examples")
        for split, df in dfs.items():
            counts = df["label_str"].value_counts().reindex(LABEL_NAMES, fill_value=0)
            lines.append(
                f"    {split:12s}: {len(df):,}  "
                + "  ".join(f"{l}={counts[l]}" for l in LABEL_NAMES)
            )
    lines += [
        "",
        "MARKET DATA (SPY 2019–2023)",
        f"  Trading days : {len(spy)}",
        f"  Up days      : {int(spy['direction'].sum())} ({spy['direction'].mean()*100:.1f}%)",
        f"  Mean return  : {spy['daily_return'].mean()*100:.4f}%",
        "",
        "KEY TAKEAWAYS",
        "  - FPB is neutral-heavy (~60-70%): use class_weight='balanced' + macro-F1",
        "  - Low FPB/FiQA vocab overlap: expect generalization gap across datasets",
        "  - Most sentences fit in 128 BERT tokens: no truncation concern",
        "  - SPY up ~55% of days: always-positive baseline for Stage 2 is ~55%",
        "",
        "NEXT STEP: python 02_baseline.py",
    ]

    out = RESULTS_DIR / "eda_summary.txt"
    out.write_text("\n".join(lines))
    log.info(f"  Saved {out}")
    print("\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Phase 1 — EDA")
    print("=" * 55)

    fpb_dfs, fiqa_dfs = load_all()
    plot_class_distribution(fpb_dfs, fiqa_dfs)
    plot_sentence_lengths(fpb_dfs, fiqa_dfs)
    plot_vocab_overlap(fpb_dfs, fiqa_dfs)
    plot_baseline_confusion(fpb_dfs, fiqa_dfs)
    spy = fetch_and_plot_market()
    save_splits(fpb_dfs, fiqa_dfs)
    write_summary(fpb_dfs, fiqa_dfs, spy)

    print("\n✅  EDA complete. Check the results/ folder for all plots.")
    print("    Next: python 02_baseline.py")
