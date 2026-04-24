"""
visualizations.py
====================
Final Visualizations

Combines results from all phases into publication-ready figures.

Run:
    python visualizations.py

Inputs (from results/ and results/market/):
    class_distribution_summary.csv
    baseline_cross_eval.csv
    alignment_metrics.csv
    daily_sentiment_all_models.csv
    bert_cross_eval_summary.csv     (copy from Drive if not present)
    finbert_summary.csv             (copy from Drive if not present)
    data/market/spy_daily.csv

Outputs (saved to results/final/):
    01_class_distribution.png
    02_model_comparison_fpb.png
    03_model_comparison_fiqa.png
    04_full_leaderboard.png
    05_directional_accuracy.png
    06_correlation_heatmap.png
    07_sentiment_vs_return_scatter.png
    08_time_series.png
    09_summary_table.png
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
MARKET_DIR  = Path("results/market")
FINAL_DIR   = Path("results/final")
FINAL_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

COLORS = {
    "negative": "#d62728",
    "neutral":  "#aec7e8",
    "positive": "#2ca02c",
}

MODEL_COLORS = {
    "Majority Baseline":  "#cccccc",
    "TF-IDF + LR":        "#7f7f7f",
    "BERT Run 1":         "#1f77b4",
    "BERT Run 2":         "#4a9fd4",
    "BERT Run 3":         "#85c1e9",
    "FinBERT Zero-Shot":  "#ff7f0e",
    "FinBERT + FiQA":     "#ffb347",
    "FinBERT + FPB":      "#e65c00",
}


def save(fig, name):
    path = FINAL_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {path}")


# ── 1. Class distribution ─────────────────────────────────────────────────────

def plot_class_distribution():
    path = RESULTS_DIR / "class_distribution_summary.csv"
    if not path.exists():
        log.warning("class_distribution_summary.csv not found, skipping")
        return

    df = pd.read_csv(path)
    label_cols = ["negative (%)", "neutral (%)", "positive (%)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution — FPB vs FiQA (Train Split)", fontsize=14)

    for ax, dataset in zip(axes, ["FPB", "FiQA"]):
        subset = df[(df["Dataset"] == dataset) & (df["Split"] == "train")]
        if subset.empty:
            continue
        row    = subset.iloc[0]
        vals   = [row[c] for c in label_cols]
        labels = ["Negative", "Neutral", "Positive"]
        colors = [COLORS["negative"], COLORS["neutral"], COLORS["positive"]]

        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(dataset)
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save(fig, "01_class_distribution.png")


# ── 2. Build leaderboard ──────────────────────────────────────────────────────

def build_leaderboard():
    rows = []
    rows.append({"model": "Majority Baseline", "fpb_f1": 0.254, "fiqa_f1": 0.167})

    tfidf_path = RESULTS_DIR / "baseline_cross_eval.csv"
    if tfidf_path.exists():
        df = pd.read_csv(tfidf_path)
        fpb_row  = df[(df["train_on"] == "fpb") & (df["eval_on"] == "fpb")]
        fiqa_row = df[(df["train_on"] == "fpb") & (df["eval_on"] == "fiqa")]
        if len(fpb_row) and len(fiqa_row):
            rows.append({
                "model":   "TF-IDF + LR",
                "fpb_f1":  fpb_row.iloc[0]["macro_f1"],
                "fiqa_f1": fiqa_row.iloc[0]["macro_f1"],
            })

    bert_path = RESULTS_DIR / "bert_cross_eval_summary.csv"
    if bert_path.exists():
        df = pd.read_csv(bert_path)
        for run in [1, 2, 3]:
            fpb_row  = df[(df["run"] == run) & (df["eval_on"] == "fpb")]
            fiqa_row = df[(df["run"] == run) & (df["eval_on"] == "fiqa")]
            if len(fpb_row) and len(fiqa_row):
                rows.append({
                    "model":   f"BERT Run {run}",
                    "fpb_f1":  fpb_row.iloc[0]["macro_f1"],
                    "fiqa_f1": fiqa_row.iloc[0]["macro_f1"],
                })
    else:
        log.warning("bert_cross_eval_summary.csv not found — using recorded values")
        rows += [
            {"model": "BERT Run 1", "fpb_f1": 0.9687, "fiqa_f1": 0.5015},
            {"model": "BERT Run 2", "fpb_f1": 0.9565, "fiqa_f1": 0.5163},
            {"model": "BERT Run 3", "fpb_f1": 0.9623, "fiqa_f1": 0.4721},
        ]

    finbert_path = RESULTS_DIR / "finbert_summary.csv"
    if finbert_path.exists():
        df = pd.read_csv(finbert_path)
        mapping = {
            "Exp A (zero-shot)": "FinBERT Zero-Shot",
            "Exp B":             "FinBERT + FiQA",
            "Exp C":             "FinBERT + FPB",
        }
        for exp_name, label in mapping.items():
            fpb_row  = df[(df["experiment"] == exp_name) & (df["eval_on"] == "fpb")]
            fiqa_row = df[(df["experiment"] == exp_name) & (df["eval_on"] == "fiqa")]
            if len(fpb_row) and len(fiqa_row):
                rows.append({
                    "model":   label,
                    "fpb_f1":  fpb_row.iloc[0]["macro_f1"],
                    "fiqa_f1": fiqa_row.iloc[0]["macro_f1"],
                })
    else:
        log.warning("finbert_summary.csv not found — using recorded values")
        rows += [
            {"model": "FinBERT Zero-Shot", "fpb_f1": 0.9629, "fiqa_f1": 0.5619},
            {"model": "FinBERT + FiQA",   "fpb_f1": 0.2055, "fiqa_f1": 0.4765},
            {"model": "FinBERT + FPB",    "fpb_f1": 0.9811, "fiqa_f1": 0.4938},
        ]

    lb = pd.DataFrame(rows)
    lb.to_csv(FINAL_DIR / "leaderboard.csv", index=False)
    return lb


# ── 3 & 4. Model comparison ───────────────────────────────────────────────────

def plot_model_comparison(lb):
    for dataset, col, num in [("FPB", "fpb_f1", "02"), ("FiQA", "fiqa_f1", "03")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors  = [MODEL_COLORS.get(m, "#333333") for m in lb["model"]]
        bars    = ax.barh(lb["model"], lb[col], color=colors,
                          edgecolor="white", height=0.6)

        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="50%")
        ax.axvline(lb[col].max(), color="gold", linestyle=":", linewidth=1.5, label="Best")

        for bar, val in zip(bars, lb[col]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)

        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Macro-F1 Score")
        ax.set_title(f"Model Comparison — {dataset} Test Set")
        ax.legend(fontsize=9)
        plt.tight_layout()
        save(fig, f"{num}_model_comparison_{dataset.lower()}.png")


def plot_full_leaderboard(lb):
    fig, ax = plt.subplots(figsize=(13, 6))
    x     = np.arange(len(lb))
    width = 0.38

    bars1 = ax.bar(x - width / 2, lb["fpb_f1"],  width,
                   label="FPB",  color="#1f77b4", edgecolor="white")
    bars2 = ax.bar(x + width / 2, lb["fiqa_f1"], width,
                   label="FiQA", color="#ff7f0e", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(lb["model"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Macro-F1 Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Full Model Leaderboard — FPB vs FiQA")
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    save(fig, "04_full_leaderboard.png")


# ── 5. Directional accuracy ───────────────────────────────────────────────────

def plot_directional_accuracy():
    path = MARKET_DIR / "alignment_metrics.csv"
    if not path.exists():
        log.warning("alignment_metrics.csv not found, skipping")
        return

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(11, 5))
    x     = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df["dir_acc_same_day"],
                   width, label="Same-Day", color="#1f77b4", edgecolor="white")
    bars2 = ax.bar(x + width / 2, df["dir_acc_next_day"],
                   width, label="Next-Day", color="#ff7f0e", edgecolor="white")

    ax.axhline(0.500, color="red",  linestyle="--", linewidth=1.5, label="Random (50%)")
    ax.axhline(0.549, color="gray", linestyle=":",  linewidth=1.5, label="Always-Up (54.9%)")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylabel("Directional Accuracy")
    ax.set_ylim(0.40, 0.85)
    ax.set_title("Directional Accuracy: Sentiment vs SPY Market Direction")
    ax.legend(fontsize=9)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    save(fig, "05_directional_accuracy.png")


# ── 6. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap():
    path = MARKET_DIR / "alignment_metrics.csv"
    if not path.exists():
        return

    df   = pd.read_csv(path)
    cols = ["pearson_same", "spearman_same", "pearson_next", "spearman_next"]
    labs = ["Pearson\n(same-day)", "Spearman\n(same-day)",
            "Pearson\n(next-day)", "Spearman\n(next-day)"]

    data = df.set_index("model")[cols].astype(float)
    vmax = max(abs(data.values.min()), abs(data.values.max()), 0.05)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labs, fontsize=10)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.set_title("Correlation: Sentiment Score vs SPY Daily Return")

    for i in range(len(data.index)):
        for j in range(len(cols)):
            val = data.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color="black")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save(fig, "06_correlation_heatmap.png")


# ── 7. Scatter ────────────────────────────────────────────────────────────────

def plot_scatter():
    path = MARKET_DIR / "daily_sentiment_all_models.csv"
    if not path.exists():
        return

    from scipy.stats import pearsonr

    df  = pd.read_csv(path)
    col = "finbert_zs_score_mean"
    if col not in df.columns:
        log.warning(f"{col} not found, skipping scatter")
        return

    mask = df[col].notna() & df["daily_return"].notna()
    x    = df.loc[mask, col]
    y    = df.loc[mask, "daily_return"] * 100
    r, p = pearsonr(x, y)
    m, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.4, s=20, color="#1f77b4", label="Trading day")
    ax.plot(np.sort(x), m * np.sort(x) + b, color="red",
            linewidth=2, label=f"Trend (r={r:.3f})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Daily Sentiment Score (FinBERT Zero-Shot)")
    ax.set_ylabel("SPY Daily Return (%)")
    ax.set_title("Sentiment Score vs SPY Return — FinBERT Zero-Shot")
    ax.legend()
    ax.text(0.05, 0.95,
            f"r = {r:.3f}\np = {p:.4f}\nn = {mask.sum()} days",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    save(fig, "07_sentiment_vs_return_scatter.png")


# ── 8. Time series ────────────────────────────────────────────────────────────

def plot_time_series():
    daily_path = MARKET_DIR / "daily_sentiment_all_models.csv"
    spy_path   = Path("data/market/spy_daily.csv")

    if not daily_path.exists() or not spy_path.exists():
        log.warning("Missing files for time series, skipping")
        return

    daily = pd.read_csv(daily_path, parse_dates=["date"])
    spy   = pd.read_csv(spy_path, index_col=0, parse_dates=True)
    spy.index = pd.to_datetime(spy.index).normalize()
    col   = "finbert_zs_score_mean"
    if col not in daily.columns:
        return

    daily = daily.sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("FinBERT Zero-Shot Sentiment vs SPY Price", fontsize=14)

    axes[0].plot(spy.index, spy["close"], color="steelblue", linewidth=1)
    axes[0].set_ylabel("SPY Close (USD)")
    axes[0].set_title("SPY Adjusted Close Price")

    colors = np.where(daily[col] >= 0, "#2ca02c", "#d62728")
    axes[1].bar(daily["date"], daily[col], color=colors, width=1.5, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Sentiment Score")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Daily Aggregated Sentiment (FinBERT Zero-Shot)")

    plt.tight_layout()
    save(fig, "08_time_series.png")


# ── 9. Summary table ──────────────────────────────────────────────────────────

def plot_summary_table(lb):
    alignment_path = MARKET_DIR / "alignment_metrics.csv"
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Project Results Summary", fontsize=15, fontweight="bold")

    # Left — classification leaderboard
    ax = axes[0]
    ax.axis("off")
    ax.set_title("Sentiment Classification (Macro-F1)", pad=10)

    table_data = [[m, f"{f:.4f}", f"{q:.4f}"]
                  for m, f, q in zip(lb["model"], lb["fpb_f1"], lb["fiqa_f1"])]
    col_labels = ["Model", "FPB F1", "FiQA F1"]
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    best_fpb  = max(range(len(lb)), key=lambda i: lb["fpb_f1"].iloc[i])
    best_fiqa = max(range(len(lb)), key=lambda i: lb["fiqa_f1"].iloc[i])
    table[best_fpb  + 1, 1].set_facecolor("#d4edda")
    table[best_fiqa + 1, 2].set_facecolor("#d4edda")

    # Right — market alignment
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Market Alignment (SPY)", pad=10)

    if alignment_path.exists():
        am = pd.read_csv(alignment_path)
        align_data = [
            [row["model"],
             f"{row['dir_acc_same_day']:.3f}",
             f"{row['dir_acc_next_day']:.3f}",
             f"{row['pearson_same']:.3f}"]
            for _, row in am.iterrows()
        ]
        col_labels2 = ["Model", "Dir Acc\n(same)", "Dir Acc\n(next)", "Pearson r\n(same)"]
        table2 = ax2.table(cellText=align_data, colLabels=col_labels2,
                           loc="center", cellLoc="center")
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 1.6)

        best_dir = max(range(len(align_data)),
                       key=lambda i: float(align_data[i][1]))
        table2[best_dir + 1, 1].set_facecolor("#d4edda")

    plt.tight_layout()
    save(fig, "09_summary_table.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6 — Final Visualizations")
    print("=" * 60)

    lb = build_leaderboard()
    log.info(f"Leaderboard: {len(lb)} models")

    plot_class_distribution()
    plot_model_comparison(lb)
    plot_full_leaderboard(lb)
    plot_directional_accuracy()
    plot_correlation_heatmap()
    plot_scatter()
    plot_time_series()
    plot_summary_table(lb)

    print(f"\nAll figures saved to {FINAL_DIR}/")
    print("\nFigures generated:")
    for f in sorted(FINAL_DIR.glob("*.png")):
        print(f"  {f.name}")
