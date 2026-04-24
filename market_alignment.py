"""
market_alignment.py
======================
Market Alignment Analysis

Correlates sentiment predictions from all trained models against
real S&P 500 (SPY) daily returns.

Run locally (all models are lightweight for inference):
    python market_alignment.py

Inputs:
    data/raw/raw_analyst_ratings.csv     — Benzinga headlines with dates + tickers
    data/market/spy_daily.csv            — SPY daily returns (from Phase 1)
    models/bert/run1_final/              — Best BERT model (Run 1)
    models/bert/run2_final/              — BERT Run 2
    models/finbert/expC_final/           — Best FinBERT model (Exp C, fine-tuned on FPB)

    Note: FinBERT zero-shot (Exp A) is loaded directly from HuggingFace.
          TF-IDF+LR model is loaded from models/tfidf_lr/tfidf_lr_fpb.pkl

Outputs (saved to results/market/):
    daily_sentiment_all_models.csv       — aggregated daily sentiment per model
    alignment_metrics.csv                — directional accuracy + correlations
    scatter_sentiment_vs_return.png      — scatter plots for each model
    time_series_sentiment_spy.png        — sentiment signal overlaid on SPY price
    directional_accuracy_bar.png         — bar chart comparing models
    correlation_heatmap.png              — Pearson/Spearman heatmap
    per_stock_alignment.csv              — per-ticker directional accuracy
"""

import logging
import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW     = Path("data/raw/raw_analyst_ratings.csv")
MARKET_DATA  = Path("data/market/spy_daily.csv")
MODELS_DIR   = Path("models")
RESULTS_DIR  = Path("results/market")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_NAMES  = ["negative", "neutral", "positive"]
MAX_LENGTH   = 128
BATCH_SIZE   = 64
DATE_START   = "2019-01-01"
DATE_END     = "2023-12-31"

# Top S&P 500 tickers to analyse individually
TOP_TICKERS  = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NVDA", "JPM"]


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    elif torch.backends.mps.is_available():
        log.info("Using Apple MPS backend")
        return "mps"
    else:
        log.info("Using CPU")
        return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and clean headlines
# ─────────────────────────────────────────────────────────────────────────────

def load_headlines() -> pd.DataFrame:
    log.info("Loading headlines...")
    df = pd.read_csv(DATA_RAW, index_col=0)

    # Parse date — format: "2020-06-05 10:30:54-04:00"
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True).dt.normalize().dt.tz_localize(None)

    # Filter to our date window
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].copy()

    # Drop rows with missing headlines
    df = df.dropna(subset=["headline"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 5]

    log.info(f"Headlines after filtering: {len(df):,} rows "
             f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load market data
# ─────────────────────────────────────────────────────────────────────────────

def load_market() -> pd.DataFrame:
    log.info("Loading SPY market data...")
    spy = pd.read_csv(MARKET_DATA, index_col=0, parse_dates=True)
    spy.index = pd.to_datetime(spy.index).normalize()
    spy = spy[(spy.index >= DATE_START) & (spy.index <= DATE_END)]
    log.info(f"SPY: {len(spy)} trading days")
    return spy


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def signed_score(label: str, score: float) -> float:
    """Convert label + confidence score to signed sentiment score [-1, +1]."""
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0.0


def transformer_inference(
    texts: list[str],
    model,
    tokenizer,
    device: str,
    label_map: dict | None = None,
) -> list[float]:
    """
    Run batch inference with a transformer model.
    Returns a list of signed scores in [-1, +1].
    label_map: optional remapping of model label indices to our convention
               (needed for FinBERT zero-shot whose label order differs)
    """
    model.eval()
    model.to(device)
    all_scores = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        for prob in probs:
            if label_map:
                # Remap indices: FinBERT native (pos=0,neg=1,neu=2) → ours (neg=0,neu=1,pos=2)
                neg_score = prob[label_map["negative"]]
                pos_score = prob[label_map["positive"]]
            else:
                neg_score = prob[0]  # our convention: 0=negative
                pos_score = prob[2]  # our convention: 2=positive
            all_scores.append(float(pos_score - neg_score))

    return all_scores


def tfidf_inference(texts: list[str], pipeline) -> list[float]:
    """Run TF-IDF + LR inference. Returns signed scores."""
    probs = pipeline.predict_proba(texts)
    # sklearn classes follow label order: [0=neg, 1=neu, 2=pos]
    return [float(p[2] - p[0]) for p in probs]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Run all models on headlines
# ─────────────────────────────────────────────────────────────────────────────

def run_all_models(headlines_df: pd.DataFrame, device: str) -> pd.DataFrame:
    """
    Run all 4 models on headlines and return DataFrame with sentiment scores.
    Columns: date, headline, stock, tfidf_score, bert_r1_score,
             bert_r2_score, finbert_zs_score, finbert_fpb_score
    """
    texts = headlines_df["headline"].tolist()
    log.info(f"Running inference on {len(texts):,} headlines...")

    result = headlines_df[["date", "headline", "stock"]].copy()

    # ── Model 1: TF-IDF + LR ─────────────────────────────────────────────────
    log.info("  [1/4] TF-IDF + LR...")
    tfidf_path = MODELS_DIR / "tfidf_lr" / "tfidf_lr_fpb.pkl"
    if tfidf_path.exists():
        with open(tfidf_path, "rb") as f:
            tfidf_pipe = pickle.load(f)
        result["tfidf_score"] = tfidf_inference(texts, tfidf_pipe)
    else:
        log.warning(f"  TF-IDF model not found at {tfidf_path}, skipping.")
        result["tfidf_score"] = np.nan

    # ── Model 2: BERT Run 1 ───────────────────────────────────────────────────
    log.info("  [2/4] BERT Run 1...")
    bert_r1_path = MODELS_DIR / "bert" / "run1_final"
    if bert_r1_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(bert_r1_path))
        model     = AutoModelForSequenceClassification.from_pretrained(str(bert_r1_path))
        result["bert_r1_score"] = transformer_inference(texts, model, tokenizer, device)
        del model
    else:
        log.warning(f"  BERT Run 1 not found at {bert_r1_path}, skipping.")
        result["bert_r1_score"] = np.nan

    # ── Model 3: BERT Run 2 ───────────────────────────────────────────────────
    log.info("  [3/4] BERT Run 2...")
    bert_r2_path = MODELS_DIR / "bert" / "run2_final"
    if bert_r2_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(bert_r2_path))
        model     = AutoModelForSequenceClassification.from_pretrained(str(bert_r2_path))
        result["bert_r2_score"] = transformer_inference(texts, model, tokenizer, device)
        del model
    else:
        log.warning(f"  BERT Run 2 not found at {bert_r2_path}, skipping.")
        result["bert_r2_score"] = np.nan

    # ── Model 4: FinBERT zero-shot ────────────────────────────────────────────
    log.info("  [4/5] FinBERT zero-shot...")
    finbert_label_map = {"negative": 1, "neutral": 2, "positive": 0}
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    result["finbert_zs_score"] = transformer_inference(
        texts, model, tokenizer, device, label_map=finbert_label_map
    )
    del model

    # ── Model 5: FinBERT fine-tuned on FPB ───────────────────────────────────
    log.info("  [5/5] FinBERT + FPB (Exp C)...")
    finbert_fpb_path = MODELS_DIR / "finbert" / "expC_final"
    if finbert_fpb_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(finbert_fpb_path))
        model     = AutoModelForSequenceClassification.from_pretrained(str(finbert_fpb_path))
        result["finbert_fpb_score"] = transformer_inference(texts, model, tokenizer, device)
        del model
    else:
        log.warning(f"  FinBERT+FPB not found at {finbert_fpb_path}, skipping.")
        result["finbert_fpb_score"] = np.nan

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Aggregate daily sentiment
# ─────────────────────────────────────────────────────────────────────────────

SCORE_COLS = [
    "tfidf_score", "bert_r1_score", "bert_r2_score",
    "finbert_zs_score", "finbert_fpb_score"
]

MODEL_LABELS = {
    "tfidf_score":       "TF-IDF + LR",
    "bert_r1_score":     "BERT Run 1",
    "bert_r2_score":     "BERT Run 2",
    "finbert_zs_score":  "FinBERT Zero-Shot",
    "finbert_fpb_score": "FinBERT + FPB",
}


def aggregate_daily(scored_df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """Average sentiment scores per trading day and merge with SPY returns."""
    trading_days = set(spy.index)
    aligned = scored_df[scored_df["date"].isin(trading_days)].copy()
    log.info(f"Headlines on trading days: {len(aligned):,} / {len(scored_df):,}")

    daily = (
        aligned.groupby("date")[SCORE_COLS]
        .agg(["mean", "count"])
        .round(6)
    )
    daily.columns = ["_".join(c) for c in daily.columns]
    daily = daily.reset_index()
    daily = daily.merge(
        spy[["daily_return", "log_return", "direction"]].reset_index(),
        left_on="date", right_on="date", how="inner"
    )
    log.info(f"Trading days with headlines: {len(daily)}")
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 6. Compute alignment metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_alignment(daily: pd.DataFrame) -> pd.DataFrame:
    """
    For each model compute:
      - Directional accuracy (same-day)
      - Directional accuracy (next-day)
      - Pearson r (same-day)
      - Spearman rho (same-day)
      - Pearson r (next-day)
      - Spearman rho (next-day)
    """
    daily = daily.copy()
    daily["next_return"]   = daily["daily_return"].shift(-1)
    daily["next_direction"] = daily["direction"].shift(-1)

    rows = []
    for col in SCORE_COLS:
        mean_col = f"{col}_mean"
        if mean_col not in daily.columns:
            continue

        scores = daily[mean_col]
        valid  = scores.notna()

        # Directional accuracy
        pred_dir      = (scores > 0).astype(int)
        dir_acc_same  = accuracy_score(daily.loc[valid, "direction"],   pred_dir[valid])
        dir_acc_next  = accuracy_score(
            daily.loc[valid & daily["next_direction"].notna(), "next_direction"],
            pred_dir[valid & daily["next_direction"].notna()]
        )

        # Pearson / Spearman — same day
        mask_same = valid & daily["daily_return"].notna()
        r_same,   p_r_same  = pearsonr( scores[mask_same], daily.loc[mask_same, "daily_return"])
        rho_same, p_rho_same = spearmanr(scores[mask_same], daily.loc[mask_same, "daily_return"])

        # Pearson / Spearman — next day
        mask_next = valid & daily["next_return"].notna()
        r_next,   p_r_next  = pearsonr( scores[mask_next], daily.loc[mask_next, "next_return"])
        rho_next, p_rho_next = spearmanr(scores[mask_next], daily.loc[mask_next, "next_return"])

        rows.append({
            "model":            MODEL_LABELS[col],
            "dir_acc_same_day": round(dir_acc_same, 4),
            "dir_acc_next_day": round(dir_acc_next, 4),
            "pearson_same":     round(r_same,   4),
            "p_pearson_same":   round(p_r_same,  4),
            "spearman_same":    round(rho_same,  4),
            "p_spearman_same":  round(p_rho_same, 4),
            "pearson_next":     round(r_next,   4),
            "p_pearson_next":   round(p_r_next,  4),
            "spearman_next":    round(rho_next,  4),
            "p_spearman_next":  round(p_rho_next, 4),
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "alignment_metrics.csv", index=False)
    log.info(f"Saved alignment_metrics.csv")

    print("\n" + "="*65)
    print("MARKET ALIGNMENT METRICS")
    print("="*65)
    print(metrics.to_string(index=False))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(daily: pd.DataFrame) -> None:
    """Scatter plot: daily sentiment score vs SPY daily return."""
    valid_cols = [c for c in SCORE_COLS if f"{c}_mean" in daily.columns]
    n = len(valid_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Daily Sentiment Score vs SPY Return", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, valid_cols):
        mean_col = f"{col}_mean"
        mask = daily[mean_col].notna() & daily["daily_return"].notna()
        x = daily.loc[mask, mean_col]
        y = daily.loc[mask, "daily_return"] * 100
        r, _ = pearsonr(x, y)

        ax.scatter(x, y, alpha=0.3, s=15, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        m, b = np.polyfit(x, y, 1)
        ax.plot(np.sort(x), m * np.sort(x) + b, color="red", linewidth=1.5)
        ax.set_title(f"{MODEL_LABELS[col]}\nr={r:.3f}", fontsize=10)
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("SPY Return (%)")

    plt.tight_layout()
    out = RESULTS_DIR / "scatter_sentiment_vs_return.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_time_series(daily: pd.DataFrame, spy: pd.DataFrame) -> None:
    """Overlay FinBERT zero-shot sentiment signal on SPY price over time."""
    col = "finbert_zs_score_mean"
    if col not in daily.columns:
        col = next((f"{c}_mean" for c in SCORE_COLS if f"{c}_mean" in daily.columns), None)
    if col is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Sentiment Signal vs SPY Price", fontsize=14, fontweight="bold")

    axes[0].plot(spy.index, spy["close"], color="steelblue", linewidth=1)
    axes[0].set_ylabel("SPY Close Price (USD)")
    axes[0].set_title("SPY Price")

    daily_sorted = daily.sort_values("date")
    colors = np.where(daily_sorted[col] >= 0, "#2ca02c", "#d62728")
    axes[1].bar(daily_sorted["date"], daily_sorted[col],
                color=colors, width=1.5, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Sentiment Score")
    axes[1].set_xlabel("Date")
    axes[1].set_title(f"Daily Sentiment — {MODEL_LABELS.get(col.replace('_mean',''), col)}")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    out = RESULTS_DIR / "time_series_sentiment_spy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_directional_accuracy(metrics: pd.DataFrame) -> None:
    """Bar chart comparing directional accuracy across models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, metrics["dir_acc_same_day"],
                   width, label="Same-Day", color="steelblue", edgecolor="white")
    bars2 = ax.bar(x + width/2, metrics["dir_acc_next_day"],
                   width, label="Next-Day", color="darkorange", edgecolor="white")

    ax.axhline(0.5,  color="red",   linewidth=1.5, linestyle="--", label="Random (50%)")
    ax.axhline(0.549, color="gray", linewidth=1.5, linestyle=":",  label="Always-Up baseline (54.9%)")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics["model"], rotation=15, ha="right")
    ax.set_ylabel("Directional Accuracy")
    ax.set_title("Directional Accuracy by Model and Horizon", fontweight="bold")
    ax.set_ylim(0.4, 0.75)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "directional_accuracy_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_correlation_heatmap(metrics: pd.DataFrame) -> None:
    """Heatmap of Pearson and Spearman correlations across models and horizons."""
    import matplotlib.colors as mcolors

    corr_cols = ["pearson_same", "spearman_same", "pearson_next", "spearman_next"]
    col_labels = ["Pearson\n(same-day)", "Spearman\n(same-day)",
                  "Pearson\n(next-day)", "Spearman\n(next-day)"]

    data = metrics.set_index("model")[corr_cols].astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    vmax = max(abs(data.values.min()), abs(data.values.max()), 0.05)
    im   = ax.imshow(data.values, cmap="RdYlGn", aspect="auto",
                     vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(corr_cols)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.set_title("Correlation Coefficients: Sentiment vs SPY Return", fontweight="bold")

    for i in range(len(data.index)):
        for j in range(len(corr_cols)):
            val = data.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color="black")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    out = RESULTS_DIR / "correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Per-stock analysis
# ─────────────────────────────────────────────────────────────────────────────

def per_stock_analysis(scored_df: pd.DataFrame) -> None:
    """
    For each ticker in TOP_TICKERS, compute directional accuracy using
    FinBERT zero-shot scores vs that stock's actual daily returns.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — skipping per-stock analysis")
        return

    log.info("Running per-stock analysis...")
    rows = []
    col  = "finbert_zs_score"

    for ticker in TOP_TICKERS:
        ticker_df = scored_df[scored_df["stock"] == ticker].copy()
        if len(ticker_df) < 20:
            log.info(f"  {ticker}: not enough headlines ({len(ticker_df)}), skipping")
            continue

        try:
            stock = yf.download(ticker, start=DATE_START, end=DATE_END,
                                 auto_adjust=True, progress=False)
            if stock.empty:
                continue
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = [c[0] for c in stock.columns]
            stock.columns  = [c.lower() for c in stock.columns]
            stock.index    = pd.to_datetime(stock.index).normalize()
            stock["ret"]   = stock["close"].pct_change()
            stock["dir"]   = (stock["ret"] > 0).astype(int)
            stock          = stock.dropna(subset=["ret"])

            ticker_df["date"] = pd.to_datetime(ticker_df["date"]).dt.normalize()
            daily_t = ticker_df.groupby("date")[col].mean().reset_index()
            daily_t = daily_t.merge(
                stock[["ret", "dir"]].reset_index().rename(columns={"index": "date"}),
                on="date", how="inner"
            )

            if len(daily_t) < 10:
                continue

            pred_dir = (daily_t[col] > 0).astype(int)
            acc = accuracy_score(daily_t["dir"], pred_dir)
            r, p = pearsonr(daily_t[col], daily_t["ret"])

            rows.append({
                "ticker":       ticker,
                "n_days":       len(daily_t),
                "dir_accuracy": round(acc, 4),
                "pearson_r":    round(r, 4),
                "p_value":      round(p, 4),
            })
            log.info(f"  {ticker}: {len(daily_t)} days | dir_acc={acc:.3f} | r={r:.3f}")

        except Exception as e:
            log.warning(f"  {ticker}: error — {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / "per_stock_alignment.csv", index=False)
        log.info(f"Saved per_stock_alignment.csv")
        print("\nPer-Stock Alignment (FinBERT Zero-Shot):")
        print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Phase 5 — Market Alignment Analysis")
    print("=" * 65)

    device = get_device()

    # Load data
    headlines_df = load_headlines()
    spy          = load_market()

    # Run all models
    scored_df = run_all_models(headlines_df, device)
    scored_df.to_csv(RESULTS_DIR / "scored_headlines.csv", index=False)
    log.info("Saved scored_headlines.csv")

    # Aggregate daily
    daily = aggregate_daily(scored_df, spy)
    daily.to_csv(RESULTS_DIR / "daily_sentiment_all_models.csv", index=False)
    log.info("Saved daily_sentiment_all_models.csv")

    # Compute alignment metrics
    metrics = compute_alignment(daily)

    # Visualizations
    log.info("Generating plots...")
    plot_scatter(daily)
    plot_time_series(daily, spy)
    plot_directional_accuracy(metrics)
    plot_correlation_heatmap(metrics)

    # Per-stock analysis
    per_stock_analysis(scored_df)
    print(f"    All outputs saved to {RESULTS_DIR}/")
