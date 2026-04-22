"""
data_utils.py
=============
Data loading, preprocessing, and alignment utilities for the
financial sentiment <-> market alignment project.

Covers:
  - Financial PhraseBank (FPB) via HuggingFace datasets
  - FiQA-2018 SA subset via HuggingFace datasets
  - S&P 500 / individual stock data via yfinance
  - Headline date-alignment helpers
  - Train/val/test splitting with stratification
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ── Label mappings ────────────────────────────────────────────────────────────

FPB_INT2STR = {0: "negative", 1: "neutral", 2: "positive"}
FPB_STR2INT = {v: k for k, v in FPB_INT2STR.items()}

# FiQA scores are continuous floats in [-1, 1]; bucket into 3 classes
FIQA_BUCKET_THRESHOLDS = (-0.1, 0.1)

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}


# ── Financial PhraseBank (FPB) ────────────────────────────────────────────────

def load_fpb(
    config: str = "sentences_allagree",
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
    cache_dir: str | None = None,
) -> DatasetDict:
    """
    Load and split Financial PhraseBank from HuggingFace.

    Args:
        config:    'sentences_allagree' (default, highest quality),
                   'sentences_66agree', 'sentences_75agree', 'sentences_50agree'
        val_size:  Fraction for validation split.
        test_size: Fraction for test split.
        seed:      Random seed.
        cache_dir: Optional cache directory.

    Returns:
        DatasetDict with keys 'train', 'validation', 'test'.
        Each example has 'sentence' (str) and 'label' (int 0/1/2).
    """
    logger.info(f"Loading FPB ({config})...")
    raw = load_dataset(
        "financial_phrasebank",
        config,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # FPB only ships a single 'train' split — we create val/test manually
    full   = raw["train"]
    n      = len(full)
    labels = np.array(full["label"])
    indices = np.arange(n)

    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=seed
    )
    adj_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=adj_val, stratify=labels[train_idx], random_state=seed
    )

    dataset = DatasetDict({
        "train":      full.select(train_idx),
        "validation": full.select(val_idx),
        "test":       full.select(test_idx),
    })

    _log_split_stats("FPB", dataset)
    return dataset


# ── FiQA-2018 Sentiment Analysis ──────────────────────────────────────────────

def load_fiqa(
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
    cache_dir: str | None = None,
) -> DatasetDict:
    """
    Load FiQA-2018 sentiment subset and convert continuous scores to 3-class labels.

    Bucketing rule:
      score < -0.1          -> 0 (negative)
      -0.1 <= score <= 0.1  -> 1 (neutral)
      score > 0.1           -> 2 (positive)

    Returns:
        DatasetDict with keys 'train', 'validation', 'test'.
        Each example has 'sentence' (str) and 'label' (int 0/1/2).
    """
    logger.info("Loading FiQA-2018...")
    raw = load_dataset("pauri32/fiqa-2018", cache_dir=cache_dir)

    neg_upper, pos_lower = FIQA_BUCKET_THRESHOLDS

    def _process(example):
        score = float(example.get("score", 0.0))
        if score < neg_upper:
            label = 0
        elif score > pos_lower:
            label = 2
        else:
            label = 1
        text = example.get("sentence") or example.get("text") or ""
        return {"sentence": _clean_text(text), "label": label}

    if "train" in raw and "test" in raw:
        train_ds = raw["train"].map(_process, remove_columns=raw["train"].column_names)
        test_ds  = raw["test"].map(_process,  remove_columns=raw["test"].column_names)

        indices = np.arange(len(train_ds))
        labels  = np.array(train_ds["label"])
        train_idx, val_idx = train_test_split(
            indices, test_size=val_size, stratify=labels, random_state=seed
        )
        dataset = DatasetDict({
            "train":      train_ds.select(train_idx),
            "validation": train_ds.select(val_idx),
            "test":       test_ds,
        })
    else:
        full = list(raw.values())[0].map(
            _process, remove_columns=list(raw.values())[0].column_names
        )
        indices = np.arange(len(full))
        labels  = np.array(full["label"])
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=seed
        )
        adj_val = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=adj_val, stratify=labels[train_idx], random_state=seed
        )
        dataset = DatasetDict({
            "train":      full.select(train_idx),
            "validation": full.select(val_idx),
            "test":       full.select(test_idx),
        })

    _log_split_stats("FiQA", dataset)
    return dataset


# ── Dataset → Pandas ──────────────────────────────────────────────────────────

def dataset_to_df(dataset: DatasetDict) -> dict[str, pd.DataFrame]:
    """Convert a DatasetDict to a dict of DataFrames with a human-readable label column."""
    result = {}
    for split, ds in dataset.items():
        df = ds.to_pandas()
        df["label_str"] = df["label"].map(ID2LABEL)
        result[split] = df
    return result


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Strip HTML tags, entities, and excess whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataset(dataset: DatasetDict) -> DatasetDict:
    """Apply _clean_text to the 'sentence' field across all splits."""
    return dataset.map(lambda ex: {"sentence": _clean_text(ex["sentence"])})


# ── Market Data ───────────────────────────────────────────────────────────────

def fetch_market_data(
    ticker: str = "SPY",
    start: str = "2019-01-01",
    end: str = "2023-12-31",
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Download daily OHLCV data via yfinance and compute return columns.

    Adds:
      daily_return : pct change in close price
      log_return   : log(close_t / close_{t-1})
      direction    : 1 if daily_return > 0, else 0

    Args:
        ticker:    Ticker symbol (default 'SPY').
        start:     Start date (YYYY-MM-DD).
        end:       End date (YYYY-MM-DD).
        save_path: If given, saves CSV to this path.

    Returns:
        DataFrame indexed by date.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Run: pip install yfinance") from exc

    logger.info(f"Fetching {ticker} ({start} -> {end})...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check ticker and date range.")

    # yfinance >=0.2 may return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"

    df["daily_return"] = df["close"].pct_change()
    df["log_return"]   = np.log(df["close"] / df["close"].shift(1))
    df["direction"]    = (df["daily_return"] > 0).astype(int)
    df = df.dropna(subset=["daily_return"])

    logger.info(f"  {len(df)} trading days loaded for {ticker}")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        logger.info(f"  Saved to {save_path}")

    return df


def fetch_multiple_tickers(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str = "2023-12-31",
    save_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch market data for a list of tickers. Returns dict of ticker -> DataFrame."""
    results = {}
    for ticker in tickers:
        path = Path(save_dir) / f"{ticker.lower()}_daily.csv" if save_dir else None
        results[ticker] = fetch_market_data(ticker, start, end, path)
    return results


# ── Headlines alignment ───────────────────────────────────────────────────────

def align_headlines_to_market(
    headlines_df: pd.DataFrame,
    market_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Join a dated headlines DataFrame to market data, keeping only trading days.

    Returns merged DataFrame with daily_return, log_return, and direction columns.
    """
    headlines_df = headlines_df.copy()
    headlines_df[date_col] = pd.to_datetime(headlines_df[date_col]).dt.normalize()

    trading_days = set(market_df.index)
    aligned = headlines_df[headlines_df[date_col].isin(trading_days)].copy()

    merged = aligned.merge(
        market_df[["daily_return", "log_return", "direction"]].reset_index(),
        on=date_col, how="left"
    )
    logger.info(
        f"Headlines: {len(headlines_df):,} total -> "
        f"{len(aligned):,} on trading days -> "
        f"{merged['daily_return'].notna().sum():,} matched"
    )
    return merged


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to CSV, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info(f"Saved {len(df):,} rows to {p}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_split_stats(name: str, dataset: DatasetDict) -> None:
    for split, ds in dataset.items():
        labels = np.array(ds["label"])
        total  = len(labels)
        counts = {ID2LABEL[i]: int((labels == i).sum()) for i in range(3)}
        pct    = {k: f"{v/total*100:.1f}%" for k, v in counts.items()}
        logger.info(
            f"{name} [{split}] {total} examples | "
            + "  ".join(f"{k}: {v} ({pct[k]})" for k, v in counts.items())
        )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 55)
    print("Smoke test: data_utils.py")
    print("=" * 55)

    fpb  = load_fpb()
    fiqa = load_fiqa()

    fpb_dfs  = dataset_to_df(fpb)
    fiqa_dfs = dataset_to_df(fiqa)

    print(f"\nFPB  train sample:\n{fpb_dfs['train'].head(3).to_string(index=False)}")
    print(f"\nFiQA train sample:\n{fiqa_dfs['train'].head(3).to_string(index=False)}")

    spy = fetch_market_data("SPY", save_path="data/market/spy_daily.csv")
    print(f"\nSPY tail:\n{spy.tail(3)}")

    print("\n✅  Smoke test passed.")
