# Do Financial News Sentiment Models Align with Market Movement?

**Author:** Emilio Guzman Ochoa | Purdue University  
**Course:** NLP — Spring 2026

---

## Overview

A two-stage NLP pipeline that:
1. Fine-tunes and benchmarks sentiment models (TF-IDF + LR, BERT, FinBERT) on financial text datasets (FPB, FiQA)
2. Evaluates whether model-predicted sentiment correlates with real S&P 500 price movements

---

## Project Structure

```
├── data_utils.py       # Data loading: FPB, FiQA, yfinance market data
├── 01_eda.py           # Phase 1: EDA — class distributions, sentence lengths, vocab overlap
├── 02_baseline.py      # Phase 2: TF-IDF + Logistic Regression baseline (coming soon)
├── setup.sh            # One-command environment bootstrap (M1 + Colab/RCAC)
├── requirements.txt    # Pinned dependencies
└── README.md
```

---

## Quickstart

```bash
# 1. Bootstrap environment
bash setup.sh
source venv/bin/activate

# 2. Run EDA
python 01_eda.py
```

Outputs are saved to `results/` (plots) and `data/processed/` (CSV splits).

---

## Datasets

| Dataset | Size | Labels | Source |
|---|---|---|---|
| Financial PhraseBank (FPB) | ~4,800 sentences | negative / neutral / positive | [HuggingFace](https://huggingface.co/datasets/takala/financial_phrasebank) |
| FiQA-2018 SA | ~1,100 sentences | negative / neutral / positive (bucketed from continuous score) | [HuggingFace](https://huggingface.co/datasets/pauri32/fiqa-2018) |

---

## Models

| Model | Type | Status |
|---|---|---|
| TF-IDF + Logistic Regression | Baseline | Planned |
| BERT-base-uncased | Transformer | Planned |
| FinBERT (ProsusAI) | Domain-adapted Transformer | Planned |

---

## Requirements

- Python 3.10+
- `datasets>=2.19.0,<3.0.0` (required — FPB uses a loading script incompatible with datasets≥3.0)
- See `setup.sh` for full dependency list
