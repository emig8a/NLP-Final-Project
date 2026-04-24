# Google Colab Setup Guide — BERT Fine-Tuning
## Do Financial News Sentiment Models Align with Market Movement?
**Author:** Emilio Guzman Ochoa | Purdue University

---

## Prerequisites
- A Google account with access to [Google Colab](https://colab.research.google.com)
- The following files from the project repository:
  - `bert_finetune.py`
  - `finbert_finetune.py`
  - `data/processed/fpb_train.csv`
  - `data/processed/fpb_validation.csv`
  - `data/processed/fpb_test.csv`
  - `data/processed/fiqa_train.csv`
  - `data/processed/fiqa_validation.csv`
  - `data/processed/fiqa_test.csv`

---

## Step 1 — Enable GPU Runtime

1. Open a new Colab notebook at [colab.research.google.com](https://colab.research.google.com)
2. In the top menu click: **Runtime → Change runtime type**
3. Under **Hardware accelerator** select **T4 GPU**
4. Click **Save**

---

## Step 2 — Mount Google Drive

Run this in the first cell to connect google drive so that model checkpoints and results are saved automatically. 

```python
from google.colab import drive
drive.mount('/content/drive')
```

A pop-up will ask you to authorize access to your Google Drive. Click **Connect to Google Drive** and follow the prompts.

---

## Step 3 — Install Dependencies

Run this in a new cell:

```python
!pip install "transformers>=4.40.0" "datasets>=2.19.0,<3.0.0" scikit-learn accelerate -q
```
---

## Step 4 — Upload Project Files

Run this in a new cell. Select the 8 files mentioned above.

```python
from google.colab import files
import os

uploaded = files.upload()

# Automatically move CSVs into the correct folder
os.makedirs("data/processed", exist_ok=True)
for filename in uploaded.keys():
    if filename.endswith(".csv"):
        os.rename(filename, f"data/processed/{filename}")

print("Files in place:")
!ls data/processed/
```

---

## Step 5 — Run  Bert Training

There are 3 training runs with different hyperparameters. Run them one at a time in separate cells:

**Run 1** — lr=2e-5, epochs=3, batch=16
```python
!python bert_finetune.py --run 1
```

**Run 2** — lr=3e-5, epochs=4, batch=32
```python
!python bert_finetune.py --run 2
```

**Run 3** — lr=2e-5, epochs=5, batch=16
```python
!python bert_finetune.py --run 3
```

---

## Expected Outputs

After all 3 runs, the following files will be saved in `results/`:

| File | Description |
|---|---|
| `bert_run{N}_fpb_metrics.txt` | Classification report on FPB test set |
| `bert_run{N}_fiqa_metrics.txt` | Classification report on FiQA test set |
| `bert_run{N}_fpb_confusion.png` | Confusion matrix on FPB |
| `bert_run{N}_fiqa_confusion.png` | Confusion matrix on FiQA |
| `bert_run{N}_training_loss.png` | Training and validation loss curve |
| `bert_run{N}_fpb_predictions.csv` | Per-sentence predictions on FPB |
| `bert_run{N}_fiqa_predictions.csv` | Per-sentence predictions on FiQA |
| `bert_cross_eval_summary.csv` | Summary table of all runs and datasets |


## Run FinBert Training

**Run 1** — zero-shot, no training needed
```python
!python finbert_finetune.py --exp A
```

**Run 2** — fine-tune on FiQA
```python
!python finbert_finetune.py --exp B
```

**Run 3** — fine-tune on FPB
```python
!python finbert_finetune.py --exp C
```
--- 

## Expected Outputs

After all 3 runs, the following files will be saved in `results/`:
