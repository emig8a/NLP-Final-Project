#!/usr/bin/env bash
# ============================================================
# setup.sh — Environment bootstrap for financial_sentiment_nlp
# Supports: MacBook Pro M1/M2 (MPS) and Linux GPU (Colab/RCAC)
# Usage:
#   Local:  bash setup.sh
#   Colab:  !bash setup.sh --colab
# ============================================================

set -euo pipefail

# ── Detect platform ─────────────────────────────────────────
COLAB=false
for arg in "$@"; do
  [[ "$arg" == "--colab" ]] && COLAB=true
done

OS=$(uname -s)
ARCH=$(uname -m)
echo ">>> Platform: $OS / $ARCH | Colab mode: $COLAB"

# ── Create & activate venv (skip in Colab) ──────────────────
if [[ "$COLAB" == false ]]; then
  if [[ ! -d "venv" ]]; then
    echo ">>> Creating virtual environment..."
    python3 -m venv venv
  fi
  source venv/bin/activate
  echo ">>> Activated venv"
fi

# ── Upgrade pip ─────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── Core ML / NLP dependencies ──────────────────────────────
echo ">>> Installing core ML packages..."
pip install --quiet \
  "transformers>=4.40.0" \
  "datasets>=2.19.0" \
  "scikit-learn>=1.4.0" \
  "accelerate>=0.30.0" \
  "evaluate>=0.4.0"

# ── PyTorch — platform-aware ────────────────────────────────
echo ">>> Installing PyTorch..."
if [[ "$COLAB" == true ]]; then
  pip install --quiet --upgrade torch torchvision torchaudio
elif [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  # Apple Silicon — MPS backend
  pip install --quiet torch torchvision torchaudio
  echo "    (MPS backend will be used on Apple Silicon)"
else
  pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ── Data & analysis ─────────────────────────────────────────
echo ">>> Installing data/analysis packages..."
pip install --quiet \
  pandas \
  numpy \
  scipy \
  matplotlib \
  seaborn \
  "yfinance>=0.2.38" \
  requests \
  beautifulsoup4 \
  tqdm

# ── Verify key imports ──────────────────────────────────────
echo ">>> Verifying installation..."
python3 - <<'EOF'
import torch, transformers, datasets, sklearn, yfinance, pandas, numpy
print(f"  torch        {torch.__version__}  (MPS: {torch.backends.mps.is_available()}, CUDA: {torch.cuda.is_available()})")
print(f"  transformers {transformers.__version__}")
print(f"  datasets     {datasets.__version__}")
print(f"  sklearn      {sklearn.__version__}")
print(f"  yfinance     {yfinance.__version__}")
print(f"  pandas       {pandas.__version__}")
print(f"  numpy        {numpy.__version__}")
EOF

echo ""
echo "✅  Setup complete. Activate your environment with:"
echo "    source venv/bin/activate"
