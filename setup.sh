#!/bin/bash
# Universal setup for TensorFlow LLM
# Supports: M4 Mac (Apple Silicon), Linux (CUDA), Linux (CPU), GitHub Codespaces
set -e

# ── Detect platform ──────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
PYTHON=${PYTHON:-python3}

echo "==> Detected: OS=$OS  ARCH=$ARCH"

# ── Python version check ─────────────────────────────────────────────────────
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 10 ]; then
  echo "ERROR: Python 3.10+ required (found $PY_VERSION)"
  exit 1
fi
echo "==> Python $PY_VERSION — OK"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment..."
  $PYTHON -m venv .venv
else
  echo "==> Virtual environment already exists, skipping creation"
fi

# Activate
if [ "$OS" = "Darwin" ] || [ "$OS" = "Linux" ]; then
  source .venv/bin/activate
else
  echo "ERROR: Unsupported OS: $OS"
  exit 1
fi

echo "==> Upgrading pip..."
pip install --upgrade pip --quiet

# ── TensorFlow — platform-specific ───────────────────────────────────────────
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
  # Apple Silicon (M1–M4): tensorflow-macos merged into tensorflow as of TF 2.13
  # Metal GPU backend is on Apple's own index, not PyPI
  echo "==> Installing TensorFlow for Apple Silicon..."
  pip install tensorflow --quiet
  echo "==> Installing Metal GPU backend (Apple index)..."
  pip install tensorflow-metal --extra-index-url https://pypi.apple.com/simple --quiet

elif [ "$OS" = "Linux" ]; then
  # Check for NVIDIA GPU
  if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "unknown")
    echo "==> NVIDIA GPU detected (CUDA $CUDA_VERSION) — installing TensorFlow with CUDA..."
    pip install "tensorflow[and-cuda]" --quiet
  else
    echo "==> No GPU detected — installing CPU-only TensorFlow..."
    pip install tensorflow --quiet
  fi

elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
  # Intel Mac
  echo "==> Intel Mac detected — installing standard TensorFlow (no Metal)..."
  pip install tensorflow --quiet

else
  echo "WARNING: Unknown platform ($OS/$ARCH) — attempting generic TensorFlow install..."
  pip install tensorflow --quiet
fi

# ── Common dependencies ───────────────────────────────────────────────────────
echo "==> Installing common dependencies..."
pip install \
  "transformers>=4.40.0" \
  "datasets>=2.19.0" \
  "tokenizers>=0.19.0" \
  "numpy>=1.26.0" \
  "scipy>=1.13.0" \
  "tqdm>=4.66.0" \
  "sentencepiece>=0.2.0" \
  --quiet

# ── Optional: W&B ─────────────────────────────────────────────────────────────
read -r -p "Install Weights & Biases for experiment tracking? [y/N] " WANDB
if [[ "$WANDB" =~ ^[Yy]$ ]]; then
  pip install wandb --quiet
  echo "==> W&B installed. Run 'wandb login' before training with --wandb."
fi

# ── Verify install ────────────────────────────────────────────────────────────
echo ""
echo "==> Verifying installation..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'  TensorFlow : {tf.__version__}')
print(f'  GPU devices: {[g.name for g in gpus] or \"none (CPU only)\"}')
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete!"
echo ""
echo "  Activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  Quick benchmark:"
echo "    python benchmark.py --size small"
echo ""
echo "  Start training (no data download needed):"
echo "    python train.py --size small --batch 4"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
