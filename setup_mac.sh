#!/bin/bash
# One-shot setup for M4 Mac (Apple Silicon)
set -e

echo "==> Creating Python 3.11 virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing TensorFlow (tensorflow-macos merged into tensorflow as of TF 2.13)..."
pip install tensorflow
echo "==> Installing Metal GPU backend (Apple's index)..."
pip install tensorflow-metal --extra-index-url https://pypi.apple.com/simple

echo "==> Installing remaining dependencies..."
pip install transformers datasets tokenizers numpy scipy tqdm wandb sentencepiece

echo ""
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Quick test:"
echo "  python benchmark.py --size small"
echo ""
echo "Start training (TinyStories, no data download needed):"
echo "  python train.py --size small --batch 4"
