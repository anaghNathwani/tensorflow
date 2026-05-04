# TensorFlow LLM

A state-of-the-art decoder-only language model built with TensorFlow, implementing the same architecture innovations as LLaMA 3 / Mistral.

**Architecture:** RoPE · Grouped Query Attention (GQA) · SwiGLU · RMSNorm · KV-cache · bfloat16

---

## Setup

### No accounts or logins needed

Everything runs offline. The built-in byte tokenizer requires no download and no HuggingFace account. Training data is bundled in `data/bundled.jsonl`. Just install Python packages and run.

### Any platform (recommended)

`setup.sh` auto-detects your OS and hardware (Apple Silicon, NVIDIA GPU, CPU-only) and installs the right TensorFlow variant:

```bash
bash setup.sh
source .venv/bin/activate
```

### M4 Mac (Apple Silicon) — shortcut

```bash
bash setup_mac.sh
source .venv/bin/activate
```

### Manual — M4 Mac

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow-macos tensorflow-metal
pip install transformers datasets tokenizers numpy scipy tqdm wandb sentencepiece
```

### Manual — Linux (CUDA)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-codespaces.txt
```

### GitHub Codespaces

Open the repo in Codespaces — setup runs automatically via `.devcontainer/devcontainer.json`. No manual steps needed.

---

## Training

### Basic training (uses bundled dataset, fully offline)

```bash
python train.py --size small --batch 4
```

### Train on your own data (.txt or .jsonl files)

```bash
python train.py --size small --data data/train.jsonl --batch 4
python train.py --size small --data data/                  # scans dir recursively
```

### Choose model size

```bash
python train.py --size small    # ~125M params — fast, fits any M2+ Mac
python train.py --size medium   # ~1B params  — fits M4 Pro/Max
python train.py --size large    # ~7B params  — for Codespaces / multi-GPU
```

### Use a custom config file

```bash
python train.py --config configs/small.json --data data/train.jsonl
```

### Resume from checkpoint

```bash
python train.py --size small --resume
```

### Change learning rate and schedule

```bash
python train.py --size small --max_lr 1e-4 --warmup 1000 --steps 50000
```

### Gradient accumulation (simulate larger batch)

```bash
python train.py --size large --batch 4 --accum 8   # effective batch = 32
```

### Change sequence length

```bash
python train.py --size small --seq 4096
```

### Custom HuggingFace tokenizer (optional — requires HF account/login)

```bash
python train.py --size small --tokenizer meta-llama/Llama-3.2-1B
python train.py --size small --tokenizer /path/to/local/tokenizer
```

### Custom output directory

```bash
python train.py --size small --output runs/experiment_1
```

### Disable mixed precision (bfloat16)

```bash
python train.py --size small --no_mp
```

### Enable Weights & Biases logging

```bash
python train.py --size small --wandb
```

### Full example (Codespaces, multi-GPU)

```bash
python train.py \
  --size large \
  --data data/ \
  --batch 16 \
  --accum 4 \
  --seq 8192 \
  --max_lr 3e-4 \
  --warmup 2000 \
  --steps 100000 \
  --output runs/large_run \
  --wandb \
  --resume
```

---

## Generation — Interactive Terminal Chat

`generate.py` opens a full interactive chat session with coloured output, streaming tokens, conversation history, and live settings control.

### Start a chat session

```bash
python3 generate.py --checkpoint checkpoints/ckpts
```

### In-session commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear the screen |
| `/settings` | Show current generation settings |
| `/set temperature 0.9` | Change any setting on the fly |
| `/reset` | Clear conversation history |
| `/quit` or `/exit` | Exit |

### Change model size

```bash
python3 generate.py --checkpoint checkpoints/ckpts --size medium
python3 generate.py --checkpoint checkpoints/ckpts --config configs/large.json
```

### Tune generation at startup

```bash
# More creative
python3 generate.py --checkpoint checkpoints/ckpts \
  --temperature 1.2 --top_k 100 --top_p 0.98

# More focused / deterministic
python3 generate.py --checkpoint checkpoints/ckpts \
  --temperature 0.3 --top_k 20 --top_p 0.85
```

### Control output length

```bash
python3 generate.py --checkpoint checkpoints/ckpts --max_tokens 512
```

### Add a system prompt

```bash
python3 generate.py --checkpoint checkpoints/ckpts \
  --system "You are a helpful coding assistant."
```

### Disable multi-turn memory (stateless mode)

```bash
python3 generate.py --checkpoint checkpoints/ckpts --no_history
```

### Custom tokenizer

```bash
python3 generate.py --checkpoint checkpoints/ckpts \
  --tokenizer /path/to/tokenizer
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/ckpts` | Checkpoint directory |
| `--size` | `small` | Model size preset |
| `--config` | — | Custom config JSON path |
| `--tokenizer` | `meta-llama/Llama-3.2-1B` | HuggingFace tokenizer |
| `--max_tokens` | `256` | Max tokens per response |
| `--temperature` | `0.8` | Sampling temperature |
| `--top_k` | `50` | Top-k filter |
| `--top_p` | `0.95` | Nucleus sampling threshold |
| `--rep_penalty` | `1.1` | Repetition penalty |
| `--system` | — | System prompt |
| `--no_history` | off | Disable conversation memory |

---

## Benchmarking

### Benchmark forward pass and generation speed

```bash
python benchmark.py --size small
python benchmark.py --size medium
python benchmark.py --size large
```

### Custom batch size and sequence length

```bash
python benchmark.py --size small --batch 4 --seq 1024
```

### Number of steps to average over

```bash
python benchmark.py --size small --steps 50
```

---

## TensorBoard

```bash
tensorboard --logdir checkpoints/logs
# Open http://localhost:6006
```

In Codespaces, port 6006 is forwarded and opens automatically.

---

## Model configs

Configs live in `configs/` as JSON files. Edit them or create your own:

```bash
# View a config
cat configs/small.json

# Train with it
python train.py --config configs/small.json --data data/train.jsonl
```

Key fields:

| Field | Description |
|---|---|
| `hidden_size` | Embedding / hidden dimension |
| `num_hidden_layers` | Number of transformer layers |
| `num_attention_heads` | Query heads |
| `num_key_value_heads` | KV heads (< query heads = GQA) |
| `intermediate_size` | SwiGLU FFN inner dimension |
| `max_position_embeddings` | Maximum context length |
| `rope_theta` | RoPE base frequency (higher = longer context) |

---

## Project structure

```
.
├── model/
│   ├── config.py        # ModelConfig dataclass + size presets
│   ├── layers.py        # RMSNorm, RoPE, SwiGLU
│   ├── attention.py     # Grouped Query Attention + KV-cache
│   └── transformer.py   # Full decoder-only model
├── training/
│   ├── trainer.py       # Training loop, checkpointing, logging
│   ├── optimizer.py     # AdamW + cosine LR with warmup
│   └── data.py          # tf.data pipeline, HuggingFace loader
├── inference/
│   └── generate.py      # Sampling, beam search, streaming
├── configs/
│   ├── small.json       # ~125M params
│   ├── medium.json      # ~1B params
│   └── large.json       # ~7B params
├── .devcontainer/
│   └── devcontainer.json  # GitHub Codespaces configuration
├── train.py             # Training entry point
├── generate.py          # Generation entry point
├── benchmark.py         # Speed benchmarking
├── setup.sh             # Universal setup (auto-detects platform)
├── setup_mac.sh         # M4 Mac shortcut
├── requirements.txt             # M4 Mac dependencies
└── requirements-codespaces.txt  # Linux/CUDA dependencies
```
