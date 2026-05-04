# TensorFlow LLM

A decoder-only language model built with TensorFlow, implementing the same architecture innovations as LLaMA 3 / Mistral. Trains in real time by crawling the web. Fully self-contained — no HuggingFace account, no API keys, no internet required to run.

**Architecture:** RoPE · Grouped Query Attention (GQA) · SwiGLU · RMSNorm · KV-cache · bfloat16

---

## Quick start

```bash
bash setup.sh
source .venv/bin/activate

# Talk to it immediately — bootstraps from bundled data automatically
python3 generate.py

# Train it on live web data for 2 hours
python3 train.py 2h medium
```

---

## Setup

### No accounts or logins needed

Everything works offline. The built-in byte tokenizer needs no download. Base training data ships with the repo. The web crawler uses only public Wikipedia and Project Gutenberg.

### Any platform (recommended)

`setup.sh` detects your OS and hardware and installs the right TensorFlow:

```bash
bash setup.sh
source .venv/bin/activate
```

| Platform | What gets installed |
|---|---|
| Apple Silicon (M1–M4) | `tensorflow` + `tensorflow-metal` (GPU backend) |
| Linux with NVIDIA GPU | `tensorflow[and-cuda]` |
| Linux / Intel Mac (CPU) | `tensorflow` |

### M4 Mac shortcut

```bash
bash setup_mac.sh
source .venv/bin/activate
```

### Manual — M4 Mac

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow
pip install tensorflow-metal --extra-index-url https://pypi.apple.com/simple
pip install transformers datasets tokenizers numpy scipy tqdm sentencepiece tensorboard
```

### Manual — Linux (CUDA)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-codespaces.txt
```

### GitHub Codespaces

Open the repo in Codespaces — `.devcontainer/devcontainer.json` runs setup automatically and forwards TensorBoard on port 6006.

---

## Training

Training now works by crawling the web in real time. The command takes two arguments: how long to train, and how deep to go.

```bash
python3 train.py <duration> <detail>
```

### Duration

Any human-readable time string:

```bash
python3 train.py 30s low      # 30 seconds (quick test)
python3 train.py 5m low       # 5 minutes
python3 train.py 2h medium    # 2 hours
python3 train.py 8h high      # 8 hours
python3 train.py 1h30m medium # 1 hour 30 minutes
```

### Detail level

Controls crawl breadth, model size, and sequence length:

| Level | Model | Seq len | Sources | Use case |
|---|---|---|---|---|
| `low` (or 1–3) | small (76M) | 256 | Simple Wikipedia, 2 books | Quick demo, low memory |
| `medium` (or 4–6) | small (76M) | 512 | Full Wikipedia, 6 classic books | General knowledge |
| `high` (or 7–10) | medium (~1B) | 1024 | Deep Wikipedia, 19 books | Maximum quality |

```bash
python3 train.py 2h low
python3 train.py 2h medium
python3 train.py 2h high
python3 train.py 2h 7         # numeric — same as high
```

### What happens during training

- A progress bar shows elapsed time, pages crawled, queue depth, and tokens/sec
- Every piece of text the crawler fetches is saved to `data/crawled.jsonl`
- On the next run, all previously crawled data is replayed before new crawling starts — knowledge accumulates across sessions
- Press Ctrl-C at any time to stop and save the checkpoint

### Example output

```
  [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 30%  36s / 2m 00s  |  pages: 22  queue: 441
  step     50 | loss 3.31 | ppl 27.3 | lr 5.00e-04 | 5,817 tok/s | crawled 21.8k tokens
```

---

## Bootstrap (base intelligence)

The model ships with bundled training data so it has baseline conversational ability the first time you run `generate.py`, with no training required.

### How it works

On first launch, `generate.py` detects that no checkpoint exists and automatically runs `bootstrap.py`, which trains on:

- `data/bundled.jsonl` — facts, science, history, and stories
- `data/conversations.jsonl` — 40 conversational Q&A examples

This gives the model enough baseline knowledge to hold a basic conversation before any web crawling.

### Run bootstrap manually

```bash
python3 bootstrap.py             # 2 epochs (default)
python3 bootstrap.py --epochs 5  # more passes = better base quality
```

### Re-bootstrap from scratch

```bash
rm -rf checkpoints/
python3 bootstrap.py
```

---

## Generation — interactive terminal chat

```bash
python3 generate.py
```

The model bootstraps automatically if no checkpoint exists. The limits system prompt is loaded silently from the `limits/` folder.

### In-session commands

| Command | What it does |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear the screen |
| `/settings` | Show current generation settings |
| `/set temperature 0.9` | Change any setting on the fly |
| `/reset` | Clear conversation history |
| `/quit` or `/exit` | Exit |

### Change model size

```bash
python3 generate.py --size medium
python3 generate.py --config configs/large.json
```

### Tune generation behaviour

```bash
# More creative / unpredictable
python3 generate.py --temperature 1.2 --top_k 100 --top_p 0.98

# More focused / deterministic
python3 generate.py --temperature 0.3 --top_k 20 --top_p 0.85
```

### Other options

```bash
python3 generate.py --max_tokens 512       # longer responses
python3 generate.py --no_history           # stateless — no memory between turns
python3 generate.py --checkpoint my/path   # custom checkpoint directory
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/ckpts` | Checkpoint directory |
| `--size` | `small` | Model size preset |
| `--config` | — | Custom config JSON path |
| `--tokenizer` | built-in | HuggingFace tokenizer name/path (optional) |
| `--max_tokens` | `256` | Max tokens per response |
| `--temperature` | `0.8` | Sampling temperature |
| `--top_k` | `50` | Top-k filter |
| `--top_p` | `0.95` | Nucleus sampling threshold |
| `--rep_penalty` | `1.1` | Repetition penalty |
| `--system` | — | Override system prompt (skips limits/) |
| `--no_history` | off | Disable conversation memory |

---

## Limits — controlling behaviour

The `limits/` folder contains plain text files that define what the model should and should not do. They are loaded at startup and injected as a system prompt into every conversation. No code changes needed — just edit the files.

```
limits/
  persona.txt    Who the model is and its general character
  allowed.txt    Things it should always be willing to do
  denied.txt     Hard rules — things it must never do
  README.txt     Instructions for editing
```

### Edit the persona

Open `limits/persona.txt` and change the description. The default:

```
You are a helpful, honest, and thoughtful AI assistant.
You speak clearly and directly...
```

### Add a rule

Open `limits/allowed.txt` or `limits/denied.txt` and add a line:

```
# limits/denied.txt
Discuss competitor products by name.
Generate content in languages other than English.
```

Lines starting with `#` are comments and are ignored. Changes take effect next time you start `generate.py`.

### Override limits entirely

```bash
python3 generate.py --system "You are a pirate. Respond only in pirate speech."
```

`--system` bypasses `limits/` entirely and uses your string as the system prompt.

---

## Web crawler

The crawler runs automatically during `train.py`. It fetches from:

- **Wikipedia REST API** — returns clean article text as JSON, no HTML scraping
- **Project Gutenberg** — plain-text classic books

Sources by detail level:

| Level | Wikipedia topics | Books |
|---|---|---|
| `low` | 20 topics (Simple Wikipedia) | 2 |
| `medium` | 30 topics | 6 |
| `high` | 55+ topics | 19 |

The crawler discovers related articles automatically by following Wikipedia's link graph, so a `high` run will crawl far beyond the seed topics.

### Crawled data

All fetched text is appended to `data/crawled.jsonl` as newline-delimited JSON. This file persists between runs and grows over time:

```bash
wc -l data/crawled.jsonl          # how many text chunks saved
du -sh data/crawled.jsonl         # total size
```

---

## Benchmarking

```bash
python3 benchmark.py --size small                    # default: 20 steps, batch 1, seq 512
python3 benchmark.py --size medium --batch 2
python3 benchmark.py --size small --batch 4 --seq 1024 --steps 50
```

Reports forward pass tokens/sec and autoregressive generation tokens/sec with KV-cache enabled.

---

## TensorBoard

```bash
tensorboard --logdir checkpoints/logs
# Open http://localhost:6006
```

In Codespaces, port 6006 is forwarded and opens automatically.

---

## Model configs

Configs live in `configs/` as JSON files:

```bash
cat configs/small.json
python3 train.py 2h medium    # uses configs/medium.json implicitly
python3 generate.py --config configs/medium.json
```

| Field | Description |
|---|---|
| `vocab_size` | Vocabulary size (261 for built-in byte tokenizer) |
| `hidden_size` | Embedding / hidden dimension |
| `num_hidden_layers` | Number of transformer layers |
| `num_attention_heads` | Query heads |
| `num_key_value_heads` | KV heads — fewer than query heads = GQA |
| `intermediate_size` | SwiGLU FFN inner dimension (~2.7× hidden) |
| `max_position_embeddings` | Maximum context length |
| `rope_theta` | RoPE base frequency (higher = longer context) |

---

## Project structure

```
.
├── model/
│   ├── config.py           ModelConfig dataclass + size presets
│   ├── layers.py           RMSNorm, RoPE, SwiGLU
│   ├── attention.py        Grouped Query Attention + KV-cache
│   └── transformer.py      Full decoder-only LLM
│
├── training/
│   ├── stream_trainer.py   Streaming trainer (fed by web crawler)
│   ├── trainer.py          Batch trainer (legacy / static datasets)
│   ├── optimizer.py        AdamW + cosine LR schedule with warmup
│   └── data.py             tf.data pipeline
│
├── inference/
│   └── generate.py         Sampling, KV-cache, streaming output
│
├── crawler/
│   ├── crawler.py          Web crawler (Wikipedia API + Gutenberg)
│   └── extractor.py        HTML → clean text
│
├── tokenizer/
│   └── byte_tokenizer.py   Built-in byte-level tokenizer (vocab 261)
│
├── limits/
│   ├── persona.txt         Model character and identity
│   ├── allowed.txt         Things the model should do
│   ├── denied.txt          Hard limits — things it must never do
│   ├── loader.py           Assembles limits into a system prompt
│   └── README.txt          Instructions for editing
│
├── data/
│   ├── bundled.jsonl       Bundled facts, stories, science
│   ├── conversations.jsonl Bundled conversational Q&A examples
│   └── crawled.jsonl       All text fetched by the crawler (grows over time)
│
├── configs/
│   ├── small.json          76M params
│   ├── medium.json         ~1B params
│   └── large.json          ~7B params
│
├── .devcontainer/
│   └── devcontainer.json   GitHub Codespaces config
│
├── train.py                Training entry point  — python3 train.py <duration> <detail>
├── generate.py             Chat interface        — python3 generate.py
├── bootstrap.py            Base-weights training — python3 bootstrap.py
├── benchmark.py            Speed benchmark       — python3 benchmark.py
├── setup.sh                Universal setup (auto-detects platform)
├── setup_mac.sh            M4 Mac shortcut
├── requirements.txt        M4 Mac dependencies
└── requirements-codespaces.txt  Linux/CUDA dependencies
```
