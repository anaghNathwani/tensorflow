"""
Bootstrap training — gives the model baseline intelligence before any
web crawling. Runs automatically the first time generate.py is launched
if no checkpoint exists.

Trains on:
  data/bundled.jsonl      — facts, science, history, stories
  data/conversations.jsonl — conversational Q&A examples

Can also be run directly:
  python3 bootstrap.py
  python3 bootstrap.py --epochs 3   # more passes over the data
"""
import os
import sys
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def main(epochs: int = 2, quiet: bool = False):
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

    from pathlib import Path
    from tokenizer.byte_tokenizer import ByteTokenizer
    from model.config import ModelConfig
    from model.transformer import TensorFlowLLM
    from training.optimizer import build_optimizer

    DATA_DIR = Path(__file__).parent / "data"
    CKPT_DIR = Path(__file__).parent / "checkpoints" / "ckpts"

    tokenizer = ByteTokenizer()
    config    = ModelConfig.small()
    model     = TensorFlowLLM(config, name="llm")

    dummy = tf.zeros((1, 16), dtype=tf.int32)
    model(dummy, use_cache=False, training=False)

    ckpt     = tf.train.Checkpoint(model=model, step=tf.Variable(0, dtype=tf.int64))
    ckpt_mgr = tf.train.CheckpointManager(ckpt, str(CKPT_DIR), max_to_keep=3)

    if ckpt_mgr.latest_checkpoint:
        ckpt.restore(ckpt_mgr.latest_checkpoint).expect_partial()
        if not quiet:
            print(f"  Existing checkpoint found — skipping bootstrap.")
        return

    if not quiet:
        print()
        print("  BootStrapping model with base knowledge...")
        print(f"  Parameters: {model.count_params()/1e6:.1f}M")

    # Load all bundled data files
    data_files = [
        DATA_DIR / "bundled.jsonl",
        DATA_DIR / "conversations.jsonl",
    ]

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    seq_len    = 256
    batch_size = 4

    # Build dataset (in memory — these files are small)
    tokens_all: list[int] = []
    for path in data_files:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", line)
                except json.JSONDecodeError:
                    text = line
                tokens_all += [bos] + tokenizer.encode(text) + [eos]

    windows = []
    for i in range(0, len(tokens_all) - seq_len, seq_len):
        windows.append(tokens_all[i:i + seq_len + 1])

    if not windows:
        if not quiet:
            print("  No bundled data found — skipping bootstrap.")
        return

    if not quiet:
        print(f"  Bundled data: {len(tokens_all):,} tokens, {len(windows)} windows")
        print(f"  Training for {epochs} epoch(s)...")

    # Build optimizer — higher LR for fast convergence on small dataset
    total_steps = len(windows) // batch_size * epochs
    optimizer   = build_optimizer(
        max_lr=5e-4, warmup_steps=max(20, total_steps // 10),
        total_steps=max(total_steps, 100),
    )

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            out  = model(x, training=True)
            logits = tf.cast(out["logits"], tf.float32)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    step  = 0
    for epoch in range(epochs):
        np.random.shuffle(windows)
        losses = []
        for i in range(0, len(windows) - batch_size, batch_size):
            batch   = windows[i:i + batch_size]
            x = tf.constant(np.array([w[:-1] for w in batch], dtype=np.int32))
            y = tf.constant(np.array([w[1:]  for w in batch], dtype=np.int32))
            loss = train_step(x, y)
            losses.append(float(loss))
            step += 1

        avg = sum(losses) / max(len(losses), 1)
        if not quiet:
            print(f"  Epoch {epoch+1}/{epochs}  —  loss {avg:.4f}  "
                  f"ppl {np.exp(min(avg,20)):.1f}")

    ckpt.step.assign(step)
    ckpt_mgr.save()
    if not quiet:
        print(f"  Bootstrap complete. Checkpoint saved to {CKPT_DIR}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()
    main(epochs=args.epochs)
