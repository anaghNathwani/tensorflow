#!/usr/bin/env python3
"""
Quick benchmark: tokens/sec for forward pass + generation on this machine.

Usage:
  python benchmark.py --size small
  python benchmark.py --size medium --batch 2
"""
import argparse
import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size",  default="small", choices=["small", "medium", "large"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq",   type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()

    import tensorflow as tf
    print("TF version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    from tokenizer.byte_tokenizer import ByteTokenizer
    from model.config import ModelConfig
    from model.transformer import TensorFlowLLM

    config = getattr(ModelConfig, args.size)()
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    model = TensorFlowLLM(config)

    # Build
    dummy = tf.zeros((args.batch, args.seq), dtype=tf.int32)
    model(dummy, training=False)
    params = model.count_params()
    print(f"Parameters: {params/1e6:.1f}M")

    @tf.function
    def fwd(x):
        return model(x, training=False)

    # Warmup
    for _ in range(3):
        fwd(dummy)

    # Forward pass throughput
    t0 = time.time()
    for _ in range(args.steps):
        fwd(dummy)
    elapsed = time.time() - t0
    tps = args.batch * args.seq * args.steps / elapsed
    print(f"Forward pass: {tps:,.0f} tokens/sec  ({elapsed/args.steps*1000:.1f} ms/step)")

    # Autoregressive generation throughput (batch=1, no KV-cache for simplicity)
    gen_tokens = 100
    input_ids = tf.constant([[1] * 32], dtype=tf.int32)
    t0 = time.time()
    past_kvs = None
    out = model(input_ids, use_cache=True, training=False)
    past_kvs = out["past_key_values"]
    for _ in range(gen_tokens):
        next_logits = out["logits"][0, -1].numpy()
        next_id = int(np.argmax(next_logits))
        next_t = tf.constant([[next_id]], dtype=tf.int32)
        out = model(next_t, past_key_values=past_kvs, use_cache=True, training=False)
        past_kvs = out["past_key_values"]
    gen_tps = gen_tokens / (time.time() - t0)
    print(f"Generation:   {gen_tps:.1f} tokens/sec  (KV-cache enabled)")


if __name__ == "__main__":
    main()
