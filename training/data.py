"""
Streaming data pipeline:
  - Reads raw .txt / .jsonl files
  - Tokenizes with a HuggingFace tokenizer (or custom BPE)
  - Packs sequences to max_seq_len with no padding waste
  - Builds a tf.data pipeline with prefetching
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Iterator, Union


def token_generator(paths: list[str], tokenizer, max_seq_len: int,
                    text_key: str = "text") -> Iterator[list[int]]:
    """Yield packed token windows from text files or JSONL."""
    import json

    buffer: list[int] = []
    bos = tokenizer.bos_token_id or 1
    eos = tokenizer.eos_token_id or 2

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get(text_key, line)
                except (json.JSONDecodeError, AttributeError):
                    text = line

                ids = [bos] + tokenizer.encode(text) + [eos]
                buffer.extend(ids)

                while len(buffer) >= max_seq_len + 1:
                    yield buffer[:max_seq_len + 1]
                    buffer = buffer[max_seq_len:]  # stride = full window


def build_dataset(paths: list[str], tokenizer, max_seq_len: int = 2048,
                  batch_size: int = 8, shuffle_buffer: int = 1000,
                  seed: int = 42) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset of (input_ids, labels) tensors.
    labels = input_ids shifted left by 1 (next-token prediction).
    """

    def _gen():
        for window in token_generator(paths, tokenizer, max_seq_len):
            arr = np.array(window, dtype=np.int32)
            yield arr[:-1], arr[1:]

    sig = (
        tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(_gen, output_signature=sig)
    ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_dataset_from_hf(dataset_name: str = "roneneldan/TinyStories",
                           split: str = "train", tokenizer=None,
                           max_seq_len: int = 2048, batch_size: int = 8,
                           max_examples: int = None) -> tf.data.Dataset:
    """Load a HuggingFace dataset and convert to tf.data."""
    from datasets import load_dataset

    ds_hf = load_dataset(dataset_name, split=split, streaming=True)
    if max_examples:
        ds_hf = ds_hf.take(max_examples)

    bos = tokenizer.bos_token_id or 1
    eos = tokenizer.eos_token_id or 2
    buffer: list[int] = []

    def _gen():
        nonlocal buffer
        for example in ds_hf:
            text = example.get("text", "")
            ids  = [bos] + tokenizer.encode(text) + [eos]
            buffer.extend(ids)
            while len(buffer) >= max_seq_len + 1:
                window = buffer[:max_seq_len + 1]
                buffer = buffer[max_seq_len:]
                arr = np.array(window, dtype=np.int32)
                yield arr[:-1], arr[1:]

    sig = (
        tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
    )
    out = tf.data.Dataset.from_generator(_gen, output_signature=sig)
    out = out.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return out
