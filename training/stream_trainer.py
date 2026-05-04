"""
Streaming trainer — reads text from a queue (fed by the web crawler)
and trains the model on it in real time. Stops when stop_event is set.
"""
import time
import threading
import queue
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"

from model.config import ModelConfig
from model.transformer import TensorFlowLLM
from training.optimizer import build_optimizer


def cross_entropy_loss(logits, labels):
    logits = tf.cast(logits, tf.float32)
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    )


class StreamTrainer:
    """
    Trains on text as it arrives from the crawler queue.
    Accumulates tokens into fixed-length windows, then trains.
    """

    def __init__(
        self,
        config: ModelConfig,
        tokenizer,
        text_queue: queue.Queue,
        stop_event: threading.Event,
        seq_len: int = 512,
        batch_size: int = 4,
        max_lr: float = 3e-4,
        warmup_steps: int = 100,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        output_dir: str = "checkpoints",
        log_every: int = 10,
        save_every: int = 200,
        mixed_precision: bool = True,
    ):
        self.config      = config
        self.tokenizer   = tokenizer
        self.text_queue  = text_queue
        self.stop_event  = stop_event
        self.seq_len     = seq_len
        self.batch_size  = batch_size
        self.log_every   = log_every
        self.save_every  = save_every
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.total_tokens = 0

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

        self.model = TensorFlowLLM(config, name="llm")
        # Build graph
        dummy = tf.zeros((1, 16), dtype=tf.int32)
        self.model(dummy, use_cache=False, training=False)

        # Use a high total_steps estimate; LR schedule adapts
        self.optimizer = build_optimizer(
            max_lr=max_lr, warmup_steps=warmup_steps,
            total_steps=500_000, weight_decay=weight_decay, grad_clip=grad_clip,
        )

        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer,
                                         step=tf.Variable(0, dtype=tf.int64))
        self.ckpt_mgr = tf.train.CheckpointManager(
            self.ckpt, str(self.output_dir / "ckpts"), max_to_keep=3
        )

        # Resume if checkpoint exists and shapes match
        latest = self.ckpt_mgr.latest_checkpoint
        if latest:
            try:
                self.ckpt.restore(latest).expect_partial()
                self.global_step = int(self.ckpt.step)
                print(f"  Resumed from step {self.global_step}")
            except ValueError:
                print(f"  Checkpoint shape mismatch — starting fresh for this model size.")

        params = self.model.count_params()
        print(f"  Model: {params/1e6:.1f}M parameters")

    @tf.function
    def _train_step(self, input_ids, labels):
        with tf.GradientTape() as tape:
            outputs = self.model(input_ids, training=True)
            loss    = cross_entropy_loss(outputs["logits"], labels)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def run(self):
        """Main training loop — runs until stop_event is set."""
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id

        token_buffer: list[int] = []
        batch_inputs: list[list[int]] = []
        batch_labels: list[list[int]] = []

        t_log = time.time()
        losses: list[float] = []

        # ── Replay saved data first ────────────────────────────────────────
        saved_tokens = self._replay_saved(token_buffer)
        if saved_tokens:
            print(f"  Replaying {saved_tokens:,} tokens from data/crawled.jsonl")

        print("  Waiting for crawled data...")

        while not self.stop_event.is_set():
            # Pull text from the crawler queue
            try:
                text = self.text_queue.get(timeout=1.0)
            except queue.Empty:
                # If we have a full batch ready, train on it even while waiting
                if len(batch_inputs) >= self.batch_size:
                    loss = self._do_batch(batch_inputs, batch_labels)
                    losses.append(loss)
                    batch_inputs, batch_labels = [], []
                    self._maybe_log(losses, t_log)
                    if losses:
                        t_log = time.time()
                continue

            # Tokenize and pack into the buffer
            ids = [bos] + self.tokenizer.encode(text) + [eos]
            token_buffer.extend(ids)
            self.total_tokens += len(ids)

            # Slice into seq_len windows
            while len(token_buffer) >= self.seq_len + 1:
                window = token_buffer[:self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len:]
                batch_inputs.append(window[:-1])
                batch_labels.append(window[1:])

                if len(batch_inputs) >= self.batch_size:
                    loss = self._do_batch(batch_inputs, batch_labels)
                    losses.append(loss)
                    batch_inputs, batch_labels = [], []
                    self._maybe_log(losses, t_log)
                    if losses:
                        t_log = time.time()

        # Final save
        self.ckpt_mgr.save()
        print(f"\n  Training stopped. Steps: {self.global_step} | "
              f"Tokens seen: {self.total_tokens:,}")

    def _replay_saved(self, token_buffer: list) -> int:
        """Load data/crawled.jsonl + bundled.jsonl into the token buffer."""
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        total = 0

        files = [
            _DATA_DIR / "bundled.jsonl",
            _DATA_DIR / "conversations.jsonl",
            _DATA_DIR / "crawled.jsonl",
        ]
        for path in files:
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            text = json.loads(line).get("text", line)
                        except json.JSONDecodeError:
                            text = line
                        ids = [bos] + self.tokenizer.encode(text) + [eos]
                        token_buffer.extend(ids)
                        total += len(ids)
            except Exception:
                pass
        self.total_tokens += total
        return total

    def _do_batch(self, inputs: list, labels: list) -> float:
        x = tf.constant(np.array(inputs, dtype=np.int32))
        y = tf.constant(np.array(labels, dtype=np.int32))
        loss = self._train_step(x, y)
        self.global_step += 1
        self.ckpt.step.assign_add(1)
        if self.global_step % self.save_every == 0:
            self.ckpt_mgr.save()
        return float(loss)

    def _maybe_log(self, losses: list[float], t_start: float):
        if self.global_step % self.log_every != 0 or not losses:
            return
        avg_loss = sum(losses[-self.log_every:]) / min(len(losses), self.log_every)
        ppl      = float(np.exp(min(avg_loss, 20)))
        elapsed  = max(time.time() - t_start, 1e-6)
        tps      = self.log_every * self.batch_size * self.seq_len / elapsed
        lr       = float(self.optimizer.learning_rate)
        print(
            f"  step {self.global_step:>6} | "
            f"loss {avg_loss:.4f} | ppl {ppl:.1f} | "
            f"lr {lr:.2e} | {tps:,.0f} tok/s | "
            f"crawled {self.total_tokens/1000:.1f}k tokens"
        )
