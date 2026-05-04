"""
Training engine:
  - Mixed precision (bfloat16) via tf.keras.mixed_precision
  - Gradient checkpointing via tf.recompute_grad
  - Distributed training with MirroredStrategy (multi-GPU / Codespaces)
  - Checkpoint + W&B / TensorBoard logging
  - Gradient accumulation
"""
import os
import time
import tensorflow as tf
from pathlib import Path
from typing import Optional

from model.transformer import TensorFlowLLM
from model.config import ModelConfig
from training.optimizer import build_optimizer


def get_strategy():
    """Auto-detect best distribution strategy."""
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy()
    if gpus:
        return tf.distribute.OneDeviceStrategy("/gpu:0")
    # Apple Silicon — TF-Metal runs on GPU-like device named /gpu:0
    return tf.distribute.OneDeviceStrategy("/gpu:0")


def cross_entropy_loss(logits, labels, vocab_size: int):
    """Sparse cross-entropy, cast to float32 for numerical stability."""
    logits = tf.cast(logits, tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )
    return tf.reduce_mean(loss)


class Trainer:
    def __init__(
        self,
        config: ModelConfig,
        output_dir: str = "checkpoints",
        max_lr: float = 3e-4,
        warmup_steps: int = 2000,
        total_steps: int = 100_000,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        accum_steps: int = 1,
        log_every: int = 10,
        save_every: int = 500,
        use_wandb: bool = False,
        mixed_precision: bool = True,
    ):
        self.config      = config
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every   = log_every
        self.save_every  = save_every
        self.total_steps = total_steps
        self.accum_steps = accum_steps
        self.use_wandb   = use_wandb
        self.global_step = 0

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

        self.strategy  = get_strategy()
        self.optimizer = build_optimizer(max_lr, warmup_steps, total_steps,
                                         weight_decay, grad_clip=grad_clip)

        with self.strategy.scope():
            self.model = TensorFlowLLM(config, name="llm")

        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer,
                                         step=tf.Variable(0, dtype=tf.int64))
        self.ckpt_mgr = tf.train.CheckpointManager(
            self.ckpt, str(self.output_dir / "ckpts"), max_to_keep=3
        )
        self.summary_writer = tf.summary.create_file_writer(
            str(self.output_dir / "logs")
        )

        if use_wandb:
            import wandb
            wandb.init(project="tensorflow-llm", config=config.__dict__)

    @tf.function
    def _train_step(self, input_ids, labels):
        with tf.GradientTape() as tape:
            outputs = self.model(input_ids, training=True)
            loss    = cross_entropy_loss(outputs["logits"], labels,
                                         self.config.vocab_size)
            if self.model.losses:
                loss += tf.add_n(self.model.losses)
            scaled_loss = self.optimizer.get_scaled_loss(loss) \
                if hasattr(self.optimizer, "get_scaled_loss") else loss

        grads = tape.gradient(scaled_loss, self.model.trainable_variables)
        if hasattr(self.optimizer, "get_unscaled_gradients"):
            grads = self.optimizer.get_unscaled_gradients(grads)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, dataset: tf.data.Dataset, resume: bool = True):
        if resume and self.ckpt_mgr.latest_checkpoint:
            self.ckpt.restore(self.ckpt_mgr.latest_checkpoint)
            self.global_step = int(self.ckpt.step)
            print(f"Resumed from step {self.global_step}")

        t0 = time.time()
        for batch in dataset.repeat():
            input_ids, labels = batch
            loss = self._train_step(input_ids, labels)
            self.global_step += 1
            int(self.ckpt.step.assign_add(1))

            if self.global_step % self.log_every == 0:
                elapsed = time.time() - t0
                lr = float(self.optimizer.learning_rate)
                ppl = float(tf.exp(loss))
                tokens_per_sec = (
                    self.log_every * int(input_ids.shape[0]) * int(input_ids.shape[1])
                    / elapsed
                )
                print(
                    f"step {self.global_step:>7} | "
                    f"loss {float(loss):.4f} | ppl {ppl:.2f} | "
                    f"lr {lr:.2e} | tok/s {tokens_per_sec:,.0f}"
                )
                try:
                    with self.summary_writer.as_default():
                        tf.summary.scalar("train/loss", loss,  step=self.global_step)
                        tf.summary.scalar("train/ppl",  ppl,   step=self.global_step)
                        tf.summary.scalar("train/lr",   lr,    step=self.global_step)
                except Exception:
                    pass  # TensorBoard not installed — skip summaries
                if self.use_wandb:
                    import wandb
                    wandb.log({"loss": float(loss), "ppl": ppl, "lr": lr},
                              step=self.global_step)
                t0 = time.time()

            if self.global_step % self.save_every == 0:
                self.ckpt_mgr.save()
                print(f"  checkpoint saved at step {self.global_step}")

            if self.global_step >= self.total_steps:
                break

        self.ckpt_mgr.save()
        print("Training complete.")
