"""AdamW + cosine LR schedule with linear warmup — standard for LLM training."""
import tensorflow as tf
import math


class CosineWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr: float, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.1):
        super().__init__()
        self.max_lr       = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = max_lr * min_lr_ratio

    def __call__(self, step):
        step  = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total  = tf.cast(self.total_steps,  tf.float32)

        linear_lr  = self.max_lr * step / warmup
        progress   = (step - warmup) / tf.maximum(total - warmup, 1.0)
        cosine_lr  = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )
        return tf.where(step < warmup, linear_lr, cosine_lr)

    def get_config(self):
        return dict(max_lr=self.max_lr, warmup_steps=self.warmup_steps,
                    total_steps=self.total_steps, min_lr_ratio=self.min_lr / self.max_lr)


def build_optimizer(max_lr: float = 3e-4, warmup_steps: int = 2000,
                    total_steps: int = 100_000, weight_decay: float = 0.1,
                    beta1: float = 0.9, beta2: float = 0.95,
                    grad_clip: float = 1.0):
    schedule = CosineWithWarmup(max_lr, warmup_steps, total_steps)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=weight_decay,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=1e-8,
        global_clipnorm=grad_clip,
    )
    return optimizer
