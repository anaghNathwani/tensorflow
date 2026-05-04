"""Core building blocks: RMSNorm, SwiGLU, RoPE."""
import tensorflow as tf
import numpy as np


class RMSNorm(tf.keras.layers.Layer):
    """Root Mean Square Layer Normalization — faster than LayerNorm, no bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(self.hidden_size,),
            initializer="ones", trainable=True,
        )

    def call(self, x):
        x = tf.cast(x, tf.float32)
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        x = x / rms
        return tf.cast(x, self.weight.dtype) * self.weight


class RotaryEmbedding(tf.keras.layers.Layer):
    """Rotary Position Embedding (RoPE) with YaRN-style long-context scaling."""

    def __init__(self, dim: int, max_seq_len: int = 8192,
                 theta: float = 500000.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        half = self.dim // 2
        freqs = 1.0 / (self.theta ** (np.arange(0, half, dtype=np.float32) / half))
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, freqs)
        self._cos = tf.constant(np.cos(freqs), dtype=tf.float32)  # (seq, dim//2)
        self._sin = tf.constant(np.sin(freqs), dtype=tf.float32)

    def _rotate_half(self, x):
        half = tf.shape(x)[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return tf.concat([-x2, x1], axis=-1)

    def call(self, q, k, position_ids=None):
        """Apply RoPE to queries and keys.

        q, k: (batch, heads, seq, head_dim)
        position_ids: (batch, seq) or None
        """
        seq_len = tf.shape(q)[2]
        if position_ids is None:
            cos = self._cos[:seq_len]   # (seq, dim//2)
            sin = self._sin[:seq_len]
        else:
            cos = tf.gather(self._cos, position_ids)  # (batch, seq, dim//2)
            sin = tf.gather(self._sin, position_ids)

        # Broadcast to (batch, 1, seq, head_dim) then tile to head_dim
        cos = tf.concat([cos, cos], axis=-1)
        sin = tf.concat([sin, sin], axis=-1)

        if position_ids is None:
            cos = cos[tf.newaxis, tf.newaxis, :, :]   # (1,1,seq,dim)
            sin = sin[tf.newaxis, tf.newaxis, :, :]
        else:
            cos = cos[:, tf.newaxis, :, :]             # (batch,1,seq,dim)
            sin = sin[:, tf.newaxis, :, :]

        orig_dtype = q.dtype
        q = tf.cast(q, tf.float32)
        k = tf.cast(k, tf.float32)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return tf.cast(q_rot, orig_dtype), tf.cast(k_rot, orig_dtype)


class SwiGLU(tf.keras.layers.Layer):
    """SwiGLU feed-forward: gate * SiLU(up) — used in LLaMA, PaLM."""

    def __init__(self, hidden_size: int, intermediate_size: int, **kwargs):
        super().__init__(**kwargs)
        self.gate_proj = tf.keras.layers.Dense(intermediate_size, use_bias=False)
        self.up_proj   = tf.keras.layers.Dense(intermediate_size, use_bias=False)
        self.down_proj = tf.keras.layers.Dense(hidden_size,       use_bias=False)

    def call(self, x):
        return self.down_proj(tf.nn.silu(self.gate_proj(x)) * self.up_proj(x))
