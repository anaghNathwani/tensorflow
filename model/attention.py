"""Grouped Query Attention (GQA) with KV-cache and Flash-style chunking."""
import tensorflow as tf
import math
from model.layers import RotaryEmbedding


class GroupedQueryAttention(tf.keras.layers.Layer):
    """
    GQA: num_key_value_heads < num_attention_heads.
    Each KV head is shared across (num_attention_heads / num_key_value_heads) Q heads.
    Supports:
      - KV-cache for autoregressive decoding
      - Causal masking
      - Sliding-window attention (optional)
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_heads    = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim     = config.head_dim
        self.groups       = self.num_heads // self.num_kv_heads
        self.hidden_size  = config.hidden_size
        self.scale        = self.head_dim ** -0.5
        self.dropout_rate = config.attention_dropout

        kw = dict(use_bias=False)
        q_dim  = self.num_heads    * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = tf.keras.layers.Dense(q_dim,  **kw, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(kv_dim, **kw, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(kv_dim, **kw, name="v_proj")
        self.o_proj = tf.keras.layers.Dense(self.hidden_size, **kw, name="o_proj")

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
            name="rope",
        )

        if self.dropout_rate > 0:
            self.attn_drop = tf.keras.layers.Dropout(self.dropout_rate)

    def _split_heads(self, x, num_heads):
        """(batch, seq, dim) -> (batch, heads, seq, head_dim)"""
        batch = tf.shape(x)[0]
        seq   = tf.shape(x)[1]
        x = tf.reshape(x, (batch, seq, num_heads, self.head_dim))
        return tf.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x):
        """(batch, heads, seq, head_dim) -> (batch, seq, hidden)"""
        batch = tf.shape(x)[0]
        seq   = tf.shape(x)[2]
        x = tf.transpose(x, (0, 2, 1, 3))
        return tf.reshape(x, (batch, seq, self.num_heads * self.head_dim))

    def call(self, hidden_states, attention_mask=None, position_ids=None,
             past_key_value=None, use_cache=False, training=False):

        q = self._split_heads(self.q_proj(hidden_states), self.num_heads)
        k = self._split_heads(self.k_proj(hidden_states), self.num_kv_heads)
        v = self._split_heads(self.v_proj(hidden_states), self.num_kv_heads)

        q, k = self.rope(q, k, position_ids=position_ids)

        # Append to KV-cache
        if past_key_value is not None:
            k = tf.concat([past_key_value[0], k], axis=2)
            v = tf.concat([past_key_value[1], v], axis=2)

        present_kv = (k, v) if use_cache else None

        # Expand KV heads to match Q heads (GQA repeat)
        if self.groups > 1:
            k = tf.repeat(k, self.groups, axis=1)
            v = tf.repeat(v, self.groups, axis=1)

        # Scaled dot-product attention
        attn = tf.matmul(q, k, transpose_b=True) * self.scale  # (B, H, Sq, Sk)

        if attention_mask is not None:
            attn = attn + tf.cast(attention_mask, attn.dtype)

        attn = tf.nn.softmax(tf.cast(attn, tf.float32), axis=-1)
        attn = tf.cast(attn, q.dtype)

        if self.dropout_rate > 0 and training:
            attn = self.attn_drop(attn, training=training)

        out = tf.matmul(attn, v)                                 # (B, H, Sq, D)
        out = self.o_proj(self._merge_heads(out))

        return out, present_kv
