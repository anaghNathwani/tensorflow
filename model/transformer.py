"""Full decoder-only transformer: embedding → N × DecoderLayer → LM head."""
import tensorflow as tf
from model.config import ModelConfig
from model.layers import RMSNorm, SwiGLU
from model.attention import GroupedQueryAttention


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: ModelConfig, layer_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = GroupedQueryAttention(config, name=f"attn_{layer_idx}")
        self.mlp       = SwiGLU(config.hidden_size, config.intermediate_size,
                                name=f"mlp_{layer_idx}")
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps,
                                  name=f"input_norm_{layer_idx}")
        self.post_norm  = RMSNorm(config.hidden_size, config.rms_norm_eps,
                                  name=f"post_norm_{layer_idx}")

    def call(self, hidden_states, attention_mask=None, position_ids=None,
             past_key_value=None, use_cache=False, training=False):
        # Pre-norm attention with residual
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(
            self.input_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            training=training,
        )
        hidden_states = residual + hidden_states

        # Pre-norm FFN with residual
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_norm(hidden_states))

        return hidden_states, present_kv


class TensorFlowLLM(tf.keras.Model):
    """
    Decoder-only LLM (LLaMA-3 architecture):
      - Rotary position embeddings (RoPE)
      - Grouped Query Attention (GQA)
      - SwiGLU feed-forward
      - RMSNorm (pre-norm)
      - No bias anywhere
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.embed_tokens = tf.keras.layers.Embedding(
            config.vocab_size, config.hidden_size, name="embed_tokens"
        )
        self.layers_ = [
            DecoderLayer(config, i, name=f"layer_{i}")
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, name="final_norm")

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = tf.keras.layers.Dense(
                config.vocab_size, use_bias=False, name="lm_head"
            )

    def _make_causal_mask(self, seq_len: int, past_len: int = 0):
        """Upper-triangular mask: -inf above diagonal, 0 on/below."""
        total = seq_len + past_len
        mask = tf.linalg.band_part(tf.ones((total, total)), -1, 0)  # lower tri
        mask = tf.cast(1 - mask, tf.float32) * -1e9
        return mask[tf.newaxis, tf.newaxis, past_len:, :]            # (1,1,seq,total)

    def call(self, input_ids, attention_mask=None, position_ids=None,
             past_key_values=None, use_cache=False, training=False,
             return_dict=True):

        batch, seq_len = tf.shape(input_ids)[0], tf.shape(input_ids)[1]
        past_len = 0 if past_key_values is None else tf.shape(past_key_values[0][0])[2]

        hidden = self.embed_tokens(input_ids)

        # Build causal mask if not supplied
        if attention_mask is None:
            attention_mask = self._make_causal_mask(seq_len, past_len)

        # Absolute position ids for RoPE
        if position_ids is None:
            position_ids = tf.range(past_len, past_len + seq_len)[tf.newaxis]
            position_ids = tf.tile(position_ids, [batch, 1])

        present_kvs = []
        for i, layer in enumerate(self.layers_):
            pkv = past_key_values[i] if past_key_values is not None else None
            hidden, present_kv = layer(
                hidden, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=pkv,
                use_cache=use_cache, training=training,
            )
            if use_cache:
                present_kvs.append(present_kv)

        hidden = self.norm(hidden)

        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = tf.matmul(hidden, tf.transpose(self.embed_tokens.embeddings))

        if return_dict:
            return {
                "logits": logits,
                "past_key_values": present_kvs if use_cache else None,
                "hidden_states": hidden,
            }
        return logits

    def count_params(self) -> int:
        return sum(tf.size(w).numpy() for w in self.trainable_variables)
