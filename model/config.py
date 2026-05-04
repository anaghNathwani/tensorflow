from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class ModelConfig:
    # Architecture
    # Default 261 = ByteTokenizer (256 bytes + 5 special tokens). No HuggingFace needed.
    vocab_size: int = 261
    hidden_size: int = 4096
    intermediate_size: int = 11008       # SwiGLU: ~2.7x hidden
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8          # GQA: 4:1 ratio
    head_dim: int = 128
    max_position_embeddings: int = 8192
    rope_theta: float = 500000.0          # LLaMA-3 extended RoPE
    rope_scaling: Optional[dict] = None

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5

    # Training
    tie_word_embeddings: bool = False
    use_cache: bool = True

    # Precision
    dtype: str = "bfloat16"               # Best on M4 / A100

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def small(cls) -> "ModelConfig":
        """~125M params — fast iteration on M4"""
        return cls(
            vocab_size=261,
            hidden_size=768, intermediate_size=2048,
            num_hidden_layers=12, num_attention_heads=12,
            num_key_value_heads=4, head_dim=64,
            max_position_embeddings=4096,
        )

    @classmethod
    def medium(cls) -> "ModelConfig":
        """~1B params — fits M4 Pro/Max unified memory"""
        return cls(
            vocab_size=261,
            hidden_size=2048, intermediate_size=5504,
            num_hidden_layers=24, num_attention_heads=16,
            num_key_value_heads=8, head_dim=128,
            max_position_embeddings=8192,
        )

    @classmethod
    def large(cls) -> "ModelConfig":
        """~7B params — for Codespaces / multi-GPU"""
        return cls(
            vocab_size=261,
            hidden_size=4096, intermediate_size=11008,
            num_hidden_layers=32, num_attention_heads=32,
            num_key_value_heads=8, head_dim=128,
            max_position_embeddings=8192,
        )
