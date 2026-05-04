"""
Byte-level tokenizer — fully self-contained, zero downloads.

Maps every UTF-8 byte to a token ID (0–255).
Special tokens live at 256+.
vocab_size = 261

This is the same foundation GPT-2 uses. It handles every language,
emoji, and code character without any vocabulary training.
"""

SPECIAL = {
    "<pad>": 256,
    "<bos>": 257,
    "<eos>": 258,
    "<unk>": 259,
    "<sep>": 260,
}

VOCAB_SIZE = 261  # 256 bytes + 5 special tokens

_ID_TO_SPECIAL = {v: k for k, v in SPECIAL.items()}


class ByteTokenizer:
    """
    Drop-in replacement for a HuggingFace tokenizer.
    All text is encoded as raw UTF-8 bytes, so vocab_size is always 261.
    """

    vocab_size      = VOCAB_SIZE
    bos_token_id    = SPECIAL["<bos>"]
    eos_token_id    = SPECIAL["<eos>"]
    pad_token_id    = SPECIAL["<pad>"]
    unk_token_id    = SPECIAL["<unk>"]
    sep_token_id    = SPECIAL["<sep>"]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        byte_vals = []
        for i in ids:
            if i in _ID_TO_SPECIAL:
                if not skip_special_tokens:
                    byte_vals.extend(_ID_TO_SPECIAL[i].encode("utf-8"))
            elif 0 <= i <= 255:
                byte_vals.append(i)
        return bytes(byte_vals).decode("utf-8", errors="replace")

    def __call__(self, text: str, **kwargs):
        return {"input_ids": self.encode(text, add_special_tokens=True)}

    def save(self, path: str):
        """Nothing to save — vocab is implicit in byte values."""
        import json, os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump({"type": "ByteTokenizer", "vocab_size": VOCAB_SIZE}, f)

    @classmethod
    def load(cls, path: str) -> "ByteTokenizer":
        return cls()
