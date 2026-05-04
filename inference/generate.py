"""
Fast autoregressive text generation:
  - KV-cache (no recomputation of past tokens)
  - Temperature / top-k / top-p (nucleus) sampling
  - Greedy and beam search
  - Repetition penalty
  - Streaming token-by-token output
"""
import tensorflow as tf
import numpy as np
from typing import Optional, Iterator


def top_p_top_k_filter(logits: np.ndarray, top_k: int = 0,
                        top_p: float = 1.0, temperature: float = 1.0,
                        repetition_penalty: float = 1.0,
                        generated_ids: list[int] = None) -> np.ndarray:
    """Filter logits with temperature, top-k, top-p, and repetition penalty."""
    logits = logits.astype(np.float32)

    # Repetition penalty
    if repetition_penalty != 1.0 and generated_ids:
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty

    logits /= max(temperature, 1e-5)

    # Top-k
    if top_k > 0:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_idx  = np.argsort(logits)[::-1]
        sorted_prob = np.exp(logits[sorted_idx] - logits[sorted_idx].max())
        sorted_prob /= sorted_prob.sum()
        cumprob     = np.cumsum(sorted_prob)
        cutoff_idx  = np.searchsorted(cumprob, top_p) + 1
        remove_idx  = sorted_idx[cutoff_idx:]
        logits[remove_idx] = -np.inf

    return logits


class Generator:
    def __init__(self, model, tokenizer, max_new_tokens: int = 512,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
                 repetition_penalty: float = 1.1):
        self.model              = model
        self.tokenizer          = tokenizer
        self.max_new_tokens     = max_new_tokens
        self.temperature        = temperature
        self.top_k              = top_k
        self.top_p              = top_p
        self.repetition_penalty = repetition_penalty

    def generate(self, prompt: str, stream: bool = False,
                 stop_sequences: list[str] = None) -> str:
        if stream:
            return "".join(self._stream(prompt, stop_sequences))
        tokens = list(self._stream(prompt, stop_sequences))
        return "".join(tokens)

    def _stream(self, prompt: str, stop_sequences=None) -> Iterator[str]:
        input_ids  = self.tokenizer.encode(prompt)
        generated  = list(input_ids)
        past_kvs   = None
        eos_id     = self.tokenizer.eos_token_id or 2
        stop_sequences = stop_sequences or []

        # Prefill — process full prompt at once
        ids_tensor = tf.constant([input_ids], dtype=tf.int32)
        out = self.model(ids_tensor, use_cache=True, training=False)
        past_kvs = out["past_key_values"]
        last_logits = out["logits"][0, -1].numpy()

        for _ in range(self.max_new_tokens):
            filtered = top_p_top_k_filter(
                last_logits.copy(), self.top_k, self.top_p,
                self.temperature, self.repetition_penalty, generated,
            )
            probs   = np.exp(filtered - filtered.max())
            probs  /= probs.sum()
            next_id = int(np.random.choice(len(probs), p=probs))

            if next_id == eos_id:
                break
            generated.append(next_id)

            token_str = self.tokenizer.decode([next_id], skip_special_tokens=True)
            yield token_str

            # Check stop sequences
            full_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if any(full_text.endswith(s) for s in stop_sequences):
                break

            # Decode only the new token (KV-cache: input is single token)
            next_tensor = tf.constant([[next_id]], dtype=tf.int32)
            out = self.model(next_tensor, past_key_values=past_kvs,
                             use_cache=True, training=False)
            past_kvs    = out["past_key_values"]
            last_logits = out["logits"][0, -1].numpy()


def beam_search(model, tokenizer, prompt: str, num_beams: int = 4,
                max_new_tokens: int = 128) -> str:
    """Simple beam search (no KV-cache — for reference/quality eval)."""
    eos_id = tokenizer.eos_token_id or 2
    input_ids = tokenizer.encode(prompt)

    beams = [(input_ids, 0.0)]  # (token_list, log_prob)
    finished = []

    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in beams:
            if seq[-1] == eos_id:
                finished.append((seq, score))
                continue
            ids_t = tf.constant([seq], dtype=tf.int32)
            logits = model(ids_t, training=False)["logits"][0, -1].numpy()
            log_probs = logits - np.log(np.sum(np.exp(logits - logits.max()))) - logits.max()
            top_k = np.argsort(log_probs)[-num_beams:]
            for tok in top_k:
                candidates.append((seq + [int(tok)], score + log_probs[tok]))

        candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        beams = candidates[:num_beams]
        if not beams:
            break

    best = (finished + beams)[0][0]
    return tokenizer.decode(best[len(input_ids):], skip_special_tokens=True)
