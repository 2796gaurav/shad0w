"""Frozen sentence encoders used ONLY at compile time for the label-free (logs) mode. Never used at runtime.

Default: BAAI/bge-small-en-v1.5 (MIT licence, 33M). Stronger, under its own licence terms:
google/embeddinggemma-300m (Gemma terms; gated on the Hugging Face Hub). Any sentence-transformers model works.
"""
from __future__ import annotations

import numpy as np

ENCODERS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-small-ml": "intfloat/multilingual-e5-small",
    "e5-base-ml": "intfloat/multilingual-e5-base",
    "gemma-300m": "google/embeddinggemma-300m",
}
DEFAULT = "bge-small"


def _device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Encoder:
    """encode(texts, side) -> L2-normalised float32 (N, D). side is "state" (texts) or "option" (option text)."""

    def __init__(self, key: str = DEFAULT):
        from sentence_transformers import SentenceTransformer
        self.key = key
        self.name = ENCODERS.get(key, key)
        self.m = SentenceTransformer(self.name, device=_device())
        self.m.max_seq_length = 128

    def _prefix(self, side):
        n = self.name.lower()
        if "e5" in n:
            return "query: "
        if "embeddinggemma" in n:
            return "task: classification | query: "
        return ""

    def encode(self, texts, side="state", bs=128):
        e = self.m.encode([self._prefix(side) + t for t in texts], batch_size=bs, convert_to_numpy=True,
                          normalize_embeddings=False, show_progress_bar=False).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
