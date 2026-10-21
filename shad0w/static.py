"""Static word vectors as table items: meaning the hashed n-grams cannot see, still tokenizer-free at serve time.

At training time every word of the training texts gets a vector from a static embedding model (potion-base-8M,
Model2Vec, MIT licence: the mean of its subword vectors), stored as int8 in a word table keyed by the word's FNV-1a
hash. At serve time a text's known words are looked up by that hash (the same hash as its word item), their int8
vectors are summed in int32, and each dimension of the mean falls into one of `buckets` quantile buckets: one item
per (dimension, bucket), added to the n-gram items of the same table. `csrc/reflex.c` (s0_decide_static) computes the
same items; the bucket test is done on integer sums against float64 edges, so C and numpy agree exactly.

Measured with the labels as the teacher (alpha 5%, 4,000 training rows): SST-2 sentiment certified 0% -> 18% of
traffic, BANKING77 70% -> 79%. Building the word table needs `pip install "shad0wllm[static]"`; serving needs nothing.

Since the deep student, new bundles no longer carry a word table: `word_vectors()` gives the vectors to training,
which folds their learned weights into the word rows of the table itself (an exact identity, see
compiler.distill._fold), so serving is one lookup per item and words the training texts never contained still carry
their meaning. Bundles trained with the bucket items (a `.s0w` file next to the table) keep loading and serving.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

from .features import MASK, _words, fnv1a

MODELS = {"potion-8M": "minishlab/potion-base-8M", "potion-32M": "minishlab/potion-base-32M"}
MAX_WORDS = 200_000


def _mix(h):
    from .vision import _mix as m
    return m(h)


@dataclass
class StaticTable:
    hashes: np.ndarray   # (W,) uint32, strictly increasing
    V: np.ndarray        # (W, D) int8
    thr: np.ndarray      # (D, NB - 1) float64: bucket edges in int8 units (compared with sums * n)
    buckets: int = 4
    model: str = "potion-8M"

    @property
    def dims(self) -> int:
        return self.V.shape[1]

    def sums(self, text: str):
        """(int32 sum of the known words' vectors, number of known words)."""
        h = np.array([fnv1a(b"w" + w) for w in _words(text)], dtype=np.uint32)
        if not len(h) or not len(self.hashes):
            return np.zeros(self.dims, np.int64), 0
        pos = np.minimum(np.searchsorted(self.hashes, h), len(self.hashes) - 1)
        rows = pos[self.hashes[pos] == h]
        return self.V[rows].astype(np.int64).sum(0), len(rows)

    def items(self, text: str) -> np.ndarray:
        s, n = self.sums(text)
        q = (s[:, None].astype(np.float64) > self.thr * float(max(n, 1))).sum(1).astype(np.uint64)
        ids = np.uint64(0xD0000000) ^ (np.arange(self.dims, dtype=np.uint64) * np.uint64(self.buckets) + q)
        return (_mix(ids) & np.uint64(MASK)).astype(np.uint32)

    def save(self, path: str) -> None:
        D, W = self.dims, len(self.hashes)
        with open(path, "wb") as f:
            f.write(b"S0SW")
            f.write(struct.pack("<IIII", 1, D, self.buckets, W))
            f.write(np.ascontiguousarray(self.thr, dtype="<f8").tobytes())
            f.write(np.ascontiguousarray(self.hashes, dtype="<u4").tobytes())
            f.write(np.ascontiguousarray(self.V, dtype=np.int8).tobytes())

    @classmethod
    def load(cls, path: str, model: str = "potion-8M") -> StaticTable:
        with open(path, "rb") as f:
            raw = f.read()
        if raw[:4] != b"S0SW":
            raise ValueError(f"{path}: not a shad0w static word table")
        ver, D, NB, W = struct.unpack_from("<IIII", raw, 4)
        if ver != 1 or not (1 <= D <= 4096 and 2 <= NB <= 16):
            raise ValueError(f"{path}: unsupported static table (version {ver}, {D} dims, {NB} buckets)")
        o = 20
        thr = np.frombuffer(raw, "<f8", D * (NB - 1), o).reshape(D, NB - 1).astype(np.float64); o += 8 * D * (NB - 1)
        hashes = np.frombuffer(raw, "<u4", W, o).astype(np.uint32); o += 4 * W
        V = np.frombuffer(raw, np.int8, W * D, o).reshape(W, D).copy(); o += W * D
        if o != len(raw):
            raise ValueError(f"{path}: wrong size for a static table")
        return cls(hashes, V, thr, NB, model)


def _load_model(model: str):
    """(subword embedding matrix, tokenizer) of a Model2Vec static model, from the Hugging Face cache or hub."""
    try:
        from huggingface_hub import snapshot_download
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer
    except ImportError as e:
        raise ImportError('static word vectors need: pip install "shad0wllm[static]" (tokenizers, safetensors, '
                          'huggingface_hub); serving does not') from e
    path = snapshot_download(MODELS.get(model, model), allow_patterns=["*.safetensors", "tokenizer.json", "*.json"])
    weights = load_file(os.path.join(path, "model.safetensors"))
    emb = next(iter(weights.values())).astype(np.float32)
    return emb, Tokenizer.from_file(os.path.join(path, "tokenizer.json"))


def word_vectors(texts, model: str = "potion-8M", max_words: int = MAX_WORDS, extra_words: int = 30_000,
                 kind: int | None = None) -> dict:
    """{word item id: float32 vector} for the words of `texts` plus up to `extra_words` whole-word tokens of the
    static model's own vocabulary (so words never seen in training still get a row when the vectors are folded into
    the table, see compiler.distill.compile_schema(dense=...)). Keys are the 22-bit ids of the 'w' items, so a text's
    word items find their vectors directly; a 22-bit collision between two words averages their vectors."""
    from .features import _kind
    kind = _kind.get() if kind is None else kind  # the featurizer being trained (shadow_compile sets it)
    emb, tok = _load_model(model)
    counts: dict[bytes, int] = {}
    for t in texts:
        for w in _words(t, kind):
            counts[w] = counts.get(w, 0) + 1
    words = sorted(counts, key=lambda w: -counts[w])[:max_words]
    seen = set(words)
    if extra_words:
        vocab = tok.get_vocab()
        whole = [w for w in vocab if w.isascii() and w.isalnum() and w.islower() and len(w) > 1]
        for w in sorted(whole, key=vocab.get)[:extra_words]:  # lower ids = more frequent in the model's training
            b = w.encode()
            if b not in seen:
                words.append(b)
                seen.add(b)
    out: dict[int, np.ndarray] = {}
    n: dict[int, int] = {}
    for w in words:
        ids = tok.encode(w.decode("utf-8", "replace"), add_special_tokens=False).ids
        if not ids:
            continue
        h = fnv1a(b"w" + w) & MASK
        v = emb[ids].mean(0)
        if h in out:
            out[h] = (out[h] * n[h] + v) / (n[h] + 1)
            n[h] += 1
        else:
            out[h], n[h] = v.astype(np.float32), 1
    if not out:
        raise ValueError("no word of the training texts has a static vector")
    return out


def build(texts, model: str = "potion-8M", buckets: int = 4, max_words: int = MAX_WORDS) -> StaticTable:
    """The word table for the words of `texts` (the training texts) and the bucket edges fitted on them."""
    emb, tok = _load_model(model)
    counts: dict[bytes, int] = {}
    for t in texts:
        for w in _words(t):
            counts[w] = counts.get(w, 0) + 1
    words = sorted(counts, key=lambda w: -counts[w])[:max_words]
    vecs, hs = [], []
    for w in words:
        ids = tok.encode(w.decode("utf-8", "replace"), add_special_tokens=False).ids
        if ids:
            vecs.append(emb[ids].mean(0))
            hs.append(fnv1a(b"w" + w))
    if not vecs:
        raise ValueError("no word of the training texts has a static vector")
    V = np.stack(vecs).astype(np.float32)
    scale = np.abs(V).max(0) / 127.0 + 1e-12
    V8 = np.clip(np.round(V / scale), -127, 127).astype(np.int8)
    order = np.argsort(np.array(hs, dtype=np.uint32), kind="stable")
    hashes = np.array(hs, dtype=np.uint32)[order]
    keep = np.concatenate([[True], hashes[1:] != hashes[:-1]])  # a 32-bit collision keeps the more frequent word
    table = StaticTable(hashes[keep], V8[order][keep], np.zeros((V8.shape[1], buckets - 1)), buckets, model)
    means = []
    for t in texts:
        s, n = table.sums(t)
        means.append(s / max(n, 1))
    table.thr = np.quantile(np.stack(means), np.linspace(0, 1, buckets + 1)[1:-1], axis=0).T.astype(np.float64)
    return table
