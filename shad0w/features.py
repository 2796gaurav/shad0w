"""Tokenizer-free feature items for the Reflex tier.

Text -> a bag of 32-bit item ids, no tokenizer model:
  * ASCII-lowercase the UTF-8 bytes,
  * words = maximal runs of [a-z0-9] or bytes >= 0x80,
  * items = word unigrams ('w'), word bigrams ('b'), char trigrams of ' '+word+' ' ('c').
Each item is hashed with FNV-1a over (type byte + payload) and reduced mod 2^BITS.
`csrc/reflex.c` implements exactly the same function; tests check they agree.
"""

from __future__ import annotations

import numpy as np

BITS = 22
MASK = (1 << BITS) - 1
FNV_OFF, FNV_PRIME = 0x811C9DC5, 0x01000193
MAX_WORD = 256  # bytes; longer words are truncated (same as csrc/reflex.c)


def fnv1a(data: bytes) -> int:
    h = FNV_OFF
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h


def _words(text: str) -> list[bytes]:
    b = text.encode("utf-8", "replace").lower()  # bytes.lower() is ASCII-only, matching C; lone surrogates become "?"
    words, cur = [], bytearray()
    for ch in b:
        if (48 <= ch <= 57) or (97 <= ch <= 122) or ch >= 0x80:
            if len(cur) < MAX_WORD:
                cur.append(ch)
        elif cur:
            words.append(bytes(cur))
            cur = bytearray()
    if cur:
        words.append(bytes(cur))
    return words


def items(text: str, bigrams: bool = True) -> np.ndarray:
    ws = _words(text)
    out = []
    for w in ws:
        out.append(fnv1a(b"w" + w) & MASK)
        p = b" " + w + b" "
        for i in range(len(p) - 2):
            out.append(fnv1a(b"c" + p[i:i + 3]) & MASK)
    if bigrams:
        for a, c in zip(ws, ws[1:]):
            out.append(fnv1a(b"b" + a + b"\x01" + c) & MASK)
    if not out:
        out.append(fnv1a(b"e") & MASK)  # empty-input item
    return np.asarray(out, dtype=np.uint32)


def items_batch(texts: list[str]) -> list[np.ndarray]:
    return [items(t) for t in texts]
