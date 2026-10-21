"""Tokenizer-free feature items for the Reflex tier.

Text -> a bag of 32-bit item ids, no tokenizer model. Two featurizers ("text kinds") exist; a bundle records its kind
in manifest meta "text_kind" (absent = 1), and the C core (s0_set_text_kind) and the JS runtime use the same one.

kind 1 (every bundle made before text_kind existed):
  * ASCII-lowercase the UTF-8 bytes,
  * words = maximal runs of [a-z0-9] or bytes >= 0x80,
  * items = word unigrams ('w'), word bigrams ('b'), char trigrams of ' '+word+' ' ('c').

kind 2 (script-aware; the default for new bundles, TEXT_KIND): identical to kind 1 on pure ASCII. Inside runs of
bytes >= 0x80 the UTF-8 is decoded (a lead byte C2..F4 followed by its continuation bytes 80..BF; anything else is one
word byte, as in kind 1) and:
  * a code point of a script written without spaces (Han, Hiragana/Katakana; CHAR_SCRIPTS)
    ends the current word and is a word of its own: a 'w' item, bigrams with its neighbours (so adjacent characters
    give character bigrams), and no 'c' trigrams (they would only be the bytes of one glyph);
  * uppercase letters of Latin-1 Supplement (U+00C0-U+00DE except U+00D7), Greek (U+0391-U+03A9 except U+03A2) and
    Cyrillic (U+0400-U+042F) are folded to lowercase before hashing. Only these ranges: other scripts' case (Latin
    Extended, Armenian, Georgian, ...) is left as is, so the C and JS runtimes need no Unicode tables;
  * everything else (Arabic, Hebrew, Devanagari, Hangul, emoji, ...) stays run-based exactly as in kind 1.
Each item is hashed with FNV-1a over (type byte + payload) and reduced mod 2^BITS.
`shad0w/_native/reflex.c` and `js/index.js` implement exactly the same functions; tests check they agree.
"""

from __future__ import annotations

import contextlib
import contextvars

import numpy as np

BITS = 22
MASK = (1 << BITS) - 1
FNV_OFF, FNV_PRIME = 0x811C9DC5, 0x01000193
MAX_WORD = 256  # bytes; longer words are truncated (same as the C core)
TEXT_KINDS = (1, 2)
TEXT_KIND = 2  # the featurizer new bundles are compiled with (shadow_compile); bundles without "text_kind" are kind 1

# code point ranges whose characters are each a word of their own under kind 2 (scripts written without spaces)
# Thai, Lao, Khmer and Myanmar are also written without spaces, but their letters are alphabetic: one word per
# letter lost information (Wongnai Thai reviews: agreement 51.1% -> 49.4%), so they keep kind 1's runs (byte trigrams).
CHAR_SCRIPTS = ((0x2E80, 0x2FDF),   # CJK radicals, Kangxi radicals
                (0x3040, 0x30FF),   # Hiragana, Katakana
                (0x31F0, 0x31FF),   # Katakana phonetic extensions
                (0x3400, 0x4DBF),   # CJK extension A
                (0x4E00, 0x9FFF),   # CJK unified ideographs
                (0xF900, 0xFAFF),   # CJK compatibility ideographs
                (0x20000, 0x2FA1F))  # CJK extensions B-F, compatibility supplement

_kind = contextvars.ContextVar("shad0w_text_kind", default=1)


@contextlib.contextmanager
def using_kind(kind: int):
    """Make `kind` the default of items() / as_items() / items_batch() inside the block (this thread / task only):
    lets code that hashes texts internally (compile_schema) train on another featurizer."""
    tok = _kind.set(check_kind(kind))
    try:
        yield
    finally:
        _kind.reset(tok)


def check_kind(kind) -> int:
    k = int(kind)
    if k not in TEXT_KINDS:
        raise ValueError(f"text_kind {kind!r} is not supported (this shad0w knows {TEXT_KINDS})")
    return k


def fnv1a(data: bytes) -> int:
    h = FNV_OFF
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h


def _bytes(text) -> bytes:
    if isinstance(text, (bytes, bytearray, memoryview)):
        return bytes(text).lower()
    return text.encode("utf-8", "replace").lower()  # bytes.lower() is ASCII-only, matching C; lone surrogates -> "?"


def is_char_script(cp: int) -> bool:
    for lo, hi in CHAR_SCRIPTS:
        if cp < lo:
            return False
        if cp <= hi:
            return True
    return False


def fold(cp: int) -> int:
    """Kind 2 lowercase for the 2-byte scripts (Latin-1 Supplement, Greek, Cyrillic); other code points unchanged."""
    if 0xC0 <= cp <= 0xDE and cp != 0xD7:
        return cp + 0x20
    if 0x391 <= cp <= 0x3A9 and cp != 0x3A2:
        return cp + 0x20
    if 0x410 <= cp <= 0x42F:
        return cp + 0x20
    if 0x400 <= cp <= 0x40F:
        return cp + 0x50
    return cp


def utf8_at(b: bytes, i: int):
    """(code point, length) of the UTF-8 sequence at b[i] (b[i] >= 0x80), or (-1, 1) when it is not a valid one:
    a lead byte C2..F4 followed by its 1-3 continuation bytes 80..BF (the C core and JS apply the same test)."""
    c = b[i]
    if 0xC2 <= c <= 0xDF:
        L, cp = 2, c & 0x1F
    elif 0xE0 <= c <= 0xEF:
        L, cp = 3, c & 0x0F
    elif 0xF0 <= c <= 0xF4:
        L, cp = 4, c & 0x07
    else:
        return -1, 1
    if i + L > len(b):
        return -1, 1
    for k in range(1, L):
        d = b[i + k]
        if d & 0xC0 != 0x80:
            return -1, 1
        cp = (cp << 6) | (d & 0x3F)
    return cp, L


def _words1(b: bytes) -> list[bytes]:
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


def _words2(b: bytes) -> list[tuple[bytes, bool]]:
    """Kind 2 words as (bytes, single): single = one character of a CHAR_SCRIPTS script (no trigram items)."""
    words, cur = [], bytearray()
    i, n = 0, len(b)

    def push(ch):
        if len(cur) < MAX_WORD:
            cur.append(ch)

    while i < n:
        ch = b[i]
        if ch < 0x80:
            if (48 <= ch <= 57) or (97 <= ch <= 122):
                push(ch)
            elif cur:
                words.append((bytes(cur), False))
                cur.clear()
            i += 1
            continue
        cp, L = utf8_at(b, i)
        if cp < 0:
            push(ch)
        elif is_char_script(cp):
            if cur:
                words.append((bytes(cur), False))
                cur.clear()
            words.append((b[i:i + L], True))
        else:
            f = fold(cp)
            for x in (chr(f).encode("utf-8") if f != cp else b[i:i + L]):
                push(x)
        i += L
    if cur:
        words.append((bytes(cur), False))
    return words


def _words(text, kind: int = 1) -> list[bytes]:
    """The words of `text` (str or UTF-8 bytes) under featurizer `kind`: the payloads of its 'w' items."""
    b = _bytes(text)
    if check_kind(kind) == 1:
        return _words1(b)
    return [w for w, _ in _words2(b)]


def items(text, bigrams: bool = True, kind: int | None = None) -> np.ndarray:
    """Item ids of one text (str, or raw UTF-8 bytes). kind: the featurizer (1 or 2); None = the current default, 1
    unless inside using_kind()."""
    kind = _kind.get() if kind is None else check_kind(kind)
    b = _bytes(text)
    ws = [(w, False) for w in _words1(b)] if kind == 1 else _words2(b)
    out = []
    for w, single in ws:
        out.append(fnv1a(b"w" + w) & MASK)
        if single:
            continue
        p = b" " + w + b" "
        for i in range(len(p) - 2):
            out.append(fnv1a(b"c" + p[i:i + 3]) & MASK)
    if bigrams:
        for (a, _), (c, _) in zip(ws, ws[1:]):
            out.append(fnv1a(b"b" + a + b"\x01" + c) & MASK)
    if not out:
        out.append(fnv1a(b"e") & MASK)  # empty-input item
    return np.asarray(out, dtype=np.uint32)


def as_items(x, kind: int | None = None) -> np.ndarray:
    """Item ids for one input: a text is hashed; item ids already computed (an image's, see shad0w.vision) pass through."""
    return items(x, kind=kind) if isinstance(x, str) else np.asarray(x, dtype=np.uint32)


def items_batch(texts: list, kind: int | None = None) -> list[np.ndarray]:
    return [as_items(t, kind) for t in texts]
