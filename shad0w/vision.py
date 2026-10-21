"""Images as decisions: pixels -> item ids for the same shad0w table, in microseconds, no neural network.

An image becomes a bag of discrete items, just as a text does, so training, the certificate, the drift guard and the
C core all stay the same:

  1. a thumbnail: `size` x `size` x `channels` integer box means (at most 2 x 2 samples per cell, 4 x 4 in the
     older "ferns/1", so the cost hardly depends on the input resolution),
  2. `ferns` random ferns (Ozuysal et al., TPAMI 2010): each is `bits` comparisons between two nearby thumbnail
     pixels, and its `bits`-bit outcome is one item,
  3. a `grid` x `grid` layout of each channel's mean, quantised to `levels` steps (colour and brightness layout),
  4. (`dhash`) a 64-bit difference hash of the thumbnail as four 16-bit bands: exact and near-duplicate images share
     these items, so repeats of an image the teacher already answered are learned almost verbatim.

`csrc/reflex.c` (s0_image_items, s0_decide_image) implements exactly the same function; tests check they agree.
Pixels must already be decoded: a raw frame, a screenshot buffer, a numpy array. Decoding a JPEG or PNG costs far
more than the decision (0.3-3 ms), so decode once (or keep thumbnails) when microseconds matter.

What it is good at: images from a constrained world (product shots, UI screens, game frames, icons, documents by
layout, camera frames from a fixed view), where the table certifies most traffic. On open-world photos it certifies
little and defers the rest to your vision model, which is the correct behaviour.
"""
from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import asdict, dataclass

import numpy as np

from .features import MASK

KIND = "ferns/2"
KINDS = {"ferns/1": 1, "ferns/2": 2}
MAX_SIZE, MAX_FERNS, MAX_GRID = 64, 4096, 16  # same limits as the C core


@dataclass(frozen=True)
class ImageSpec:
    """How an image becomes items. Stored in the bundle: a table only answers for images featurised the same way."""
    size: int = 32       # thumbnail side
    channels: int = 3    # 3 = RGB, 1 = grey
    ferns: int = 256     # one item each
    bits: int = 8        # comparisons per fern
    radius: int = 2      # how far apart (in thumbnail pixels) the two pixels of a comparison may be
    grid: int = 4        # grid x grid cells of mean colour
    levels: int = 4      # quantisation steps of a cell mean
    margin: int = 2      # a comparison is true when p1 > p2 + margin (ignores sensor noise)
    seed: int = 0
    version: int = 2     # 1 = "ferns/1" (4 x 4 samples per thumbnail cell, no dHash); 2 = "ferns/2"
    dhash: bool = True   # four near-duplicate items (version 2 only)

    def __post_init__(self):
        ok = (4 <= self.size <= MAX_SIZE and self.channels in (1, 3) and 1 <= self.ferns <= MAX_FERNS
              and 1 <= self.bits <= 16 and self.radius >= 1 and 0 <= self.grid <= min(MAX_GRID, self.size)
              and 1 <= self.levels <= 256 and self.margin >= 0 and 0 <= self.seed < 2**32 and self.version in (1, 2)
              and (not self.dhash or (self.version >= 2 and self.size >= 9)))
        if not ok:
            raise ValueError(f"invalid ImageSpec {self}")

    @property
    def n_items(self) -> int:
        return self.ferns + self.grid * self.grid * self.channels + 4 * bool(self.dhash)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dhash"] = bool(d["dhash"])
        return {"kind": f"ferns/{self.version}", **d}

    @classmethod
    def from_dict(cls, d: dict) -> ImageSpec:
        kind = d.get("kind", "ferns/1")
        if kind not in KINDS:
            raise ValueError(f"image featurizer {kind!r} is not supported by this shad0w (it knows {sorted(KINDS)})")
        fields = {k: v for k, v in d.items() if k != "kind"}
        fields["version"] = KINDS[kind]
        fields.setdefault("dhash", False)  # a ferns/1 spec never had one
        return cls(**fields)


def _u32(x):
    return np.asarray(x, dtype=np.uint64) & np.uint64(0xFFFFFFFF)


def _mix(h):
    """reflex.c mix(): the slot scramble, reused to place image items in the 22-bit item space."""
    h = _u32(h)
    h ^= h >> np.uint64(16); h = (h * np.uint64(0x7FEB352D)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(15); h = (h * np.uint64(0x846CA68B)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(16)
    return h


def _pairs(spec: ImageSpec):
    """The fern comparisons as flat thumbnail offsets (a, b), each (ferns, bits). Same generator as s0_image_new."""
    n = spec.ferns * spec.bits
    st = (spec.seed * 0x9E3779B9 + 0x7F4A7C15) & 0xFFFFFFFF
    draws = np.empty(6 * n, dtype=np.uint64)
    for i in range(6 * n):  # splitmix-style, 32-bit
        st = (st + 0x9E3779B9) & 0xFFFFFFFF
        z = st
        z = ((z ^ (z >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
        z = ((z ^ (z >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
        draws[i] = z ^ (z >> 16)
    d = draws.reshape(n, 6)
    S, C, r = spec.size, spec.channels, spec.radius
    y1, x1 = (d[:, 0] % S).astype(np.int64), (d[:, 1] % S).astype(np.int64)
    dy, dx = (d[:, 2] % (2 * r + 1)).astype(np.int64) - r, (d[:, 3] % (2 * r + 1)).astype(np.int64) - r
    c1, c2 = (d[:, 4] % C).astype(np.int64), (d[:, 5] % C).astype(np.int64)
    y2, x2 = np.clip(y1 + dy, 0, S - 1), np.clip(x1 + dx, 0, S - 1)
    a = (y1 * S + x1) * C + c1
    b = (y2 * S + x2) * C + c2
    return a.reshape(spec.ferns, spec.bits), b.reshape(spec.ferns, spec.bits)


_PAIRS: dict = {}


def pixels(img) -> np.ndarray:
    """An image as a uint8 array (H, W) or (H, W, C) with C in 1, 3, 4. Accepts a numpy array, a PIL image, or encoded
    bytes / a file path / a data: URL (decoded with Pillow, which costs far more than the decision itself)."""
    if isinstance(img, np.ndarray):
        a = img
    elif hasattr(img, "mode") and hasattr(img, "size") and hasattr(img, "convert"):  # PIL.Image
        a = np.asarray(img.convert("RGB") if img.mode not in ("L", "RGB", "RGBA") else img)
    else:
        a = np.asarray(_open(img).convert("RGB"))
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating) and a.size and float(np.nanmax(a)) <= 1.0:
            a = a * 255.0
        a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (3, 4)) or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"expected an image of shape (H, W), (H, W, 3) or (H, W, 4), got {a.shape}")
    return a


def _open(src):
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError('decoding image files needs Pillow: pip install "shad0wllm[vision]" (or pass a numpy array)') from e
    if isinstance(src, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(src)))
    if isinstance(src, str) and src.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1])))
    return Image.open(src)


def thumbnail(a: np.ndarray, spec: ImageSpec) -> np.ndarray:
    """(size, size, channels) uint8 integer box means, exactly as s0_image_items computes them."""
    h, w = a.shape[:2]
    S, C = spec.size, spec.channels
    src = a if a.ndim == 3 else a[:, :, None]
    if C == 1:
        if src.shape[2] >= 3:
            s16 = src[:, :, :3].astype(np.int32)
            src = ((s16[:, :, 0] + 2 * s16[:, :, 1] + s16[:, :, 2]) >> 2)[:, :, None]
        else:
            src = src[:, :, :1].astype(np.int32)
    else:
        src = (src[:, :, :3] if src.shape[2] >= 3 else np.repeat(src[:, :, :1], 3, axis=2)).astype(np.int32)

    samp = 2 if spec.version >= 2 else 4

    def cells(n):
        out = []
        for i in range(S):
            lo, hi = i * n // S, (i + 1) * n // S
            hi = max(hi, lo + 1)
            idx = np.arange(lo, hi, max(1, (hi - lo) // samp))
            out.append(idx[:samp] if spec.version >= 2 else idx)  # ferns/2: at most 2 samples per axis
        return out

    ys, xs = cells(h), cells(w)
    if all(len(c) == 1 for c in ys) and all(len(c) == 1 for c in xs):  # one sample per cell: plain indexing
        return src[np.concatenate(ys)][:, np.concatenate(xs)].astype(np.uint8)
    th = np.empty((S, S, C), dtype=np.uint8)
    for i, yy in enumerate(ys):
        rows = src[yy]
        for j, xx in enumerate(xs):
            blk = rows[:, xx].reshape(-1, C)
            th[i, j] = blk.sum(0) // len(blk)
    return th


def image_items(img, spec: ImageSpec | None = None) -> np.ndarray:
    """The item ids of one image (uint32, spec.n_items of them). Uses the C core when it is available."""
    spec = spec or ImageSpec()
    a = pixels(img)
    nat = _native_image(spec)
    if nat is not None:
        return nat.items(a)
    return image_items_py(a, spec)


def image_items_py(a: np.ndarray, spec: ImageSpec) -> np.ndarray:
    """The numpy reference implementation of s0_image_items."""
    th = thumbnail(pixels(a), spec)
    flat = th.reshape(-1).astype(np.int32)
    if spec not in _PAIRS:
        _PAIRS[spec] = _pairs(spec)
    pa, pb = _PAIRS[spec]
    bits = flat[pa] > flat[pb] + spec.margin
    code = (bits.astype(np.uint64) << np.arange(spec.bits, dtype=np.uint64)).sum(1)
    fern = (np.arange(spec.ferns, dtype=np.uint64) << np.uint64(spec.bits)) | code
    out = [_mix(np.uint64(0xF0000000) ^ fern)]
    G, S, C, L = spec.grid, spec.size, spec.channels, spec.levels
    if G:
        q = np.empty((G, G, C), dtype=np.uint64)
        for gi in range(G):
            for gj in range(G):
                blk = th[gi * S // G:(gi + 1) * S // G, gj * S // G:(gj + 1) * S // G].reshape(-1, C).astype(np.uint64)
                q[gi, gj] = (blk.sum(0) // len(blk)) * np.uint64(L) >> np.uint64(8)
        j = np.arange(G * G * C, dtype=np.uint64)
        out.append(_mix(np.uint64(0xE0000000) ^ (j * np.uint64(L) + q.reshape(-1))))
    if spec.dhash:
        out.append(_mix(np.uint64(0xB0000000) ^ _dhash_bands(th, spec)))
    return (np.concatenate(out) & np.uint64(MASK)).astype(np.uint32)


def _dhash_bands(th: np.ndarray, spec: ImageSpec) -> np.ndarray:
    """The four 16-bit bands of the thumbnail's 64-bit difference hash, tagged by band (as s0_image_items)."""
    S = spec.size
    t = th.astype(np.uint32)
    grey = t[:, :, 0] if spec.channels == 1 else (t[:, :, 0] + 2 * t[:, :, 1] + t[:, :, 2]) >> 2
    g = np.empty((8, 9), dtype=np.uint32)
    for r in range(8):
        for c in range(9):
            blk = grey[r * S // 8:(r + 1) * S // 8, c * S // 9:(c + 1) * S // 9]
            g[r, c] = blk.sum() // blk.size
    bits = (g[:, :8] > g[:, 1:]).reshape(4, 16).astype(np.uint64)
    return (np.arange(4, dtype=np.uint64) << np.uint64(16)) | (bits << np.arange(16, dtype=np.uint64)).sum(1)


def image_digest(img) -> str:
    """sha256 of the decoded pixels and their shape: the image's identity in logs (the image itself is not logged)."""
    a = np.ascontiguousarray(pixels(img))
    h = hashlib.sha256(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def is_image(x) -> bool:
    """True for inputs this module treats as an image (a numpy array or a PIL image); text stays text."""
    return isinstance(x, np.ndarray) or (hasattr(x, "mode") and hasattr(x, "size") and hasattr(x, "convert"))


_NATIVE: dict = {}


def _native_image(spec: ImageSpec):
    if spec in _NATIVE:
        return _NATIVE[spec]
    try:
        from .native import NativeImage
        nat = NativeImage(spec)
    except OSError:
        nat = None
    _NATIVE[spec] = nat
    return nat


# ---------- a second, slower tier: a frozen image encoder, its embedding quantised to items ----------------------

EMBED_KIND = "embed/1"
_EMBEDDERS: dict = {}


def register_embedder(name: str, fn) -> None:
    """Make `fn(image) -> 1-D float vector` available as EmbedSpec(model=name), e.g. an ONNX or a hosted encoder."""
    _EMBEDDERS[name] = fn


def _clip_embedder(model_id: str):
    """A Hugging Face CLIP / SigLIP image tower on CPU (needs torch + transformers): image -> L2-normalised vector."""
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as e:
        raise ImportError(f"EmbedSpec(model={model_id!r}) needs torch and transformers (or register_embedder())") from e
    model = AutoModel.from_pretrained(model_id).eval()
    proc = AutoProcessor.from_pretrained(model_id)

    def embed(img):
        from PIL import Image
        im = img if hasattr(img, "convert") else Image.fromarray(pixels(img)).convert("RGB")
        with torch.no_grad():
            e = model.get_image_features(**proc(images=im, return_tensors="pt"))
        e = (e if torch.is_tensor(e) else e.pooler_output)[0].float().numpy()
        return e / (np.linalg.norm(e) + 1e-9)
    return embed


def embed_vector(img, model: str) -> np.ndarray:
    """The embedding of one image under `model` (a registered embedder, or a Hugging Face CLIP/SigLIP id)."""
    fn = _EMBEDDERS.get(model)
    if fn is None:
        fn = _EMBEDDERS[model] = _clip_embedder(model)
    return np.asarray(fn(img), dtype=np.float32).ravel()


@dataclass(frozen=True)
class EmbedSpec:
    """A frozen encoder's embedding as items: each dimension falls in one of `buckets` quantile buckets, and each
    (dimension, bucket) is one item. The bucket edges are fitted on the training images at train time (`fit`), so a
    trained spec carries them; on CLIP ViT-B/32 with a gemma4 teacher on CIFAR-10 this certified 59% of photos at
    3.4% disagreement, as much as a float logistic probe on the same embedding. Costs the encoder's time (CLIP B/32:
    about 25 ms on a laptop CPU), so it runs only on images the cheaper tiers pass on."""
    model: str = "openai/clip-vit-base-patch32"
    buckets: int = 4
    edges: tuple | None = None  # (dims, buckets - 1) after fit

    def __post_init__(self):
        if not 2 <= self.buckets <= 16:
            raise ValueError("EmbedSpec.buckets must be in 2..16")

    @property
    def fitted(self) -> bool:
        return self.edges is not None

    def fit(self, vectors) -> EmbedSpec:
        V = np.asarray(vectors, dtype=np.float32)
        q = np.quantile(V, np.linspace(0, 1, self.buckets + 1)[1:-1], axis=0).T  # (dims, buckets-1)
        return EmbedSpec(self.model, self.buckets, tuple(map(tuple, np.round(q, 6).tolist())))

    def items(self, vector) -> np.ndarray:
        if not self.fitted:
            raise ValueError("this EmbedSpec has no bucket edges yet: it is fitted when the table is trained")
        v = np.asarray(vector, dtype=np.float32).ravel()
        E = np.asarray(self.edges, dtype=np.float32)
        if len(v) != len(E):
            raise ValueError(f"embedding has {len(v)} dimensions, the spec was fitted on {len(E)}")
        q = (v[:, None] > E).sum(1).astype(np.uint64)
        ids = np.uint64(0xD0000000) ^ (np.arange(len(v), dtype=np.uint64) * np.uint64(self.buckets) + q)
        return (_mix(ids) & np.uint64(MASK)).astype(np.uint32)

    def to_dict(self) -> dict:
        return {"kind": EMBED_KIND, "model": self.model, "buckets": self.buckets,
                "edges": None if self.edges is None else [list(r) for r in self.edges]}

    @classmethod
    def from_dict(cls, d: dict) -> EmbedSpec:
        if d.get("kind") != EMBED_KIND:
            raise ValueError(f"not an {EMBED_KIND} spec: {d.get('kind')!r}")
        e = d.get("edges")
        return cls(d["model"], int(d["buckets"]), None if e is None else tuple(map(tuple, e)))


def spec_from_dict(d: dict):
    """An ImageSpec or EmbedSpec from its manifest dict."""
    return EmbedSpec.from_dict(d) if d.get("kind") == EMBED_KIND else ImageSpec.from_dict(d)
