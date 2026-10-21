"""ctypes binding for the C core (shad0w/_native/reflex.c).

Wheels ship it as the extension module shad0w._reflex. In a source checkout, `make` builds it in place.
SHAD0W_NATIVE_LIB points at any other build. Without it, shad0w uses numpy and makes identical decisions.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from importlib.machinery import EXTENSION_SUFFIXES

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def find_library() -> str | None:
    env = os.environ.get("SHAD0W_NATIVE_LIB")
    if env:
        return env
    # the extension built for this interpreter first: a checkout shared between machines can also hold
    # e.g. a macOS build next to the Linux one, and loading that fails
    for suffix in EXTENSION_SUFFIXES:
        p = os.path.join(_HERE, "_reflex" + suffix)
        if os.path.exists(p):
            return p
    ext = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    hits = sorted(glob.glob(os.path.join(_HERE, "libreflex.*")), key=lambda p: not p.endswith(ext))  # `make` builds
    return hits[0] if hits else None


def available() -> bool:
    return find_library() is not None


class NativeReflex:
    def __init__(self, table: str, lib: str | None = None, static: str | None = None, text_kind: int = 1):
        lib = lib or find_library()
        if lib is None:
            raise OSError("shad0w C core not found (pip wheels include it; in a checkout run `make`)")
        self.lib = ctypes.CDLL(os.path.abspath(lib))
        self.lib.s0_load.restype = ctypes.c_void_p
        self.lib.s0_load.argtypes = [ctypes.c_char_p]
        self.lib.s0_decide.restype = ctypes.c_int
        self.lib.s0_decide.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self.lib.s0_bench.restype = ctypes.c_double
        self.lib.s0_bench.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
                                      ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_double)]
        self.lib.s0_num_classes.restype = ctypes.c_uint32
        self.lib.s0_num_classes.argtypes = [ctypes.c_void_p]
        self.lib.s0_free.argtypes = [ctypes.c_void_p]
        self._has_items = hasattr(self.lib, "s0_decide_items")  # an older build of the C core has no item entry points
        if self._has_items:
            self.lib.s0_decide_items.restype = ctypes.c_int
            self.lib.s0_decide_items.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self._counts = hasattr(self.lib, "s0_decide_counts")  # newer cores also report how many items had a row
        if self._counts:
            self.lib.s0_decide_counts.restype = ctypes.c_int
            self.lib.s0_decide_counts.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                                  ctypes.POINTER(ctypes.c_uint32)]
            self.lib.s0_decide_items_counts.restype = ctypes.c_int
            self.lib.s0_decide_items_counts.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
                                                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                                        ctypes.POINTER(ctypes.c_uint32)]
        self._n = (ctypes.c_uint32 * 2)()
        self.unseen = 0.0  # share of the last input's items without a table row (set by decide / decide_items)
        self.m = self.lib.s0_load(os.fsencode(table))
        if not self.m:
            raise OSError(f"cannot load {table}: not a valid shad0w table")
        self.text_kind = int(text_kind)
        if self.text_kind != 1:  # the featurizer this table was trained with (manifest meta "text_kind")
            if not hasattr(self.lib, "s0_set_text_kind"):
                self.lib.s0_free(self.m)
                self.m = None
                raise OSError("this build of the shad0w C core has no text_kind 2: rebuild it (`make`)")
            self.lib.s0_set_text_kind.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.s0_set_text_kind(self.m, self.text_kind)
        self.sw, self.has_static = None, False
        if static is not None and hasattr(self.lib, "s0_static_load"):
            self.lib.s0_static_load.restype = ctypes.c_void_p
            self.lib.s0_static_load.argtypes = [ctypes.c_char_p]
            self.lib.s0_static_free.argtypes = [ctypes.c_void_p]
            self.lib.s0_decide_static.restype = ctypes.c_int
            self.lib.s0_decide_static.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            self.sw = self.lib.s0_static_load(os.fsencode(static))
            if not self.sw:
                raise OSError(f"cannot load {static}: not a valid shad0w static word table")
            self.has_static = True
        self.K = self.lib.s0_num_classes(self.m)
        self.hidden = 0  # width of a deep table's hidden layer (0 = linear; such a table reports radius NaN)
        if hasattr(self.lib, "s0_hidden"):
            self.lib.s0_hidden.restype = ctypes.c_uint32
            self.lib.s0_hidden.argtypes = [ctypes.c_void_p]
            self.hidden = int(self.lib.s0_hidden(self.m))
        self._p = (ctypes.c_float * self.K)()
        self._r = ctypes.c_float()

    def decide(self, text: str):
        """Not thread-safe per instance (shared output buffers); Model guards it with a lock."""
        b = text.encode("utf-8", "replace")
        if self._counts:
            y = self.lib.s0_decide_counts(self.m, self.sw, b, len(b), self._p, ctypes.byref(self._r), self._n)
            self.unseen = 1.0 - self._n[1] / self._n[0] if self._n[0] else 1.0
        elif self.has_static:
            y = self.lib.s0_decide_static(self.m, self.sw, b, len(b), self._p, ctypes.byref(self._r))
        else:
            y = self.lib.s0_decide(self.m, b, len(b), self._p, ctypes.byref(self._r))
        return y, np.array(self._p, dtype=np.float32), self._r.value

    def decide_items(self, its: np.ndarray):
        """Decide from item ids (an image's, see shad0w.vision): same outputs as decide()."""
        if not self._has_items:
            raise OSError("this build of the shad0w C core has no s0_decide_items: rebuild it (`make`)")
        a = np.ascontiguousarray(its, dtype=np.uint32)
        ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        if self._counts:
            y = self.lib.s0_decide_items_counts(self.m, ptr, len(a), self._p, ctypes.byref(self._r), self._n)
            self.unseen = 1.0 - self._n[1] / self._n[0] if self._n[0] else 1.0
        else:
            y = self.lib.s0_decide_items(self.m, ptr, len(a), self._p, ctypes.byref(self._r))
        return y, np.array(self._p, dtype=np.float32), self._r.value

    def bench(self, texts: list[str], reps: int = 5):
        enc = [t.encode("utf-8", "replace") for t in texts]
        arr = (ctypes.c_char_p * len(enc))(*enc)
        lens = (ctypes.c_int * len(enc))(*[len(e) for e in enc])
        per = (ctypes.c_double * len(enc))()
        mean = self.lib.s0_bench(self.m, arr, lens, len(enc), reps, per)
        per = np.frombuffer(per, dtype=np.float64).copy()
        return {"mean_ns": mean, "p50_ns": float(np.percentile(per, 50)),
                "p99_ns": float(np.percentile(per, 99))}

    def __del__(self):
        if getattr(self, "m", None):
            self.lib.s0_free(self.m)
            self.m = None
        if getattr(self, "sw", None):
            self.lib.s0_static_free(self.sw)
            self.sw = None


class S0ImageSpec(ctypes.Structure):
    _fields_ = [(k, ctypes.c_int32) for k in ("size", "channels", "ferns", "bits", "radius", "grid", "levels", "margin")] + \
               [("seed", ctypes.c_uint32), ("version", ctypes.c_int32), ("dhash", ctypes.c_int32)]


class NativeImage:
    """The C image featurizer (s0_image_items) for one ImageSpec. Read-only after construction: share it."""

    def __init__(self, spec, lib: str | None = None):
        lib = lib or find_library()
        if lib is None:
            raise OSError("shad0w C core not found")
        self.lib = ctypes.CDLL(os.path.abspath(lib))
        if not hasattr(self.lib, "s0_image_new"):
            raise OSError("this build of the shad0w C core has no image support: rebuild it (`make`)")
        self.lib.s0_image_new.restype = ctypes.c_void_p
        self.lib.s0_image_new.argtypes = [ctypes.POINTER(S0ImageSpec)]
        self.lib.s0_image_free.argtypes = [ctypes.c_void_p]
        self.lib.s0_image_num_items.restype = ctypes.c_int
        self.lib.s0_image_num_items.argtypes = [ctypes.c_void_p]
        self.lib.s0_image_items.restype = ctypes.c_int
        self.lib.s0_image_items.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
        self.lib.s0_decide_image.restype = ctypes.c_int
        self.lib.s0_decide_image.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float)]
        self.lib.s0_bench_image.restype = ctypes.c_double
        self.lib.s0_bench_image.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.spec = spec
        cs = S0ImageSpec(*(int(getattr(spec, k)) for k in ("size", "channels", "ferns", "bits", "radius", "grid", "levels",
                                                           "margin", "seed", "version", "dhash")))
        self.f = self.lib.s0_image_new(ctypes.byref(cs))
        if not self.f:
            raise OSError(f"the C core rejected {spec}")
        self.n = self.lib.s0_image_num_items(self.f)

    @staticmethod
    def _shape(a: np.ndarray):
        a = np.ascontiguousarray(a, dtype=np.uint8)
        h, w = a.shape[:2]
        c = 1 if a.ndim == 2 else a.shape[2]
        return a, w, h, c

    def items(self, a: np.ndarray) -> np.ndarray:
        a, w, h, c = self._shape(a)
        out = np.empty(self.n, dtype=np.uint32)
        n = self.lib.s0_image_items(self.f, a.ctypes.data, w, h, c, w * c, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)))
        if n < 0:
            raise ValueError(f"the C core rejected an image of shape {a.shape}")
        return out

    def decide(self, table: NativeReflex, a: np.ndarray):
        """Pixels in, decision out, in one C call: (option index, probabilities, radius)."""
        a, w, h, c = self._shape(a)
        y = self.lib.s0_decide_image(table.m, self.f, a.ctypes.data, w, h, c, w * c, table._p, ctypes.byref(table._r))
        if y < 0:
            raise ValueError(f"the C core rejected an image of shape {a.shape}")
        return y, np.array(table._p, dtype=np.float32), table._r.value

    def bench(self, table: NativeReflex, images: list, reps: int = 20) -> float:
        """Mean ns per s0_decide_image (pixels -> decision) over same-shape images."""
        arrs = [self._shape(a)[0] for a in images]
        _, w, h, c = self._shape(arrs[0])
        ptrs = (ctypes.c_void_p * len(arrs))(*[a.ctypes.data for a in arrs])
        return self.lib.s0_bench_image(table.m, self.f, ptrs, len(arrs), w, h, c, w * c, reps)

    def __del__(self):
        if getattr(self, "f", None):
            self.lib.s0_image_free(self.f)
            self.f = None
