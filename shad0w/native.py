"""ctypes binding for the C core (shad0w/_native/reflex.c).

Wheels ship it as the extension module shad0w._reflex. In a source checkout, `make` builds it in place.
SHAD0W_NATIVE_LIB points at any other build. Without it, shad0w uses numpy and makes identical decisions.
"""

from __future__ import annotations

import ctypes
import glob
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def find_library() -> str | None:
    env = os.environ.get("SHAD0W_NATIVE_LIB")
    if env:
        return env
    for pattern in ("_reflex*.so", "_reflex*.pyd", "_reflex*.dylib", "libreflex.*"):
        hits = sorted(glob.glob(os.path.join(_HERE, pattern)))
        if hits:
            return hits[0]
    return None


def available() -> bool:
    return find_library() is not None


class NativeReflex:
    def __init__(self, table: str, lib: str | None = None):
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
        self.m = self.lib.s0_load(os.fsencode(table))
        if not self.m:
            raise OSError(f"cannot load {table}: not a valid shad0w table")
        self.K = self.lib.s0_num_classes(self.m)
        self._p = (ctypes.c_float * self.K)()
        self._r = ctypes.c_float()

    def decide(self, text: str):
        """Not thread-safe per instance (shared output buffers); Model guards it with a lock."""
        b = text.encode("utf-8")
        y = self.lib.s0_decide(self.m, b, len(b), self._p, ctypes.byref(self._r))
        return y, np.array(self._p, dtype=np.float32), self._r.value

    def bench(self, texts: list[str], reps: int = 5):
        enc = [t.encode("utf-8") for t in texts]
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
