"""Systems under test. Each one answers a task's test and calibration rows one text at a time and returns, per row,
the chosen option, its confidence and the wall-clock time of that single decision (batch size 1, measured around the
call from Python with time.perf_counter_ns, after a warm-up).

Learned systems fit on the task's train rows (ground-truth labels). Decision models answer zero-shot through
llama.cpp's /v1/systemone endpoint on the CPU (`--device none`, no GPU, no Metal).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP = 20


def _timed(fn, texts):
    for t in texts[:WARMUP]:
        fn(t)
    out = []
    for t in texts:
        t0 = time.perf_counter_ns()
        r = fn(t)
        out.append((r, (time.perf_counter_ns() - t0) / 1000.0))
    return out


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


# ---------------------------------------------------------------------------------------------------------- shad0w
class Shad0w:
    """shad0w's table, compiled from the train rows, served by the C core (shad0w.native)."""
    kind = "learned"

    def __init__(self, student: str):
        self.student = student
        self.name = f"shad0w-{student}"
        self.info = {"family": "shad0w", "student": student, "threads": 1}

    def fit(self, task, workdir):
        from shad0w.compiler.distill import compile_schema
        from shad0w.native import NativeReflex
        t0 = time.time()
        comp = compile_schema([t for t, _ in task.train], np.array([y for _, y in task.train]), task.options,
                              student=self.student)
        self.train_s = time.time() - t0
        path = os.path.join(workdir, f"{task.name}-{self.name}.s0")
        comp.rx.save(path)
        self.size_bytes = os.path.getsize(path)
        self.nat = NativeReflex(path)
        self.hidden = comp.rx.hidden
        self.info.update({"hidden": self.hidden, "chosen": comp.student})
        return self

    def answer(self, text):
        y, p, _ = self.nat.decide(text)
        return int(y), float(p[y]), p

    def c_core_ns(self, texts):
        """Mean and p50 nanoseconds per decision measured inside the C core (no Python call overhead)."""
        return self.nat.bench(texts, reps=5)


# ------------------------------------------------------------------------------------------- classic CPU baselines
class TfidfLR:
    """Word 1-2 grams + char 2-5 grams TF-IDF, multinomial logistic regression (scikit-learn)."""
    kind = "learned"
    name = "tfidf-logreg"

    def __init__(self):
        self.info = {"family": "classic", "threads": 1}

    def fit(self, task, workdir):
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        t0 = time.time()
        X = [t for t, _ in task.train]
        y = [y for _, y in task.train]
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200_000)
        M = hstack([self.w.fit_transform(X), self.c.fit_transform(X)]).tocsr()
        self.lr = LogisticRegression(C=10.0, max_iter=2000).fit(M, y)
        self._hstack = hstack
        self.train_s = time.time() - t0
        self.size_bytes = int(self.lr.coef_.size * 8 + (len(self.w.vocabulary_) + len(self.c.vocabulary_)) * 24)
        return self

    def answer(self, text):
        M = self._hstack([self.w.transform([text]), self.c.transform([text])]).tocsr()
        p = self.lr.predict_proba(M)[0]
        full = np.zeros(len(self.lr.classes_) if self.lr.classes_.max() < len(p) else self.lr.classes_.max() + 1)
        full[self.lr.classes_] = p
        y = int(np.argmax(full))
        return y, float(full[y]), full


class EmbedLR:
    """A sentence embedding model (CPU, one thread at serving time) + logistic regression on the embeddings."""
    kind = "learned"

    def __init__(self, model_id: str, short: str):
        self.model_id, self.name = model_id, short
        self.info = {"family": "embedding", "model": model_id, "threads": 1}

    def _load(self):
        if self.model_id.startswith("minishlab/"):  # Model2Vec static model: mean of token vectors, no transformer
            from huggingface_hub import snapshot_download
            from safetensors.numpy import load_file
            from tokenizers import Tokenizer
            p = snapshot_download(self.model_id)
            emb = next(iter(load_file(os.path.join(p, "model.safetensors")).values())).astype(np.float32)
            tok = Tokenizer.from_file(os.path.join(p, "tokenizer.json"))
            self.size_bytes_model = emb.nbytes

            def enc(texts):
                out = np.zeros((len(texts), emb.shape[1]), np.float32)
                for i, t in enumerate(texts):
                    ids = tok.encode(t, add_special_tokens=False).ids
                    if ids:
                        out[i] = emb[ids].mean(0)
                return out
            self.encode = enc
        else:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(self.model_id, device="cpu")
            self.size_bytes_model = sum(p.numel() * p.element_size() for p in m.parameters())
            self.encode = lambda texts: m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def fit(self, task, workdir):
        import torch
        from sklearn.linear_model import LogisticRegression
        self._load()
        t0 = time.time()
        torch.set_num_threads(os.cpu_count() or 4)  # training may use the whole CPU; serving is timed on one thread
        E = self.encode([t for t, _ in task.train])
        self.lr = LogisticRegression(C=10.0, max_iter=3000).fit(E, [y for _, y in task.train])
        self.train_s = time.time() - t0
        torch.set_num_threads(1)
        self.size_bytes = int(self.size_bytes_model + self.lr.coef_.size * 8)
        return self

    def answer(self, text):
        p = self.lr.predict_proba(self.encode([text]))[0]
        full = np.zeros(max(len(p), int(self.lr.classes_.max()) + 1))
        full[self.lr.classes_] = p
        y = int(np.argmax(full))
        return y, float(full[y]), full


# ------------------------------------------------------------------------------- decision models (zero-shot, CPU)
GGUF = {
    "julia-1": ("ggml-org/Julia-1-GGUF", "Julia-1-Q8_0.gguf", "Julia-1 144M (mmBERT-small encoder)"),
    "laya": ("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf", "Laya 421M (ModernBERT-large encoder)"),
    "kev-0.8b": ("ggml-org/Kev-0.8B-GGUF", "Kev-0.8B-Q8_0.gguf", "Kev 0.8B (Qwen3.5 + pointer head)"),
    "kev-4b": ("ggml-org/Kev-4B-GGUF", "Kev-4B-Q4_K_M.gguf", "Kev 4B (Qwen3.5 + pointer head), Q4_K_M"),
    "lev": ("ggml-org/lev-GGUF", "lev-Q4_K_M.gguf", "Lev, Q4_K_M"),
    "clef-flash": ("ggml-org/Clef-Flash-GGUF", "Clef-Flash-Q4_K_M.gguf", "Clef-flash 9B (Qwen + LoRA + routing head), Q4_K_M"),
}


# One neutral instruction for every decision model and task, with each option's short description as its criterion.
# Chosen before any test row was seen: on 60 calibration rows Julia-1 scored 0.25 on SMS spam with the task-specific
# question ("Is this SMS message spam?") and 0.75 with this one; on SST-2 both gave 0.77.
INSTRUCTION = "Classify the following text into exactly one of the categories."
MAX_CHARS = 4000  # longer inputs are truncated for decision models (context window); shad0w and baselines see all of it


class DecisionModel:
    """An open decision model served by llama.cpp's llama-server on the CPU only, asked through /v1/systemone with the
    task's instruction and option descriptions. Nothing is trained: this is the model as shipped."""
    kind = "zero-shot"

    def __init__(self, key: str, threads: int, port: int = 18820):
        self.key, self.threads, self.port = key, threads, port
        repo, fname, desc = GGUF[key]
        self.name = key
        self.info = {"family": "decision model", "model": desc, "gguf": f"{repo}/{fname}", "threads": threads,
                     "runtime": "llama.cpp llama-server, --device none"}
        self.repo, self.fname = repo, fname
        self.proc = None

    def start(self):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(self.repo, self.fname)
        self.size_bytes = os.path.getsize(path)
        ver = subprocess.run(["llama-server", "--version"], capture_output=True, text=True)
        self.info["llama_cpp"] = (ver.stdout + ver.stderr).strip().splitlines()[0]
        log = open(os.path.join(HERE, "results", f"llama-{self.key}-t{self.threads}.log"), "w")
        self.proc = subprocess.Popen(["llama-server", "-m", path, "--port", str(self.port), "-ngl", "0", "--device", "none",
                                      "-t", str(self.threads), "-tb", str(self.threads), "--ctx-size", "8192",
                                      "-b", "8192", "-ub", "8192",  # the default 512-token batch rejects long option lists
                                      "--parallel", "1"], stdout=log, stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).read():
                    return self
            except Exception:
                time.sleep(0.5)
        raise RuntimeError(f"llama-server did not start for {self.key}")

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=30)
            self.proc = None

    def fit(self, task, workdir):
        self.task, self.train_s = task, 0.0
        return self

    def answer(self, text):
        t = self.task
        body = {"model": self.key, "state": text[:MAX_CHARS],
                "questions": {"q": {"type": "choice", "instructions": INSTRUCTION, "criteria": t.descriptions}}}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/systemone", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        try:
            a = json.loads(urllib.request.urlopen(req, timeout=600).read())["answers"]["q"]
        except Exception as e:  # counted as a wrong answer with zero confidence, never dropped
            self.errors = getattr(self, "errors", 0) + 1
            self.info["errors"] = self.errors
            self.info["last_error"] = type(e).__name__ + ": " + str(e)[:120]
            return -1, 0.0, np.zeros(len(t.options))
        probs = a.get("probabilities") or {}
        p = np.array([float(probs.get(o, 0.0)) for o in t.options])
        choice = a.get("choice")
        y = t.options.index(choice) if choice in t.options else int(np.argmax(p))
        return y, float(p[y]) if p.sum() > 0 else float(a.get("confidence", 0.0)), p


# ------------------------------------------------------- decision models that ship as Python packages (zero-shot, CPU)
PY_DM = {
    "decider-0.8b": ("decider", "Mapika/decider-0.8b", "Decider 0.8B (Qwen3.5-0.8B-Base + decision head)"),
    "gliner2.5-decide": ("gliner2", "fastino/GLiNER2.5-Decide", "GLiNER2.5-Decide (encoder, 0.3B class)"),
}


class PyDecisionModel(DecisionModel):
    """A decision model whose own Python package runs it, served by bench/py_server.py (same /v1/systemone shape, CPU
    only). Its Python comes from $SHAD0W_BENCH_DM_PYTHON (decider needs Python >= 3.11)."""

    def __init__(self, key: str, threads: int, port: int = 18822):
        self.key, self.threads, self.port = key, threads, port
        backend, model, desc = PY_DM[key]
        self.backend, self.model = backend, model
        self.name = key
        self.info = {"family": "decision model", "model": desc, "hf": model, "threads": threads,
                     "runtime": f"{backend} (Python package), CPU"}
        self.proc = None

    def start(self):
        py = os.environ.get("SHAD0W_BENCH_DM_PYTHON", "python")
        log = open(os.path.join(HERE, "results", f"py-{self.key}-t{self.threads}.log"), "w")
        self.proc = subprocess.Popen([py, os.path.join(HERE, "py_server.py"), "--backend", self.backend, "--model", self.model,
                                      "--port", str(self.port), "--threads", str(self.threads)], stdout=log, stderr=subprocess.STDOUT)
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(self.model)
            self.size_bytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs
                                  if f.endswith((".safetensors", ".bin")))
        except Exception:
            self.size_bytes = 0
        for _ in range(1200):
            if self.proc.poll() is not None:
                raise RuntimeError(f"py_server exited for {self.key}; see results/py-{self.key}-t{self.threads}.log")
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).read():
                    return self
            except Exception:
                time.sleep(0.5)
        raise RuntimeError(f"py_server did not start for {self.key}")
