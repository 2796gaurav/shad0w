"""Settings for shad0w, resolved once and from one place.

Precedence (highest first):
    1. code: keyword arguments you pass to `Shadow(...)`, `decision(...)`, `Gateway(...)` or a CLI flag
    2. environment: SHAD0W_ALPHA, SHAD0W_MODE, SHAD0W_NEVER_SERVE="fraud,self_harm", ...
    3. a config file: ./shad0w.toml (or the path in SHAD0W_CONFIG), with optional per-question sections:
           alpha = 0.05
           mode = "shadow"
           [questions.intent]
           alpha = 0.02
    4. the defaults below

`shad0w config --explain` prints every effective value and where it came from. Standard library only.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass, fields
from typing import Any

__all__ = ["DEFAULTS", "Settings", "resolve", "explain", "find_config", "load_toml", "HELP"]

DEFAULTS: dict[str, Any] = {
    "alpha": 0.05,              # certified disagreement bound used by train()
    "delta": 0.1,               # certificate failure probability (0.1 = 90% confidence)
    "min_rows": 1000,           # logged answers needed before train() will run
    "cal_fraction": 0.3,        # share of the log held out to certify (never trained on)
    "max_cal": 3000,            # cap on the calibration slice
    "audit_rate": 0.01,         # share of decisions re-asked to your LLM as spot checks (uniform over all traffic)
    "auto_train": 0,            # retrain in the background every N new logged answers (0 = off)
    "retrain": "gated",         # "gated": keep the old bundle unless the new one is at least as good; "always"
    "mode": "serve",            # "serve" | "shadow" (compute, but always return your LLM's answer) | "off"
    "canary": 1.0,              # share of certified answers the table may actually serve (gradual rollout)
    "never_serve": [],          # labels that always go to your LLM
    "min_confidence": None,     # manual confidence floor; can only make serving stricter than the certificate
    "force_threshold": None,    # serve above this confidence even when uncertified (voids the certificate; loud)
    "drift_window": 500,        # decisions in the drift guard's sliding window
    "drift_margin": 0.03,       # estimated error rise (absolute) that raises the drift flag
    "cost_per_call": None,      # what one LLM call costs you (dashboard: money saved)
    "llm_latency_ms": None,     # assumed LLM latency until measured (dashboard: time saved)
    "folder": "shad0w",         # where logs and bundles live: <folder>/<question>/{log.jsonl, bundle/}
    "exposed": False,           # treat inputs as adversarial (also defer low-radius answers)
    "trace": None,              # JSON Lines file receiving every decision
    "capture": ["header", "model", "tools", "decisions"],  # proxy: how a request is recognised as a decision
    "timeout": 120.0,           # proxy: upstream request timeout in seconds
    "on_new_option": "defer",   # your option list gained options the table never learned: "defer" everything to
                                # your LLM until retrained (safe), or "serve" the options the table does know
    "max_mb": None,             # size budget per question's table in MB; keeps the most informative features (None = all)
    "rename": {},               # {"old_label": "new_label"}: rename options without retraining (logs follow at train)
}

HELP = {
    "alpha": "Max share of table answers that may differ from your LLM. Lower = safer, fewer calls saved.",
    "delta": "Chance the certificate itself is wrong because of an unlucky sample. 0.1 = 90% confidence.",
    "min_rows": "Logged LLM answers needed before training. More rows → larger certified share.",
    "cal_fraction": "Share of logged rows held back to certify, never trained on.",
    "max_cal": "Upper limit on the calibration slice.",
    "audit_rate": "Share of decisions re-asked to your LLM to measure live agreement (and to calibrate honestly).",
    "auto_train": "Retrain in the background every N new logged answers. 0 turns it off.",
    "retrain": "'gated' keeps the old table unless the new one certifies at least 80% of its share; 'always' swaps.",
    "mode": "'serve' answers from the table; 'shadow' computes but always returns your LLM's answer; 'off' skips the table.",
    "canary": "Share of eligible traffic the table may answer (1.0 = all). Use 0.1 for a gradual rollout.",
    "never_serve": "Labels always sent to your LLM, for example fraud or self_harm.",
    "min_confidence": "Manual floor on confidence. Can only make serving stricter than the certificate.",
    "force_threshold": "Serve above this confidence even below the certified threshold. Those answers are NOT certified.",
    "drift_window": "Decisions in the drift guard's sliding window.",
    "drift_margin": "Rise in estimated error (absolute) that raises the drift flag.",
    "cost_per_call": "Cost of one LLM call, for the dashboard's money-saved figure.",
    "llm_latency_ms": "Assumed LLM latency until measured, for the dashboard's time-saved figure.",
    "folder": "Where logs and bundles live.",
    "exposed": "Treat inputs as adversarial: also defer answers a few edits could flip.",
    "trace": "JSON Lines file that receives every decision.",
    "capture": "Proxy: which request shapes count as decisions (header, model, tools, decisions, json_schema).",
    "timeout": "Proxy: upstream request timeout in seconds.",
    "on_new_option": "Options added after training: 'defer' asks your LLM until you retrain; 'serve' keeps serving known ones.",
    "max_mb": "Size budget for each table in MB. Many options or a huge vocabulary? Cap it; the certificate is computed on the capped table.",
    "rename": "Rename labels without retraining, e.g. {lost_card = \"card_lost\"}. Applied to the table and, at the next train, to the log.",
}

ENV_PREFIX = "SHAD0W_"
CONFIG_ENV = "SHAD0W_CONFIG"
CONFIG_FILE = "shad0w.toml"
MODES = ("serve", "shadow", "off")
RETRAIN = ("gated", "always")
ON_NEW_OPTION = ("defer", "serve")
CAPTURE = ("header", "model", "tools", "json_schema", "decisions")


@dataclass(frozen=True)
class Settings:
    alpha: float = 0.05
    delta: float = 0.1
    min_rows: int = 1000
    cal_fraction: float = 0.3
    max_cal: int = 3000
    audit_rate: float = 0.01
    auto_train: int = 0
    retrain: str = "gated"
    mode: str = "serve"
    canary: float = 1.0
    never_serve: tuple = ()
    min_confidence: float | None = None
    force_threshold: float | None = None
    drift_window: int = 500
    drift_margin: float = 0.03
    cost_per_call: float | None = None
    llm_latency_ms: float | None = None
    folder: str = "shad0w"
    exposed: bool = False
    trace: str | None = None
    capture: tuple = ("header", "model", "tools", "decisions")
    timeout: float = 120.0
    on_new_option: str = "defer"
    max_mb: float | None = None
    rename: tuple = ()  # ((old, new), ...): hashable form of the rename map

    @property
    def rename_map(self) -> dict[str, str]:
        return dict(self.rename)

    def asdict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def find_config(path: str | None = None) -> str | None:
    """The config file to read: an explicit path, $SHAD0W_CONFIG, or ./shad0w.toml; None when none exists."""
    p = path or os.environ.get(CONFIG_ENV) or CONFIG_FILE
    if path or os.environ.get(CONFIG_ENV):
        if not os.path.exists(p):
            raise FileNotFoundError(f"config file {p} not found")
        return p
    return p if os.path.exists(p) else None


@functools.lru_cache(maxsize=16)
def _load_toml_cached(path: str, mtime: float) -> dict:
    try:
        import tomllib  # 3.11+
    except ModuleNotFoundError:  # pragma: no cover - 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            raise ValueError(f"reading {path} needs Python 3.11+ or `pip install tomli`") from None
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_toml(path: str) -> dict:
    return _load_toml_cached(os.path.abspath(path), os.path.getmtime(path))


def _coerce(key: str, raw: str) -> Any:
    """Turn an environment string into the type of the default."""
    d = DEFAULTS[key]
    s = raw.strip()
    if s.lower() in ("", "none", "null"):
        return None
    if isinstance(d, bool):
        return s.lower() in ("1", "true", "yes", "on")
    if isinstance(d, int) and not isinstance(d, bool):
        if s.lower() in ("off", "false", "no"):
            return 0
        return int(float(s))
    if isinstance(d, float) or key in ("min_confidence", "force_threshold", "cost_per_call", "llm_latency_ms", "max_mb"):
        return float(s)
    if isinstance(d, list):
        return [x.strip() for x in s.split(",") if x.strip()]
    if isinstance(d, dict):  # "old=new,old2=new2"
        return dict(p.split("=", 1) for p in (x.strip() for x in s.split(",")) if "=" in p)
    return s


def _norm(key: str, v: Any) -> Any:
    if key in ("never_serve", "capture"):
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        return tuple(str(x) for x in (v or ()))
    if key == "rename":
        if isinstance(v, str):
            v = dict(p.split("=", 1) for p in (x.strip() for x in v.split(",")) if "=" in p)
        if isinstance(v, dict):
            v = v.items()
        return tuple(sorted((str(a).strip(), str(b).strip()) for a, b in (v or ())))
    if key == "auto_train" and (v is None or v is False):
        return 0
    if key == "trace" and v is not None and not isinstance(v, str):
        return v  # a TraceWriter passed in code
    return v


def _validate(s: dict) -> None:
    def between(k, lo, hi, allow_none=False):
        v = s[k]
        if v is None and allow_none:
            return
        if not (isinstance(v, (int, float)) and lo <= v <= hi):
            raise ValueError(f"{k} must be between {lo} and {hi}, got {v!r}")
    between("alpha", 1e-6, 0.999999)
    between("delta", 1e-6, 0.999999)
    between("audit_rate", 0.0, 1.0)
    between("canary", 0.0, 1.0)
    between("cal_fraction", 0.01, 0.9)
    between("min_confidence", 0.0, 1.0, allow_none=True)
    between("force_threshold", 0.0, 1.0, allow_none=True)
    between("max_mb", 0.01, 4096, allow_none=True)
    if s["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {s['mode']!r}")
    if s["on_new_option"] not in ON_NEW_OPTION:
        raise ValueError(f"on_new_option must be one of {ON_NEW_OPTION}, got {s['on_new_option']!r}")
    olds = [a for a, _ in s["rename"]]
    if len(set(olds)) != len(olds) or any(not a or not b or a == b for a, b in s["rename"]):
        raise ValueError(f"rename must map distinct non-empty labels to different non-empty labels, got {dict(s['rename'])}")
    if s["retrain"] not in RETRAIN:
        raise ValueError(f"retrain must be one of {RETRAIN}, got {s['retrain']!r}")
    if int(s["min_rows"]) < 100:
        raise ValueError("min_rows must be at least 100")
    if int(s["drift_window"]) < 10:
        raise ValueError("drift_window must be at least 10")
    bad = set(s["capture"]) - set(CAPTURE)
    if bad:
        raise ValueError(f"capture: unknown {sorted(bad)}; choose from {CAPTURE}")
    if s["force_threshold"] is not None:
        from .observe import log
        log.warning("force_threshold=%s is set: answers served below the certified threshold are NOT covered by the "
                    "certificate (they carry certified=False, flag='manual_threshold')", s["force_threshold"])


def _layers(question: str | None, path: str | None, overrides: dict) -> list[tuple[str, dict]]:
    """[(source, {key: value}), ...] from lowest to highest precedence."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown shad0w setting(s): {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    layers: list[tuple[str, dict]] = [("default", dict(DEFAULTS))]
    cfg = find_config(path)
    if cfg:
        doc = load_toml(cfg)
        top = {k: v for k, v in doc.items() if k in DEFAULTS}
        layers.append((f"toml:{cfg}", top))
        qsec = (doc.get("questions") or {}).get(question or "", {}) if question else {}
        if qsec:
            layers.append((f"toml:{cfg} [questions.{question}]", {k: v for k, v in qsec.items() if k in DEFAULTS}))
    env = {}
    for k in DEFAULTS:
        raw = os.environ.get(ENV_PREFIX + k.upper())
        if raw is not None:
            env[k] = _coerce(k, raw)
    if env:
        layers.append(("env", env))
    code = {k: v for k, v in overrides.items() if v is not None}
    if code:
        layers.append(("code", code))
    return layers


def explain(question: str | None = None, path: str | None = None, **overrides) -> list[tuple[str, Any, str]]:
    """Every setting with its effective value and where it came from."""
    out: dict[str, tuple[Any, str]] = {}
    for source, vals in _layers(question, path, overrides):
        for k, v in vals.items():
            src = source if source != "env" else f"env:{ENV_PREFIX}{k.upper()}"
            out[k] = (_norm(k, v), src)
    return [(k, out[k][0], out[k][1]) for k in DEFAULTS]


def resolve(question: str | None = None, path: str | None = None, **overrides) -> Settings:
    """The effective settings for one question (or the defaults when question is None)."""
    s = {k: v for k, v, _ in explain(question, path, **overrides)}
    _validate(s)
    return Settings(**s)
