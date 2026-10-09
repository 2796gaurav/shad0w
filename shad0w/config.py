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

`shad0w.configure(...)` sets process-wide values from code: they sit just below the keywords of one call and above
the environment. `shad0w.configure()` with no arguments clears them.

API keys never go in a file or in these settings: `api_key_env` names the environment variable that holds the key
(an `api_key =` line in shad0w.toml is an error). The key itself is wrapped in `Secret`, which prints as 'sk-…3f9a'.

`shad0w config --explain` prints every effective value and where it came from. Standard library only.
"""
from __future__ import annotations

import difflib
import functools
import logging
import os
import re
from dataclasses import dataclass, fields
from typing import Any

__all__ = ["DEFAULTS", "Settings", "resolve", "explain", "find_config", "load_toml", "HELP", "configure", "Secret",
           "did_you_mean"]

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
    "llm": None,                # default "provider/model" for decision() when llm= is not passed
    "base_url": None,           # default base URL for that LLM (an OpenAI-compatible server)
    "api_key_env": None,        # NAME of the environment variable holding the LLM key (never the key itself)
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
    "llm": "Default LLM for decision(), as \"provider/model\", e.g. \"openai/gpt-6-luna\".",
    "base_url": "Base URL of an OpenAI-compatible server for that LLM (default: the provider's).",
    "api_key_env": "Name of the environment variable that holds the LLM key, e.g. \"OPENAI_API_KEY\". Never the key itself.",
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
    llm: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None

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
        try:
            return tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path}: not valid TOML: {e}") from None


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
    if key == "auto_train" and v is True:
        raise ValueError("auto_train takes a number of new answers between retrains (e.g. 1000), not True")
    if isinstance(DEFAULTS.get(key), bool) and isinstance(v, str):  # "no" / "false" from code means False
        return v.strip().lower() in ("1", "true", "yes", "on")
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
    if int(s["auto_train"]) < 0:
        raise ValueError(f"auto_train must be 0 (off) or a positive number of answers, got {s['auto_train']!r}")
    if s["timeout"] is not None and not (isinstance(s["timeout"], (int, float)) and s["timeout"] > 0):
        raise ValueError(f"timeout must be a positive number of seconds (or none), got {s['timeout']!r}")
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


def did_you_mean(word: str, known) -> str:
    """ " (did you mean 'api_key'?)" when a close match exists, else ""."""
    m = difflib.get_close_matches(str(word), list(known), n=1, cutoff=0.6)
    return f" (did you mean {m[0]!r}?)" if m else ""


def unknown_keywords(names, known, where: str) -> TypeError:
    """A TypeError naming the unknown keywords, with a suggestion for each."""
    bad = sorted(names)
    hints = "; ".join(f"{b!r}{did_you_mean(b, known)}" for b in bad)
    return TypeError(f"{where} got unexpected keyword argument(s): {hints}")


_KEY_LINE = ("{where}: found `{key} = ...`. Never put the key itself in a file: store the env var name in api_key_env, "
             "not the key itself, e.g. api_key_env = \"OPENAI_API_KEY\"")


def _check_no_key(doc: dict, cfg: str) -> None:
    secs = [("", doc)] + [(f" [questions.{q}]", v) for q, v in (doc.get("questions") if isinstance(doc.get("questions"), dict) else {}).items()
                                    if isinstance(v, dict)]
    for sec, d in secs:
        bad = [k for k in d if str(k).lower().replace("-", "_").endswith(("api_key", "apikey"))]
        if bad:
            raise ValueError(_KEY_LINE.format(where=f"{cfg}{sec}", key=bad[0]))


_WARNED: set = set()


def _warn_unknown(cfg: str, sec: str, keys: list) -> None:
    """A misspelt key in shad0w.toml is ignored, so say so once (with a suggestion) instead of silently using defaults."""
    for k in keys:
        if (cfg, sec, k) in _WARNED:
            continue
        _WARNED.add((cfg, sec, k))
        logging.getLogger("shad0w").warning("%s%s: unknown setting %r ignored%s", cfg, sec, k, did_you_mean(str(k), DEFAULTS))


def _layers(question: str | None, path: str | None, overrides: dict) -> list[tuple[str, dict]]:
    """[(source, {key: value}), ...] from lowest to highest precedence."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise unknown_keywords(unknown, DEFAULTS, "shad0w settings")
    layers: list[tuple[str, dict]] = [("default", dict(DEFAULTS))]
    cfg = find_config(path)
    if cfg:
        doc = load_toml(cfg)
        _check_no_key(doc, cfg)
        top = {k: v for k, v in doc.items() if k in DEFAULTS}
        _warn_unknown(cfg, "", [k for k in doc if k not in DEFAULTS and k != "questions"])
        layers.append((f"toml:{cfg}", top))
        qs = doc.get("questions") or {}
        if not isinstance(qs, dict):
            raise ValueError(f"{cfg}: `questions` must be a table ([questions.<name>] sections), got {qs!r}")
        qsec = qs.get(question or "", {}) if question else {}
        if qsec and not isinstance(qsec, dict):
            raise ValueError(f"{cfg}: [questions.{question}] must be a table of settings, got {qsec!r}")
        if qsec:
            _warn_unknown(cfg, f" [questions.{question}]", [k for k in qsec if k not in DEFAULTS])
            layers.append((f"toml:{cfg} [questions.{question}]", {k: v for k, v in qsec.items() if k in DEFAULTS}))
    env = {}
    for k in DEFAULTS:
        raw = os.environ.get(ENV_PREFIX + k.upper())
        if raw is not None:
            try:
                env[k] = _coerce(k, raw)
            except ValueError as e:
                raise ValueError(f"{ENV_PREFIX + k.upper()}={raw!r}: {e}") from None
    if env:
        layers.append(("env", env))
    if _CONFIGURED:
        layers.append(("configure()", dict(_CONFIGURED)))
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


# -- API keys and process-wide defaults ------------------------------------------------------------------------------

class Secret:
    """An API key that never prints. Holds a string, a zero-arg callable (a rotating or vault key, called on every
    request) or the NAME of an environment variable (read on every request). repr/str show 'sk-…3f9a' at most; pickling
    keeps only the variable name, never a literal key."""

    __slots__ = ("_value", "env", "source")

    def __init__(self, value=None, *, env: str | None = None, source: str = ""):
        if value is not None and not (isinstance(value, str) or callable(value)):
            raise TypeError("api_key must be a string or a zero-argument function that returns one")
        self._value, self.env = value, env
        self.source = source or (f"env {env}" if env else ("api_key=<function>" if callable(value) else "api_key="))

    def get(self) -> str | None:
        """The key, now (a callable is called; an environment variable is read)."""
        v = self._value
        if callable(v):
            v = v()
        elif v is None and self.env:
            v = os.environ.get(self.env)
        return None if v is None else str(v).strip()

    @property
    def is_callable(self) -> bool:
        return callable(self._value)

    def hint(self) -> str:
        """'sk-…3f9a': the prefix and the last 4 characters, enough to tell keys apart, never enough to use one."""
        if self.is_callable:
            return "<function>"
        return mask(self.get())

    def describe(self) -> str:
        """'set via OPENAI_API_KEY (sk-…3f9a)' / 'not set (OPENAI_API_KEY is empty)'."""
        where = self.env if self.env and self._value is None else self.source
        if self.is_callable:
            return f"set via {where} (called on every request)"
        v = self.get()
        return f"set via {where} ({mask(v)})" if v else f"not set ({where} is empty)"

    def __repr__(self) -> str:
        return repr(self.hint())

    __str__ = __repr__

    def __eq__(self, other) -> bool:
        if isinstance(other, Secret):
            return self.get() == other.get()
        return isinstance(other, str) and not self.is_callable and self.get() == other

    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        return self.is_callable or bool(self.get())

    def __reduce__(self):  # never pickle a literal key; an environment variable name is safe
        return (_unpickle_secret, (self.env, self.source))

    def __getstate__(self):
        return None


def _unpickle_secret(env, source):
    return Secret(env=env, source=source) if env else Secret("", source=source + " (dropped when pickled)")


def mask(v: str | None) -> str:
    if not v:
        return "<empty>"
    m = re.match(r"^([A-Za-z]{1,8}[-_])", v)
    head = m.group(1) if m and len(v) > len(m.group(1)) + 8 else ""
    return f"{head}…{v[-4:]}" if len(v) >= 12 else "…"


def redact(text: str, *secrets) -> str:
    """`text` with every known key value replaced by its mask (for error messages)."""
    for s in secrets:
        if isinstance(s, Secret) and not s.is_callable:
            v = s.get()
        else:
            v = s if isinstance(s, str) else None
        if v and len(v) >= 6:
            text = text.replace(v, mask(v))
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}", lambda m: m.group(1) + "…", text)


_CONFIGURED: dict[str, Any] = {}
_CONFIGURED_KEY: list = [None]  # [Secret | None]


def configure(*, llm: str | None = None, api_key=None, api_key_env: str | None = None, base_url: str | None = None,
              **settings) -> dict:
    """Process-wide defaults, set once from code (for example at app start-up):

        shad0w.configure(llm="openai/gpt-6-luna", api_key_env="OPENAI_API_KEY", alpha=0.02)

    Every decision() created afterwards uses them unless it passes its own. They rank below the keywords of a single
    call and above SHAD0W_* variables and shad0w.toml. api_key may be a string or a zero-argument function (called
    on every request). Calling configure() with no arguments clears everything set before. Returns the current
    defaults, with the key masked."""
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise unknown_keywords(unknown, list(DEFAULTS) + ["api_key"], "shad0w.configure()")
    given = {"llm": llm, "api_key_env": api_key_env, "base_url": base_url, **settings}
    given = {k: v for k, v in given.items() if v is not None}
    if not given and api_key is None:
        _CONFIGURED.clear()
        _CONFIGURED_KEY[0] = None
        return {}
    trial = {**_CONFIGURED, **given}
    s = {k: _norm(k, v) for k, v in {**DEFAULTS, **trial}.items()}
    _validate(s)
    _CONFIGURED.clear()
    _CONFIGURED.update(trial)
    if api_key is not None:
        _CONFIGURED_KEY[0] = Secret(api_key, source="shad0w.configure(api_key=...)")
    out = dict(_CONFIGURED)
    if _CONFIGURED_KEY[0] is not None:
        out["api_key"] = _CONFIGURED_KEY[0]
    return out


def configured_key() -> Secret | None:
    return _CONFIGURED_KEY[0]
