"""Structured extraction: a JSON schema of enum fields compiled into one certified table per field.

    schema = {"area": ["centre", "north", "south", "none"], "price": ["cheap", "moderate", "expensive", "none"],
              "parking": ["yes", "no", "none"]}
    records = [{"text": "...", "area": "north", "price": "cheap", "parking": "none"}, ...]   # your LLM's JSON, logged
    ex = shad0w.extract.compile(schema, records, alpha=0.05)                                  # one table per field
    e = ex.decide("a cheap place up north")      # Extraction(values={"price": "cheap", "area": "north"}, deferred=["parking"])
    if e.deferred:                               # partial deferral: ask the LLM only for what the tables could not certify
        e = ex.merge(e, your_llm(ex.prompt(text, e.deferred)))
    e.values                                     # {"area": ..., "price": ..., "parking": ...}

Each field carries the ordinary shad0w certificate (shadow_compile): among values a field's table serves itself,
disagreement with the teacher <= alpha with probability >= 1 - delta. The text is hashed once and shared by all
fields, so a record costs about one table decision plus a few microseconds per extra field.

Record-level guarantee (record_alpha=...): alpha and delta are split across the F fields by Bonferroni
(alpha / F, delta / F each). Then, on traffic like the calibration traffic and with probability >= 1 - delta, the
share of records in which SOME certified field disagrees with the teacher is at most record_alpha: by the union bound
P(any certified field wrong) <= sum_f P(field f certified and wrong) <= sum_f alpha / F. Among the records that are
served completely (no field deferred) the realised rate is at most record_alpha / (share served completely); the
certificate reports the empirical value. Fields the schema marks as free text are not compiled: they are always
deferred (they are what the LLM is for).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum

from .api import Model
from .api import load as _load_model
from .llm import match_option, normalize_options

EXTRACT_FILE = "extract.json"


@dataclass
class Extraction:
    values: dict                      # certified field -> value (plus merged LLM values after merge())
    deferred: list                    # fields the tables could not certify (ask your LLM for these only)
    hints: dict = field(default_factory=dict)        # the tables' uncertified guesses for the deferred fields
    confidence: dict = field(default_factory=dict)   # field -> table confidence
    flags: dict = field(default_factory=dict)        # field -> why it was deferred
    latency_us: float = 0.0
    sources: dict = field(default_factory=dict)      # field -> "table" | "llm"

    @property
    def complete(self) -> bool:
        return not self.deferred

    def __getitem__(self, k):
        return self.values[k]


def _field_options(name: str, options) -> dict | None:
    """{option: description} for an enum field; None for a free-text field ("text", str, None, {"type": "string"})."""
    if options is None or options is str or options == "text" or options == "string":
        return None
    if isinstance(options, dict) and options.get("type") in ("string", "text", "number", "integer") and "criteria" not in options:
        return None
    _, qtype, crit, _ = normalize_options(options, name)
    return {"yes": None, "no": None} if qtype == "yesno" else crit


class Extractor:
    """Per-field certified tables over one shared hashing pass (build it with `compile` or `load`)."""

    def __init__(self, model: Model, fields: dict, free: list | None = None, cert: dict | None = None):
        self.model, self.cert = model, cert or {}
        self.fields = {f: list(o) for f, o in fields.items()}   # enum fields -> option names
        self.free = list(free or [])                             # free-text fields: always deferred
        self._questions = {f: {} for f in self.fields}

    # -- persistence --------------------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        self.model.meta["extract"] = {"fields": self.fields, "free": self.free}
        self.model.save(path)
        with open(os.path.join(path, EXTRACT_FILE), "w", encoding="utf-8") as f:
            json.dump(self.cert, f, indent=2)

    @classmethod
    def load(cls, path: str, native: bool = True) -> Extractor:
        m = _load_model(path, native=native)
        meta = m.meta.get("extract") or {"fields": {n: q.options for n, q in m.questions.items()}, "free": []}
        p = os.path.join(path, EXTRACT_FILE)
        cert = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        return cls(m, meta["fields"], meta.get("free"), cert)

    # -- deciding -----------------------------------------------------------------------------------------------
    def decide(self, text: str, observe: bool = True) -> Extraction:
        """Microseconds: every enum field whose table is certified on this text is filled; the rest are deferred."""
        t0 = time.perf_counter()
        ans = self.model.decide(text, questions=self._questions, probabilities=False, observe=observe)["answers"]
        out = Extraction({}, [], latency_us=0.0)
        for f in self.fields:
            r = ans[f]
            v = r["choice"] if "choice" in r else ("yes" if r["answer"] else "no")
            out.confidence[f] = r["confidence"]
            if r["certified"]:
                out.values[f] = v
                out.sources[f] = "table"
            else:
                out.deferred.append(f)
                out.hints[f] = v
                out.flags[f] = r["flag"]
        out.deferred.extend(self.free)
        out.latency_us = (time.perf_counter() - t0) * 1e6
        return out

    def prompt(self, text: str, fields: list | None = None) -> str:
        """A prompt asking your LLM for ONLY the given fields (default: all), as one JSON object."""
        fields = list(self.fields) + self.free if fields is None else list(fields)
        lines = []
        for f in fields:
            if f in self.fields:
                lines.append(f'  "{f}": one of {json.dumps(self.fields[f])}')
            else:
                lines.append(f'  "{f}": free text (or null)')
        return ("Extract the following fields from the text and answer with one JSON object and nothing else:\n{\n"
                + ",\n".join(lines) + "\n}\n\nText:\n" + text)

    def merge(self, extraction: Extraction, llm_reply) -> Extraction:
        """Fill the deferred fields from your LLM's reply (a JSON string or a parsed dict). Enum values are matched
        to the field's options (case, spacing and code fences forgiven); a value that matches no option, or a field
        the LLM left out, stays deferred. Certified values are never overwritten."""
        data = llm_reply
        if isinstance(data, str):
            s = data.strip()
            if s.startswith("```"):
                s = s.strip("`")
                s = s[4:] if s.lower().startswith("json") else s
            try:
                data = json.loads(s)
            except ValueError:
                data = {}
        if not isinstance(data, dict):
            data = {}
        still = []
        for f in extraction.deferred:
            if f not in data:
                still.append(f)
                continue
            v = data[f]
            if f in self.fields:
                v = match_option(v.value if isinstance(v, Enum) else v, self.fields[f], f)
                if v is None:
                    still.append(f)
                    continue
            extraction.values[f] = v
            extraction.sources[f] = "llm"
        extraction.deferred = still
        return extraction

    def __call__(self, text: str, llm=None) -> Extraction:
        """decide(); when fields are deferred and `llm` (callable(prompt) -> reply) is given, ask it for those only."""
        e = self.decide(text)
        if e.deferred and llm is not None:
            e = self.merge(e, llm(self.prompt(text, e.deferred)))
        return e

    def __repr__(self) -> str:
        return f"Extractor(fields={list(self.fields)}, free={self.free})"


def compile(schema: dict, records: list, alpha: float = 0.05, delta: float = 0.1, record_alpha: float | None = None,
            teacher: str = "unspecified", **shadow_kw) -> Extractor:
    """Compile every enum field of `schema` ({field: options}; options as for shad0w.decision: a list, a
    {value: description} dict, an Enum, Literal, bool; None / str / "text" marks a free-text field, never compiled)
    from records {"text": ..., field: value, ...} (a record may miss some fields) into one certified table per field.

    alpha, delta   per-field certificate (shadow_compile)
    record_alpha   optional record-level guarantee: the fields are certified at record_alpha / F and delta / F each,
                   so P(some certified field of a record disagrees with the teacher) <= record_alpha (see module doc)
    Other keywords go to shadow_compile (cal_fraction, max_cal, cal_records, seed, max_mb, static, fit_extra, ...)."""
    from .shadow import shadow_compile
    fields, free = {}, []
    for name, options in schema.items():
        crit = _field_options(name, options)
        if crit is None:
            free.append(name)
        else:
            fields[name] = crit
    if not fields:
        raise ValueError("the schema has no enum field to compile (every field is free text)")
    F = len(fields)
    a_f, d_f = (record_alpha / F, delta / F) if record_alpha is not None else (alpha, delta)
    sh_schema = {f: {"type": "choice", "criteria": crit} for f, crit in fields.items()}
    rows = [{**{k: (str(v) if k in fields and v is not None else v) for k, v in r.items()}} for r in records]
    model, cert = shadow_compile(sh_schema, rows, alpha=a_f, delta=d_f, teacher=teacher, **shadow_kw)
    cert["extract"] = {"fields": list(fields), "free": free, "field_alpha": a_f, "field_delta": d_f,
                       "record_alpha": record_alpha,
                       "statement": ("Per field: among values the field's table serves, disagreement with the teacher "
                                     f"<= {a_f:.4g} with probability >= {1 - d_f:.4g}." +
                                     (f" Record level (Bonferroni over {F} fields): the share of records in which some "
                                      f"certified field disagrees with the teacher is <= {record_alpha:.4g}, with probability "
                                      f">= {1 - delta:.4g}." if record_alpha is not None else ""))}
    ex = Extractor(model, {f: list(c) for f, c in fields.items()}, free, cert)
    model.meta["extract"] = {"fields": ex.fields, "free": free}
    return ex


def load(path: str, native: bool = True) -> Extractor:
    return Extractor.load(path, native=native)
