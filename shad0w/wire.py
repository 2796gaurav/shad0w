"""The two "decision model" wire formats, in one place. Standard library only.

  decisions   OpenAI's Decisions API:  POST /v1/decisions
              {"model": "gpt-6-luna", "input": "text" | [user messages], "questions": [
                 {"type": "choice", "name": "intent", "instructions": "...", "choices": [{"value": "refund", "description": "..."}]},
                 {"type": "predicate", "name": "urgent", "instructions": "..."},
                 {"type": "score", "name": "severity", "instructions": "...", "levels": [{"label": "...", "description": "..."}]}]}
              -> {"answers": [{"type": "choice", "name": "intent", "choice": "refund", "confidence": 0.9,
                               "probabilities": [{"value": "refund", "probability": 0.9}, ...]},
                              {"type": "predicate", "name": "urgent", "probability": 0.12}, ...], "model": ..., "usage": ...}

  systemone   the System One format spoken by Jev, Kev, Laya, Ollaya and llama.cpp:  POST /v1/systemone
              {"model": "kev-4b", "state": "text" | {...}, "questions": {
                 "intent": {"type": "choice", "instructions": "...", "criteria": {"refund": "...", ...}},
                 "urgent": {"type": "noul", "instructions": "..."},
                 "severity": {"type": "score", "instructions": "...", "criteria": ["low", "high"]}}}
              -> {"answers": {"intent": {"type": "choice", "choice": "refund", "confidence": 0.9, "probabilities": {...}},
                              "urgent": {"type": "noul", "probability": 0.12}}, "latency_ms": 12}

Both carry the question name and the finite option list inside the request, so a proxy can recognise a decision
with no header and no code change. Field names for the OpenAI format were taken from the openai SDK 3.26 types.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WireQ", "dialect_of", "parse", "subset", "parse_answers", "answer_value", "format_answer",
           "format_response", "teacher_request", "teacher_answer", "DEC_CHOICES", "DEC_VALUE", "DEC_CHOSEN", "DEC_LEVELS"]

# OpenAI Decisions API field names (openai>=3.26: types/decision_create_params.py, types/decision.py)
DEC_CHOICES = "choices"      # request: choice question options
DEC_VALUE = "value"          # request: option value; response: probabilities[].value
DEC_CHOSEN = "choice"        # response: the chosen value
DEC_LEVELS = "levels"        # request: score levels [{label, description}]

DIALECTS = ("decisions", "systemone")


@dataclass
class WireQ:
    name: str
    qtype: str                                   # "choice" | "yesno" | "score"
    criteria: dict[str, str | None] = field(default_factory=dict)
    instructions: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def options(self) -> list[str]:
        return list(self.criteria)


def dialect_of(path: str) -> str | None:
    p = path.split("?", 1)[0].rstrip("/")
    if p.endswith("/decisions"):
        return "decisions"
    if p.endswith("/systemone"):
        return "systemone"
    return None


def _text_of_messages(msgs) -> str | None:
    """The last user message's text parts (OpenAI `input` list form)."""
    for m in reversed(msgs or []):
        if not isinstance(m, dict) or m.get("role", "user") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") in ("input_text", "text")]
            return "\n".join(x for x in parts if x)
    return None


def _state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)


def parse(dialect: str, req: dict) -> tuple[str, list[WireQ]]:
    """(text, questions) for a request in either format. Raises ValueError on a malformed request."""
    if not isinstance(req, dict):
        raise ValueError("request body must be a JSON object")
    if dialect == "decisions":
        inp = req.get("input")
        text = inp if isinstance(inp, str) else _text_of_messages(inp)
        qs_raw = req.get("questions")
        if text is None or not isinstance(qs_raw, list):
            raise ValueError("a decisions request needs `input` (text or user messages) and a `questions` list")
        out = []
        for i, q in enumerate(qs_raw):
            if not isinstance(q, dict):
                raise ValueError(f"questions[{i}] must be an object")
            t = q.get("type")
            name = q.get("name") or f"q{i}"
            if t == "choice":
                opts = q.get(DEC_CHOICES) or q.get("options") or []
                crit = {}
                for o in opts:
                    if isinstance(o, dict) and DEC_VALUE in o:
                        crit[_val(o[DEC_VALUE])] = o.get("description")
                    elif isinstance(o, (str, bool)):
                        crit[_val(o)] = None
                if len(crit) < 2:
                    raise ValueError(f"question {name!r}: a choice needs at least two {DEC_CHOICES}")
                out.append(WireQ(name, "choice", crit, q.get("instructions"), q))
            elif t == "predicate":
                out.append(WireQ(name, "yesno", {"yes": None, "no": None}, q.get("instructions"), q))
            elif t == "score":
                levels = [str(lv.get("label", i)) if isinstance(lv, dict) else str(lv) for i, lv in enumerate(q.get(DEC_LEVELS) or [])]
                out.append(WireQ(name, "score", {lv: None for lv in levels}, q.get("instructions"), q))
            else:
                raise ValueError(f"question {name!r}: unknown type {t!r} (predicate, choice, score)")
        return text, out
    if dialect == "systemone":
        if "state" not in req or not isinstance(req.get("questions"), dict):
            raise ValueError("a systemone request needs `state` and a `questions` object")
        text = _state_text(req["state"])
        out = []
        for name, q in req["questions"].items():
            if not isinstance(q, dict):
                raise ValueError(f"question {name!r} must be an object")
            t = q.get("type", "choice")
            crit_raw = q.get("criteria")
            if t == "choice":
                if isinstance(crit_raw, dict):
                    crit = {str(k): (None if v is None else str(v)) for k, v in crit_raw.items()}
                elif isinstance(crit_raw, list):
                    crit = {str(k): None for k in crit_raw}
                else:
                    crit = {}
                if len(crit) < 2:
                    raise ValueError(f"question {name!r}: a choice needs criteria with at least two options")
                out.append(WireQ(str(name), "choice", crit, q.get("instructions"), q))
            elif t in ("noul", "boolean", "yesno", "predicate"):
                out.append(WireQ(str(name), "yesno", {"yes": None, "no": None}, q.get("instructions"), q))
            elif t == "score":
                levels = list(crit_raw) if isinstance(crit_raw, (list, dict)) else []
                out.append(WireQ(str(name), "score", {str(lv): None for lv in levels}, q.get("instructions"), q))
            else:
                raise ValueError(f"question {name!r}: unknown type {t!r} (choice, noul, score)")
        return text, out
    raise ValueError(f"unknown dialect {dialect!r}")


def _val(v) -> str:
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


def subset(dialect: str, req: dict, names: list[str]) -> dict:
    """The same request with only the named questions (order kept)."""
    keep = set(names)
    out = dict(req)
    if dialect == "decisions":
        out["questions"] = [q for i, q in enumerate(req.get("questions") or []) if (q.get("name") or f"q{i}") in keep]
    else:
        out["questions"] = {k: v for k, v in (req.get("questions") or {}).items() if k in keep}
    return out


def parse_answers(dialect: str, body: dict) -> dict[str, dict]:
    """Answers by question name from a response in either format."""
    ans = body.get("answers") if isinstance(body, dict) else None
    if dialect == "decisions":
        out = {}
        for i, a in enumerate(ans or []):
            if isinstance(a, dict):
                out[a.get("name") or f"q{i}"] = a
        return out
    return {str(k): v for k, v in (ans or {}).items() if isinstance(v, dict)}


def answer_value(q: WireQ, a: dict):
    """The option name (choice), a bool (yesno) or a float (score) carried by one answer; None when unusable."""
    if not isinstance(a, dict) or a.get("type") == "refusal":
        return None
    if q.qtype == "choice":
        for k in (DEC_CHOSEN, "value", "answer", "label"):
            if k in a and a[k] is not None:
                v = _val(a[k])
                if v in q.criteria:
                    return v
                low = {o.lower(): o for o in q.criteria}
                return low.get(v.lower())
        return None
    if q.qtype == "yesno":
        for k in ("probability", "noul", "p"):
            if isinstance(a.get(k), (int, float)):
                return float(a[k]) >= 0.5
        v = a.get("answer", a.get(DEC_CHOSEN, a.get("value")))
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("yes", "true", "y", "1")
        return None
    if q.qtype == "score":
        v = a.get("score")
        return float(v) if isinstance(v, (int, float)) else None
    return None


def format_answer(dialect: str, q: WireQ, *, choice=None, yes: bool | None = None, probability: float | None = None,
                  probabilities: dict | None = None, confidence: float | None = None, certified: bool = False,
                  flag: str | None = None, source: str = "table") -> dict:
    """One answer in the requested format, carrying the table's probabilities and a `shad0w` block."""
    meta = {"source": source, "certified": certified, "flag": flag}
    if dialect == "decisions":
        if q.qtype == "yesno":
            p = probability if probability is not None else (1.0 if yes else 0.0)
            return {"type": "predicate", "name": q.name, "probability": float(p), "shad0w": meta}
        probs = [{DEC_VALUE: k, "probability": float(v)} for k, v in (probabilities or {}).items()]
        return {"type": "choice", "name": q.name, DEC_CHOSEN: choice, "confidence": float(confidence or 0.0),
                "probabilities": probs, "shad0w": meta}
    if q.qtype == "yesno":
        p = probability if probability is not None else (1.0 if yes else 0.0)
        return {"type": "noul", "noul": float(p), "answer": "yes" if p >= 0.5 else "no", "probability": float(p),
                "confidence": float(confidence if confidence is not None else max(p, 1 - p)), "shad0w": meta}
    return {"type": "choice", "choice": choice, "confidence": float(confidence or 0.0),
            "probabilities": {k: float(v) for k, v in (probabilities or {}).items()}, "shad0w": meta}


def format_response(dialect: str, req: dict, answers, latency_us: float, extra: dict | None = None) -> dict:
    """The whole response. `answers` is a list (decisions, request order) or a dict by name (systemone)."""
    meta = {"latency_us": round(latency_us, 2), **(extra or {})}
    if dialect == "decisions":
        return {"object": "decision", "model": req.get("model") or "shad0w", "answers": list(answers),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "shad0w": meta}
    return {"model": req.get("model") or "shad0w", "answers": dict(answers), "latency_ms": round(latency_us / 1e3, 3), "shad0w": meta}


def teacher_request(dialect: str, model: str, text: str, q: WireQ) -> tuple[str, dict]:
    """(path, body) to ask one question of a decision-model server."""
    instr = q.instructions or (f"Answer the question '{q.name}' about the text."
                               if q.qtype == "yesno" else f"Classify the text ({q.name}). Pick exactly one option.")
    if dialect == "decisions":
        if q.qtype == "yesno":
            question = {"type": "predicate", "name": q.name, "instructions": instr}
        else:
            question = {"type": "choice", "name": q.name, "instructions": instr,
                        DEC_CHOICES: [{DEC_VALUE: k, **({"description": v} if v else {})} for k, v in q.criteria.items()]}
        return "/decisions", {"model": model, "input": text, "questions": [question]}
    if q.qtype == "yesno":
        question = {"type": "noul", "instructions": instr}
    else:
        question = {"type": "choice", "instructions": instr, "criteria": {k: (v or k) for k, v in q.criteria.items()}}
    return "/systemone", {"model": model, "state": text, "questions": {q.name: question}}


def teacher_answer(dialect: str, body: dict, q: WireQ):
    """The teacher's answer for `q` from a decision-model reply: option name, bool, or None."""
    return answer_value(q, parse_answers(dialect, body).get(q.name, {}))


def answer_for(dialect: str, body: dict, q: WireQ) -> dict | None:
    return parse_answers(dialect, body).get(q.name)


def is_decision_path(path: str) -> bool:
    return dialect_of(path) is not None


def to_json_text(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)
