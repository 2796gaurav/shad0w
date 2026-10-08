"""The two decision-model wire formats: parse, subset, answers, formatting, teacher requests."""
import pytest

from shad0w import wire

DEC = {"model": "gpt-6-luna", "input": "my card was stolen", "questions": [
    {"type": "choice", "name": "intent", "instructions": "Which intent?",
     "choices": [{"value": "refund", "description": "money back"}, {"value": "lost_card"}]},
    {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
    {"type": "score", "name": "severity", "instructions": "How bad?", "levels": [{"label": "low"}, {"label": "high"}]}]}
S1 = {"model": "kev-4b", "state": {"message": "my card was stolen"}, "questions": {
    "intent": {"type": "choice", "instructions": "Which intent?", "criteria": {"refund": "money back", "lost_card": "lost"}},
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "severity": {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]}}}


def test_dialect_of():
    assert wire.dialect_of("/v1/decisions") == "decisions" and wire.dialect_of("/decisions?x=1") == "decisions"
    assert wire.dialect_of("/v1/systemone") == "systemone" and wire.dialect_of("/v1/chat/completions") is None


def test_parse_both():
    text, qs = wire.parse("decisions", DEC)
    assert text == "my card was stolen" and [q.qtype for q in qs] == ["choice", "yesno", "score"]
    assert qs[0].options == ["refund", "lost_card"] and qs[0].criteria["refund"] == "money back" and qs[2].options == ["low", "high"]
    text, qs = wire.parse("systemone", S1)
    assert text == '{"message": "my card was stolen"}' and [q.name for q in qs] == ["intent", "urgent", "severity"]
    assert qs[1].qtype == "yesno" and qs[0].criteria["lost_card"] == "lost"
    msgs = {**DEC, "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"},
                                                           {"type": "input_image", "image_url": "data:..."}]}]}
    assert wire.parse("decisions", msgs)[0] == "hello"
    alias = {**DEC, "questions": [{"type": "choice", "name": "q", "instructions": "", "options": [{"value": "a"}, {"value": "b"}]}]}
    assert wire.parse("decisions", alias)[1][0].options == ["a", "b"]
    with pytest.raises(ValueError):
        wire.parse("decisions", {"model": "m", "questions": []})
    with pytest.raises(ValueError):
        wire.parse("decisions", {**DEC, "questions": [{"type": "choice", "name": "x", "choices": [{"value": "only"}]}]})


def test_subset_and_answers():
    sub = wire.subset("decisions", DEC, ["urgent"])
    assert [q["name"] for q in sub["questions"]] == ["urgent"] and sub["input"] == DEC["input"]
    sub = wire.subset("systemone", S1, ["severity", "intent"])
    assert list(sub["questions"]) == ["intent", "severity"]
    body = {"answers": [{"type": "choice", "name": "intent", "choice": "lost_card", "confidence": 0.9,
                         "probabilities": [{"value": "lost_card", "probability": 0.9}]},
                        {"type": "predicate", "name": "urgent", "probability": 0.8},
                        {"type": "score", "name": "severity", "score": 0.7},
                        {"type": "refusal", "name": "x"}]}
    _, qs = wire.parse("decisions", DEC)
    a = wire.parse_answers("decisions", body)
    assert wire.answer_value(qs[0], a["intent"]) == "lost_card" and wire.answer_value(qs[1], a["urgent"]) is True
    assert wire.answer_value(qs[2], a["severity"]) == 0.7 and wire.answer_value(qs[0], a["x"]) is None
    assert wire.answer_value(qs[0], {"type": "choice", "choice": "Lost_Card"}) == "lost_card"
    assert wire.answer_value(qs[0], {"type": "choice", "choice": "nope"}) is None
    _, sq = wire.parse("systemone", S1)
    s1 = wire.parse_answers("systemone", {"answers": {"intent": {"choice": "refund"}, "urgent": {"noul": 0.2}}})
    assert wire.answer_value(sq[0], s1["intent"]) == "refund" and wire.answer_value(sq[1], s1["urgent"]) is False
    assert wire.answer_value(sq[1], {"answer": "yes"}) is True


def test_format_roundtrip():
    _, qs = wire.parse("decisions", DEC)
    a = wire.format_answer("decisions", qs[0], choice="refund", probabilities={"refund": 0.7, "lost_card": 0.3}, confidence=0.7,
                           certified=True)
    assert a["type"] == "choice" and a["choice"] == "refund" and a["probabilities"][0] == {"value": "refund", "probability": 0.7}
    assert a["shad0w"]["certified"] is True
    p = wire.format_answer("decisions", qs[1], yes=True, probability=0.9, confidence=0.9)
    assert p["type"] == "predicate" and p["probability"] == 0.9
    r = wire.format_response("decisions", DEC, [a, p], 12.5)
    assert r["object"] == "decision" and r["model"] == "gpt-6-luna" and len(r["answers"]) == 2 and r["shad0w"]["latency_us"] == 12.5
    _, sq = wire.parse("systemone", S1)
    s = wire.format_answer("systemone", sq[1], yes=False, probability=0.1, confidence=0.9)
    assert s["type"] == "noul" and s["answer"] == "no" and s["probability"] == 0.1
    r = wire.format_response("systemone", S1, {"urgent": s}, 1000.0)
    assert r["latency_ms"] == 1.0 and r["answers"]["urgent"]["answer"] == "no"


def test_teacher_request_and_answer():
    q = wire.WireQ("intent", "choice", {"refund": "money back", "lost_card": None}, "Which?")
    path, body = wire.teacher_request("decisions", "gpt-6-luna", "text", q)
    assert path == "/decisions" and body["questions"][0]["choices"] == [{"value": "refund", "description": "money back"}, {"value": "lost_card"}]
    assert wire.teacher_answer("decisions", {"answers": [{"type": "choice", "name": "intent", "choice": "refund"}]}, q) == "refund"
    path, body = wire.teacher_request("systemone", "kev", "text", q)
    assert path == "/systemone" and body["questions"]["intent"]["criteria"] == {"refund": "money back", "lost_card": "lost_card"}
    yn = wire.WireQ("urgent", "yesno")
    assert wire.teacher_request("decisions", "m", "t", yn)[1]["questions"][0]["type"] == "predicate"
    assert wire.teacher_request("systemone", "m", "t", yn)[1]["questions"]["urgent"]["type"] == "noul"
    assert wire.teacher_answer("systemone", {"answers": {"urgent": {"type": "noul", "probability": 0.9}}}, yn) is True
