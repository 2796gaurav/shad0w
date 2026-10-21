"""LLMTeacher(stabilise=n): every input is asked in 1 + n option orders and the majority answer is returned, on all
three transports (chat JSON / structured outputs, the OpenAI Decisions API, System One)."""
import json

import pytest

import shad0w
from shad0w.llm import LLMTeacher, majority, option_orders

from .mock_llm import MockLLM

OPTIONS = {"refund": "wants money back", "lost_card": "card lost or stolen", "balance": "asks balance", "transfer": "move money"}


@pytest.fixture
def mock(request):
    m = MockLLM(getattr(request, "param", "structured"))
    yield m
    m.close()


def _orders_sent(requests):
    """The option order of each chat request: from the system prompt's option lines and from the enum."""
    out = []
    for r in requests:
        body = r["body"]
        lines = [l[2:].split(":")[0] for l in body["messages"][0]["content"].splitlines() if l.startswith("- ")]
        enum = body.get("response_format", {}).get("json_schema", {}).get("schema", {}).get("properties", {}).get("answer", {}).get("enum")
        out.append((lines, enum))
    return out


def test_option_orders_are_reproducible_permutations():
    a = option_orders(list(OPTIONS), "my card was stolen", 3)
    b = option_orders(list(OPTIONS), "my card was stolen", 3)
    assert a == b and a[0] == list(OPTIONS) and len(a) == 4
    assert all(sorted(o) == sorted(OPTIONS) for o in a)
    assert any(o != a[0] for o in a[1:])
    assert option_orders(list(OPTIONS), "another text", 3)[1:] != a[1:]


def test_majority_breaks_ties_with_the_first_asking():
    assert majority(["a", "b", "b"]) == "b"
    assert majority(["a", "b"]) == "a"
    assert majority(["a", "b", "b", "c", "c"]) == "b"  # tie between b and c: b was asked earlier
    assert majority([True, False, True]) is True


def test_chat_transport_permutes_prompt_and_enum(mock):
    t = shad0w.llm_teacher(OPTIONS, model="mock-1", base_url=mock.url, api_key="k", stabilise=2)
    assert t("someone stole my card, block my card") == "lost_card"
    assert len(mock.requests) == 3 and t.calls == 3
    sent = _orders_sent(mock.requests)
    assert sent[0] == (list(OPTIONS), list(OPTIONS))  # the first asking is the canonical order
    for lines, enum in sent:
        assert lines == enum and sorted(lines) == sorted(OPTIONS)  # prompt and schema agree, same options
    assert any(lines != list(OPTIONS) for lines, _ in sent[1:])
    assert t.info == {"teacher": "custom/mock-1", "teacher_stabilised": 2}
    assert shad0w.llm_teacher(OPTIONS, model="mock-1", base_url=mock.url, api_key="k").info == {"teacher": "custom/mock-1"}


def test_descriptions_follow_their_option(mock):
    t = shad0w.llm_teacher(OPTIONS, model="mock-1", base_url=mock.url, api_key="k", stabilise=3)
    t("i want a refund")
    for r in mock.requests:
        for line in r["body"]["messages"][0]["content"].splitlines():
            if line.startswith("- "):
                name, _, desc = line[2:].partition(": ")
                assert OPTIONS[name] == desc


def test_majority_vote_over_order_dependent_answers():
    """A model that always picks the FIRST listed option flips with every order; the vote still settles."""
    calls = []

    def first_listed(messages):
        calls.append(messages)
        lines = [l[2:].split(":")[0] for l in messages[0]["content"].splitlines() if l.startswith("- ")]
        return json.dumps({"answer": lines[0]})

    t = LLMTeacher(OPTIONS, "x", complete=first_listed, stabilise=4)
    ans = t("whatever")
    assert ans in OPTIONS and len(calls) == 5
    answers = [json.loads(first_listed(m))["answer"] for m in list(calls)]
    assert ans == majority(answers)
    # stabilise=0 is the old behaviour: one call, the first option
    t0 = LLMTeacher(OPTIONS, "x", complete=first_listed)
    assert t0("whatever") == "refund"


def test_failed_extra_askings_are_left_out_but_the_first_must_succeed():
    n = {"i": 0}

    def flaky(messages):
        n["i"] += 1
        return "???" if n["i"] % 2 == 0 else json.dumps({"answer": "balance"})

    t = LLMTeacher(OPTIONS, "x", complete=flaky, stabilise=2, retries=0)
    assert t("hi") == "balance" and t.failures == 1
    n["i"] = 1  # the first asking fails
    with pytest.raises(shad0w.TeacherError):
        t("hi")


def test_decisions_api_transport(mock):
    t = shad0w.llm_teacher(OPTIONS, model="openai-decisions/gpt-6-luna", base_url=mock.url, api_key="k", stabilise=2)
    ans, probs = t.answer_with_probabilities("someone stole my card, block my card")
    assert ans == "lost_card" and list(probs) == list(OPTIONS) and abs(sum(probs.values()) - 1) < 1e-6
    assert probs["lost_card"] > 0.8
    sent = [[c["value"] for c in r["body"]["questions"][0]["choices"]] for r in mock.requests]
    assert len(sent) == 3 and sent[0] == list(OPTIONS) and any(o != sent[0] for o in sent[1:])
    assert all(sorted(o) == sorted(OPTIONS) for o in sent)
    descs = {c["value"]: c.get("description") for r in mock.requests for c in r["body"]["questions"][0]["choices"]}
    assert descs == OPTIONS


def test_systemone_transport(mock):
    t = shad0w.llm_teacher(OPTIONS, model="systemone/kev-4b", base_url=mock.url, stabilise=2)
    ans, probs = t.answer_with_probabilities("i want a refund")
    assert ans == "refund" and list(probs) == list(OPTIONS) and probs["refund"] > 0.8
    sent = [list(r["body"]["questions"]["decision"]["criteria"]) for r in mock.requests]
    assert len(sent) == 3 and sent[0] == list(OPTIONS) and any(o != sent[0] for o in sent[1:])
    assert all(r["path"].endswith("/systemone") for r in mock.requests)


def test_yesno_still_votes(mock):
    t = shad0w.llm_teacher(bool, model="mock-1", base_url=mock.url, api_key="k", question="spam", stabilise=2,
                           system="Is this spam? Reply with JSON {\"answer\": \"yes\"} or {\"answer\": \"no\"}.")
    assert t("free money click here") is True and len(mock.requests) == 3
    enums = [r["body"]["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] for r in mock.requests]
    assert all(sorted(e) == ["no", "yes"] for e in enums)


def test_pickle_keeps_stabilise():
    import pickle
    t = LLMTeacher(OPTIONS, "mock-1", base_url="http://127.0.0.1:9/v1", api_key="k", stabilise=1)
    t2 = pickle.loads(pickle.dumps(t))
    assert t2.stabilise == 1 and t2.info["teacher_stabilised"] == 1


# ---- majority-of-k teacher votes for calibration rows (shad0w.audit.majority_labels, `shad0w votes`) ----

def test_majority_labels_votes_and_tolerates_failures():
    from shad0w.audit import majority_labels
    seq = iter(["refund", "balance", "refund", "???", "balance", "balance", "x", "x", "x"])

    def teacher(text):
        a = next(seq)
        if a in ("???", "x"):
            raise shad0w.TeacherError("no")
        return a
    out = majority_labels(teacher, ["t1", "t2", "t3"], 3)
    assert out[0] == {"answer": "refund", "votes": 3, "agreement": 2 / 3, "answers": ["refund", "balance", "refund"]}
    assert out[1]["answer"] == "balance" and out[1]["agreement"] == 1.0 and len(out[1]["answers"]) == 2
    assert out[2] == {"answer": None, "votes": 3, "agreement": None, "answers": []}
    # a logged first answer counts as the first vote: only k - 1 more askings
    calls = []
    out = majority_labels(lambda t: calls.append(t) or "balance", ["a", "b"], 3, first=["refund", "balance"], workers=2)
    assert len(calls) == 4 and out[0]["answer"] == "balance" and out[1]["agreement"] == 1.0
    with pytest.raises(ValueError):
        majority_labels(teacher, ["a"], 0)


def test_votes_cli_writes_audit_rows(tmp_path, mock):
    import os
    import subprocess
    import sys

    from .test_shadow import synth
    rows, _ = synth(400, 3, teacher_noise=0.0)
    log = tmp_path / "log.jsonl"
    log.write_text("".join(json.dumps({"text": r["text"], "intent": r["intent"], "source": "teacher"}) + "\n" for r in rows))
    out = tmp_path / "cal.jsonl"
    env = {**os.environ, "PYTHONPATH": os.getcwd(), "MOCK_KEY": "k"}
    env.pop("SHAD0W_CONFIG", None)
    r = subprocess.run([sys.executable, "-m", "shad0w", "votes", "--question", "intent", "--log", str(log), "--n", "40",
                        "--votes", "3", "--llm", "mock-1", "--out", str(out), "--workers", "3", "--replace", "--stabilise", "1"],
                       capture_output=True, text=True, timeout=300, env={**env, "SHAD0W_BASE_URL": mock.url, "SHAD0W_API_KEY_ENV": "MOCK_KEY"})
    assert r.returncode == 0, r.stderr + r.stdout
    cal = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(cal) == 40 and all(c["source"] == "audit" and c["teacher_votes"] == 3 and c["teacher_stabilised"] == 1 for c in cal)
    assert all(c["intent"] in ("refund", "lost_card", "balance", "transfer") and c["vote_agreement"] == 1.0 for c in cal)
    assert len(mock.requests) == 40 * 3 * 2  # 3 votes x (1 + 1 stabilise) askings per text
    assert "majority-of-3 rows" in r.stdout and "rewrote" in r.stdout
    new_log = [json.loads(l) for l in log.read_text().splitlines()]
    texts = {c["text"] for c in cal}
    dup = sum(r["text"] in texts for r in rows)  # synth repeats texts: every row with a sampled text leaves the fit set
    assert (log.parent / "log.jsonl.bak").exists() and len(new_log) == 400 - dup + 40
    assert sum(r["source"] == "audit" for r in new_log) == 40 and not any(r["text"] in texts and r["source"] != "audit" for r in new_log)
