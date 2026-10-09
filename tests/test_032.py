"""0.3.2 features: Enum / Literal options, `shad0w init` (runnable app.py), warm_start, Decision.why, repr,
decide_many / adecide_many, `shad0w status`, and the JavaScript keys / decideMany (skipped without node)."""
import asyncio
import enum
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Literal

import pytest

import shad0w

from .mock_llm import MockLLM, classify
from .test_shadow import VOCAB, synth

OPTS = list(VOCAB)
KEY = "sk-test-abcdefghijkl3f9a"
ROOT = os.path.dirname(os.path.dirname(__file__))
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("OPENAI_API_KEY", "SHAD0W_BASE_URL", "SHAD0W_LLM", "SHAD0W_API_KEY_ENV", "SHAD0W_CONFIG"):
        monkeypatch.delenv(k, raising=False)
    shad0w.configure()
    yield
    shad0w.configure()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A decision trained from warm-started rows (no LLM at all)."""
    folder = tmp_path_factory.mktemp("ws")
    d = shad0w.decision("intent", options=OPTS, folder=str(folder / "intent"), fallback="needs_review", audit_rate=0)
    rows, _ = synth(2000, 7, teacher_noise=0.01)
    rep = d.warm_start(rows + [{"text": "weird", "intent": "astrology"}, {"text": "", "intent": "refund"}])
    cert = d.train()
    return d, folder, rep, cert


# -- enums ------------------------------------------------------------------------------------------------------------

class Intent(str, enum.Enum):  # enum.StrEnum needs 3.11
    REFUND = "refund"
    LOST_CARD = "lost_card"
    BALANCE = "balance"
    TRANSFER = "transfer"


class Route(enum.Enum):
    billing = "invoices and payments"
    tech = "something is broken"


def test_enum_options_and_round_trip(tmp_path):
    d = shad0w.decision("intent", options=Intent, llm=lambda t: classify(t), folder=str(tmp_path / "a"))
    assert d.schema["criteria"] == {o: None for o in OPTS}
    d2 = shad0w.decision("route", options=Route, llm=lambda t: "tech", folder=str(tmp_path / "b"))
    assert d2.schema["criteria"] == {"billing": "invoices and payments", "tech": "something is broken"}
    d3 = shad0w.decision("lit", options=Literal["a", "b"], llm=lambda t: "a", folder=str(tmp_path / "c"))
    assert list(d3.schema["criteria"]) == ["a", "b"]

    @shad0w.decide(folder=str(tmp_path / "d"))
    def intent(text) -> Intent:
        return Intent(classify(text))

    a = intent("hi block my card thanks")
    assert a is Intent.LOST_CARD
    row = json.loads((tmp_path / "d" / "log.jsonl").read_text().splitlines()[-1])
    assert row["intent"] == "lost_card"

    @shad0w.decide(folder=str(tmp_path / "e"))
    def route(text) -> Route:
        return Route.billing if "invoice" in text else "tech"  # a member or its option name both work

    assert route("my invoice is wrong") is Route.billing and route("it crashed") is Route.tech
    assert json.loads((tmp_path / "e" / "log.jsonl").read_text().splitlines()[0])["route"] == "billing"


def test_enum_round_trip_through_the_table(trained, tmp_path):
    d, folder, *_ = trained

    @shad0w.decide(name="intent", folder=str(folder / "intent"))
    def intent(text) -> Intent:
        raise AssertionError("the table should answer")

    assert intent("hi block my card thanks") is Intent.LOST_CARD


# -- warm start, why, repr, decide_many ---------------------------------------------------------------------------------

def test_warm_start_then_train_certifies(trained, tmp_path):
    d, folder, rep, cert = trained
    assert rep["imported"] == 2000 and rep["skipped_unknown_label"] == 1 and rep["skipped_no_text"] == 1
    assert rep["unknown_labels"] == {"astrology": 1} and rep["ready_to_train"]
    assert cert["accepted"] and cert["certified_share_on_calibration"] > 0.5
    rows = [json.loads(x) for x in (folder / "intent" / "log.jsonl").read_text().splitlines()]
    assert {r["source"] for r in rows} == {"import"}
    csv = tmp_path / "labels.csv"
    csv.write_text("message,label\nmy card was stolen,lost_card\nrefund please,Refund\nhm,nope\n", encoding="utf-8")
    d2 = shad0w.decision("intent", options=OPTS, folder=str(tmp_path / "w"), fallback="x")
    r = d2.warm_start(str(csv), text="message", label="label")
    assert r["imported"] == 2 and r["unknown_labels"] == {"nope": 1}
    jl = tmp_path / "old.jsonl"
    jl.write_text(json.dumps({"text": "send money", "intent": "transfer"}) + "\n")
    assert d2.warm_start(str(jl))["imported"] == 1
    yn = shad0w.decision("urgent", options=bool, folder=str(tmp_path / "y"), fallback=False)
    r = yn.warm_start([{"text": "now", "urgent": "yes"}, {"text": "later", "urgent": False}, {"text": "x", "urgent": "maybe"}])
    assert r["imported"] == 2 and r["skipped_unknown_label"] == 1


def test_why_repr_and_html(trained):
    d, *_ = trained
    served = d("hi block my card thanks")
    assert served.source == "table" and served.why.startswith("certified")
    unsure = d("zebra lettuce quantum harmonica")
    assert unsure.source == "fallback" and "fallback" in unsure.why
    r = repr(d)
    assert "'intent'" in r and "options=4" in r and "serving: certifies" in r and "alpha=0.05" in r
    assert "<table>" in d._repr_html_() and "serving" in d._repr_html_()
    fresh = shad0w.Shadow(None, teacher=lambda t: "a", question="q", log=None)
    assert "logging 0/1,000" in repr(fresh) and "llm=<lambda>" in repr(fresh)


def test_decide_many_keeps_order_and_only_sends_deferred_texts(trained):
    d, *_ = trained
    calls, live, peak = [], [0], [0]
    lock = threading.Lock()

    def slow_llm(text):
        with lock:
            calls.append(text)
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return "balance"
    texts = [f"zebra lettuce {i}" if i % 2 else "hi block my card thanks" for i in range(16)]
    out = d.decide_many(texts, concurrency=8, teacher=slow_llm)
    assert [o.source for o in out] == ["table" if i % 2 == 0 else "teacher" for i in range(16)]
    assert [o.answer for o in out] == ["lost_card" if i % 2 == 0 else "balance" for i in range(16)]
    assert sorted(calls) == sorted(texts[1::2]) and peak[0] > 1  # the slow calls overlap (not timed: CI runners vary)

    async def allm(text):
        await asyncio.sleep(0.05)
        return "transfer"
    out = asyncio.run(d.adecide_many(texts, concurrency=8, teacher=allm))
    assert [o.answer for o in out] == ["lost_card" if i % 2 == 0 else "transfer" for i in range(16)]


def test_decide_many_without_a_table(tmp_path):
    seen = []
    d = shad0w.decision("intent", options=OPTS, llm=lambda t: (seen.append(t), classify(t))[1], folder=str(tmp_path / "n"))
    texts = [r["text"] for r in synth(40, 3)[0]]
    out = d.decide_many(texts, concurrency=4)
    assert [o.answer for o in out] == [classify(t) for t in texts] and len(seen) == 40 and d.log_rows() == 40


# -- CLI: init and status ---------------------------------------------------------------------------------------------

def run(*args, env=None, cwd=None):
    e = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "SHAD0W_CONFIG", "SHAD0W_BASE_URL")}
    e.update(env or {})
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=e, cwd=cwd, timeout=180,
                          encoding="utf-8")


def test_init_writes_a_runnable_app(tmp_path):
    r = run("-m", "shad0w", "init", "--llm", "openai/gpt-6-luna", "--options", "refund,lost_card,balance,transfer",
            cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    toml = (tmp_path / "shad0w.toml").read_text()
    assert 'api_key_env = "OPENAI_API_KEY"' in toml and 'llm = "openai/gpt-6-luna"' in toml and "sk-" not in toml
    assert "decision(" in (tmp_path / "app.py").read_text()
    mock = MockLLM()
    try:
        r = run("app.py", "please block my card", "i want a refund", env={"OPENAI_API_KEY": KEY, "SHAD0W_BASE_URL": mock.url},
                cwd=str(tmp_path))
    finally:
        mock.close()
    assert r.returncode == 0, r.stderr
    assert "-> lost_card  via teacher" in r.stdout and "-> refund" in r.stdout and KEY not in r.stdout
    assert mock.requests[-1]["headers"]["authorization"] == f"Bearer {KEY}"
    assert len((tmp_path / "shad0w" / "intent" / "log.jsonl").read_text().splitlines()) == 2
    # (run from elsewhere: next to a ./shad0w data folder, an *editable* install would import that folder instead)
    assert run("-m", "shad0w", "init", "--dir", str(tmp_path)).stdout.count("kept existing") == 2
    r = run("-m", "shad0w", "init", "--files", "--dir", str(tmp_path / "old"))
    assert r.returncode == 0 and (tmp_path / "old" / "schema.json").exists()


def test_status_lines(trained, tmp_path):
    d, folder, *_ = trained
    small = shad0w.decision("urgent", options=bool, folder=str(folder / "urgent"), fallback=False)
    small.warm_start([{"text": f"t{i}", "urgent": i % 2 == 0} for i in range(60)])
    r = run("-m", "shad0w", "status", "--folder", str(folder))
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    intent = next(x for x in lines if x.startswith("intent"))
    urgent = next(x for x in lines if x.startswith("urgent"))
    assert "certified" in intent and "alpha=0.05" in intent and intent.endswith("-> serving")
    assert "60/1,000 rows" in urgent and "log 940 more answers" in urgent
    r = run("-m", "shad0w", "status", "--folder", str(tmp_path / "none"))
    assert r.returncode == 0 and "no decisions" in r.stdout


def test_status_live_agreement(tmp_path):
    rows, _ = synth(1500, 8, teacher_noise=0.0)
    d = shad0w.decision("intent", options=OPTS, folder=str(tmp_path / "intent"), fallback="x", audit_rate=0)
    d.warm_start(rows)
    d.train()
    time.sleep(0.01)
    with open(tmp_path / "intent" / "log.jsonl", "a") as f:
        for r in synth(50, 9, teacher_noise=0.0)[0]:
            f.write(json.dumps({**r, "source": "audit", "ts": time.time() + 1}) + "\n")
    out = run("-m", "shad0w", "status", "--folder", str(tmp_path)).stdout
    assert "live agreement" in out and "(n=" in out


# -- JavaScript -------------------------------------------------------------------------------------------------------

@needs_node
def test_js_keys_configure_and_decide_many():
    mock = MockLLM()
    script = r"""
const s = require(process.argv[1]);
(async () => {
  const base = process.argv[2];
  const out = {};
  let n = 0;
  const d = await s.decision("intent", {options: ["refund", "lost_card", "balance", "transfer"], llm: "openai/gpt-6-luna",
    baseURL: base, apiKey: () => `sk-test-rotating-key-${String(++n).padStart(4, "0")}`, log: () => {}});
  await d.decide("i want a refund"); await d.decide("block my card");
  out.calls = n; out.str = String(d);
  try { await s.decision("intent", {options: ["a", "b"], llm: () => "a", apiKey: "x"}); } catch (e) { out.fnKey = e.message; }
  try { await s.decision("intent", {options: ["a", "b"], llm: "openai/x", apikey: "x"}); } catch (e) { out.typo = e.message; }
  delete process.env.OPENAI_API_KEY;
  try { await s.decision("intent", {options: ["a", "b"], llm: "openai/x"}); } catch (e) { out.missing = e.message; }
  s.configure({ llm: "openai/gpt-6-luna", apiKey: "sk-test-configured-0003", baseURL: base });
  const c = await s.decision("intent", {options: ["refund", "lost_card", "balance", "transfer"], log: () => {}});
  out.conf = (await c.decide("send money to bob")).answer;
  out.masked = s.configure({}).apiKey; s.configure();
  let llmCalls = 0;
  const slow = async (t) => { llmCalls++; await new Promise(r => setTimeout(r, 20)); return t.length % 2 ? "a" : "b"; };
  const m = await s.decision({options: ["a", "b"], llm: slow, log: () => {}});
  const t0 = Date.now();
  const res = await m.decideMany(["x", "xy", "xyz", "xyzw", "x", "xy", "xyz", "xyzw"], {concurrency: 8});
  out.many = res.map(r => r.answer); out.manyMs = Date.now() - t0; out.llmCalls = llmCalls; out.why = res[0].why;
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e); process.exit(1); });
"""
    try:
        r = subprocess.run(["node", "-e", script, os.path.join(ROOT, "js", "index.js"), mock.url], capture_output=True,
                           text=True, timeout=60, env={**os.environ, "OPENAI_API_KEY": ""})
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout)
        auths = [q["headers"].get("authorization") for q in mock.requests]
    finally:
        mock.close()
    assert out["calls"] == 2 and auths[:2] == ["Bearer sk-test-rotating-key-0001", "Bearer sk-test-rotating-key-0002"]
    assert "called on every request" in out["str"] and "sk-test-rotating" not in out["str"]
    assert "provider/model" in out["fnKey"] and 'did you mean "apiKey"' in out["typo"]
    assert "apiKey" in out["missing"] and "apiKeyEnv" in out["missing"] and "configure" in out["missing"]
    assert out["conf"] == "transfer" and auths[-1] == "Bearer sk-test-configured-0003" and out["masked"] == "sk-…0003"
    assert out["many"] == ["a", "b"] * 4 and out["llmCalls"] == 8 and out["manyMs"] < 150
    assert out["why"] == "no trained table yet: asked your model"


@needs_node
def test_esm_build_is_in_sync():
    src = open(os.path.join(ROOT, "js", "index.mjs"), encoding="utf-8").read()
    assert "function configure(" in src and "decideMany(" in src and "export { Table" in src.replace("\n", " ")


def test_import_openai_chat_and_decisions(tmp_path):
    chat = tmp_path / "chat.jsonl"
    lines = []
    for r in synth(30, 5, teacher_noise=0.0)[0]:
        req = {"model": "gpt-6-luna", "messages": [{"role": "system", "content": "classify"}, {"role": "user", "content": r["text"]}],
               "response_format": {"type": "json_schema", "json_schema": {"name": "intent", "schema": {
                   "type": "object", "properties": {"answer": {"type": "string", "enum": OPTS}}}}}}
        resp = {"choices": [{"message": {"role": "assistant", "content": json.dumps({"answer": r["intent"]})}}]}
        lines.append({"custom_id": "x", "request": {"body": req}, "response": {"status_code": 200, "body": resp}})
    lines.append({"request": {"messages": [{"role": "user", "content": "hm"}]},
                  "response": {"choices": [{"message": {"content": "no idea, sorry"}}]}})
    chat.write_text("".join(json.dumps(x) + "\n" for x in lines))
    r = run("-m", "shad0w", "import", "--from", "openai-chat", "--file", str(chat), "--question", "intent",
            "--options", ",".join(OPTS), "--dir", str(tmp_path / "d"))
    assert r.returncode == 0, r.stderr
    assert "30 answers imported" in r.stdout and "(1 exchanges without a usable answer" in r.stdout
    rows = [json.loads(x) for x in (tmp_path / "d" / "intent" / "log.jsonl").read_text().splitlines()]
    assert len(rows) == 30 and rows[0]["source"] == "import" and rows[0]["intent"] in OPTS

    dec = tmp_path / "dec.jsonl"
    req = {"model": "gpt-6-luna", "input": "please block my card",
           "questions": [{"type": "choice", "name": "intent", "choices": [{"value": o} for o in OPTS]}]}
    resp = {"object": "decision", "answers": [{"type": "choice", "name": "intent", "choice": "lost_card"}]}
    dec.write_text(json.dumps({"request": req, "response": resp}) + "\n")
    r = run("-m", "shad0w", "import", "--from", "openai-decisions", "--file", str(dec), "--question", "intent",
            "--dir", str(tmp_path / "e"))
    assert r.returncode == 0, r.stderr
    row = json.loads((tmp_path / "e" / "intent" / "log.jsonl").read_text())
    assert row["text"] == "please block my card" and row["intent"] == "lost_card"


def test_decide_decorator_can_call_an_llm_for_you(tmp_path):
    m = MockLLM()
    try:
        @shad0w.decide(llm="openai/gpt-6-luna", api_key=KEY, base_url=m.url, folder=str(tmp_path / "s"), audit_rate=0)
        def intent(text) -> Literal["refund", "lost_card", "balance", "transfer"]:
            raise AssertionError("the body is not run when llm= is given")

        assert intent("hi block my card thanks") == "lost_card"
        assert intent.shadow.teacher.api_key == KEY

        @shad0w.decide(llm="openai/gpt-6-luna", api_key=KEY, base_url=m.url, folder=str(tmp_path / "e"), audit_rate=0)
        def as_enum(text) -> Intent: ...

        assert as_enum("hi block my card thanks") is Intent.LOST_CARD
    finally:
        m.close()

    @shad0w.decide(llm=lambda t: "tech", folder=str(tmp_path / "f"))
    def route(text) -> Route: ...

    assert route("it crashed") is Route.tech


def test_failed_spot_check_is_not_left_running(trained):
    d = trained[0]
    def broken(text):
        raise RuntimeError("upstream down")
    d.audit_rate = 1.0
    try:
        for text in ["block my card please", "refund my order", "what is my balance"]:
            d.decide(text, teacher=broken)
        d.flush(5)
        assert not [t for t in d._audits if getattr(t, "is_alive", lambda: False)()]
        assert len(d._audits) == 0
    finally:
        d.audit_rate = 0


def test_adecide_accepts_probabilities(trained):
    d = trained[0]
    out = asyncio.run(d.adecide("hi block my card thanks", probabilities=True))
    assert out.top is not None or out.source != "table"


def test_recertify_keeps_entry_for_questions_without_fresh_rows(trained, tmp_path):
    from shad0w.shadow import certify_bundle, read_certificate
    folder = trained[1]
    src = os.path.join(str(folder), "intent", "bundle")
    dst = str(tmp_path / "b")
    shutil.copytree(src, dst)
    cert = certify_bundle(dst, [{"text": "x", "other_question": "a"}])
    assert cert["questions"]["intent"]["recertified"] is False
    assert read_certificate(dst)["questions"]["intent"]["threshold"] == cert["questions"]["intent"]["threshold"]


def test_shadow_mode_never_serves_even_with_force_threshold(trained, tmp_path):
    d = trained[0]
    sh = shad0w.Shadow(d.bundle_path if hasattr(d, "bundle_path") else os.path.join(str(trained[1]), "intent", "bundle"),
                       teacher=lambda t: "refund", question="intent", log=str(tmp_path / "l.jsonl"),
                       mode="shadow", force_threshold=0.0, audit_rate=0)
    outs = [sh.decide(t) for t in ["block my card", "refund please", "zzz qqq"]]
    assert all(o.source == "teacher" for o in outs)


def test_config_reports_invalid_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAD0W_MODE", "on")
    r = subprocess.run([sys.executable, "-c", "import sys; from shad0w.__main__ import main; sys.exit(main(['config']))"],
                       cwd=str(tmp_path), capture_output=True, text=True)
    assert r.returncode == 2 and "invalid setting" in r.stderr
