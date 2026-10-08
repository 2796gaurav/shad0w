"""Options change after training: added options defer everything, removed options are never served, renames apply
without retraining, the next train() learns the new option set (the retrain gate never blocks it), and the proxy follows
the options each request names. Python and JavaScript behave the same."""
import json
import os
import random
import shutil
import subprocess
import threading
import urllib.request

import pytest

import shad0w
from shad0w.cascade import rows_for_training
from shad0w.proxy import proxy

from .test_shadow import FILLER, VOCAB

OLD = ["refund", "lost_card", "balance"]
ROOT = os.path.dirname(os.path.dirname(__file__))


def texts(k, n, seed=0):
    rng = random.Random(seed)
    return [" ".join([rng.choice(FILLER), rng.choice(VOCAB[k]), rng.choice(FILLER)]) for _ in range(n)]


def oracle(t):
    return next((k for k, ps in VOCAB.items() if any(p in t for p in ps)), "balance")


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    """A trained 3-option table under <folder>/intent/bundle."""
    f = tmp_path_factory.mktemp("opts") / "intent"
    d = shad0w.decision("intent", options=OLD, llm=lambda t: oracle(t) if oracle(t) in OLD else "balance",
                        folder=str(f), audit_rate=0)
    for i, k in enumerate(OLD):
        for t in texts(k, 400, seed=i):
            d(t)
    d.train()
    return f


def copy(folder, tmp_path):
    dst = tmp_path / "intent"
    shutil.copytree(folder, dst)
    return dst


def test_added_option_defers_everything(folder, tmp_path):
    f = copy(folder, tmp_path)
    asked = []
    d = shad0w.decision("intent", options=OLD + ["transfer"], llm=lambda t: asked.append(t) or oracle(t), folder=str(f), audit_rate=0)
    assert d.options_added == ("transfer",) and d.options_removed == ()
    out = [d(t) for t in texts("transfer", 50) + texts("refund", 50)]
    assert all(o.source == "teacher" and o.flag == "options_changed" for o in out) and len(asked) == 100
    assert d.stats()["options_added"] == ["transfer"]


def test_on_new_option_serve_keeps_known_answers(folder, tmp_path):
    f = copy(folder, tmp_path)
    d = shad0w.decision("intent", options=OLD + ["transfer"], llm=oracle, folder=str(f), audit_rate=0, on_new_option="serve")
    assert any(o.source == "table" for o in (d(t) for t in texts("refund", 50)))


def test_removed_option_is_never_served(folder, tmp_path):
    f = copy(folder, tmp_path)
    d = shad0w.decision("intent", options=["refund", "lost_card"], llm=lambda t: "refund", folder=str(f), audit_rate=0)
    assert d.options_removed == ("balance",)
    out = [d(t) for t in texts("balance", 50)]
    assert not any(o.source == "table" for o in out) and {o.flag for o in out} == {"option_removed"}
    assert any(d(t).source == "table" for t in texts("lost_card", 30))  # the others keep serving


def test_rename_applies_without_retraining_and_to_the_log(folder, tmp_path):
    f = copy(folder, tmp_path)
    d = shad0w.decision("intent", options=["refund", "card_lost", "balance"], llm=oracle, folder=str(f), audit_rate=0,
                        rename={"lost_card": "card_lost"})
    assert d.options_added == () and d.options_removed == ()
    served = [o for o in (d(t) for t in texts("lost_card", 40)) if o.source == "table"]
    assert served and {o.answer for o in served} == {"card_lost"}
    rows, _ = rows_for_training([{"text": "x", "intent": "lost_card"}], "intent", {"type": "choice", "criteria": {"card_lost": None, "a": None}},
                                rename={"lost_card": "card_lost"})
    assert rows[0]["intent"] == "card_lost"


def test_retrain_learns_the_new_option_even_when_gated(folder, tmp_path):
    f = copy(folder, tmp_path)
    d = shad0w.decision("intent", options=OLD + ["transfer"], llm=oracle, folder=str(f), audit_rate=0)
    for t in texts("transfer", 400, seed=9):
        d(t)
    cert = d.train(gate=True)
    assert cert["accepted"] and d.options_added == ()
    assert "transfer" in d.model.questions["intent"].options
    served = [o for o in (d(t) for t in texts("transfer", 100, seed=3)) if o.source == "table"]
    assert len(served) > 50 and {o.answer for o in served} == {"transfer"}


def test_settings_from_env_and_validation(monkeypatch):
    from shad0w import config
    monkeypatch.setenv("SHAD0W_RENAME", "a=b, c=d")
    monkeypatch.setenv("SHAD0W_ON_NEW_OPTION", "serve")
    s = config.resolve("q")
    assert s.rename_map == {"a": "b", "c": "d"} and s.on_new_option == "serve"
    with pytest.raises(ValueError):
        config.resolve("q", on_new_option="maybe")
    with pytest.raises(ValueError):
        config.resolve("q", rename={"a": "a"})


def test_proxy_follows_the_options_a_request_names(folder, tmp_path):
    """Decisions-API requests list their choices: a new choice makes the proxy forward the question upstream."""
    copy(folder, tmp_path)
    seen = []

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append([q["name"] for q in req["questions"]])
            text = req["input"]
            ans = [{"name": q["name"], "type": "choice", "choice": oracle(text)} for q in req["questions"]]
            body = json.dumps({"id": "d", "object": "decision", "model": req["model"], "answers": ans}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    srv = proxy(f"http://127.0.0.1:{up.server_address[1]}/v1", port=0, folder=str(tmp_path), audit_rate=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1/decisions"

    def ask(choices, text):
        body = {"model": "gpt-6-luna", "input": text,
                "questions": [{"type": "choice", "name": "intent", "choices": [{"value": c} for c in choices]}]}
        r = urllib.request.urlopen(urllib.request.Request(base, data=json.dumps(body).encode(),
                                                          headers={"content-type": "application/json"}))
        return r.headers["x-shad0w-source"], json.loads(r.read())["answers"][0]["choice"]

    try:
        src, ans = ask(OLD, "hi block my card thanks")
        assert src == "table" and ans == "lost_card" and not seen
        src, ans = ask(OLD + ["transfer"], "hi block my card thanks")  # the request now offers a new option
        assert src == "teacher" and seen == [["intent"]]
    finally:
        srv.shutdown()
        up.shutdown()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_options_change(folder):
    script = """
const {Shadow, Bundle} = require(process.argv[1]);
(async () => {
  const b = await Bundle.load(process.argv[2]);
  const teacher = async () => "x";
  const add = new Shadow(b, {teacher, question: "intent", options: ["refund", "lost_card", "balance", "transfer"], auditRate: 0});
  const rem = new Shadow(b, {teacher, question: "intent", options: ["refund", "lost_card"], auditRate: 0});
  const ren = new Shadow(b, {teacher, question: "intent", options: ["refund", "card_lost", "balance"],
                             rename: {lost_card: "card_lost"}, auditRate: 0});
  const a = await add.decide("hi block my card thanks");
  const r = await rem.decide("hi what is my balance thanks");
  const n = await ren.decide("hi block my card thanks");
  console.log(JSON.stringify({added: add.optionsAdded, a: [a.source, a.flag], removed: rem.optionsRemoved, r: [r.source, r.flag],
                              n: [n.source, n.answer], renAdded: ren.optionsAdded}));
})().catch((e) => { console.error(e); process.exit(1); });
"""
    r = subprocess.run(["node", "-e", script, os.path.join(ROOT, "js", "index.js"), str(folder / "bundle")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["added"] == ["transfer"] and out["a"] == ["teacher", "options_changed"]
    assert out["removed"] == ["balance"] and out["r"] == ["teacher", "option_removed"]
    assert out["n"] == ["table", "card_lost"] and out["renAdded"] == []
