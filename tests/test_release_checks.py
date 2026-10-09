"""Pre-release audit regressions: tied confidences, narrowed options, non-finite tables, tiny logs, odd unicode."""
import glob
import json

import numpy as np
import pytest

from shad0w.api import _load_reflex
from shad0w.reliability import SelectiveRiskController
from shad0w.shadow import shadow_compile

from .test_shadow import SCHEMA, VOCAB, synth

OPTS = list(VOCAB)


@pytest.fixture(scope="module")
def model():
    rows, _ = synth(1500, 3)
    m, _ = shadow_compile(SCHEMA, rows)
    return m


@pytest.mark.parametrize("procedure", ["fixed-sequence", "bonferroni", "auto"])
def test_tied_confidences_are_counted_in_the_test(procedure):
    """500 clean distinct rows plus 500 rows tied at 0.8 with 30% disagreement: a threshold at 0.8 serves the whole
    tie, so it may only be chosen if the tie passes the bound (it does not)."""
    viol = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        conf = np.r_[rng.uniform(0.9, 1.0, 500), np.full(500, 0.8)]
        ok = np.r_[np.ones(500, bool), rng.random(500) > 0.3]
        acc = SelectiveRiskController(0.05, 0.1, procedure).fit(conf, ok).accept(conf)
        viol += bool(acc.any() and (~ok[acc]).mean() > 0.05)
    assert viol <= 10


def test_narrowed_options_never_certify_beyond_the_full_answer(model):
    q = model.questions["intent"]
    rows, _ = synth(200, 9)
    for r in rows:
        full = q.decide(r["text"], observe=False)
        for drop in OPTS:
            nar = q.decide(r["text"], options=[o for o in OPTS if o != drop], observe=False)
            if nar["certified"]:
                assert full["certified"] and nar["choice"] == full["choice"]
                assert nar["confidence"] == full["confidence"]
    none = q.decide(rows[0]["text"], options=["not-an-option"], observe=False)
    assert none["certified"] is False and none["flag"] == "options_changed"


def test_non_finite_tables_are_refused(model, tmp_path):
    model.save(str(tmp_path))
    path = glob.glob(str(tmp_path / "**" / "*.s0"), recursive=True)[0]
    data = bytearray(open(path, "rb").read())
    data[-4:] = np.float32(np.nan).tobytes()  # last entry of G
    open(path, "wb").write(bytes(data))
    with pytest.raises(ValueError, match="non-finite"):
        _load_reflex(path)
    from shad0w import native
    if native.available():
        with pytest.raises(OSError):
            native.NativeReflex(path)


@pytest.mark.parametrize("n", [100, 101, 120])
def test_tiny_logs_compile(n):
    rows, _ = synth(n, 2)
    m, cert = shadow_compile(SCHEMA, rows)
    assert cert["questions"]["intent"]["n_fit"] > 0


def test_lone_surrogates_do_not_crash(model):
    assert model.questions["intent"].decide("refund \ud800 please", observe=False)["choice"] in OPTS


# -- runtime ---------------------------------------------------------------------------------------------------------

def _trained(tmp_path, **kw):
    import shad0w
    d = shad0w.decision("intent", options=OPTS, folder=str(tmp_path / "intent"), llm=lambda t: synth_label(t), audit_rate=0, **kw)
    rows, _ = synth(1500, 5, teacher_noise=0.0)
    d.warm_start([{"text": r["text"], "intent": r["intent"]} for r in rows])
    d.train()
    return d


def synth_label(text):
    for k, words in VOCAB.items():
        if any(w in text for w in words):
            return k
    return OPTS[0]


def test_function_teacher_answers_are_checked_before_logging(tmp_path):
    import enum

    import shad0w
    d = shad0w.decision("g", options=["a", "b"], folder=str(tmp_path / "g"), llm=lambda t: "zzz", audit_rate=0)
    d("x")
    assert d.log_rows() == 0  # "zzz" matches no option: never becomes training data

    class Intent(enum.Enum):
        a = 1
        b = 2
    e = shad0w.decision("e", options=Intent, folder=str(tmp_path / "e"), llm=lambda t: Intent.a, audit_rate=0)
    e("x")
    assert json.loads(open(e.log).read())["e"] == "a"

    async def anext_(t):
        return "a"
    s = shad0w.decision("s", options=["a", "b"], folder=str(tmp_path / "s"), llm=anext_, audit_rate=0)
    with pytest.raises(TypeError, match="adecide"):
        s.decide("x")


def test_never_serve_matches_yes_no_answers(tmp_path):
    import shad0w
    d = shad0w.decision("spam", options=bool, folder=str(tmp_path / "spam"), llm=lambda t: "refund" in t, audit_rate=0,
                        never_serve=["yes"])
    rows, _ = synth(1500, 6, teacher_noise=0.0)
    d.warm_start([{"text": r["text"], "spam": r["intent"] == "refund"} for r in rows])
    d.train()
    served_yes = [x for x in (d(r["text"]) for r in rows[:300]) if x.source == "table" and x.answer is True]
    assert not served_yes


def test_new_options_served_are_not_called_certified(tmp_path):
    import shad0w
    _trained(tmp_path)
    d2 = shad0w.decision("intent", options=OPTS + ["fraud"], folder=str(tmp_path / "intent"), llm=synth_label,
                         audit_rate=0, on_new_option="serve")
    out = [d2(r["text"]) for r in synth(200, 7)[0]]
    table = [x for x in out if x.source == "table"]
    assert table and all(not x.certified and x.flag == "new_options_served" for x in table)


def test_record_without_a_teacher_never_claims_uniform_audit_rows(tmp_path):
    import shad0w
    d = shad0w.decision("intent", options=OPTS, folder=str(tmp_path / "r"), audit_rate=1.0)
    for r in synth(50, 8)[0]:
        d.record(r["text"], r["intent"])
    assert {json.loads(x)["source"] for x in open(d.log)} == {"teacher"}


def test_settings_reject_bad_shapes(monkeypatch):
    from shad0w import config
    with pytest.raises(ValueError, match="auto_train"):
        config.resolve(auto_train=True)
    with pytest.raises(ValueError, match="timeout"):
        config.resolve(timeout=-1)
    assert config.resolve(exposed="no").exposed is False
    monkeypatch.setenv("SHAD0W_ALPHA", "abc")
    with pytest.raises(ValueError, match="SHAD0W_ALPHA"):
        config.resolve()


def test_certify_refuses_tiny_sets(tmp_path):
    from shad0w.__main__ import main
    d = _trained(tmp_path)
    p = tmp_path / "few.jsonl"
    p.write_text("\n".join(json.dumps({"text": r["text"], "intent": r["intent"]}) for r in synth(5, 9)[0]))
    assert main(["certify", "--bundle", d.bundle_path, "--data", str(p)]) == 2


def test_proxy_spot_checks_keep_their_own_key(tmp_path):
    """Keep-alive: a chat spot check runs after the connection already parsed the next request; it must still go
    to /chat/completions with the FIRST request's key."""
    import http.client
    import importlib
    import threading
    import time
    cascade = importlib.import_module("shad0w.cascade")
    from shad0w.proxy import proxy

    from .mock_llm import MockLLM
    rows, _ = synth(3000, 21, teacher_noise=0.01)
    m, cert = shadow_compile(SCHEMA, rows)
    from shad0w.shadow import write_certificate
    m.save(str(tmp_path / "intent" / "bundle"))
    write_certificate(str(tmp_path / "intent" / "bundle"), cert)
    mock = MockLLM()
    srv = proxy(mock.url, port=0, folder=str(tmp_path), audit_rate=1.0, model="mock-1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    orig = cascade.Shadow._audit

    def slow(self, *a):
        time.sleep(0.3)
        return orig(self, *a)
    cascade.Shadow._audit = slow
    try:
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
        chat = {"model": "mock-1", "messages": [{"role": "user", "content": "hi block my card thanks"}]}
        c.request("POST", "/v1/chat/completions", json.dumps(chat), {"authorization": "Bearer sk-AAAA1111AAAA", "x-shad0w-question": "intent",
                                                                      "content-type": "application/json"})
        c.getresponse().read()
        c.request("POST", "/v1/models", b"{}", {"authorization": "Bearer sk-BBBB2222BBBB", "content-type": "application/json"})
        c.getresponse().read()
        time.sleep(1.0)
    finally:
        cascade.Shadow._audit = orig
        srv.shutdown()
        mock.close()
    spot = [r for r in mock.requests if r["body"].get("messages") == chat["messages"]]
    assert spot and all(r["headers"].get("authorization") == "Bearer sk-AAAA1111AAAA" for r in spot)


# -- second review ---------------------------------------------------------------------------------------------------

def test_renamed_and_numeric_answers_are_still_logged(tmp_path):
    import shad0w
    d = shad0w.decision("intent", options=["money_back", "lost_card", "balance", "transfer"], folder=str(tmp_path / "r"),
                        llm=lambda t: "refund", audit_rate=0, rename={"refund": "money_back"})
    d("x")
    assert json.loads(open(d.log).read())["intent"] == "refund"  # the old name; training maps it to the new one
    y = shad0w.decision("spam", options=bool, folder=str(tmp_path / "y"), llm=lambda t: 1, audit_rate=0)
    y("x")
    assert json.loads(open(y.log).read())["spam"] is True


def test_served_but_uncovered_answers_are_spot_checked(tmp_path):
    import shad0w
    _trained(tmp_path)
    d2 = shad0w.decision("intent", options=OPTS + ["fraud"], folder=str(tmp_path / "intent"), llm=synth_label,
                         audit_rate=1.0, on_new_option="serve")
    out = [d2(r["text"]) for r in synth(40, 11)[0]]
    d2.flush(5)
    assert any(x.source == "table" for x in out)
    assert any(json.loads(line)["source"] == "audit" for line in open(d2.log))


def test_record_can_keep_audit_rows_uniform(tmp_path):
    import shad0w
    d = shad0w.decision("intent", options=OPTS, folder=str(tmp_path / "p"), audit_rate=1.0)
    d.record("hi", "refund", source=d._src())
    assert json.loads(open(d.log).read())["source"] == "audit"


def test_proxy_refuses_malformed_chat_bodies_with_400(tmp_path):
    import threading
    import urllib.error
    import urllib.request

    from shad0w.proxy import proxy

    from .mock_llm import MockLLM
    mock = MockLLM()
    srv = proxy(mock.url, port=0, folder=str(tmp_path), audit_rate=0.0, model="mock-1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for body in ({"messages": "str"}, {"messages": ["str"]}, {"messages": [{"role": "user", "content": "x"}], "tools": ["x"]}):
            req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json", "x-shad0w-question": "intent"})
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req, timeout=5)
            assert e.value.code == 400
    finally:
        srv.shutdown()
        mock.close()


# -- fresh-install review ----------------------------------------------------------------------------------------------

def test_empty_input_is_never_served(model):
    for t in ("", "   ", "\n"):
        r = model.questions["intent"].decide(t, observe=False)
        assert r["certified"] is False and r["flag"] == "empty_input"


def test_unknown_toml_keys_are_reported(tmp_path, caplog):
    from shad0w import config
    p = tmp_path / "shad0w.toml"
    p.write_text("alpah = 0.1\n")
    with caplog.at_level("WARNING", logger="shad0w"):
        config.resolve(path=str(p))
    assert "alpah" in caplog.text and "alpha" in caplog.text
    p.write_text("alpha = \n")
    with pytest.raises(ValueError, match="shad0w.toml"):
        config.resolve(path=str(p))


def test_proxy_cli_checks_bundle_paths(tmp_path):
    from shad0w.__main__ import main
    with pytest.raises(SystemExit):
        main(["proxy", "--upstream", "http://127.0.0.1:9/v1", "--bundle", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["proxy", "--upstream", "http://127.0.0.1:9/v1", "--bundle", f"intent={tmp_path}"])
