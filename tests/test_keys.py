"""0.3.2 API keys: explicit api_key / api_key_env / configure(), precedence, never leaking (repr, vars, pickle, doctor,
config, errors), rotating keys, clear TypeErrors, shad0w.toml rules, and the proxy keeping each client's key."""
import json
import os
import pickle
import subprocess
import sys
import threading
import urllib.request

import pytest

import shad0w
from shad0w import config
from shad0w.config import Secret, redact
from shad0w.llm import resolve_key
from shad0w.proxy import Gateway, key_id, proxy

from .mock_llm import MockLLM
from .test_shadow import SCHEMA, VOCAB

KEY = "sk-test-abcdefghijkl3f9a"
OPTS = list(VOCAB)


@pytest.fixture
def mock():
    m = MockLLM()
    yield m
    m.close()


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("OPENAI_API_KEY", "SHAD0W_BASE_URL", "SHAD0W_LLM", "SHAD0W_API_KEY_ENV", "SHAD0W_CONFIG", "MY_KEY"):
        monkeypatch.delenv(k, raising=False)
    shad0w.configure()
    yield
    shad0w.configure()


def auth(m, i=-1):
    return m.requests[i]["headers"].get("authorization")


def test_explicit_api_key_reaches_the_server_and_never_prints(tmp_path, mock):
    d = shad0w.decision("intent", options=OPTS, llm="openai/gpt-6-luna", api_key=KEY, base_url=mock.url,
                        folder=str(tmp_path / "i"), audit_rate=0)
    assert d("hi block my card thanks").answer == "lost_card"
    assert auth(mock) == f"Bearer {KEY}"
    for shown in (repr(d), str(d), d._repr_html_(), repr(d.teacher), str(d.teacher.api_key), repr(vars(d.teacher)),
                  repr(vars(d)), d.teacher.key_source):
        assert KEY not in shown
    assert "sk-…3f9a" in repr(d) and "sk-…3f9a" in repr(d.teacher)
    assert KEY.encode() not in pickle.dumps(d.teacher)
    assert d.teacher.api_key == KEY and d.teacher.api_key.get() == KEY  # still usable on purpose
    assert KEY not in (tmp_path / "i" / "log.jsonl").read_text()


def test_callable_key_is_called_on_every_request(tmp_path, mock):
    calls = []

    def rotating():
        calls.append(1)
        return f"sk-test-rotating-key-{len(calls):04d}"
    t = shad0w.llm_teacher(OPTS, model="openai/gpt-6-luna", base_url=mock.url, api_key=rotating)
    t("i want a refund")
    t("send money to bob")
    assert len(calls) == 2
    assert auth(mock, -2) == "Bearer sk-test-rotating-key-0001" and auth(mock) == "Bearer sk-test-rotating-key-0002"
    assert "called on every request" in t.key_source and "rotating" not in repr(t)


def test_api_key_env_and_precedence(monkeypatch, mock, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-provider-env-0001")
    monkeypatch.setenv("MY_KEY", "sk-test-my-own-var-0002")
    assert resolve_key("OPENAI_API_KEY").get() == "sk-test-provider-env-0001"
    shad0w.configure(api_key_env="MY_KEY")
    assert resolve_key("OPENAI_API_KEY").get() == "sk-test-my-own-var-0002"           # configure > provider env
    shad0w.configure(api_key="sk-test-configured-0003")
    assert resolve_key("OPENAI_API_KEY").get() == "sk-test-configured-0003"           # configure key > configure env
    assert resolve_key("OPENAI_API_KEY", api_key_env="OPENAI_API_KEY").get() == "sk-test-provider-env-0001"  # explicit env
    assert resolve_key("OPENAI_API_KEY", api_key="sk-test-explicit-0004", api_key_env="MY_KEY").get() == "sk-test-explicit-0004"
    d = shad0w.decision("intent", options=OPTS, llm="openai/gpt-6-luna", base_url=mock.url, folder=str(tmp_path / "a"), audit_rate=0)
    d("i want a refund")
    assert auth(mock) == "Bearer sk-test-configured-0003"
    assert shad0w.configure() == {}
    assert resolve_key("OPENAI_API_KEY").describe().startswith("set via OPENAI_API_KEY")


def test_configure_sets_llm_and_settings(tmp_path, mock):
    out = shad0w.configure(llm="openai/gpt-6-luna", api_key=KEY, base_url=mock.url, audit_rate=0.0)
    assert KEY not in repr(out) and out["llm"] == "openai/gpt-6-luna"
    d = shad0w.decision("intent", options=OPTS, folder=str(tmp_path / "c"))
    assert d.teacher is not None and d.settings.audit_rate == 0.0
    d("send money to my friend")
    assert auth(mock) == f"Bearer {KEY}"
    rows = {k: src for k, _, src in config.explain()}
    assert rows["llm"] == "configure()"
    with pytest.raises(TypeError, match="did you mean 'audit_rate'"):
        shad0w.configure(audit_rat=0.1)
    with pytest.raises(ValueError):
        shad0w.configure(mode="sometimes")


def test_missing_key_message_names_every_way(tmp_path):
    with pytest.raises(ValueError) as e:
        shad0w.decision("intent", options=OPTS, llm="openai/gpt-6-luna", folder=str(tmp_path / "m"))
    msg = str(e.value)
    assert "api_key=" in msg and "api_key_env=" in msg and "shad0w.configure" in msg and "OPENAI_API_KEY" in msg
    with pytest.raises(ValueError, match="MY_KEY is not set"):
        shad0w.decision("intent", options=OPTS, llm="openai/gpt-6-luna", api_key_env="MY_KEY", folder=str(tmp_path / "m"))


def test_llm_keywords_with_a_function_llm_raise(tmp_path):
    for kw in ({"api_key": KEY}, {"base_url": "http://x/v1"}, {"api_key_env": "MY_KEY"}, {"temperature": 0.5}):
        with pytest.raises(TypeError, match="provider/model"):
            shad0w.decision("intent", options=OPTS, llm=lambda t: "refund", folder=str(tmp_path / "f"), **kw)
    with pytest.raises(TypeError, match="provider/model"):
        shad0w.Shadow(None, teacher=lambda t: "refund", api_key=KEY)
    with pytest.raises(TypeError, match="not both"):
        shad0w.Shadow(None, teacher=lambda t: "refund", llm="openai/gpt-6-luna")


def test_unknown_keyword_suggests_the_right_one(tmp_path):
    with pytest.raises(TypeError, match="did you mean 'api_key'"):
        shad0w.decision("intent", options=OPTS, llm="openai/gpt-6-luna", apikey=KEY, folder=str(tmp_path / "u"))
    with pytest.raises(TypeError, match="did you mean 'audit_rate'"):
        shad0w.decision("intent", options=OPTS, llm=lambda t: "refund", auditrate=0.1, folder=str(tmp_path / "u"))
    with pytest.raises(TypeError, match="did you mean 'canary'"):
        shad0w.Shadow(None, teacher=lambda t: "refund", canery=0.5)


def test_shadow_builds_the_llm_teacher_itself(tmp_path, mock):
    sh = shad0w.Shadow(None, llm="openai/gpt-6-luna", question="intent", schema=SCHEMA["intent"], api_key=KEY,
                       base_url=mock.url, log=str(tmp_path / "log.jsonl"), audit_rate=0)
    assert sh.decide("transfer money to savings").answer == "transfer" and auth(mock) == f"Bearer {KEY}"
    with pytest.raises(ValueError, match="options"):
        shad0w.Shadow(None, llm="openai/gpt-6-luna", api_key=KEY)


def test_toml_llm_settings_and_key_line(tmp_path, monkeypatch, mock):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MY_KEY", KEY)
    (tmp_path / "shad0w.toml").write_text(f'llm = "openai/gpt-6-luna"\napi_key_env = "MY_KEY"\nbase_url = "{mock.url}"\n'
                                          "audit_rate = 0.0\n")
    d = shad0w.decision("intent", options=OPTS)
    assert d.teacher.base_url == mock.url and "MY_KEY" in d.teacher.key_source
    d("i lost my card")
    assert auth(mock) == f"Bearer {KEY}"
    monkeypatch.setenv("SHAD0W_API_KEY_ENV", "OTHER")  # env beats the file
    assert config.resolve().api_key_env == "OTHER"
    (tmp_path / "shad0w.toml").write_text('api_key = "sk-test-in-a-file-0001"\n')
    with pytest.raises(ValueError, match="store the env var name in api_key_env, not the key itself"):
        config.resolve()
    (tmp_path / "shad0w.toml").write_text('[questions.intent]\napi_key = "sk-test-in-a-file-0001"\n')
    with pytest.raises(ValueError, match="api_key_env"):
        config.resolve("intent")


def test_secret_masking_and_redaction():
    s = Secret(KEY)
    assert repr(s) == "'sk-…3f9a'" and str(s) == "'sk-…3f9a'" and f"{s}" == "'sk-…3f9a'"
    assert repr(Secret("short")) == "'…'" and Secret(lambda: KEY).hint() == "<function>"
    assert KEY not in redact(f"HTTP 401: Incorrect API key provided: {KEY}", s)
    assert "abcdefghijkl" not in redact("authorization: Bearer abcdefghijklmnop")
    with pytest.raises(TypeError):
        Secret(123)


def run_cli(*args, env=None, cwd=None):
    e = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "SHAD0W_CONFIG")}
    e.update(env or {})
    e["PYTHONIOENCODING"] = "utf-8"  # a Windows pipe defaults to cp1252
    return subprocess.run([sys.executable, "-m", "shad0w", *args], capture_output=True, text=True, env=e, cwd=cwd,
                          timeout=120, encoding="utf-8")


def test_doctor_and_config_show_where_the_key_is_but_not_the_key(tmp_path):
    r = run_cli("doctor", "--llm", "openai/gpt-6-luna", env={"OPENAI_API_KEY": KEY}, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "key: set via OPENAI_API_KEY (sk-…3f9a)" in r.stdout and KEY not in r.stdout + r.stderr
    (tmp_path / "shad0w.toml").write_text('llm = "openai/gpt-6-luna"\napi_key_env = "MY_KEY"\n')
    r = run_cli("config", env={"MY_KEY": KEY}, cwd=str(tmp_path))
    assert "llm key: set via MY_KEY (sk-…3f9a)" in r.stdout and KEY not in r.stdout
    r = run_cli("doctor", cwd=str(tmp_path))  # llm from shad0w.toml, MY_KEY unset
    assert r.returncode == 1 and "not set (MY_KEY is empty)" in r.stdout


# -- proxy ------------------------------------------------------------------------------------------------------------

def _post(base, body, headers):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, dict(r.headers)


def _chat(text):
    return {"model": "mock-1", "messages": [{"role": "user", "content": text}]}


def test_two_proxy_clients_reach_the_upstream_with_their_own_keys(tmp_path, mock):
    srv = proxy(mock.url, port=0, folder=str(tmp_path), audit_rate=0.0, model="mock-1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        a, b = "sk-test-client-aaaa-0001", "sk-test-client-bbbb-0002"
        for i in range(3):
            for key in (a, b):
                st, h = _post(base, _chat(f"zebra lettuce {i}"), {"X-Shad0w-Question": "intent", "Authorization": f"Bearer {key}"})
                assert st == 200 and h["x-shad0w-source"] == "teacher"
                assert auth(mock) == f"Bearer {key}"
        gw = srv.gateway
        sh = gw.shadow("intent")
        ta = gw.teacher_for("intent", sh, "decisions", "gpt-6-luna", a)
        tb = gw.teacher_for("intent", sh, "decisions", "gpt-6-luna", b)
        assert ta is not tb and ta.api_key == a and tb.api_key == b
        assert gw.teacher_for("intent", sh, "decisions", "gpt-6-luna", a) is ta
        assert all(a not in repr(k) and b not in repr(k) for k in gw._teachers)
        assert key_id(a) != key_id(b) and a not in key_id(a)
    finally:
        srv.shutdown()


def test_proxy_api_key_file_is_reread_on_every_request(tmp_path, mock):
    from shad0w.__main__ import _key_file
    p = tmp_path / "key"
    p.write_text("sk-test-from-file-0001\n")
    gw = Gateway(mock.url, folder=str(tmp_path / "d"), api_key=_key_file(str(p)), audit_rate=0.0)
    assert gw.upstream_key() == "sk-test-from-file-0001"
    p.write_text("sk-test-from-file-0002")
    assert gw.upstream_key() == "sk-test-from-file-0002"
    assert "sk-test-from-file" not in repr(gw.api_key) and "file" in gw.api_key.describe()
    with pytest.raises(FileNotFoundError):
        _key_file(str(tmp_path / "missing"))


def test_proxy_flags(tmp_path):
    r = run_cli("proxy", "--api-key-env", "X", "--api-key-file", "y", cwd=str(tmp_path))
    assert r.returncode == 2 and "not both" in r.stderr
    r = run_cli("proxy", "--help")
    assert "--api-key-file" in r.stdout and "--api-key " not in r.stdout


def test_teacher_error_says_why_the_llm_was_asked(tmp_path, mock):
    d = shad0w.decision("intent", options=OPTS, llm="openai/gpt-6-luna", api_key=KEY, base_url=mock.url,
                        folder=str(tmp_path / "e"), audit_rate=0, retries=0)
    with pytest.raises(shad0w.TeacherError) as e:
        d("boom")
    assert "why your LLM was asked: no trained table yet" in str(e.value) and KEY not in str(e.value)


def test_pickle_keeps_an_env_name_and_drops_a_literal_key():
    os.environ["MY_KEY"] = KEY
    try:
        t = shad0w.llm_teacher(OPTS, model="openai/gpt-6-luna", api_key_env="MY_KEY")
        assert pickle.loads(pickle.dumps(t)).api_key.get() == KEY
        t2 = shad0w.llm_teacher(OPTS, model="openai/gpt-6-luna", api_key=KEY)
        assert not pickle.loads(pickle.dumps(t2)).api_key.get()
    finally:
        del os.environ["MY_KEY"]
