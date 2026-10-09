"""Access tokens for `shad0w proxy` / `shad0w serve`: every route but /v1/health needs the token when one is set, the
header that carried it never goes upstream, and a non-local proxy holding its own LLM key refuses to start without one."""
import base64
import json
import os
import subprocess
import sys
import threading

import pytest

from shad0w import access
from shad0w.config import Secret
from shad0w.proxy import proxy
from shad0w.server import serve
from shad0w.shadow import shadow_compile

from .mock_llm import MockLLM
from .test_proxy import call, chat
from .test_shadow import SCHEMA, synth

TOKEN = "proxy-token-1234"
KEY = "sk-test-proxy-own-key-9999"


@pytest.fixture
def mock():
    m = MockLLM()
    yield m
    m.close()


def start(tmp_path, mock, **kw):
    srv = proxy(mock.url, port=0, folder=str(tmp_path / "d"), audit_rate=0.0, model="mock-1", **kw)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv


def basic(password, user="me"):
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def test_is_local():
    assert access.is_local("127.0.0.1") and access.is_local("localhost") and access.is_local("::1")
    assert access.is_local("127.0.0.5")
    assert not access.is_local("0.0.0.0") and not access.is_local("") and not access.is_local("::")
    assert not access.is_local("192.168.1.4") and not access.is_local("example.com")


def test_an_own_key_open_to_the_network_is_refused(tmp_path, mock, capsys):
    with pytest.raises(ValueError, match="refusing to listen on 0.0.0.0"):
        proxy(mock.url, host="0.0.0.0", port=0, folder=str(tmp_path / "a"), api_key=KEY)
    srv = proxy(mock.url, host="0.0.0.0", port=0, folder=str(tmp_path / "b"), api_key=KEY, allow_open=True)
    srv.server_close()
    assert "WARNING listening on 0.0.0.0" in capsys.readouterr().err
    srv = proxy(mock.url, host="0.0.0.0", port=0, folder=str(tmp_path / "c"), api_key=KEY, access_token=TOKEN)
    srv.server_close()
    assert "WARNING" not in capsys.readouterr().err
    srv = proxy(mock.url, host="0.0.0.0", port=0, folder=str(tmp_path / "e"))  # clients bring their own keys: a warning
    srv.server_close()
    out = capsys.readouterr()
    assert "WARNING listening on 0.0.0.0" in out.err and "access token: none" in out.out


def test_every_route_but_health_needs_the_token(tmp_path, mock):
    base, srv = start(tmp_path, mock, access_token=TOKEN)
    try:
        assert call(base, "/v1/health")[0] == 200
        for path, body in (("/", None), ("/v1/stats", None), ("/metrics", None), ("/v1/models", None),
                           ("/v1/chat/completions", chat("hello")), ("/v1/playground", {"text": "x"})):
            st, h, b = call(base, path, body)
            assert st == 401, path
            assert h["www-authenticate"].startswith("Basic") and json.loads(b)["error"]["code"] == "invalid_access_token"
            assert call(base, path, body, {"X-Shad0w-Token": "wrong"})[0] == 401
        assert mock.requests == []  # nothing reached the LLM
        assert call(base, "/v1/stats", headers={"X-Shad0w-Token": TOKEN})[0] == 200
        assert call(base, "/", headers={"Authorization": basic(TOKEN)})[0] == 200  # a browser logging in
        assert call(base, "/", headers={"Authorization": basic("nope")})[0] == 401
    finally:
        srv.shutdown()


def test_the_token_header_never_goes_upstream(tmp_path, mock):
    # clients bring their own keys: the token travels in X-Shad0w-Token, the client's key is forwarded
    base, srv = start(tmp_path, mock, access_token=TOKEN)
    try:
        st, *_ = call(base, "/v1/chat/completions", chat("hello"), {"X-Shad0w-Token": TOKEN, "Authorization": "Bearer sk-client-1"})
        h = mock.requests[-1]["headers"]
        assert st == 200 and h["authorization"] == "Bearer sk-client-1" and "x-shad0w-token" not in h
    finally:
        srv.shutdown()
    # the proxy holds the key: an OpenAI SDK sends the token as its api_key, and the upstream sees only the real key
    base, srv = start(tmp_path, mock, access_token=TOKEN, api_key=KEY)
    try:
        st, *_ = call(base, "/v1/chat/completions", chat("hello"), {"Authorization": f"Bearer {TOKEN}"})
        h = mock.requests[-1]["headers"]
        assert st == 200 and h["authorization"] == f"Bearer {KEY}" and TOKEN not in json.dumps(h)
        st, *_ = call(base, "/v1/chat/completions", chat("hello"), {"Authorization": f"Bearer {KEY}"})  # the key is not the token
        assert st == 401
    finally:
        srv.shutdown()


def test_token_from_env_file_and_rotation(tmp_path, monkeypatch):
    monkeypatch.delenv(access.TOKEN_ENV, raising=False)
    assert access.token_secret() is None
    monkeypatch.setenv(access.TOKEN_ENV, TOKEN)
    assert access.token_secret().get() == TOKEN and TOKEN not in repr(access.token_secret())
    with pytest.raises(ValueError, match="NOPE_TOKEN is not set"):
        access.token_secret(env="NOPE_TOKEN")
    p = tmp_path / "tok"
    p.write_text("first\n")
    s = access.token_secret(file=str(p))
    assert access.authorize({"x-shad0w-token": "first"}, "/v1/stats", s)[0]
    p.write_text("second")  # a mounted secret rotated in place
    assert not access.authorize({"x-shad0w-token": "first"}, "/v1/stats", s)[0]
    assert access.authorize({"x-shad0w-token": "second"}, "/v1/stats", s)[0]
    p.write_text("")  # emptied: fail closed
    assert not access.authorize({"x-shad0w-token": ""}, "/v1/stats", s)[0]
    assert access.authorize({}, "/v1/stats", None) == (True, None)
    assert access.authorize({"authorization": "Basic !!!"}, "/", Secret("t"))[0] is False


def test_serve_needs_the_token_too(tmp_path):
    rows, _ = synth(1500, 5, teacher_noise=0.01)
    m, _ = shadow_compile(SCHEMA, rows, alpha=0.05)
    m.save(str(tmp_path / "b"))
    srv = serve(str(tmp_path / "b"), port=0, run=False, access_token=TOKEN)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert call(base, "/v1/health")[0] == 200
        assert call(base, "/v1/decide", {"state": "my card was stolen"})[0] == 401
        assert call(base, "/v1/decide", {"state": "my card was stolen"}, {"X-Shad0w-Token": TOKEN})[0] == 200
    finally:
        srv.shutdown()


def test_cli_refuses_an_open_key_and_explains(tmp_path):
    env = {**os.environ, "OWN_KEY": KEY, "PYTHONIOENCODING": "utf-8"}
    env.pop(access.TOKEN_ENV, None)
    r = subprocess.run([sys.executable, "-m", "shad0w", "proxy", "--host", "0.0.0.0", "--port", "0", "--api-key-env", "OWN_KEY"],
                       capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(tmp_path), timeout=60)
    assert r.returncode == 2 and "refusing to listen on 0.0.0.0" in r.stderr and KEY not in r.stderr + r.stdout
    r = subprocess.run([sys.executable, "-m", "shad0w", "proxy", "--token-env", "A", "--token-file", "b"],
                       capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(tmp_path), timeout=60)
    assert r.returncode == 2 and "not both" in r.stderr
