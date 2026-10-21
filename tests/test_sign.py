"""Signed certificates: HMAC-SHA256 over the canonical certificate.json, keyed by $SHAD0W_CERT_KEY, checked by
`shad0w verify` together with the bundle's teacher."""
import json
import os
import subprocess
import sys

import pytest

from shad0w.shadow import shadow_compile, write_certificate
from shad0w.sign import KEY_ENV, SIG_FILE, SignatureError, canonical_json, sign_bundle, verify_bundle

from .test_shadow import SCHEMA, synth

KEY = "a-test-key-that-is-long-enough-0123456789"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    d = tmp_path_factory.mktemp("sig")
    rows, _ = synth(600, 5, teacher_noise=0.01)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05, teacher="mock-teacher-v1")
    m.save(str(d / "bundle"))
    write_certificate(str(d / "bundle"), cert)
    return str(d / "bundle")


def run(*args, env=None):
    e = {**os.environ, **(env or {})}
    e.pop(KEY_ENV, None)
    e.update(env or {})
    return subprocess.run([sys.executable, "-m", "shad0w", *args], capture_output=True, text=True, env=e, timeout=120)


def test_canonical_json_is_order_and_whitespace_free():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1}) == b'{"a":[1,2],"b":1}'
    assert canonical_json({"s": "é"}) == '{"s":"é"}'.encode()


def test_sign_and_verify(bundle, monkeypatch):
    monkeypatch.setenv(KEY_ENV, KEY)
    sig = sign_bundle(bundle)
    assert sig["algorithm"] == "hmac-sha256" and len(sig["signature"]) == 64 and sig["teacher"] == "mock-teacher-v1"
    assert os.path.exists(os.path.join(bundle, SIG_FILE))
    rep = verify_bundle(bundle)
    assert rep["ok"] and rep["teacher"] == "mock-teacher-v1" and rep["key_id"] == sig["key_id"]
    assert verify_bundle(bundle, teacher="mock-teacher-v1")["ok"]
    with pytest.raises(SignatureError, match="expected 'other'"):
        verify_bundle(bundle, teacher="other")
    # re-indenting the certificate does not break the signature; changing a value does
    p = os.path.join(bundle, "certificate.json")
    cert = json.load(open(p))
    json.dump(cert, open(p, "w"), indent=None, sort_keys=False)
    assert verify_bundle(bundle)["ok"]
    cert["alpha"] = 0.5
    json.dump(cert, open(p, "w"), indent=2)
    with pytest.raises(SignatureError, match="does not match"):
        verify_bundle(bundle)
    cert["alpha"] = 0.05
    json.dump(cert, open(p, "w"), indent=2)
    assert verify_bundle(bundle)["ok"]


def test_wrong_or_missing_key(bundle, monkeypatch):
    monkeypatch.setenv(KEY_ENV, KEY)
    sign_bundle(bundle)
    with pytest.raises(SignatureError, match="does not match"):
        verify_bundle(bundle, key="another-key-that-is-also-long-enough")
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=KEY_ENV):
        sign_bundle(bundle)
    with pytest.raises(ValueError, match=KEY_ENV):
        verify_bundle(bundle)
    with pytest.raises(ValueError, match="too short"):
        sign_bundle(bundle, key="short")


def test_manifest_teacher_must_match_certificate(bundle, monkeypatch, tmp_path):
    import shutil
    monkeypatch.setenv(KEY_ENV, KEY)
    b = str(tmp_path / "b")
    shutil.copytree(bundle, b)
    man_p = os.path.join(b, "manifest.json")
    man = json.load(open(man_p))
    man["meta"]["teacher"] = "someone-else"
    json.dump(man, open(man_p, "w"))
    sign_bundle(b)
    with pytest.raises(SignatureError, match="compiled from 'someone-else'"):
        verify_bundle(b)


def test_cli_sign_verify(bundle, tmp_path):
    import shutil
    b = str(tmp_path / "cli")
    shutil.copytree(bundle, b)
    if os.path.exists(os.path.join(b, SIG_FILE)):
        os.remove(os.path.join(b, SIG_FILE))
    r = run("verify", "--bundle", b, env={KEY_ENV: KEY})
    assert r.returncode == 1 and "unsigned" in r.stderr
    r = run("sign", "--bundle", b)
    assert r.returncode == 2 and KEY_ENV in r.stderr  # no key, no default
    r = run("sign", "--bundle", b, env={KEY_ENV: KEY})
    assert r.returncode == 0 and json.loads(r.stdout)["algorithm"] == "hmac-sha256"
    r = run("verify", "--bundle", b, "--teacher", "mock-teacher-v1", env={KEY_ENV: KEY})
    assert r.returncode == 0 and r.stdout.startswith("ok:")
    r = run("verify", "--bundle", b, "--teacher", "gpt-6-luna", env={KEY_ENV: KEY})
    assert r.returncode == 1 and "expected 'gpt-6-luna'" in r.stderr
    # certify --sign re-signs the fresh certificate
    rows, _ = synth(300, 9, teacher_noise=0.01)
    data = tmp_path / "fresh.jsonl"
    data.write_text("".join(json.dumps({"text": r["text"], "intent": r["intent"]}) + "\n" for r in rows))
    r = run("certify", "--bundle", b, "--data", str(data), "--sign", env={KEY_ENV: KEY})
    assert r.returncode == 0 and "signed ->" in r.stderr and json.loads(r.stdout)["teacher"] == "mock-teacher-v1"
    r = run("verify", "--bundle", b, env={KEY_ENV: KEY})
    assert r.returncode == 0
