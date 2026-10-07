"""The Shadow wrapper: log-only start, certified cascade, audits, decorator, and strict table loading."""
import json
import os
import struct
import subprocess
import sys

import pytest

import shad0w
from shad0w.shadow import shadow_compile

from .test_shadow import SCHEMA, VOCAB, synth


def teacher(text):
    for k, phrases in VOCAB.items():
        if any(p in text for p in phrases):
            return k
    return "balance"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    rows, _ = synth(3000, 11, teacher_noise=0.01)
    m, _ = shadow_compile(SCHEMA, rows, alpha=0.05)
    path = str(tmp_path_factory.mktemp("cascade"))
    m.save(path)
    return path


def test_log_only_mode_writes_teacher_log(tmp_path):
    log = tmp_path / "log.jsonl"
    sh = shad0w.Shadow(None, teacher=teacher, question="intent", log=str(log))
    d = sh.decide("hi block my card thanks")
    assert d.answer == "lost_card" and d.source == "teacher" and d.flag == "no_bundle"
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["text"] == "hi block my card thanks" and row["intent"] == "lost_card" and row["source"] == "teacher"


def test_cascade_serves_certified_and_defers_the_rest(bundle, tmp_path):
    log = tmp_path / "log.jsonl"
    sh = shad0w.Shadow(bundle, teacher=teacher, log=str(log), audit_rate=0.0)
    texts = [r["text"] for r in synth(500, 12)[0]] + ["completely unrelated words about the weather"]
    out = [sh.decide(t) for t in texts]
    sources = {d.source for d in out}
    assert "table" in sources, "a clean teacher should let the table serve some traffic"
    for d in out:
        assert d.certified == (d.source == "table")
        assert d.answer in VOCAB
    s = sh.stats()
    assert s["table"] + s["teacher"] == len(texts) and 0 < s["offload"] <= 1
    logged = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    assert len(logged) == s["teacher"]  # every deferred answer is logged for the next compile


def test_audits_measure_live_disagreement(bundle):
    sh = shad0w.Shadow(bundle, teacher=teacher, audit_rate=1.0, seed=0)
    for r in synth(400, 13)[0]:
        sh.decide(r["text"])
    s = sh.stats()
    assert s["audits"] == s["table"] > 0
    assert 0.0 <= s["audit_disagreement"] <= s["audit_disagreement_upper"] <= 1.0


def test_decorator(bundle):
    @shad0w.cascade(bundle, audit_rate=0.0)
    def classify(text):
        """docstring kept"""
        return teacher(text)

    assert classify("please send money to my friend now") == "transfer"
    assert classify.__doc__ == "docstring kept" and classify.shadow.stats()["table"] + classify.shadow.stats()["teacher"] == 1


def test_bad_arguments(bundle):
    with pytest.raises(TypeError):
        shad0w.Shadow(bundle, teacher="not callable")
    with pytest.raises(ValueError):
        shad0w.Shadow(bundle, teacher=teacher, question="nope")
    with pytest.raises(ValueError):
        shad0w.Shadow(bundle, teacher=teacher, audit_rate=2)


@pytest.mark.parametrize("mutate", ["magic", "version", "truncate", "pad", "shape"])
def test_corrupt_tables_are_refused(bundle, tmp_path, mutate):
    import shutil
    b = tmp_path / "b"
    shutil.copytree(bundle, b)
    p = b / "intent.s0"
    data = bytearray(p.read_bytes())
    if mutate == "magic":
        data[:4] = b"XXXX"
    elif mutate == "version":
        data[4:8] = struct.pack("<I", 99)
    elif mutate == "truncate":
        data = data[:-7]
    elif mutate == "pad":
        data += b"\0" * 3
    elif mutate == "shape":
        data[12:16] = struct.pack("<I", 100000)
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        shad0w.load(str(b), native=False)
    if shad0w.api.__dict__.get("NativeReflex") is None:
        from shad0w.native import NativeReflex, available
        if available():
            with pytest.raises(OSError):
                NativeReflex(str(p))


def test_cli_version_and_init(tmp_path):
    r = subprocess.run([sys.executable, "-m", "shad0w", "--version"], capture_output=True, text=True)
    assert r.stdout.strip() == f"shad0w {shad0w.__version__}"
    r = subprocess.run([sys.executable, "-m", "shad0w", "init", "--dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "schema.json").read_text())["intent"]["type"] == "choice"
    assert os.path.exists(tmp_path / "teacher_log.jsonl")
