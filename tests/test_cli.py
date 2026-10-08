"""The command line: train, try, stats, config, doctor, friendly errors."""
import json
import os
import subprocess
import sys

import pytest

from .mock_llm import MockLLM
from .test_shadow import synth


def run(*args, env=None, cwd=None):
    e = {**os.environ, **(env or {})}
    e.pop("SHAD0W_CONFIG", None)
    return subprocess.run([sys.executable, "-m", "shad0w", *args], capture_output=True, text=True, env=e, cwd=cwd, timeout=300)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    log = d / "log.jsonl"
    rows, _ = synth(1200, 41, teacher_noise=0.0)
    log.write_text("".join(json.dumps({"text": r["text"], "intent": r["intent"], "source": "teacher"}) + "\n" for r in rows))
    out = d / "bundle"
    r = run("train", "--log", str(log), "--out", str(out))
    assert r.returncode == 0, r.stderr
    return d, log, out


def test_train_try_stats(trained):
    d, log, out = trained
    r = run("try", "--bundle", str(out), "hi block my card thanks", "zebra lettuce")
    assert r.returncode == 0 and ("table" in r.stdout or "your LLM" in r.stdout) and "needs" in r.stdout
    r = run("stats", "--log", str(log))
    assert r.returncode == 0 and "ready to train" in r.stdout
    r = run("report", "--bundle", str(out))
    assert r.returncode == 0 and json.loads(r.stdout)["questions"]["intent"]["calibration"] == "held-out-split"
    r = run("train", "--log", str(log), "--out", str(out), "--gate")
    assert r.returncode in (0, 3), r.stderr


def test_friendly_errors(tmp_path):
    r = run("try", "--bundle", str(tmp_path / "nope"), "hi")
    assert r.returncode == 2 and "Traceback" not in r.stderr and "no bundle" in r.stderr
    r = run("stats", "--log", str(tmp_path / "missing.jsonl"))
    assert r.returncode == 2 and "Traceback" not in r.stderr and "nothing logged" in r.stderr
    r = run("serve", "--bundle", str(tmp_path / "nope"))
    assert r.returncode == 2 and "Traceback" not in r.stderr
    r = run("train", "--log", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "b"))
    assert r.returncode == 2 and "Traceback" not in r.stderr
    code = ("import sys; sys.modules['sklearn'] = None; sys.modules['scipy'] = None; from shad0w.__main__ import main; "
            f"sys.exit(main(['train', '--log', {str(tmp_path / 'l.jsonl')!r}, '--out', {str(tmp_path / 'b')!r}, '--min-rows', '100']))")
    rows, _ = synth(150, 42, teacher_noise=0.0)
    (tmp_path / "l.jsonl").write_text("".join(json.dumps({"text": r["text"], "intent": r["intent"]}) + "\n" for r in rows))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "shad0wllm[compile]" in r.stderr and "Traceback" not in r.stderr


def test_config_command(tmp_path):
    r = run("config", "--explain", env={"SHAD0W_ALPHA": "0.02"}, cwd=str(tmp_path))
    assert r.returncode == 0 and "env:SHAD0W_ALPHA" in r.stdout and "default" in r.stdout and "0.02" in r.stdout
    r = run("config", "--init", cwd=str(tmp_path))
    assert r.returncode == 0 and (tmp_path / "shad0w.toml").exists()
    assert run("config", "--init", cwd=str(tmp_path)).returncode == 2  # never overwrites
    (tmp_path / "shad0w.toml").write_text('mode = "shadow"\n[questions.intent]\ncanary = 0.25\n')
    r = run("config", "--question", "intent", cwd=str(tmp_path))
    assert "shadow" in r.stdout and "0.25" in r.stdout and "[questions.intent]" in r.stdout


def test_doctor(trained, tmp_path):
    d, log, out = trained
    mock = MockLLM()
    try:
        r = run("doctor", "--bundle", str(out), "--log", str(log), "--upstream", mock.url, "--llm", "ollama/x", cwd=str(tmp_path))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "C core" in r.stdout and "upstream reachable" in r.stdout and "bundle question 'intent'" in r.stdout
        r = run("doctor", "--bundle", str(tmp_path / "nope"), "--llm", "openai/gpt-6-luna",
                env={"OPENAI_API_KEY": ""}, cwd=str(tmp_path))
        assert r.returncode == 1 and "[!!]" in r.stdout
    finally:
        mock.close()


def test_calibrate_cli(trained, tmp_path):
    d, log, out = trained
    rows, gold = synth(300, 43, teacher_noise=0.0)
    data = tmp_path / "labels.jsonl"
    data.write_text("".join(json.dumps({"text": r["text"], "intent": g}) + "\n" for r, g in zip(rows, gold)))
    import shutil
    b = tmp_path / "b"
    shutil.copytree(out, b)
    r = run("calibrate", "--bundle", str(b), "--data", str(data))
    assert r.returncode == 0 and "threshold" in r.stdout
