"""Running without an LLM (fallback=), the default decision name, and the async path evaluating the policy once."""
import asyncio
import json
import os
import shutil
import subprocess

import pytest

import shad0w
from shad0w.shadow import shadow_compile

from .mock_llm import classify
from .test_shadow import SCHEMA, synth

ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    rows, _ = synth(3000, 51, teacher_noise=0.01)
    m, _ = shadow_compile(SCHEMA, rows, alpha=0.05)
    p = tmp_path_factory.mktemp("fb") / "b"
    m.save(str(p))
    return str(p)


def test_without_an_llm_the_fallback_answers_and_is_never_logged(bundle, tmp_path):
    log = tmp_path / "log.jsonl"
    sh = shad0w.Shadow(bundle, teacher=None, question="intent", log=str(log), fallback="needs_review", audit_rate=0)
    served = sh.decide("hi block my card thanks")
    unsure = sh.decide("zebra lettuce quantum harmonica")
    assert served.source == "table"
    assert unsure.source == "fallback" and unsure.answer == "needs_review" and unsure.certified is False
    assert not log.exists() or not log.read_text().strip()  # fallback answers are not training data
    assert sh.stats()["fallback"] >= 1


def test_callable_fallback_and_no_bundle(tmp_path):
    sh = shad0w.Shadow(None, teacher=None, question="intent", fallback=lambda t: "rule:" + t[:4])
    d = sh.decide("abcdef")
    assert (d.source, d.answer, d.flag) == ("fallback", "rule:abcd", "no_bundle")


def test_no_teacher_and_no_fallback_explains_itself(bundle):
    sh = shad0w.Shadow(bundle, teacher=None, question="intent", audit_rate=0)
    with pytest.raises(RuntimeError, match="fallback"):
        sh.decide("zebra lettuce quantum harmonica")


def test_decision_name_is_optional(tmp_path):
    d = shad0w.decision(options=["a", "b"], llm=lambda t: "a", folder=str(tmp_path / "x"))
    assert d.question == "decision" and d("hello").answer == "a"
    row = json.loads((tmp_path / "x" / "log.jsonl").read_text().splitlines()[0])
    assert row["decision"] == "a"


def test_async_with_a_canary_never_runs_a_sync_teacher_on_the_loop(bundle):
    """canary < 1 draws a random number per decision: adecide must draw it once, then use a worker thread."""
    import threading
    loop_threads = set()

    def teacher(text):
        loop_threads.add(threading.current_thread() is threading.main_thread())
        return classify(text)

    sh = shad0w.Shadow(bundle, teacher=teacher, question="intent", canary=0.5, audit_rate=0, seed=1)

    async def run():
        return [await sh.adecide("hi block my card thanks") for _ in range(60)]

    out = asyncio.run(run())
    assert {d.source for d in out} == {"table", "teacher"}
    assert loop_threads == {False}  # every teacher call ran off the event-loop thread


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_decision_without_a_name():
    script = """
const {decision} = require(process.argv[1]);
(async () => { const d = await decision({options: ["a", "b"], llm: async () => "a", log: () => {}});
  const r = await d.decide("hello"); console.log(JSON.stringify([d.question, r.answer, r.source])); })();
"""
    r = subprocess.run(["node", "-e", script, os.path.join(ROOT, "js", "index.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == ["decision", "a", "teacher"]
