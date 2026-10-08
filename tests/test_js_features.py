"""The 0.3 JavaScript features: explain/peek/record, timeouts, the AI SDK middleware and decision model, the System One
and Decisions API teachers, and bundle-load errors (skipped without node)."""
import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from shad0w.shadow import shadow_compile

from .mock_llm import classify
from .test_shadow import SCHEMA, synth

ROOT = os.path.dirname(os.path.dirname(__file__))
INDEX = os.path.join(ROOT, "js", "index.js")
OPTIONS = ["refund", "lost_card", "balance", "transfer"]
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    rows, _ = synth(3000, 41, teacher_noise=0.01)
    m, _ = shadow_compile(SCHEMA, rows, alpha=0.05)
    out = tmp_path_factory.mktemp("js") / "b"
    m.save(str(out))
    return str(out)


def node(script, *args, timeout=60):
    r = subprocess.run(["node", "-e", script, INDEX, *map(str, args)], capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


class DecisionServer:
    """A tiny server speaking both /v1/systemone (TypeSafe shape) and /v1/decisions (OpenAI shape)."""

    def __init__(self):
        self.requests = []
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["content-length"])))
                srv.requests.append({"path": self.path, "body": req, "auth": self.headers.get("authorization")})
                if self.path.endswith("/systemone"):
                    answers = {}
                    for name, q in req["questions"].items():
                        if q["type"] == "noul":
                            answers[name] = {"type": "noul", "answer": "urgent" in req["state"], "probability": 0.9 if "urgent" in req["state"] else 0.1}
                        else:
                            answers[name] = {"type": "choice", "choice": classify(req["state"]), "confidence": 0.9}
                    return self._send(200, {"model": req.get("model"), "answers": answers, "latency_ms": 1})
                if self.path.endswith("/decisions"):
                    answers = []
                    for q in req["questions"]:
                        if q["type"] == "predicate":
                            answers.append({"type": "predicate", "name": q.get("name"), "probability": 0.9 if "urgent" in req["input"] else 0.1})
                        else:
                            ans = classify(req["input"])
                            values = [c["value"] for c in q["choices"]]
                            answers.append({"type": "choice", "name": q.get("name"), "choice": ans, "confidence": 0.9,
                                            "probabilities": [{"value": v, "probability": 0.9 if v == ans else 0.1 / max(1, len(values) - 1)} for v in values]})
                    return self._send(200, {"model": req.get("model"), "answers": answers,
                                            "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}})
                self._send(404, {"error": "nf"})

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@needs_node
def test_explain_peek_record_and_threshold(bundle):
    out = node("""
const {Shadow, Bundle, explainFlag} = require(process.argv[1]);
(async () => {
  const rows = [];
  const b = await Bundle.load(process.argv[2]);
  const sh = new Shadow(b, {question: "intent", log: (r) => rows.push(r), auditRate: 0});
  const ex = sh.explain("hi block my card thanks");
  const peekYes = await sh.peek("hi block my card thanks");
  const peekNo = await sh.peek("weather in paris tomorrow");
  const rec = await sh.record("weather in paris tomorrow", "balance");
  const none = new Shadow(null, {question: "intent"});
  console.log(JSON.stringify({ex, peekYes, peekNo, rec, rows, stats: sh.stats(), noneEx: none.explain("x"),
    why: explainFlag("canary"), opts: sh.options(), thr: sh.threshold()}));
})().catch(e => { console.error(e); process.exit(1); });
""", bundle)
    ex = out["ex"]
    assert set(ex) == {"text", "answer", "confidence", "threshold", "certified", "flag", "why", "top"}
    assert ex["answer"] == "lost_card" and ex["certified"] is True and ex["flag"] is None
    assert ex["threshold"] == out["thr"] and 0 < ex["threshold"] <= 1 and ex["confidence"] >= ex["threshold"]
    assert ex["why"].startswith("certified") and ex["top"][0][0] == "lost_card" and len(ex["top"]) == 3
    assert out["peekYes"]["source"] == "table" and out["peekYes"]["answer"] == "lost_card" and out["peekYes"]["question"] == "intent"
    assert abs(sum(out["peekYes"]["probabilities"].values()) - 1) < 1e-6
    assert out["peekNo"] is None
    assert out["rec"]["source"] == "teacher" and out["rec"]["answer"] == "balance" and out["rec"]["certified"] is False
    assert [r["intent"] for r in out["rows"]] == ["balance"] and out["rows"][0]["source"] == "teacher"
    assert out["stats"]["table"] == 1 and out["stats"]["teacher"] == 1
    assert out["noneEx"]["flag"] == "no_bundle" and "no trained table" in out["noneEx"]["why"]
    assert "canary" in out["why"] and out["opts"] == OPTIONS


@needs_node
def test_openai_teacher_timeout_is_quick():
    out = node("""
const {openaiTeacher} = require(process.argv[1]);
(async () => {
  let aborted = 0;
  const never = (url, init) => new Promise((_, rej) => { init.signal.addEventListener("abort", () => { aborted++; const e = new Error("aborted"); e.name = "AbortError"; rej(e); }); });
  const t = openaiTeacher({options: ["a", "b"], model: "mock-1", baseURL: "http://127.0.0.1:9", fetch: never, timeout: 150, retries: 1});
  const t0 = Date.now();
  let msg = null;
  try { await t("hello"); } catch (e) { msg = e.message; }
  console.log(JSON.stringify({ms: Date.now() - t0, msg, aborted}));
})().catch(e => { console.error(e); process.exit(1); });
""")
    assert out["aborted"] == 2 and "timeout after 150 ms" in out["msg"]
    assert out["ms"] < 5000


@needs_node
def test_bundle_load_errors(bundle, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text("{not json", encoding="utf-8")
    out = node("""
const {decision} = require(process.argv[1]);
(async () => {
  const missing = await decision("intent", {options: ["a", "b"], bundle: process.argv[2] + "/nope"});
  let corrupt = null;
  try { await decision("intent", {options: ["a", "b"], bundle: process.argv[3]}); } catch (e) { corrupt = e.message; }
  console.log(JSON.stringify({missingIsNull: missing.bundle === null, corrupt}));
})().catch(e => { console.error(e); process.exit(1); });
""", bundle, bad)
    assert out["missingIsNull"] is True
    assert out["corrupt"] and "JSON" in out["corrupt"]


@needs_node
def test_ai_sdk_middleware(bundle):
    out = node("""
const {Shadow, Bundle, shad0wMiddleware} = require(process.argv[1]);
(async () => {
  const rows = [];
  const b = await Bundle.load(process.argv[2]);
  const sh = new Shadow(b, {question: "intent", log: (r) => rows.push(r), auditRate: 0});
  const mw = shad0wMiddleware(sh, {specificationVersion: "v3"});
  let calls = 0;
  const doGenerate = async () => { calls++; return {content: [{type: "text", text: "The answer is balance."}], finishReason: "stop", usage: {inputTokens: 9, outputTokens: 2, totalTokens: 11}, warnings: []}; };
  const prompt = (t) => [{role: "system", content: "classify"}, {role: "user", content: [{type: "text", text: t}]}];
  const certified = await mw.wrapGenerate({doGenerate, params: {prompt: prompt("hi block my card thanks")}});
  const deferred = await mw.wrapGenerate({doGenerate, params: {prompt: prompt("weather in paris tomorrow")}});
  console.log(JSON.stringify({version: mw.specificationVersion, certified, deferred, calls, rows}));
})().catch(e => { console.error(e); process.exit(1); });
""", bundle)
    assert out["version"] == "v3"
    assert out["certified"]["content"] == [{"type": "text", "text": "lost_card"}]
    assert out["certified"]["providerMetadata"]["shad0w"]["source"] == "table"
    assert out["deferred"]["content"][0]["text"] == "The answer is balance." and out["deferred"]["usage"]["inputTokens"] == 9
    assert out["calls"] == 1
    assert out["rows"] == [{"text": "weather in paris tomorrow", "intent": "balance", "source": "teacher", "ts": out["rows"][0]["ts"]}]


@needs_node
def test_ai_sdk_decision_model(bundle):
    out = node("""
const {Shadow, Bundle, decisionModel} = require(process.argv[1]);
(async () => {
  const rows = [];
  const b = await Bundle.load(process.argv[2]);
  const sh = new Shadow(b, {question: "intent", log: (r) => rows.push(r), auditRate: 0});
  const seen = [];
  const fallback = {specificationVersion: "v4", provider: "fake", modelId: "fake-1", supportedQuestionTypes: ["choice", "score", "boolean"],
    doDecide: async ({state, questions}) => { seen.push(Object.keys(questions)); const answers = {};
      for (const [id, q] of Object.entries(questions)) answers[id] = q.type === "choice" ? {type: "choice", choice: "balance", probabilities: {balance: 0.8}} : q.type === "boolean" ? {type: "boolean", probability: 0.2} : {type: "score", score: 1.5};
      return {answers, usage: {inputTokens: 5, outputTokens: 0}, warnings: [], response: {modelId: "fake-1"}}; } };
  const questions = {intent: {type: "choice", instructions: "what?", criteria: {refund: null, lost_card: null, balance: null, transfer: null}},
                     severity: {type: "score", instructions: "how bad?", criteria: ["low", "high"]},
                     urgent: {type: "boolean", instructions: "urgent?"}};
  const dm = decisionModel(sh, {fallback});
  const certified = await dm.doDecide({state: "hi block my card thanks", questions});
  const deferred = await dm.doDecide({state: {message: "weather in paris tomorrow"}, questions});
  const alone = decisionModel(sh);
  const refused = await alone.doDecide({state: "weather in paris tomorrow", questions: {intent: questions.intent}});
  console.log(JSON.stringify({spec: [dm.specificationVersion, dm.provider, dm.modelId, dm.supportedQuestionTypes, alone.supportedQuestionTypes], certified, deferred, refused, seen, rows}));
})().catch(e => { console.error(e); process.exit(1); });
""", bundle)
    assert out["spec"][0] == "v4" and out["spec"][1] == "shad0w" and out["spec"][2] == "shad0w:intent"
    assert out["spec"][3] == ["choice", "score", "boolean"] and out["spec"][4] == ["choice", "boolean"]
    c = out["certified"]
    assert c["answers"]["intent"]["type"] == "choice" and c["answers"]["intent"]["choice"] == "lost_card"
    assert set(c["answers"]["intent"]["probabilities"]) == set(OPTIONS)
    assert c["answers"]["severity"]["type"] == "score" and c["answers"]["urgent"]["type"] == "boolean"
    assert c["providerMetadata"]["shad0w"] == {"table": ["intent"], "fallback": ["severity", "urgent"]}
    assert out["seen"][0] == ["severity", "urgent"]  # the certified question never reached the fallback
    d = out["deferred"]
    assert d["answers"]["intent"]["choice"] == "balance" and out["seen"][1] == ["intent", "severity", "urgent"]
    assert out["refused"]["answers"]["intent"]["type"] == "refusal" and out["refused"]["warnings"]
    assert [r["intent"] for r in out["rows"]] == ["balance"]  # the fallback's choice was logged for training


@needs_node
def test_systemone_and_decisions_teachers(bundle):
    srv = DecisionServer()
    try:
        out = node("""
const {systemoneTeacher, decisionsTeacher, decision} = require(process.argv[1]);
(async () => {
  const base = process.argv[2];
  const s1 = systemoneTeacher({options: ["refund", "lost_card", "balance", "transfer"], question: "intent", model: "kev-0.8b", baseURL: base});
  const s1yes = systemoneTeacher({options: {type: "yesno", instructions: "urgent?"}, question: "urgent", baseURL: base + "/v1"});
  const d1 = decisionsTeacher({options: {refund: "money back", lost_card: "card lost or stolen", balance: null, transfer: null}, question: "intent", baseURL: base + "/v1", apiKey: "k"});
  const d1yes = decisionsTeacher({options: {type: "yesno"}, question: "urgent", baseURL: base + "/v1", apiKey: "k"});
  const viaDecision = await decision("intent", {options: ["refund", "lost_card", "balance", "transfer"], llm: "systemone/kev-0.8b", baseURL: base, bundle: process.argv[3] + "/nope"});
  const viaDecision2 = await decision("intent", {options: ["refund", "lost_card", "balance", "transfer"], llm: "openai-decisions/gpt-6-luna", baseURL: base + "/v1", apiKey: "k"});
  console.log(JSON.stringify({s1: await s1("hi block my card thanks"), s1yes: [await s1yes("urgent please"), await s1yes("later")],
    d1: await d1("i want a refund now"), d1yes: [await d1yes("urgent please"), await d1yes("later")],
    v: (await viaDecision.decide("send money to bob")).answer, v2: (await viaDecision2.decide("send money to bob")).answer,
    names: [s1.teacherName, d1.teacherName]}));
})().catch(e => { console.error(e); process.exit(1); });
""", srv.url, bundle)
    finally:
        srv.close()
    assert out["s1"] == "lost_card" and out["s1yes"] == [True, False]
    assert out["d1"] == "refund" and out["d1yes"] == [True, False]
    assert out["v"] == "transfer" and out["v2"] == "transfer"
    assert out["names"] == ["systemone/kev-0.8b", "openai-decisions/gpt-6-luna"]
    s1_req = [r for r in srv.requests if r["path"].endswith("/systemone")][0]["body"]
    assert s1_req["model"] == "kev-0.8b" and s1_req["questions"]["intent"]["type"] == "choice"
    assert list(s1_req["questions"]["intent"]["criteria"]) == OPTIONS
    noul = [r for r in srv.requests if r["path"].endswith("/systemone") and "urgent" in r["body"]["questions"]][0]["body"]
    assert noul["questions"]["urgent"]["type"] == "noul"
    dec = [r for r in srv.requests if r["path"].endswith("/decisions")][0]
    assert dec["auth"] == "Bearer k" and dec["body"]["model"] == "gpt-6-luna"
    q = dec["body"]["questions"][0]
    assert q["type"] == "choice" and q["name"] == "intent"
    assert q["choices"][:2] == [{"value": "refund", "description": "money back"}, {"value": "lost_card", "description": "card lost or stolen"}]
    assert q["choices"][2] == {"value": "balance"}
    pred = [r for r in srv.requests if r["path"].endswith("/decisions") and r["body"]["questions"][0]["type"] == "predicate"]
    assert pred and pred[0]["body"]["questions"][0]["name"] == "urgent"
