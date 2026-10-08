import json
import threading
import time
import urllib.request

import numpy as np
import pytest

import shad0w
from shad0w.native import available as _native_available

RNG = np.random.default_rng(0)
TEMPL = {
    "refund": ["i want my money back for {x}", "please refund the {x} charge", "refund {x} now"],
    "lost_card": ["i lost my card at {x}", "my card was stolen near {x}", "can't find my card since {x}"],
    "transfer": ["send money to {x}", "transfer funds to {x} please", "how do i wire cash to {x}"],
}
FILL = ["the store", "monday", "my sister", "paris", "the atm", "work", "an order", "amazon", "jim"]


def data(n=600):
    texts, intent, urgent = [], [], []
    for _ in range(n):
        k = RNG.choice(list(TEMPL))
        t = RNG.choice(TEMPL[k]).format(x=RNG.choice(FILL))
        u = int(RNG.random() < 0.5)
        if u:
            t = "URGENT " + t + " asap"
        texts.append(t)
        intent.append(k)
        urgent.append(u)
    return texts, intent, urgent


SCHEMA = {"intent": {"type": "choice", "criteria": {"refund": "money back", "lost_card": None, "transfer": None}},
          "urgent": {"type": "yesno", "instructions": "Is it urgent?"}}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    texts, intent, urgent = data()
    m = shad0w.compile(SCHEMA, labeled={"intent": (texts, intent), "urgent": (texts, urgent)})
    path = str(tmp_path_factory.mktemp("bundle"))
    m.save(path)
    return path


def test_compile_save_load_decide(bundle):
    m = shad0w.load(bundle, native=False)
    out = m.decide("please refund the amazon charge")["answers"]
    assert out["intent"]["choice"] == "refund"
    assert set(out["intent"]["probabilities"]) == {"refund", "lost_card", "transfer"}
    assert out["urgent"]["answer"] is False
    assert shad0w.load(bundle, native=False).decide("URGENT i lost my card at paris asap")["answers"]["urgent"]["answer"]


@pytest.mark.skipif(not _native_available(), reason="C core not built")
def test_native_matches_python(bundle):
    a, b = shad0w.load(bundle, native=True), shad0w.load(bundle, native=False)
    for t in ["refund now!!", "transfer funds to jim please", "", "URGENT my card was stolen near work asap", "ünïcode café"]:
        x, y = a.decide(t)["answers"], b.decide(t)["answers"]
        assert x["intent"]["choice"] == y["intent"]["choice"]
        assert abs(x["intent"]["confidence"] - y["intent"]["confidence"]) < 1e-5
        assert x["intent"]["radius"] == y["intent"]["radius"]


def test_exposed_channel_flags_low_radius(bundle):
    m = shad0w.load(bundle, native=False)
    for q in m.questions.values():
        q.r_min = 1e9  # nothing is robust enough -> every exposed answer must be flagged
    out = m.decide("refund the store charge", exposed=True)["answers"]["intent"]
    assert out["certified"] is False and out["flag"] in ("low_radius", "low_confidence")


def test_server_roundtrip(bundle):
    from shad0w.server import serve
    srv = serve(bundle, "127.0.0.1", 0, run=False)  # port 0: any free port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    for _ in range(50):
        try:
            urllib.request.urlopen(base + "/v1/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    req = urllib.request.Request(base + "/v1/decide",
                                 data=json.dumps({"state": "send money to my sister"}).encode(),
                                 headers={"content-type": "application/json"})
    # functional check, not a benchmark: shared CI runners can stall a cold first call for milliseconds,
    # so warm up, then take the best of a few calls and allow a generous bound
    lat = []
    for _ in range(5):
        r = json.loads(urllib.request.urlopen(req, timeout=5).read())
        assert r["answers"]["intent"]["choice"] == "transfer"
        lat.append(r["latency_us"])
    assert min(lat) < 50_000
    stats = json.loads(urllib.request.urlopen(base + "/v1/stats", timeout=5).read())
    assert stats["questions"]["intent"]["decisions"] == 5  # counted per question
    assert b"shad0w_decisions_total" in urllib.request.urlopen(base + "/metrics", timeout=5).read()
    srv.shutdown()


def test_observe_false_does_not_touch_guard(bundle):
    m = shad0w.load(bundle, native=False)
    q = m.questions["intent"]
    n0 = len(q.guard.buf)
    q.decide("refund the store charge", observe=False)
    m.decide("refund the store charge", observe=False)
    assert len(q.guard.buf) == n0
    m.decide("refund the store charge")
    assert len(q.guard.buf) == n0 + 1


def test_drift_guard_reconfigure_and_raise():
    from shad0w.api import DriftGuard
    g = DriftGuard(np.linspace(0.5, 1.0, 100), np.array([False] * 10 + [True] * 90), window=10, margin=0.0)
    for _ in range(20):
        g.update(0.0)
    assert g.raised and g.state()["window"] == 10
    g.reconfigure(window=500, margin=0.5)
    assert g.window == 500 and not g.raised and len(g.buf) == 0


def test_calibrate_accepts_yes_no_strings(bundle):
    texts, intent, urgent = data(300)
    a = shad0w.load(bundle, native=False)
    b = shad0w.load(bundle, native=False)
    ra = a.calibrate("urgent", texts, urgent)
    rb = b.calibrate("urgent", texts, ["yes" if u else "no" for u in urgent])
    assert ra["threshold"] == rb["threshold"] and ra["accuracy"] == rb["accuracy"] and 0 <= ra["certified_share"] <= 1
    from shad0w.api import to_bool
    assert [to_bool(v) for v in ("yes", "No", True, 0, "1", "false", "Y")] == [True, False, True, False, True, False, True]


def test_server_decisions_systemone_playground(bundle):
    from shad0w.server import serve
    srv = serve(bundle, "127.0.0.1", 0, run=False)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    out = post("/v1/decisions", {"model": "x", "input": "send money to my sister", "questions": [
        {"type": "choice", "name": "intent", "instructions": "", "choices": [{"value": "refund"}, {"value": "transfer"}]},
        {"type": "predicate", "name": "urgent", "instructions": ""},
        {"type": "score", "name": "severity", "instructions": "", "levels": [{"label": "a"}, {"label": "b"}]},
        {"type": "choice", "name": "unknown", "instructions": "", "choices": [{"value": "x"}, {"value": "y"}]}]})
    a = out["answers"]
    assert out["object"] == "decision" and [x["name"] for x in a] == ["intent", "urgent", "severity", "unknown"]
    assert a[0]["choice"] == "transfer" and {p["value"] for p in a[0]["probabilities"]} == {"refund", "transfer"}
    assert a[1]["type"] == "predicate" and 0 <= a[1]["probability"] <= 1
    assert a[2]["shad0w"]["flag"] == "unsupported" and a[3]["shad0w"]["flag"] == "unsupported"
    out = post("/v1/systemone", {"model": "x", "state": "URGENT send money to my sister asap", "questions": {
        "intent": {"type": "choice", "criteria": {"refund": "r", "transfer": "t", "lost_card": "l"}},
        "urgent": {"type": "noul"}}})
    assert out["answers"]["intent"]["choice"] == "transfer" and out["answers"]["urgent"]["type"] == "noul"
    assert out["answers"]["urgent"]["answer"] == "yes" and "latency_ms" in out
    out = post("/v1/playground", {"question": "intent", "text": "send money to my sister"})
    assert out["explain"]["answer"] == "transfer" and "threshold" in out["explain"] and out["llm"] is None
    srv.shutdown()


def test_uncertified_bundle_manifest_is_strict_json(tmp_path):
    """A question that certified nothing has an infinite threshold; the manifest must still be valid JSON (null)."""
    import json as _json
    import math as _math

    from shad0w.api import load
    from shad0w.shadow import shadow_compile
    rows = [{"text": f"word{i % 7} filler {i}", "q": ["a", "b"][i % 2]} for i in range(300)]  # labels unrelated to text
    model, _ = shadow_compile({"q": {"type": "choice", "criteria": {"a": None, "b": None}}}, rows, alpha=0.001)
    model.questions["q"].threshold = float("inf")
    model.save(str(tmp_path))
    raw = (tmp_path / "manifest.json").read_text()
    assert "Infinity" not in raw and _json.loads(raw)["questions"]["q"]["threshold"] is None
    m = load(str(tmp_path), native=False)
    assert _math.isinf(m.questions["q"].threshold)
    assert m.decide("word1 filler")["answers"]["q"]["certified"] is False
