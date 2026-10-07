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
