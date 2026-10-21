"""text_kind 2 (script-aware items): Python == C == JS item for item on a multilingual corpus, kind 1 unchanged, pure
ASCII identical under both kinds, and a text_kind=2 bundle round-trips through save / load / the C core / JS."""
import ctypes
import json
import os
import random
import shutil
import subprocess

import numpy as np
import pytest

from shad0w import native
from shad0w.features import MASK, MAX_WORD, TEXT_KIND, _words, fnv1a, items, items_batch, using_kind
from shad0w.mathutil import softmax
from shad0w.reflex import Reflex

ROOT = os.path.dirname(os.path.dirname(__file__))

FRAGMENTS = [
    "I lost my card!!", "Where is my REFUND?", "transfer 50€ to Ana", "naïve café crème", "Über ÜBER über", "Straße",
    "我的银行卡丢了", "请帮我查询余额", "中文abc", "abc中文def", "2024年3月5日", "你好，世界！", "繁體中文測試",
    "東京は日本の首都です。", "カタカナとひらがな", "ﾊﾝｶｸｶﾀｶﾅ", "今日はいい天気ですね", "ㇰㇱㇲ",
    "ภาษาไทยไม่มีช่องว่าง", "ສະບາຍດີ", "ភាសាខ្មែរ", "မြန်မာဘာသာ",
    "한국어 텍스트입니다", "Привет, МИР! Ёлка ЁЖ", "ЀЍЏ Ѐѐ", "Ωμέγα ΑΛΦΑ Σίσυφος", "Ά Έ ΐ",
    "مرحبا بالعالم", "שלום עולם", "नमस्ते दुनिया", "🎉 party 🎉🎉", "emoji👍mixed", "𠀀𠀁 rare han 𪚥",
    "ÀÉÎÕÜ ×÷ Þþ ß", "x" * 400, "中" * 300, "Ü" * 200, "", "   ", "!!!", "a", "A1 b2 C3",
]


def corpus():
    rng = random.Random(0)
    out = list(FRAGMENTS)
    while len(out) < 320:
        k = rng.randint(1, 4)
        sep = rng.choice([" ", "", ", ", "\n", "。", "-"])
        out.append(sep.join(rng.choice(FRAGMENTS) for _ in range(k)))
    return out


RAW = [b"\xff\xfe", b"ab\xffcd", b"\xe4\xb8", b"\xe4\xb8\xad\xe6", b"\xc3", b"x\xc3(y", b"\xed\xa0\x80z", b"\xf5\x80\x80\x80",
       b"\xe0\x80\x80", "中文".encode() + b"\x80\x80" + "über".encode(), b"\xf0\x9f\x8e", b"ok \xc0\xaf ok"]


def _kind1_reference(text) -> list[int]:
    """The featurizer as it was before text_kind existed (kept verbatim): kind 1 must never move."""
    b = text if isinstance(text, bytes) else text.encode("utf-8", "replace")
    b = b.lower()
    ws, cur = [], bytearray()
    for ch in b:
        if (48 <= ch <= 57) or (97 <= ch <= 122) or ch >= 0x80:
            if len(cur) < MAX_WORD:
                cur.append(ch)
        elif cur:
            ws.append(bytes(cur))
            cur = bytearray()
    if cur:
        ws.append(bytes(cur))
    out = []
    for w in ws:
        out.append(fnv1a(b"w" + w) & MASK)
        p = b" " + w + b" "
        out += [fnv1a(b"c" + p[i:i + 3]) & MASK for i in range(len(p) - 2)]
    out += [fnv1a(b"b" + a + b"\x01" + c) & MASK for a, c in zip(ws, ws[1:])]
    return out or [fnv1a(b"e") & MASK]


FROZEN_KIND1 = {  # computed once with the pre-text_kind featurizer
    "Über 中文abc": [259146, 2889737, 238525, 2382119, 1119719, 3173471, 1576074, 1407786, 2545753, 2481341, 2816709,
                    1752187, 3557008, 3518360, 30614, 789259, 3496286],
    "hi": [3417961, 3591849, 3713695],
    "": [795360],
}


def test_kind1_is_unchanged():
    for t in corpus() + RAW:
        assert items(t, kind=1).tolist() == _kind1_reference(t), t
        assert items(t).tolist() == _kind1_reference(t), t  # the default stays kind 1 outside using_kind
    for t, want in FROZEN_KIND1.items():
        assert items(t, kind=1).tolist() == want, t


def test_ascii_is_identical_under_both_kinds():
    rng = random.Random(1)
    alphabet = "abcXYZ0129 .,!?-_'\"\n\t"
    texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))) for _ in range(300)]
    texts += [t for t in corpus() if t.isascii()]
    for t in texts:
        assert items(t, kind=2).tolist() == items(t, kind=1).tolist(), t


def test_kind2_words():
    assert [w.decode() for w in _words("中文abc我", 2)] == ["中", "文", "abc", "我"]
    assert [w.decode() for w in _words("ÜBER Über über", 2)] == ["über"] * 3
    assert [w.decode() for w in _words("ПРИВЕТ Ёж ΑΛΦΑ", 2)] == ["привет", "ёж", "αλφα"]
    assert [w.decode() for w in _words("ภาษา", 2)] == ["ภาษา"]  # alphabetic scripts keep runs (Thai lost accuracy per letter)
    assert [w.decode() for w in _words("한국어 텍스트", 2)] == ["한국어", "텍스트"]  # Hangul stays run-based
    assert [w.decode() for w in _words("× Σ", 2)] == ["×", "σ"]  # U+00D7 is not a letter
    assert _words(b"\xff\xfe", 2) == [b"\xff\xfe"]  # invalid UTF-8: word bytes, as in kind 1
    its = items("中文", kind=2)
    assert len(its) == 3  # two character words + one character bigram, no trigrams
    assert items("Über", kind=2).tolist() == items("über", kind=2).tolist()
    assert items("Über", kind=1).tolist() != items("über", kind=1).tolist()
    with using_kind(2):
        assert items("中文").tolist() == its.tolist() and items_batch(["中文"])[0].tolist() == its.tolist()
    assert items("中文").tolist() == items("中文", kind=1).tolist()
    assert TEXT_KIND == 2
    with pytest.raises(ValueError):
        items("x", kind=3)


def _random_table(texts, kinds, path, K=4, seed=0):
    vocab_ids = sorted({int(h) for t in texts for k in kinds for h in items(t, kind=k)})
    vocab = {h: i for i, h in enumerate(vocab_ids)}
    rng = np.random.default_rng(seed)
    rx = Reflex.compile(vocab, rng.standard_normal((K, len(vocab))).astype(np.float32), np.zeros(K, np.float32),
                        list("abcd")[:K])
    rx.save(path)
    return rx


@pytest.mark.skipif(not native.available(), reason="C core not built")
def test_c_matches_python_kind2(tmp_path):
    texts = corpus()
    # leave the last 40 texts' items out of the table, so the unseen counts are exercised too
    rx = _random_table(texts[:-40] + RAW, (1, 2), str(tmp_path / "t.s0"))
    for kind in (1, 2):
        nat = native.NativeReflex(str(tmp_path / "t.s0"), text_kind=kind)
        nat.lib.s0_text_kind.argtypes = [ctypes.c_void_p]
        assert nat.lib.s0_text_kind(nat.m) == kind
        for t in texts:
            its = items(t, kind=kind)
            y, p, r = nat.decide(t)
            pp = softmax(rx.logits(its))
            hits, n_all = rx.rows(its)
            assert nat._n[0] == len(its) and nat._n[1] == len(hits), (kind, t)
            assert y == int(np.argmax(pp)) and np.allclose(p, pp, atol=1e-5), (kind, t)
            want_r = rx.radius(its)
            assert r == want_r or (np.isinf(r) and np.isinf(want_r))
        # raw bytes, including invalid UTF-8, straight into the C entry point
        lib, P, R, N = nat.lib, (ctypes.c_float * 4)(), ctypes.c_float(), (ctypes.c_uint32 * 2)()
        for b in RAW:
            its = items(b, kind=kind)
            y = lib.s0_decide_counts(nat.m, None, b, len(b), P, ctypes.byref(R), N)
            pp = softmax(rx.logits(its))
            assert N[0] == len(its) and N[1] == len(rx.rows(its)[0]) and y == int(np.argmax(pp)), (kind, b)
            assert np.allclose(np.array(P), pp, atol=1e-5)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_items_match_python(tmp_path):
    texts = corpus()
    cases = [{"t": t} for t in texts] + [{"b": list(b)} for b in RAW]
    (tmp_path / "c.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    script = ("const S=require(process.argv[1]);const c=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));"
              "console.log(JSON.stringify(c.map(x=>{const v=x.t!==undefined?x.t:Uint8Array.from(x.b);"
              "return [S.items(v,1),S.items(v,2),S.words(v,2).map(w=>Array.from(w))]})))")
    r = subprocess.run(["node", "-e", script, os.path.join(ROOT, "js", "index.js"), str(tmp_path / "c.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    js = json.loads(r.stdout)
    for x, (k1, k2, w2) in zip(texts + RAW, js):
        assert k1 == items(x, kind=1).tolist(), x
        assert k2 == items(x, kind=2).tolist(), x
        assert w2 == [list(w) for w in _words(x, 2)], x


def _multilingual_rows(n, seed):
    rng = random.Random(seed)
    vocab = {"refund": ["退款", "返金してください", "คืนเงิน", "Rückerstattung", "возврат денег"],
             "lost_card": ["银行卡丢了", "カードをなくした", "บัตรหาย", "Karte verloren", "потерял карту"],
             "balance": ["查询余额", "残高を確認", "ยอดเงินคงเหลือ", "Kontostand", "баланс счёта"]}
    filler = ["请", "お願いします", "ครับ", "bitte", "пожалуйста", "今天", "hi", "thanks"]
    rows = []
    for _ in range(n):
        y = rng.choice(list(vocab))
        rows.append({"text": f"{rng.choice(filler)}{rng.choice(vocab[y])}{rng.choice(filler)}", "intent": y})
    return rows


def test_bundle_roundtrip_text_kind_2(tmp_path):
    import shad0w
    from shad0w.shadow import shadow_compile

    schema = {"intent": {"type": "choice", "criteria": {"refund": None, "lost_card": None, "balance": None}}}
    rows = _multilingual_rows(1500, 0)
    m, cert = shadow_compile(schema, rows, alpha=0.05)
    assert cert["text_kind"] == 2 and m.meta["text_kind"] == 2 and m.questions["intent"].text_kind == 2
    m.save(str(tmp_path / "b"))
    man = json.loads((tmp_path / "b" / "manifest.json").read_text(encoding="utf-8"))
    assert man["meta"]["text_kind"] == 2
    py = shad0w.load(str(tmp_path / "b"), native=False)
    assert py.questions["intent"].text_kind == 2
    probe = [r["text"] for r in _multilingual_rows(200, 1)] + ["", "Über", "中文abc"]
    want = [m.decide(t)["answers"]["intent"] for t in probe]
    got = [py.decide(t)["answers"]["intent"] for t in probe]
    assert [(a["choice"], a["certified"]) for a in want] == [(a["choice"], a["certified"]) for a in got]
    if native.available():
        nat = shad0w.load(str(tmp_path / "b"))
        assert nat.native and nat.questions["intent"]._native.text_kind == 2
        for t, a in zip(probe, want):
            b = nat.decide(t)["answers"]["intent"]
            assert b["choice"] == a["choice"] and b["certified"] == a["certified"] and abs(b["confidence"] - a["confidence"]) < 1e-5
    # kind 1 for the same rows: a different featurizer, and no "text_kind" in the manifest (old runtimes read it)
    m1, cert1 = shadow_compile(schema, rows, alpha=0.05, text_kind=1)
    m1.save(str(tmp_path / "b1"))
    assert cert1["text_kind"] == 1 and "text_kind" not in json.loads((tmp_path / "b1" / "manifest.json").read_text())["meta"]
    assert shad0w.load(str(tmp_path / "b1"), native=False).questions["intent"].text_kind == 1
    # an unknown featurizer is refused
    man["meta"]["text_kind"] = 9
    (tmp_path / "b" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError):
        shad0w.load(str(tmp_path / "b"), native=False)
