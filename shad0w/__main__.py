"""shad0w command line.

Start:
  shad0w init                                                                       # starter schema.json + example log

Shadow mode (learn from the model you already run, certify agreement with it, no human labels):
  shad0w shadow    --schema schema.json --data teacher_log.jsonl --teacher NAME [--alpha 0.05] --out bundle/
  shad0w certify   --bundle bundle/ --data fresh_teacher_log.jsonl [--alpha 0.05]   # re-certify on new traffic
  shad0w report    --bundle bundle/                                                 # print the certificate

Other modes:
  shad0w compile   --schema schema.json --data labels.jsonl --out bundle/           # from human labels
  shad0w compile   --schema schema.json --unlabeled logs.txt --out bundle/          # from unlabelled logs
  shad0w calibrate --bundle bundle/ --data real_labels.jsonl [--alpha 0.05]         # certificate against real labels

Serving:
  shad0w decide --bundle bundle/ "some text" [--exposed]
  shad0w serve  --bundle bundle/ [--port 8010]                                      # POST /v1/decide
  shad0w bench  --bundle bundle/ --texts file.txt

Files:
  schema.json        {"intent": {"type": "choice", "criteria": {"refund": "money back", "lost_card": null}},
                      "urgent": {"type": "yesno"}}
  teacher_log.jsonl  {"text": "...", "intent": "refund", "urgent": true}   one line per decision your model made
"""

import argparse
import json
import sys
import time


def _rows_by_question(path, schema_questions):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = {}
    for q in schema_questions:
        rs = [r for r in rows if q in r]
        if rs:
            out[q] = ([r["text"] for r in rs], [r[q] for r in rs])
    return rows, out


_SCHEMA = {"intent": {"type": "choice", "criteria": {
    "refund": "the customer wants money back", "lost_card": "a card is lost or stolen",
    "balance": "the customer asks about their balance", "other": None}}}
_LOG = [{"text": "my card was stolen yesterday", "intent": "lost_card"},
        {"text": "can I get my money back for this order", "intent": "refund"},
        {"text": "how much is left in my account", "intent": "balance"}]


def _init(d):
    import os
    os.makedirs(d, exist_ok=True)
    wrote = []
    for name, body in (("schema.json", json.dumps(_SCHEMA, indent=2) + "\n"),
                       ("teacher_log.jsonl", "".join(json.dumps(r) + "\n" for r in _LOG))):
        p = os.path.join(d, name)
        if os.path.exists(p):
            print(f"kept existing {p}")
            continue
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
        wrote.append(p)
    print("wrote " + ", ".join(wrote) if wrote else "nothing to write")
    print("next: log your model's answers into teacher_log.jsonl (1,000+ lines), then\n"
          "  shad0w shadow --schema schema.json --data teacher_log.jsonl --teacher my-model --out bundle/")
    return 0


def main(argv=None):
    from . import __version__

    ap = argparse.ArgumentParser(prog="shad0w", description="shad0w: certified microsecond decisions, compiled "
                                 "from the model you already run.", epilog="docs: https://2796gaurav.github.io/shad0w")
    ap.add_argument("--version", action="version", version=f"shad0w {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    it = sub.add_parser("init", help="write a starter schema.json and an example teacher_log.jsonl")
    it.add_argument("--dir", default=".")

    sh = sub.add_parser("shadow", help="compile from a teacher's logged answers and certify agreement with it")
    sh.add_argument("--schema", required=True)
    sh.add_argument("--data", required=True, help="jsonl: text + one field per question holding the teacher's answer")
    sh.add_argument("--teacher", default="unspecified", help="name/version of the model whose answers were logged")
    sh.add_argument("--alpha", type=float, default=0.05, help="certified disagreement bound (0.01-0.10)")
    sh.add_argument("--delta", type=float, default=0.1, help="certificate failure probability")
    sh.add_argument("--out", required=True)

    ce = sub.add_parser("certify", help="re-certify a bundle against fresh teacher-labelled traffic")
    ce.add_argument("--bundle", required=True)
    ce.add_argument("--data", required=True)
    ce.add_argument("--alpha", type=float)
    ce.add_argument("--delta", type=float, default=0.1)
    ce.add_argument("--teacher")

    rp = sub.add_parser("report", help="print the bundle's certificate")
    rp.add_argument("--bundle", required=True)

    c = sub.add_parser("compile", help="compile from human labels or from unlabelled logs")
    c.add_argument("--schema", required=True)
    c.add_argument("--data", help="jsonl with human labels")
    c.add_argument("--unlabeled", help="text file, one unlabelled message per line (label-free mode)")
    c.add_argument("--encoder", default="bge-small",
                   help="compile-time encoder for label-free mode (shad0w/compiler/encoders.py key or a sentence-transformers name)")
    c.add_argument("--out", required=True)
    c.add_argument("--alpha", type=float, default=0.05)

    k = sub.add_parser("calibrate", help="certificate against REAL labels (~300 per question)")
    k.add_argument("--bundle", required=True)
    k.add_argument("--data", required=True)
    k.add_argument("--alpha", type=float)

    d = sub.add_parser("decide")
    d.add_argument("--bundle", required=True)
    d.add_argument("text")
    d.add_argument("--exposed", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("--bundle", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8010)
    b = sub.add_parser("bench")
    b.add_argument("--bundle", required=True)
    b.add_argument("--texts", required=True)
    a = ap.parse_args(argv)

    from . import api

    if a.cmd == "init":
        return _init(a.dir)
    if a.cmd == "shadow":
        from .shadow import read_jsonl, shadow_compile, write_certificate
        schema = json.load(open(a.schema, encoding="utf-8"))
        t = time.time()
        m, cert = shadow_compile(schema, read_jsonl(a.data), alpha=a.alpha, delta=a.delta, teacher=a.teacher)
        m.save(a.out)
        write_certificate(a.out, cert)
        print(f"compiled and certified {list(m.questions)} in {time.time() - t:.1f}s -> {a.out}")
        for q, s_ in cert["questions"].items():
            print(f"  {q}: agreement with teacher {s_['agreement_with_teacher']:.1%}; certified share "
                  f"{s_['certified_share_on_calibration']:.1%} at alpha={a.alpha}")
    elif a.cmd == "certify":
        from .shadow import certify_bundle, read_jsonl
        cert = certify_bundle(a.bundle, read_jsonl(a.data), alpha=a.alpha, delta=a.delta, teacher=a.teacher)
        print(json.dumps(cert, indent=2))
    elif a.cmd == "report":
        from .shadow import read_certificate
        cert = read_certificate(a.bundle)
        print(json.dumps(cert, indent=2) if cert else "no certificate in this bundle (run shad0w shadow, shad0w certify or shad0w calibrate)")
    elif a.cmd == "compile":
        schema = json.load(open(a.schema, encoding="utf-8"))
        labeled = _rows_by_question(a.data, schema)[1] if a.data else {}
        unl = [l.strip() for l in open(a.unlabeled, encoding="utf-8") if l.strip()] if a.unlabeled else None
        t = time.time()
        m = api.compile(schema, labeled=labeled or None, unlabeled=unl, alpha=a.alpha, encoder=a.encoder)
        m.save(a.out)
        print(f"compiled {list(m.questions)} in {time.time() - t:.1f}s -> {a.out}")
    elif a.cmd == "calibrate":
        m = api.load(a.bundle, native=False)
        _, by_q = _rows_by_question(a.data, m.questions)
        for q, (texts, labels) in by_q.items():
            print(q, json.dumps(m.calibrate(q, texts, labels, a.alpha)))
        m.save(a.bundle)
    elif a.cmd == "decide":
        print(json.dumps(api.load(a.bundle).decide(a.text, exposed=a.exposed), indent=2, ensure_ascii=False))
    elif a.cmd == "serve":
        from .server import serve
        serve(a.bundle, host=a.host, port=a.port)
    elif a.cmd == "bench":
        m = api.load(a.bundle)
        texts = [l.strip() for l in open(a.texts, encoding="utf-8") if l.strip()]
        for t in texts[:50]:
            m.decide(t)
        ts = []
        for t in texts:
            t0 = time.perf_counter_ns()
            m.decide(t)
            ts.append(time.perf_counter_ns() - t0)
        ts.sort()
        print(json.dumps({"n": len(ts), "p50_us": ts[len(ts) // 2] / 1e3, "p99_us": ts[int(len(ts) * 0.99)] / 1e3,
                          "questions": len(m.questions)}))


if __name__ == "__main__":
    sys.exit(main())
