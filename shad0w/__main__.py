"""shad0w command line.

Start:
  shad0w init                                                                       # starter schema.json + example log
  shad0w proxy --upstream https://api.openai.com/v1                                 # OpenAI-compatible gateway, zero code
  shad0w train --log shad0w/intent/log.jsonl --out shad0w/intent/bundle             # compile + certify from logged answers
  shad0w try   --bundle shad0w/intent/bundle "my card was stolen"                   # see what the table does, and why
  shad0w stats --log shad0w/intent/log.jsonl     |    shad0w watch                  # what has been logged / live counters

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


def _train(a):
    import os

    from .cascade import rows_for_training
    from .shadow import read_jsonl, shadow_compile, write_certificate
    log_path, out = a.log, a.out
    if a.dir:
        if not a.question:
            qs = sorted(d for d in os.listdir(a.dir) if os.path.exists(os.path.join(a.dir, d, "log.jsonl")))
            if len(qs) != 1:
                print(f"--dir {a.dir} holds questions {qs}; pick one with --question", file=sys.stderr)
                return 2
            a.question = qs[0]
        log_path = log_path or os.path.join(a.dir, a.question, "log.jsonl")
        out = out or os.path.join(a.dir, a.question, "bundle")
    if not log_path or not out:
        print("give --log and --out (or --dir [--question])", file=sys.stderr)
        return 2
    records = read_jsonl(log_path)
    names = [a.question] if a.question else sorted({k for r in records for k in r} - {"text", "source", "ts"})
    full = json.load(open(a.schema, encoding="utf-8")) if a.schema else {}
    old = os.path.join(out, "schema.json")
    if not full and os.path.exists(old):
        full = json.load(open(old, encoding="utf-8"))
    schema, rows_all = {}, []
    for n in names:
        rows, sch = rows_for_training(records, n, full.get(n))
        print(f"{n}: {len(rows):,} usable answers, {len(sch.get('criteria', {'yes': 1, 'no': 1}))} options")
        if len(rows) < a.min_rows:
            print(f"  not enough yet: need {a.min_rows:,} (--min-rows to override; 100 is the hard minimum)", file=sys.stderr)
            return 1
        schema[n] = sch
        rows_all.extend(rows)
    t = time.time()
    m, cert = shadow_compile(schema, rows_all, alpha=a.alpha, delta=a.delta, teacher=a.teacher)
    m.save(out)
    write_certificate(out, cert)
    with open(os.path.join(out, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"trained in {time.time() - t:.1f}s -> {out}")
    for q, s_ in cert["questions"].items():
        print(f"  {q}: the table will answer {s_['certified_share_on_calibration']:.1%} of traffic like this, disagreeing with "
              f"your model on at most {a.alpha:.0%} of those (with {1 - a.delta:.0%} confidence)")
    return 0


def _try(a):
    from .cascade import Shadow
    sh = Shadow(a.bundle, teacher=None, question=a.question, audit_rate=0)
    texts = a.text or None

    def show(t):
        e = sh.explain(t)
        t0 = time.perf_counter_ns()
        sh.model.decide(t, questions={sh.question: {}}, probabilities=False)
        us = (time.perf_counter_ns() - t0) / 1e3
        mark = "\u2713 table" if e["certified"] else "\u2192 your LLM"
        print(f"  {mark}  {e['answer']}  (confidence {e['confidence']:.3f}, needs {e['threshold']:.3f})  {us:.1f} \u00b5s")
        print(f"     {e['why']}")
        print("     top: " + ", ".join(f"{k} {v:.2f}" for k, v in e["top"]))
    if sh.model is None:
        print(f"no bundle at {a.bundle}", file=sys.stderr)
        return 2
    if texts:
        for t in texts:
            print(t)
            show(t)
        return 0
    print(f"question {sh.question!r} - type a message (empty line to quit)")
    while True:
        try:
            t = input("> ").strip()
        except EOFError:
            break
        if not t:
            break
        show(t)
    return 0


def _stats(a):
    import os
    from collections import Counter
    if a.url:
        import urllib.request
        print(json.dumps(json.load(urllib.request.urlopen(a.url.rstrip("/") + "/v1/stats")), indent=2))
        return 0
    rows = [json.loads(l) for l in open(a.log, encoding="utf-8") if l.strip()]
    names = sorted({k for r in rows for k in r} - {"text", "source", "ts"})
    src = Counter(r.get("source", "teacher") for r in rows)
    print(f"{a.log}: {len(rows):,} answers logged ({', '.join(f'{k} {v:,}' for k, v in src.items())})")
    for n in names:
        c = Counter(str(r[n]) for r in rows if n in r)
        tot = sum(c.values())
        print(f"\n{n}: {tot:,} answers, {len(c)} different")
        for k, v in c.most_common(a.top):
            print(f"  {k:<32} {v:>7,}  {'#' * max(1, round(40 * v / max(c.values())))}")
        if len(c) > a.top:
            print(f"  ... {len(c) - a.top} more")
        ready = "ready to train" if tot >= 1000 else f"{1000 - tot:,} more answers before training (100 minimum to try)"
        print(f"  -> {ready}")
    if a.trace and os.path.exists(a.trace):
        tr = [json.loads(l) for l in open(a.trace, encoding="utf-8") if l.strip()]
        s = Counter(r["source"] for r in tr)
        print(f"\ntrace: {len(tr):,} decisions, table answered {s.get('table', 0) / max(1, len(tr)):.1%}")
    return 0


def _watch(a):
    import urllib.request
    url = a.url.rstrip("/") + "/v1/stats"
    try:
        while True:
            try:
                s = json.load(urllib.request.urlopen(url, timeout=5))
            except OSError as e:
                print(f"cannot reach {url}: {e}")
                time.sleep(a.every)
                continue
            out = ["\x1b[2J\x1b[H" + f"shad0w  {url}   up {s['uptime_s']:.0f}s",
                   f"decisions {s['decisions']:,}   table {s['offload']:.1%}   LLM calls saved {s['llm_calls_saved']:,}"
                   + (f"   time saved {s['time_saved_s']:.1f}s" if s.get("time_saved_s") is not None else "")
                   + (f"   cost saved {s['cost_saved']:.2f}" if s.get("cost_saved") is not None else ""), ""]
            for q, v in s["questions"].items():
                up, al = v.get("audit_disagreement_upper"), v.get("alpha")
                safe = "" if up is None or al is None else (f"   live bound {up:.1%} {'<=' if up <= al else '>'} alpha {al:.0%}")
                out.append(f"{q:<20} {v['decisions']:>8,}  table {v['offload']:6.1%}   table p50 {v['table_p50_us'] or 0:.1f}us"
                           f"   LLM p50 {v['llm_p50_ms'] or 0:.0f}ms{safe}")
            out.append("")
            for r in s["recent"][:12]:
                out.append(f"  {r['source']:<8} {str(r['answer'])[:24]:<24} {(r.get('text') or '')[:70]}")
            print("\n".join(out), flush=True)
            time.sleep(a.every)
    except KeyboardInterrupt:
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
    s.add_argument("--cost-per-call", type=float)

    px = sub.add_parser("proxy", help="OpenAI-compatible gateway: answers marked decisions from the table, forwards the rest")
    px.add_argument("--upstream", default="https://api.openai.com/v1", help="the real API base URL (OpenAI, Groq, Ollama, vLLM, ...)")
    px.add_argument("--dir", default="shad0w", help="where logs and bundles live: <dir>/<question>/{log.jsonl,bundle/}")
    px.add_argument("--model", help="upstream model for requests that use model='shad0w/<question>'")
    px.add_argument("--api-key-env", help="send this environment variable as the upstream key (default: forward the client's)")
    px.add_argument("--host", default="127.0.0.1")
    px.add_argument("--port", type=int, default=8010)
    px.add_argument("--audit-rate", type=float, default=0.01)
    px.add_argument("--auto-train", type=int, help="retrain a question every N new LLM answers (needs shad0wllm[compile])")
    px.add_argument("--alpha", type=float, default=0.05)
    px.add_argument("--cost-per-call", type=float, help="cost of one LLM call, to show money saved")
    px.add_argument("--schema", help="optional schema.json naming each question's options")
    px.add_argument("--no-text", action="store_true", help="keep request text out of the dashboard")

    tr = sub.add_parser("train", help="compile + certify a bundle from logged LLM answers (auto-detects questions and options)")
    tr.add_argument("--log", help="JSON Lines log of your model's answers")
    tr.add_argument("--out", help="bundle folder to write")
    tr.add_argument("--dir", help="proxy/decision folder layout instead of --log/--out: <dir>/<question>/")
    tr.add_argument("--question")
    tr.add_argument("--schema", help="optional schema.json (default: options = the answers seen in the log)")
    tr.add_argument("--teacher", default="unspecified")
    tr.add_argument("--alpha", type=float, default=0.05)
    tr.add_argument("--delta", type=float, default=0.1)
    tr.add_argument("--min-rows", type=int, default=1000)

    ty = sub.add_parser("try", help="type messages and see what the table answers, how fast, and why")
    ty.add_argument("--bundle", required=True)
    ty.add_argument("--question")
    ty.add_argument("text", nargs="*")

    st = sub.add_parser("stats", help="summarise a log (what has been learned, is it ready to train) or a running server")
    st.add_argument("--log")
    st.add_argument("--trace")
    st.add_argument("--url", help="a running `shad0w serve` / `shad0w proxy`")
    st.add_argument("--top", type=int, default=15)

    w = sub.add_parser("watch", help="live counters from a running `shad0w serve` / `shad0w proxy`")
    w.add_argument("--url", default="http://127.0.0.1:8010")
    w.add_argument("--every", type=float, default=1.0)
    a = ap.parse_args(argv)

    from . import api

    if a.cmd == "init":
        return _init(a.dir)
    if a.cmd == "train":
        return _train(a)
    if a.cmd == "try":
        return _try(a)
    if a.cmd == "stats":
        if not (a.log or a.url):
            ap.error("stats needs --log or --url")
        return _stats(a)
    if a.cmd == "watch":
        return _watch(a)
    if a.cmd == "proxy":
        import os

        from .proxy import proxy
        key = os.environ.get(a.api_key_env) if a.api_key_env else None
        if a.api_key_env and not key:
            ap.error(f"{a.api_key_env} is not set")
        schema = json.load(open(a.schema, encoding="utf-8")) if a.schema else None
        srv = proxy(a.upstream, host=a.host, port=a.port, folder=a.dir, model=a.model, api_key=key, audit_rate=a.audit_rate,
                    auto_train=a.auto_train, alpha=a.alpha, cost_per_call=a.cost_per_call, keep_text=not a.no_text, schema=schema)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
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
        from .observe import Metrics
        from .server import serve
        serve(a.bundle, host=a.host, port=a.port, metrics=Metrics(cost_per_call=a.cost_per_call))
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
