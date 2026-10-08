"""shad0w command line.

Start:
  shad0w init                                                                       # starter schema.json + example log
  shad0w doctor [--bundle B] [--upstream URL] [--llm provider/model]                # is everything in place?
  shad0w config [--question Q] [--init]                                             # effective settings and their source
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
    from . import config as _config
    st = _config.resolve(a.question, path=a.config, alpha=a.alpha, delta=a.delta, min_rows=a.min_rows)
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
        if len(rows) < st.min_rows:
            print(f"  not enough yet: need {st.min_rows:,} (--min-rows to override; 100 is the hard minimum)", file=sys.stderr)
            return 1
        schema[n] = sch
        rows_all.extend(rows)
    cal = [r for r in rows_all if r.get("source") == "audit"]
    fit_rows, cal_rows = rows_all, None
    if len(cal) >= 100:
        fit_rows, cal_rows = [r for r in rows_all if r.get("source") != "audit"], cal
        print(f"certifying on {len(cal):,} uniform spot-check rows; fitting on the other {len(fit_rows):,}")
    t = time.time()
    m, cert = shadow_compile(schema, fit_rows, alpha=st.alpha, delta=st.delta, teacher=a.teacher, cal_records=cal_rows,
                             cal_fraction=st.cal_fraction, max_cal=st.max_cal, drift_window=st.drift_window,
                             drift_margin=st.drift_margin)
    if a.gate and os.path.exists(os.path.join(out, "certificate.json")):
        from .shadow import read_certificate
        old = read_certificate(out) or {}
        for q, s_ in cert["questions"].items():
            o = (old.get("questions") or {}).get(q)
            if o and o.get("threshold") is not None and (s_["threshold"] is None or
                                                          s_["certified_share_on_calibration"] < 0.8 * o["certified_share_on_calibration"]):
                print(f"{q}: new bundle certifies {s_['certified_share_on_calibration']:.1%} vs "
                      f"{o['certified_share_on_calibration']:.1%} before; kept the old bundle (drop --gate to replace)", file=sys.stderr)
                return 3
    m.save(out)
    write_certificate(out, cert)
    with open(os.path.join(out, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"trained in {time.time() - t:.1f}s -> {out}")
    for q, s_ in cert["questions"].items():
        print(f"  {q}: the table will answer {s_['certified_share_on_calibration']:.1%} of traffic like this, disagreeing with "
              f"your model on at most {st.alpha:.0%} of those (with {1 - st.delta:.0%} confidence; "
              f"calibration: {s_.get('calibration', 'held-out-split')})")
    return 0


def _need_bundle(path):
    import os
    if not os.path.exists(os.path.join(path, "manifest.json")):
        raise FileNotFoundError(f"no bundle at {path} (expected manifest.json; run `shad0w train` first)")


def _try(a):
    from .cascade import Shadow
    _need_bundle(a.bundle)
    sh = Shadow(a.bundle, teacher=None, question=a.question, audit_rate=0)
    texts = a.text or None

    def show(t):
        e = sh.explain(t)
        t0 = time.perf_counter_ns()
        sh.model.decide(t, questions={sh.question: {}}, probabilities=False)
        us = (time.perf_counter_ns() - t0) / 1e3
        mark = "\u2713 table" if e.get("would_serve", e["certified"]) else "\u2192 your LLM"
        print(f"  {mark}  {e['answer']}  (confidence {e['confidence']:.3f}, needs {e['threshold']:.3f})  {us:.1f} \u00b5s")
        print(f"     {e['why']}" + (f"  [{e['policy_flag']}]" if e.get("policy_flag") not in (None, e["flag"]) else ""))
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
    from . import config as _config
    need = _config.resolve().min_rows
    if not os.path.exists(a.log):
        raise FileNotFoundError(f"nothing logged yet at {a.log}")
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
        ready = "ready to train" if tot >= need else f"{need - tot:,} more answers before training (100 minimum to try)"
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


def _config_cmd(a):
    import os

    from . import config as _config
    if a.init:
        p = a.config or _config.CONFIG_FILE
        if os.path.exists(p):
            print(f"{p} exists; not overwriting", file=sys.stderr)
            return 2
        lines = ["# shad0w settings. Precedence: code / flags > SHAD0W_* environment > this file > defaults.",
                 "# Every key is optional. Per-question overrides go under [questions.<name>].", ""]
        for k, v in _config.DEFAULTS.items():
            if k in ("trace", "folder", "capture", "timeout"):
                continue
            val = json.dumps(v) if not isinstance(v, (list, tuple)) else json.dumps(list(v))
            if v is None:
                lines.append(f"# {k} = ...   # {_config.HELP[k]}")
            else:
                lines.append(f"# {k} = {val}   # {_config.HELP[k]}")
        lines += ["", "# [questions.intent]", "# alpha = 0.02", ""]
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"wrote {p}")
        return 0
    rows = _config.explain(a.question, path=a.config)
    w = max(len(k) for k, _, _ in rows)
    for k, v, src in rows:
        print(f"{k:<{w}}  {json.dumps(list(v) if isinstance(v, tuple) else v):<28}  {src}")
    cfg = _config.find_config(a.config)
    print(f"\nconfig file: {cfg or 'none (create one with: shad0w config --init)'}")
    return 0


def _doctor(a):
    import importlib.util
    import os
    import platform
    import shutil
    import subprocess
    import urllib.request

    from . import __version__
    from . import config as _config
    from .llm import PROVIDERS
    from .native import available, find_library
    rows, fails = [], 0

    def ok(label, detail=""):
        rows.append(("ok", label, detail))

    def warn(label, detail=""):
        rows.append(("--", label, detail))

    def fail(label, detail=""):
        nonlocal fails
        fails += 1
        rows.append(("!!", label, detail))

    py = platform.python_version()
    (ok if tuple(sys.version_info[:2]) >= (3, 10) else fail)(f"Python {py}", f"shad0w {__version__}")
    if available():
        ok("C core", find_library() or "built in")
    else:
        warn("C core", "numpy fallback (same answers, slower)")
    missing = [m for m in ("scipy", "sklearn") if importlib.util.find_spec(m) is None]
    if missing:
        warn("training deps (scipy, scikit-learn)", f"missing {missing}: pip install \"shad0wllm[compile]\"")
    else:
        ok("training deps (scipy, scikit-learn)", "installed")
    has_torch = importlib.util.find_spec("torch") is not None
    (ok if has_torch else warn)("torch (optional, faster training)", "installed" if has_torch else "not installed: scipy solver is used")
    try:
        import tomllib  # noqa: F401
        ok("TOML config reader", "tomllib")
    except ModuleNotFoundError:
        if importlib.util.find_spec("tomli"):
            ok("TOML config reader", "tomli")
        else:
            warn("TOML config reader", "pip install tomli (needed to read shad0w.toml on 3.10)")
    node = shutil.which("node")
    if node:
        try:
            v = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            v = "?"
        ok("Node (JavaScript package)", v)
    else:
        warn("Node (JavaScript package)", "not found; only needed for the npm package and its tests")
    try:
        cfg = _config.find_config(a.config)
        st = _config.resolve(path=a.config)
        ok("config", f"{cfg or 'defaults'}; mode={st.mode} alpha={st.alpha} audit_rate={st.audit_rate} canary={st.canary}")
    except Exception as e:
        fail("config", str(e))
    if a.llm:
        provider = a.llm.split("/", 1)[0]
        env = PROVIDERS.get(provider, ("", None))[1] if provider in PROVIDERS else None
        if provider not in PROVIDERS:
            warn(f"provider {provider!r}", "unknown preset; pass base_url= in code")
        elif env and not os.environ.get(env):
            fail(f"{provider} API key", f"{env} is not set")
        else:
            ok(f"{provider} API key", env or "no key needed")
    if a.upstream:
        try:
            req = urllib.request.Request(a.upstream.rstrip("/") + "/models")
            urllib.request.urlopen(req, timeout=3).close()
            ok("upstream reachable", a.upstream)
        except urllib.error.HTTPError as e:
            ok("upstream reachable", f"{a.upstream} (HTTP {e.code})")
        except Exception as e:
            fail("upstream reachable", f"{a.upstream}: {e}")
    if a.bundle:
        try:
            from .api import load
            from .shadow import read_certificate
            m = load(a.bundle, native=False)
            cert = read_certificate(a.bundle)
            for q, qq in m.questions.items():
                thr, fin = qq.threshold, qq.threshold != float("inf")
                detail = f"{len(qq.options)} options, threshold {thr:.3f}, alpha {qq.alpha}" if fin else "certifies nothing (threshold inf)"
                (ok if qq.calibrated and fin else warn)(f"bundle question {q!r}", detail)
            (ok if cert else warn)("certificate.json", "present" if cert else "missing: run shad0w train / shadow / certify")
        except Exception as e:
            fail("bundle", f"{a.bundle}: {e}")
    if a.log:
        n = sum(1 for l in open(a.log, encoding="utf-8") if l.strip()) if os.path.exists(a.log) else 0
        st = _config.resolve(path=a.config)
        (ok if n >= st.min_rows else warn)("log", f"{n:,} rows ({'ready to train' if n >= st.min_rows else f'need {st.min_rows:,}'})")
    w = max(len(r[1]) for r in rows)
    for mark, label, detail in rows:
        print(f"[{mark}] {label:<{w}}  {detail}")
    print("\nall good" if not fails else f"\n{fails} problem(s); fix the [!!] lines")
    return 1 if fails else 0


def main(argv=None):
    try:
        return _main(argv)
    except FileNotFoundError as e:
        print(f"shad0w: {e}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] in ("sklearn", "scipy", "torch", "sentence_transformers"):
            print(f"shad0w: {e.name} is not installed; training needs: pip install \"shad0wllm[compile]\"", file=sys.stderr)
            return 2
        raise
    except ImportError as e:
        print(f"shad0w: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"shad0w: {e}", file=sys.stderr)
        return 2


def _main(argv=None):
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
    s.add_argument("--llm-latency-ms", type=float, help="assumed LLM latency, to show time saved")

    px = sub.add_parser("proxy", help="OpenAI-compatible gateway: answers marked decisions from the table, forwards the rest")
    px.add_argument("--upstream", default="https://api.openai.com/v1", help="the real API base URL (OpenAI, Groq, Ollama, vLLM, ...)")
    px.add_argument("--dir", help="where logs and bundles live: <dir>/<question>/{log.jsonl,bundle/} (default shad0w)")
    px.add_argument("--model", help="upstream model for requests that use model='shad0w/<question>'")
    px.add_argument("--api-key-env", help="send this environment variable as the upstream key (default: forward the client's)")
    px.add_argument("--host", default="127.0.0.1")
    px.add_argument("--port", type=int, default=8010)
    px.add_argument("--audit-rate", type=float, help="share of decisions spot-checked against the LLM (default 0.01)")
    px.add_argument("--auto-train", type=int, help="retrain a question every N new LLM answers (needs shad0wllm[compile])")
    px.add_argument("--alpha", type=float, help="certified disagreement bound (default 0.05)")
    px.add_argument("--delta", type=float, help="certificate failure probability (default 0.1)")
    px.add_argument("--min-rows", type=int, help="logged answers needed before auto-train runs (default 1000)")
    px.add_argument("--mode", choices=("serve", "shadow", "off"), help="serve (default), shadow (always return the LLM's answer), off")
    px.add_argument("--canary", type=float, help="share of eligible traffic the table may answer, 0-1 (default 1)")
    px.add_argument("--never-serve", help="comma-separated labels that always go to the LLM")
    px.add_argument("--min-confidence", type=float, help="manual confidence floor on top of the certificate")
    px.add_argument("--capture", action="append", help="how requests are recognised as decisions: header, model, tools, "
                    "decisions, json_schema (repeatable; default header,model,tools,decisions)")
    px.add_argument("--timeout", type=float, help="upstream timeout in seconds (default 120)")
    px.add_argument("--trace", help="JSON Lines file receiving every decision")
    px.add_argument("--bundle", action="append", metavar="Q=PATH", help="use a bundle at PATH for question Q (repeatable)")
    px.add_argument("--cost-per-call", type=float, help="cost of one LLM call, to show money saved")
    px.add_argument("--llm-latency-ms", type=float, help="assumed LLM latency until measured, to show time saved")
    px.add_argument("--schema", help="optional schema.json naming each question's options")
    px.add_argument("--no-text", action="store_true", help="keep request text out of the dashboard")
    px.add_argument("--config", help="path to a shad0w.toml (default ./shad0w.toml or $SHAD0W_CONFIG)")

    tr = sub.add_parser("train", help="compile + certify a bundle from logged LLM answers (auto-detects questions and options)")
    tr.add_argument("--log", help="JSON Lines log of your model's answers")
    tr.add_argument("--out", help="bundle folder to write")
    tr.add_argument("--dir", help="proxy/decision folder layout instead of --log/--out: <dir>/<question>/")
    tr.add_argument("--question")
    tr.add_argument("--schema", help="optional schema.json (default: options = the answers seen in the log)")
    tr.add_argument("--teacher", default="unspecified")
    tr.add_argument("--alpha", type=float, help="certified disagreement bound (default 0.05)")
    tr.add_argument("--delta", type=float, help="certificate failure probability (default 0.1)")
    tr.add_argument("--min-rows", type=int, help="logged answers needed (default 1000; 100 is the hard minimum)")
    tr.add_argument("--gate", action="store_true", help="keep the old bundle unless the new one certifies >= 80%% of its share")
    tr.add_argument("--config", help="path to a shad0w.toml")

    cf = sub.add_parser("config", help="print every effective setting and where it came from, or write a starter shad0w.toml")
    cf.add_argument("--explain", action="store_true", help="(default) show the effective settings")
    cf.add_argument("--question", help="also apply the [questions.<name>] section")
    cf.add_argument("--config", help="path to a shad0w.toml")
    cf.add_argument("--init", action="store_true", help="write a commented shad0w.toml with every setting")

    dr = sub.add_parser("doctor", help="check Python, the C core, training deps, Node, config, keys, upstream and a bundle")
    dr.add_argument("--bundle")
    dr.add_argument("--log")
    dr.add_argument("--upstream", help="an OpenAI-compatible base URL to probe")
    dr.add_argument("--llm", help="provider/model whose API key should be set")
    dr.add_argument("--config")

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
    if a.cmd == "config":
        return _config_cmd(a)
    if a.cmd == "doctor":
        return _doctor(a)
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
        bundles = dict(b.split("=", 1) for b in (a.bundle or []) if "=" in b)
        capture = [c for x in (a.capture or []) for c in x.split(",") if c] or None
        never = [x.strip() for x in a.never_serve.split(",") if x.strip()] if a.never_serve else None
        srv = proxy(a.upstream, host=a.host, port=a.port, folder=a.dir, model=a.model, api_key=key, audit_rate=a.audit_rate,
                    auto_train=a.auto_train, alpha=a.alpha, delta=a.delta, min_rows=a.min_rows, mode=a.mode, canary=a.canary,
                    never_serve=never, min_confidence=a.min_confidence, capture=capture, timeout=a.timeout, trace=a.trace,
                    cost_per_call=a.cost_per_call, llm_latency_ms=a.llm_latency_ms, keep_text=not a.no_text, schema=schema,
                    bundles=bundles, config=a.config)
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
        _need_bundle(a.bundle)
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
        _need_bundle(a.bundle)
        m = api.load(a.bundle, native=False)
        _, by_q = _rows_by_question(a.data, m.questions)
        for q, (texts, labels) in by_q.items():
            print(q, json.dumps(m.calibrate(q, texts, labels, a.alpha)))
        m.save(a.bundle)
    elif a.cmd == "decide":
        _need_bundle(a.bundle)
        print(json.dumps(api.load(a.bundle).decide(a.text, exposed=a.exposed), indent=2, ensure_ascii=False))
    elif a.cmd == "serve":
        from .observe import Metrics
        from .server import serve
        _need_bundle(a.bundle)
        serve(a.bundle, host=a.host, port=a.port, metrics=Metrics(cost_per_call=a.cost_per_call, llm_latency_ms=a.llm_latency_ms))
    elif a.cmd == "bench":
        _need_bundle(a.bundle)
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
