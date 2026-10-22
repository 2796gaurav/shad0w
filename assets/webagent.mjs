// The web-agent playground: a shad0w table runs a shopping errand from the pages' raw HTML, in this tab, and hands
// steps back to the LLM (played here by a script) when it isn't certified or a safety check fires.
import { Bundle } from "./shad0w.mjs";
import { Episode, Shop, makeTask, MAX_STEPS } from "./webagent-core.mjs";

const $ = (id) => document.getElementById(id);
const [bundle, info] = await Promise.all([Bundle.load(new URL("bundle", location.href).href), fetch("webagent.json").then((r) => r.json())]);
const KNOWN = new Set(info.known_ids), LLM_S = info.collect.llm_s;
const pct = (x, d = 0) => `${(100 * x).toFixed(d)}%`;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const get = (o, p) => p.split(".").reduce((a, k) => (a == null ? a : a[k]), o);
document.querySelectorAll("[data-w]").forEach((e) => { const v = get(info, e.dataset.w); e.textContent = typeof v === "number" ? Math.round(v).toLocaleString("en-US") : v; });

// ---------------------------------------------------------------- settings
const opts = () => ({ shift: $("oShift").checked, novel: $("oNovel").checked, progress: $("oProg").checked, risky: $("oRisky").checked, known: KNOWN });
const SHOP_CSS = `<style>
body{font:15px/1.45 system-ui,sans-serif;margin:0;background:#fff;color:#14213d}
nav{display:flex;gap:14px;padding:10px 16px;background:#14213d}nav a{color:#fff;text-decoration:none;font-weight:600}nav a:first-child{color:#fca311}
main{padding:14px 18px}h1{font-size:21px;margin:4px 0 12px}h2{font-size:15px}label{display:block;font-size:13px;margin:8px 0 3px}
input,select{font:inherit;padding:6px 8px;border:1px solid #9aa3b5;border-radius:6px;min-width:180px}input[type=checkbox]{min-width:0;display:inline-block;margin-right:6px}
label[for=terms]{display:inline}button{font:600 14px system-ui;padding:7px 14px;margin:10px 8px 0 0;border-radius:7px;border:0;background:#14213d;color:#fff}
ul{padding-left:18px}li{margin:4px 0}a{color:#14213d}.alert,.empty{background:#fde8c4;border-left:4px solid #fca311;padding:6px 10px}
.cookie{position:fixed;left:10px;right:10px;bottom:10px;background:#fff;border:2px solid #14213d;border-radius:10px;padding:8px 14px;box-shadow:0 8px 30px #0004;z-index:5}
.modal{position:fixed;inset:0;background:#000a;display:grid;place-content:center;text-align:center;z-index:6}.modal p{background:#fff;margin:0;padding:18px 22px 6px;border-radius:12px 12px 0 0;font-weight:600}
.modal button{margin:0;border-radius:0 0 12px 12px;width:100%}footer{padding:12px 18px;color:#6b7385;font-size:12px}
.hit{outline:3px solid #fca311!important;outline-offset:3px;box-shadow:0 0 0 7px #fca31155!important}</style>`;
function show(html, hitId) {
  let h = html.replace("<head>", "<head>" + SHOP_CSS);
  if (hitId) h = h.replace(new RegExp(`id=(["'])${hitId}\\1`), (m) => `${m} class="hit"`).replace(/class="hit" class="(\w+)"/, 'class="hit $1"');
  $("wFrame").srcdoc = h;
}

// ---------------------------------------------------------------- one errand, step by step
let ep, busy = false, stop = false, log = [];
function firstSeed() { for (let s = 1; s < 500; s++) { const a = new Shop(s), t = makeTask(s); if (a.banner && a.loginWall && !a.oos[t.slug].has(t.variant)) return s; } return 1; }
function fresh(seed) {
  ep = new Episode(bundle, seed, opts()); log = [];
  $("wTask").textContent = `#${seed}: ${ep.task.text}${ep.app.oos[ep.task.slug].has(ep.task.variant) ? "" : ""}`;
  $("wUrl").textContent = "bazaar.example" + ep.app.path; show(ep.html);
  $("wDec").innerHTML = '<span class="muted">Press <b>Run</b> or <b>Step</b>.</span>'; $("wText").textContent = "";
  $("wTl").innerHTML = ""; stats();
}
function timeIt(text) { const t0 = performance.now(); let n = 0; while (performance.now() - t0 < 4) { for (let i = 0; i < 20; i++) bundle.decide(text, { probabilities: false }); n += 20; } return (performance.now() - t0) * 1000 / n; }
const label = (a) => a.replace("click:", "click ").replace("type:", "type ").replace("select:", "choose ").replace(/=.*/, "").replace("fail:out-of-stock", "stop: out of stock");
function stats() {
  const tUs = log.filter((x) => x.by === "table").reduce((s, x) => s + x.us, 0), llm = log.filter((x) => x.by === "llm").length;
  const done = ep.over ? (ep.ok ? `<b class="wa-ok">✓ errand done right</b>` : `<b class="wa-bad">✗ ${ep.final ? "finished wrong" : `stuck: gave up after ${MAX_STEPS} steps`}</b>`) : "";
  $("wStats").innerHTML = `<div><b>${ep.steps}</b><span>steps</span></div><div><b class="o">${ep.served}</b><span>by the table</span></div>`
    + `<div><b class="l">${llm}</b><span>by the LLM</span></div><div><b>${tUs < 1000 ? Math.round(tUs) + " µs" : (tUs / 1000).toFixed(1) + " ms"}</b><span>table time, all its steps</span></div>`
    + `<div><b>${(llm * LLM_S).toFixed(1)} s</b><span>LLM time (at ${LLM_S} s a call)</span></div><div><b>${(ep.steps * LLM_S).toFixed(0)} s</b><span>if the LLM did every step</span></div>${done ? `<div class="wa-done">${done}</div>` : ""}`;
}
async function step() {
  if (ep.over) return;
  const dec = ep.decide(), us = timeIt(dec.text), el = dec.action.includes(":") && !dec.action.startsWith("fail") ? dec.action.split(":")[1].split("=")[0] : null;
  $("wText").textContent = dec.text; show(ep.html, el);
  const fast = $("oFast").checked || reduce;
  if (dec.by === "table") {
    $("wDec").innerHTML = `<div class="wa-v t"><i>⚡</i><div><b>${esc(label(dec.action))}</b><span>the table · certified · ${us < 10 ? us.toFixed(1) : Math.round(us)} µs in this tab</span></div></div>`;
  } else {
    const ms = fast ? 250 : LLM_S * 1000;
    const why = dec.a.certified || dec.reason !== "not certified" ? dec.reason : `not certified (the table's guess: ${label(dec.a.choice)})`;
    $("wDec").innerHTML = `<div class="wa-v l"><i>…</i><div><b>Asking the LLM</b><span>${esc(why)}</span></div></div><div class="wa-wait"><i style="animation-duration:${ms}ms"></i></div>`;
    await sleep(ms);
    $("wDec").innerHTML = `<div class="wa-v l"><i>✓</i><div><b>${esc(label(dec.action))}</b><span>the LLM · ${esc(why)} · ~${LLM_S} s</span></div></div>`;
  }
  log.push({ by: dec.by, us, action: dec.action, why: dec.reason });
  const last = $("wTl").lastElementChild, cls = `wa-chip ${dec.by === "table" ? "t" : "l"}`;
  if (last && last.dataset.a === dec.action && last.className === cls) {   // a repeated step: one chip with a count
    last.dataset.n = +last.dataset.n + 1; last.textContent = `${label(dec.action)} ×${last.dataset.n}`;
  } else {
    const chip = document.createElement("span"); chip.className = cls; chip.textContent = label(dec.action); chip.dataset.a = dec.action; chip.dataset.n = 1;
    chip.title = dec.by === "table" ? "the table, certified" : `the LLM: ${dec.reason}`; $("wTl").append(chip);
  }
  await sleep(dec.by === "table" ? (fast ? 180 : 520) : 200);
  ep.apply(dec);
  $("wUrl").textContent = "bazaar.example" + ep.app.path; show(ep.html); stats();
}
async function runAll() {
  if (busy) { stop = true; return; }
  if (ep.over) fresh(ep.seed);
  busy = true; stop = false; $("wRun").textContent = "■ Stop";
  while (!ep.over && !stop) await step();
  busy = false; $("wRun").textContent = "▶ Run the errand";
}
$("wRun").onclick = runAll;
$("wStep").onclick = async () => { if (!busy) { busy = true; await step(); busy = false; } };
$("wNew").onclick = () => { stop = true; setTimeout(() => fresh(1 + Math.floor(Math.random() * 1e6)), busy ? 300 : 0); };
for (const id of ["oShift", "oNovel", "oProg", "oRisky"]) $(id).addEventListener("change", () => { stop = true; setTimeout(() => fresh(ep.seed), busy ? 300 : 0); });
fresh(firstSeed());

// ---------------------------------------------------------------- 100 errands
function cfgText() {
  const o = opts(), on = [o.novel && "novelty check", o.progress && "progress check", o.risky && '"Place order" to the LLM'].filter(Boolean);
  return `${o.shift ? "Changed site" : "Unchanged site"} · ${on.length ? on.join(", ") : "no checks"}`;
}
const upd = () => { $("wBatchCfg").textContent = cfgText(); }; ["oShift", "oNovel", "oProg", "oRisky"].forEach((id) => $(id).addEventListener("change", upd)); upd();
$("wBatch").onclick = async () => {
  $("wBatch").disabled = true; $("wBatchOut").innerHTML = '<div class="panel"><p>Running…</p></div>'; await sleep(30);
  const o = opts(), base = 1 + Math.floor(Math.random() * 1e6); let ok = 0, served = 0, steps = 0, llm = 0, cap = 0; const t0 = performance.now();
  for (let s = 0; s < 100; s++) { const e = new Episode(bundle, base + s, o); while (!e.over) e.apply(e.decide()); ok += e.ok; served += e.served; steps += e.steps; llm += e.llm; cap += !e.ok && !e.final; }
  const ms = performance.now() - t0, ref = (o.shift ? info.shift : info.eval).llm_only.llm_calls_per_episode;
  $("wBatchOut").innerHTML = [
    [ok === 100 ? "✓" : "✗", `${ok} / 100`, `errands done right${cap ? ` · ${cap} got stuck and gave up after ${MAX_STEPS} steps` : ""}`],
    ["⚡", pct(served / steps), `of ${steps.toLocaleString("en-US")} steps answered by the table (all 100 errands: ${Math.round(ms)} ms in this tab, LLM turns included)`],
    ["…", (llm / 100).toFixed(2), `LLM calls per errand. The real LLM alone needed ${ref.toFixed(1)} on ${o.shift ? "the changed" : "this"} site, about ${Math.round(ref * LLM_S)} s an errand`]]
    .map(([i, b, p]) => `<div class="panel ${i === "✗" ? "bad" : ""}"><i>${i}</i><b>${b}</b><p>${p}</p></div>`).join("");
  $("wBatch").disabled = false;
};

// ---------------------------------------------------------------- measured numbers
const row = (name, r, extra = "") => `<tr><th>${name}</th><td class="${r.success < 1 ? "bad" : ""}">${pct(r.success)}</td><td>${r.llm_calls_per_episode.toFixed(2)}</td><td>${pct(r.served_by_table)}</td>${extra}</tr>`;
const E = info.eval;
$("tEval").innerHTML = `<thead><tr><th></th><th>errands done right</th><th>LLM calls per errand</th><th>steps by the table</th></tr></thead><tbody>`
  + row("LLM only", E.llm_only) + row("shad0w, LLM when not certified", E.shad0w_then_llm) + row('same, "Place order" always to the LLM', E.shad0w_then_llm_confirm_risky) + "</tbody>";
const S = info.shift, G5 = info.guards_5000, G7 = info.guards_7000;
const two = (name, a, b) => `<tr><th>${name}</th><td class="${a.success < 1 ? "bad" : ""}">${pct(a.success)}</td><td>${a.llm_calls_per_episode.toFixed(2)}</td><td>${pct(a.served_by_table)}</td>`
  + (b ? `<td>${pct(b.success)} · ${b.llm_calls_per_episode.toFixed(2)} · ${pct(b.served_by_table)}</td>` : "<td class='muted'>–</td>") + "</tr>";
$("tShift").innerHTML = `<thead><tr><th></th><th>errands done right</th><th>LLM calls per errand</th><th>steps by the table</th><th>100 fresh errands (right · calls · table)</th></tr></thead><tbody>`
  + two("LLM only", S.llm_only) + two("shad0w, no checks", S.shad0w_then_llm) + two('shad0w, "Place order" to the LLM', S.shad0w_then_llm_confirm_risky)
  + two("+ progress check", G5.guard_progress, G7.guard_progress) + two("+ novelty check", G5.guard_novel, G7.guard_novel) + two("+ both checks", G5.guard_both, G7.guard_both) + "</tbody>";
const U = info.unchanged_guards.unchanged_guard_both;
$("wUnch").textContent = `${pct(U.success)} of errands done right, ${pct(U.served)} of steps by the table, ${U.llm_calls} LLM calls.`;
const M = info.m2w, P = info.pages["alpha_0.1"];
$("fTask").textContent = pct(M.cross_task.top1); $("fSite").textContent = pct(M.cross_website.top1); $("fBase").textContent = pct(M.cross_website.overlap_baseline_top1);
$("fPgSeen").textContent = pct(P.held_out_tasks_seen_sites.agreement_all); $("fPgNew").textContent = pct(P.unseen_websites.agreement_all);
$("wFail").innerHTML = [
  ["✗", pct(M.cross_website.top1), "right element first time, on websites it never saw (out of ~600 per page)"],
  ["✗", pct(M.cross_task.top1), "right element first time, new tasks on websites it trained on"],
  ["0", "0%", "of those steps certified, even at a 20% error budget. It knows it doesn't know."]]
  .map(([i, b, p]) => `<div class="panel"><i>${i}</i><b>${b}</b><p>${p}</p></div>`).join("");
