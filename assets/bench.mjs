// The Decision Bench page: every number comes from results.json (bench/score.py's output, copied at build time).
const $ = (id) => document.getElementById(id);
const R = await (await fetch("results.json")).json();
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x, d = 1) => x == null ? "–" : `${(100 * x).toFixed(d)}%`;
const fmtT = (us) => us == null ? "–" : us < 1000 ? `${us < 10 ? us.toFixed(1) : Math.round(us)} µs` : us < 1e6 ? `${(us / 1000).toFixed(us < 1e4 ? 1 : 0)} ms` : `${(us / 1e6).toFixed(1)} s`;
const NAME = { "shad0w-auto": "shad0w", "shad0w-linear": "shad0w (linear)", "shad0w-deep": "shad0w (deep)", "julia-1": "Julia-1", "laya": "Laya",
  "kev-0.8b": "Kev-0.8B", "decider-0.8b": "Decider-0.8B", "gliner2.5-decide": "GLiNER2.5-Decide" };
const TASK = { banking77: ["BANKING77", "77 banking intents", "Which of 77 intents is this support message? (\"I lost my card\")"],
  clinc150: ["CLINC150", "150 intents + out of scope", "Which of 150 assistant intents, or none of them?"],
  ag_news: ["AG News", "4 news topics", "World, sports, business or sci/tech, from a headline and lead."],
  sst2: ["SST-2", "sentiment", "Is this movie review positive or negative? Negation and tone matter."],
  sms_spam: ["SMS spam", "spam or not", "Is this text message spam?"],
  prompt_injection: ["Prompt injection", "attack or safe", "Is this input trying to hijack an LLM?"] };
const ORDER = ["banking77", "clinc150", "ag_news", "sst2", "sms_spam", "prompt_injection"].filter((t) => R.tasks[t]);
const isMe = (s) => s.startsWith("shad0w");
const dms = (t) => Object.values(R.tasks[t].systems).filter((s) => !isMe(s.system));

// ---------------------------------------------------------------- headline tiles
{
  const me = ORDER.map((t) => R.tasks[t].systems["shad0w-auto"]);
  const dmT = ORDER.flatMap((t) => dms(t).map((s) => s.p50_us));
  const wins = ORDER.filter((t) => R.tasks[t].systems["shad0w-auto"].accuracy >= Math.max(...dms(t).map((s) => s.accuracy))).length;
  const cov = me.map((s) => s.certified_coverage), dmCert = ORDER.flatMap((t) => dms(t)).filter((s) => s.certified_coverage > 0).length;
  const tile = (big, small) => `<div><b>${big}</b><small>${small}</small></div>`;
  $("heads").innerHTML = tile(`${fmtT(Math.min(...me.map((s) => s.p50_us)))}–${fmtT(Math.max(...me.map((s) => s.p50_us)))}`, "shad0w per decision, one thread, on every task") +
    tile(`${fmtT(Math.min(...dmT))}–${fmtT(Math.max(...dmT))}`, `open decision models per decision (CPU, ${R.hardware.dm_threads} threads)`) +
    tile(`${wins} of ${ORDER.length}`, "tasks where shad0w is at least as accurate as the best decision model") +
    tile(`${pct(Math.min(...cov), 0)}–${pct(Math.max(...cov), 0)}`, `of rows shad0w serves under the 5% certificate; decision models: ${dmCert ? `${dmCert} task-model pairs` : "none"} certify any`);
  $("threads").textContent = R.hardware.dm_threads;
  $("hw").textContent = `${R.hardware.cpu}, ${R.hardware.os.split("-")[0]}, ${R.hardware.date}, shad0w ${R.hardware.shad0w}`;
}

// ---------------------------------------------------------------- tabs
let cur = ORDER[0];
$("tabs").innerHTML = ORDER.map((t) => `<button type="button" role="tab" data-t="${t}"><b>${TASK[t]?.[0] ?? t}</b><small>${TASK[t]?.[1] ?? ""}</small></button>`).join("");
$("tabs").addEventListener("click", (e) => { const b = e.target.closest("[data-t]"); if (b) show(b.dataset.t); });
try { const h = location.hash.slice(1); if (R.tasks[h]) cur = h; } catch {}

// ---------------------------------------------------------------- the chart: accuracy vs log time
const tip = $("tip");
function chart(t) {
  const sys = Object.values(R.tasks[t].systems), box = $("chart"), W = Math.max(300, box.clientWidth), H = 380;
  const m = { l: 48, r: 18, t: 16, b: 40 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const lo = Math.min(...sys.map((s) => s.accuracy_ci95?.[0] ?? s.accuracy)), yMin = Math.max(0, Math.floor((lo - 0.05) * 10) / 10);
  const x = (us) => m.l + (Math.log10(us) - 0) / 7 * iw, y = (a) => m.t + (1 - (a - yMin) / (1 - yMin)) * ih;
  let g = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Accuracy against time per decision for ${esc(TASK[t]?.[0])}">`;
  for (let k = 0; k <= 7; k++) { const xx = x(10 ** k); g += `<line x1="${xx}" x2="${xx}" y1="${m.t}" y2="${m.t + ih}" class="bn-grid"/><text x="${xx}" y="${H - 16}" class="bn-ax" text-anchor="middle">${["1 µs", "10 µs", "100 µs", "1 ms", "10 ms", "100 ms", "1 s", "10 s"][k]}</text>`; }
  for (let a = yMin; a <= 1.0001; a += 0.1) { const yy = y(a); g += `<line x1="${m.l}" x2="${m.l + iw}" y1="${yy}" y2="${yy}" class="bn-grid"/><text x="${m.l - 8}" y="${yy + 4}" class="bn-ax" text-anchor="end">${Math.round(a * 100)}%</text>`; }
  g += `<text x="${m.l + iw}" y="${H - 2}" class="bn-ax" text-anchor="end">time per decision →</text>`;
  const pts = sys.slice().sort((a, b) => isMe(a.system) - isMe(b.system));
  // labels: try right, left, above, below; skip a label that would collide (the tooltip still names the point)
  const boxes = pts.map((s) => ({ x: x(s.p50_us) - 10, y: y(s.accuracy) - 10, w: 20, h: 20 })), placed = [];
  const hit = (b) => [...boxes, ...placed].some((o) => b.x < o.x + o.w && o.x < b.x + b.w && b.y < o.y + o.h && o.y < b.y + b.h);
  const labelAt = (s, r) => { const cx = x(s.p50_us), cy = y(s.accuracy), name = NAME[s.system] ?? s.system, w = name.length * 7.7 + 6, h = 17;
    for (const [lx, ly, anchor] of [[cx + r + 5, cy + 4, "start"], [cx - r - 5, cy + 4, "end"], [cx, cy - r - 6, "middle"], [cx, cy + r + 15, "middle"]]) {
      const bx = anchor === "start" ? lx : anchor === "end" ? lx - w : lx - w / 2, b = { x: bx, y: ly - 13, w, h };
      if (bx < m.l || bx + w > W - 2 || hit(b)) continue;
      placed.push(b); return `<text x="${lx}" y="${ly}" text-anchor="${anchor}" class="bn-lab${isMe(s.system) ? " me" : ""}">${esc(name)}</text>`; }
    return ""; };
  const labels = [];
  for (const s of pts.slice().sort((a, b) => (b.system === "shad0w-auto") - (a.system === "shad0w-auto") || b.accuracy - a.accuracy)) {
    const me = isMe(s.system), main = s.system === "shad0w-auto"; if (main || !me) labels.push(labelAt(s, main ? 9 : 7)); }
  for (const s of pts) {
    const cx = x(s.p50_us), cy = y(s.accuracy), me = isMe(s.system), main = s.system === "shad0w-auto";
    if (s.accuracy_ci95) g += `<line x1="${cx}" x2="${cx}" y1="${y(s.accuracy_ci95[1])}" y2="${y(s.accuracy_ci95[0])}" class="bn-ci${me ? " me" : ""}"/>`;
    const r = main ? 9 : me ? 6 : 7;
    g += me ? `<path d="M${cx} ${cy - r}L${cx + r} ${cy}L${cx} ${cy + r}L${cx - r} ${cy}Z" class="bn-pt me${main ? "" : " alt"}" data-s="${s.system}"/>`
            : `<circle cx="${cx}" cy="${cy}" r="${r}" class="bn-pt dm" data-s="${s.system}"/>`;
  }
  for (const s of pts) g += `<circle cx="${x(s.p50_us)}" cy="${y(s.accuracy)}" r="16" class="bn-hit" data-s="${s.system}"/>`;
  box.innerHTML = g.replace(/(<circle[^>]*class="bn-hit"[\s\S]*)$/, labels.join("") + "$1") + "</svg>";
  const showTip = (e) => { const el = e.target.closest("[data-s]"); if (!el) { tip.hidden = true; return; } const s = R.tasks[t].systems[el.dataset.s];
    tip.innerHTML = `<b>${esc(NAME[s.system] ?? s.system)}</b><span>${esc(s.info?.model ?? (s.info?.chosen ? `${s.info.chosen} table` : ""))}</span>
      <span>accuracy <b>${pct(s.accuracy)}</b> (95%: ${pct(s.accuracy_ci95?.[0])}–${pct(s.accuracy_ci95?.[1])})</span>
      <span>time per decision <b>${fmtT(s.p50_us)}</b>${s.single_thread_p50_us ? ` · one thread ${fmtT(s.single_thread_p50_us)}` : ""}</span>
      <span>certified <b>${pct(s.certified_coverage, 0)}</b>${s.served_error != null ? ` · error on served ${pct(s.served_error)}` : ""}</span>
      <span>${s.size_mb >= 1 ? `${s.size_mb.toFixed(s.size_mb < 10 ? 1 : 0)} MB` : `${Math.round(s.size_mb * 1024)} KB`} · ${s.n_test} test rows</span>`;
    tip.hidden = false; const r = el.getBoundingClientRect();
    tip.style.left = `${Math.min(innerWidth - 300, Math.max(8, r.left + scrollX + 14))}px`; tip.style.top = `${r.top + scrollY + 14}px`; };
  box.onmousemove = showTip; box.onclick = showTip; box.onmouseleave = () => { tip.hidden = true; };
}

// ---------------------------------------------------------------- cascade: shad0w in front of each model
function cascade(t) {
  const c = R.tasks[t].cascades || {};
  $("cascade").innerHTML = Object.entries(c).sort((a, b) => b[1].accuracy - a[1].accuracy).map(([dm, v]) => `
    <div class="bn-cas">
      <div class="bn-cas-h"><b>shad0w → ${esc(NAME[dm] ?? dm)}</b><span><b class="o">${v.speedup.toLocaleString("en-US", { maximumFractionDigits: 1 })}× faster</b> · shad0w served ${pct(v.served_by_shad0w, 0)}</span></div>
      <div class="bn-bar"><i style="width:${100 * v.decision_model_alone_accuracy}%"></i><em>${esc(NAME[dm] ?? dm)} alone</em><b>${pct(v.decision_model_alone_accuracy)}</b></div>
      <div class="bn-bar me"><i style="width:${100 * v.accuracy}%"></i><em>with shad0w in front</em><b>${pct(v.accuracy)}</b></div>
    </div>`).join("") || `<p class="muted">No cascade for this task.</p>`;
}

// ---------------------------------------------------------------- table
let sortKey = "accuracy", sortDir = -1;
const COLS = [["system", "system"], ["accuracy", "accuracy (95% interval)"], ["certified_coverage", "certified at α = 5%"], ["served_error", "error on served"],
  ["p50_us", "time per decision"], ["single_thread_p50_us", "one thread"], ["size_mb", "size"], ["n_test", "rows"]];
function table(t) {
  const sys = Object.values(R.tasks[t].systems).sort((a, b) => sortKey === "system" ? sortDir * String(NAME[a.system]).localeCompare(NAME[b.system]) : sortDir * ((a[sortKey] ?? -1) - (b[sortKey] ?? -1)));
  $("table").innerHTML = `<tr>${COLS.map(([k, n]) => `<th><button type="button" data-k="${k}">${n}${sortKey === k ? (sortDir < 0 ? " ↓" : " ↑") : ""}</button></th>`).join("")}</tr>` + sys.map((s) => `
    <tr${s.system === "shad0w-auto" ? ' class="hl"' : ""}><td><b>${esc(NAME[s.system] ?? s.system)}</b><small>${esc(s.info?.model ?? (s.info?.chosen ? `auto chose the ${s.info.chosen} table` : s.info?.student ? `${s.info.student} table` : ""))}</small></td>
      <td><b>${pct(s.accuracy)}</b> <small>${pct(s.accuracy_ci95?.[0])}–${pct(s.accuracy_ci95?.[1])}</small></td><td>${pct(s.certified_coverage, 0)}</td><td>${s.served_error == null ? "–" : pct(s.served_error)}</td>
      <td>${fmtT(s.p50_us)}</td><td>${isMe(s.system) ? fmtT(s.p50_us) : fmtT(s.single_thread_p50_us)}</td><td>${s.size_mb >= 1 ? `${s.size_mb.toFixed(s.size_mb < 10 ? 1 : 0)} MB` : `${Math.round(s.size_mb * 1024)} KB`}</td><td>${s.n_test}${s.subset ? "*" : ""}</td></tr>`).join("");
  const sub = Object.values(R.tasks[t].systems).filter((s) => s.subset).map((s) => NAME[s.system]);
  $("note").textContent = (sub.length ? `* ${sub.join(", ")} took over 1 s per decision here and answered the first 150 test rows. ` : "") +
    "shad0w times are the C core in-process on one thread; decision models run over localhost HTTP on the CPU. Accuracy is against the dataset's true labels.";
}
$("table").addEventListener("click", (e) => { const b = e.target.closest("[data-k]"); if (!b) return; sortDir = sortKey === b.dataset.k ? -sortDir : -1; sortKey = b.dataset.k; table(cur); });

// ---------------------------------------------------------------- summary across tasks
$("summary").innerHTML = `<tr><th>task</th><th>shad0w: accuracy / certified / time</th><th>most accurate decision model</th><th>fastest decision model</th><th>shad0w in front of the most accurate</th></tr>` +
  ORDER.map((t) => { const me = R.tasks[t].systems["shad0w-auto"], d = dms(t), best = d.slice().sort((a, b) => b.accuracy - a.accuracy)[0], fast = d.slice().sort((a, b) => a.p50_us - b.p50_us)[0], c = R.tasks[t].cascades?.[best.system];
    return `<tr><td><button type="button" class="linkish" data-t="${t}">${TASK[t]?.[0] ?? t}</button></td><td><b>${pct(me.accuracy)}</b> / ${pct(me.certified_coverage, 0)} / ${fmtT(me.p50_us)}</td>
      <td>${esc(NAME[best.system])}: ${pct(best.accuracy)}, ${fmtT(best.p50_us)}${best.subset ? " (150 rows)" : ""}</td><td>${esc(NAME[fast.system])}: ${pct(fast.accuracy)}, ${fmtT(fast.p50_us)}</td>
      <td>${c ? `${pct(c.accuracy)}, <b class="o">${c.speedup.toLocaleString("en-US", { maximumFractionDigits: 1 })}× faster</b>` : "–"}</td></tr>`; }).join("");
$("summary").addEventListener("click", (e) => { const b = e.target.closest("[data-t]"); if (b) { show(b.dataset.t); $("tabs").scrollIntoView({ behavior: "smooth", block: "start" }); } });

function show(t) {
  cur = t;
  $("tabs").querySelectorAll("button").forEach((b) => { b.classList.toggle("on", b.dataset.t === t); b.setAttribute("aria-selected", b.dataset.t === t); });
  $("what").innerHTML = `<b>${esc(TASK[t]?.[0] ?? t)}.</b> ${esc(TASK[t]?.[2] ?? "")}`;
  chart(t); cascade(t); table(t);
  try { history.replaceState(null, "", `#${t}`); } catch {}
}
addEventListener("resize", () => chart(cur));
show(cur);
