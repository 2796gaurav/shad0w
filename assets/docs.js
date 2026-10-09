// shad0w docs: search, "on this page" scroll-spy, table and heading polish, and the interactive figures
// (<div class="viz" data-viz="...">fallback text</div>, see work/plan/DOCS_CONTRACT.md). Vanilla JS, no libraries.
// Every figure keeps its fallback text for screen readers and no-JS readers, starts only when scrolled near
// (IntersectionObserver), pauses off screen, and respects prefers-reduced-motion. Loaded on docs pages only.

const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const D = Object.assign({ alpha: 0.05, delta: 0.1, min_rows: 1000, cal_fraction: 0.3, audit_rate: 0.01, canary: 1 }, window.SHAD0W_DEFAULTS || {});
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (v, d = 0) => (100 * v).toFixed(d).replace(/\.0+$/, "") + "%";
const NS = "http://www.w3.org/2000/svg";
const S = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
};
const ease = { out: (k) => 1 - Math.pow(1 - k, 4), crawl: (k) => k * k * (1.6 - 0.6 * k), lin: (k) => k };

/* ------------------------------------------------------------------ page chrome */

function scrollSpy() {
  const links = $$(".toc a[href^='#']").filter((a) => a.getAttribute("href").length > 1);
  if (!links.length) return;
  const map = new Map(links.map((a) => [decodeURIComponent(a.getAttribute("href").slice(1)), a]));
  const heads = $$(".prose h2[id], .prose h3[id]").filter((h) => map.has(h.id));
  let cur = null, ticking = false;
  const update = () => {
    ticking = false;
    const y = 110;
    let hit = heads[0];
    for (const h of heads) { if (h.getBoundingClientRect().top - y <= 0) hit = h; else break; }
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) hit = heads[heads.length - 1];
    if (hit === cur) return;
    cur = hit;
    links.forEach((a) => { a.classList.remove("on"); a.removeAttribute("aria-current"); });
    const a = hit && map.get(hit.id);
    if (a) {
      a.classList.add("on"); a.setAttribute("aria-current", "location");
      const toc = a.closest(".toc"), r = a.getBoundingClientRect(), tr = toc.getBoundingClientRect();
      if (toc.scrollHeight > toc.clientHeight && (r.top < tr.top + 40 || r.bottom > tr.bottom - 40)) toc.scrollTop += r.top - tr.top - tr.height / 2;
    }
  };
  addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  update();
}

function tables() {
  // wide tables scroll sideways inside their wrapper; tables that fit drop the wrapper's overflow so the header can stick
  const fit = () => $$(".prose .tablewrap").forEach((w) => {
    w.classList.remove("fits");
    const t = w.querySelector("table");
    if (t && t.scrollWidth <= w.clientWidth + 1) w.classList.add("fits");
  });
  fit();
  let to;
  addEventListener("resize", () => { clearTimeout(to); to = setTimeout(fit, 120); });
}

function headingLinks() {
  document.addEventListener("click", async (e) => {
    const a = e.target.closest(".prose .headerlink");
    if (!a) return;
    const url = location.href.split("#")[0] + a.getAttribute("href");
    try { await navigator.clipboard.writeText(url); a.classList.add("copied"); setTimeout(() => a.classList.remove("copied"), 1200); } catch (err) { /* navigation still works */ }
  });
}

/* ------------------------------------------------------------------ search */

function search() {
  const input = $("#dsearch"), box = $("#dresults");
  if (!input || !box) return;
  let index = null, loading = null, results = [], active = -1;
  const load = () => loading || (loading = fetch(new URL("search.json", location.href)).then((r) => r.json()).then((j) => {
    index = [];
    j.forEach((p) => p.h.forEach(([h, id, text], i) => index.push({
      page: p.n, title: p.t, sec: p.s, head: i === 0 && !id ? p.n : h, url: p.u + (id ? "#" + id : ""), text,
      lc: { title: (p.t + " " + p.n).toLowerCase(), head: h.toLowerCase(), text: text.toLowerCase() } })));
  }).catch(() => { index = []; }));
  const terms = (q) => q.toLowerCase().split(/[^\w.@/-]+/).filter((t) => t.length > 1 || /\d/.test(t));
  const score = (it, ts) => {
    let s = 0;
    for (const t of ts) {
      const inT = it.lc.title.includes(t), inH = it.lc.head.includes(t), n = it.lc.text.split(t).length - 1;
      if (!inT && !inH && !n) return 0;
      s += (inT ? 6 : 0) + (inH ? 10 : 0) + Math.min(n, 4) + (it.lc.head.startsWith(t) ? 4 : 0);
    }
    return s;
  };
  const mark = (text, ts) => {
    let out = esc(text);
    ts.forEach((t) => { out = out.replace(new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi"), "\u0001$1\u0002"); });
    return out.replace(/\u0001/g, "<mark>").replace(/\u0002/g, "</mark>");
  };
  const snippet = (text, ts) => {
    const lc = text.toLowerCase();
    let at = -1;
    for (const t of ts) { const i = lc.indexOf(t); if (i >= 0 && (at < 0 || i < at)) at = i; }
    if (at < 0) return text.slice(0, 120) + (text.length > 120 ? "…" : "");
    const a = Math.max(0, at - 40);
    return (a ? "…" : "") + text.slice(a, a + 130) + (a + 130 < text.length ? "…" : "");
  };
  const close = () => { box.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); active = -1; };
  const setActive = (i) => {
    const opts = $$(".dr-item", box);
    if (!opts.length) return;
    active = (i + opts.length) % opts.length;
    opts.forEach((o, k) => o.setAttribute("aria-selected", k === active ? "true" : "false"));
    input.setAttribute("aria-activedescendant", opts[active].id);
    opts[active].scrollIntoView({ block: "nearest" });
  };
  const render = async () => {
    const q = input.value.trim();
    if (!q) { close(); return; }
    await load();
    const ts = terms(q);
    results = ts.length ? index.map((it) => [score(it, ts), it]).filter((x) => x[0] > 0).sort((a, b) => b[0] - a[0]).slice(0, 8).map((x) => x[1]) : [];
    box.innerHTML = results.length ? results.map((r, i) => `<a class="dr-item" id="dr-${i}" role="option" aria-selected="false" href="${esc(r.url)}">
        <span class="dr-path">${esc(r.sec)} › ${esc(r.page)}</span><b>${mark(r.head, ts)}</b><span class="dr-snip">${mark(snippet(r.text, ts), ts)}</span></a>`).join("")
      + `<div class="dr-hint"><kbd>↑</kbd><kbd>↓</kbd> to move · <kbd>Enter</kbd> to open · <kbd>Esc</kbd> to close</div>`
      : `<div class="dr-empty">No results for “${esc(q)}”. Try a setting name like <code>alpha</code> or a word like <em>proxy</em>.</div>`;
    box.hidden = false; input.setAttribute("aria-expanded", "true");
    if (results.length) setActive(0);
  };
  input.addEventListener("focus", load, { once: true });
  input.addEventListener("input", render);
  input.addEventListener("focus", () => { if (input.value.trim()) render(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setActive(active + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive(active - 1); }
    else if (e.key === "Enter") { const o = $$(".dr-item", box)[active]; if (o) { e.preventDefault(); location.href = o.href; close(); } }
    else if (e.key === "Escape") { if (!box.hidden) close(); else { input.value = ""; input.blur(); } }
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".dsearch")) close(); });
  document.addEventListener("keydown", (e) => {
    const typing = e.target.closest && e.target.closest("input, textarea, select, [contenteditable]");
    if ((e.key === "/" && !typing) || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k")) {
      e.preventDefault();
      const menu = input.closest(".sidebar");
      if (menu) menu.scrollIntoView({ block: "nearest" });
      input.focus(); input.select();
    }
  });
}

/* ------------------------------------------------------------------ figure frame */

// Wraps a figure: header (eyebrow + title + optional controls), stage, live caption. The fallback text stays in the
// DOM for screen readers (visually hidden); the drawing itself is aria-hidden and the controls are real buttons.
function frame(el, { kicker, title, controls = "" }) {
  const alt = el.innerHTML.trim();
  el.innerHTML = `<div class="viz-head"><div><span class="viz-k">${esc(kicker)}</span><b class="viz-t">${title}</b></div>${controls}</div>
    <div class="viz-stage"></div><p class="viz-cap" aria-live="polite"></p>${alt ? `<div class="sr-only">${alt}</div>` : ""}`;
  el.classList.add("ready");
  return { stage: $(".viz-stage", el), cap: $(".viz-cap", el), head: $(".viz-head", el) };
}

// requestAnimationFrame loop that runs only while the figure is on screen, the tab is visible and not paused
function loop(el, tick) {
  let on = false, visible = false, paused = false, raf = 0, last = 0;
  const step = (t) => { raf = 0; if (!on) return; tick(t, t - (last || t)); last = t; raf = requestAnimationFrame(step); };
  const sync = () => {
    const want = visible && !paused && !document.hidden;
    if (want && !on) { on = true; last = 0; raf = requestAnimationFrame(step); }
    if (!want && on) { on = false; if (raf) cancelAnimationFrame(raf); raf = 0; }
  };
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; sync(); }).observe(el);
  document.addEventListener("visibilitychange", sync);
  return { pause(p) { paused = p; sync(); }, get paused() { return paused; } };
}

const pauseBtn = () => `<button type="button" class="viz-btn viz-pause" aria-pressed="false" aria-label="Pause the animation"><span class="ic-pause" aria-hidden="true"></span><span class="lbl">Pause</span></button>`;
function wirePause(el, ctl) {
  const b = $(".viz-pause", el);
  if (!b) return;
  b.addEventListener("click", () => {
    ctl.pause(!ctl.paused);
    b.setAttribute("aria-pressed", ctl.paused); b.classList.toggle("on", ctl.paused);
    $(".lbl", b).textContent = ctl.paused ? "Play" : "Pause";
    b.setAttribute("aria-label", ctl.paused ? "Play the animation" : "Pause the animation");
  });
}

// a segmented control: [{v, label}], returns the element; onPick(v) on click / arrow keys
function seg(name, items, cur, onPick) {
  const w = document.createElement("div");
  w.className = "viz-seg"; w.setAttribute("role", "radiogroup"); w.setAttribute("aria-label", name);
  w.innerHTML = items.map((it) => `<button type="button" role="radio" data-v="${esc(it.v)}" aria-checked="${it.v === cur}" tabindex="${it.v === cur ? 0 : -1}">${it.label}</button>`).join("");
  const pick = (b, focus) => {
    $$("button", w).forEach((x) => { const on = x === b; x.setAttribute("aria-checked", on); x.tabIndex = on ? 0 : -1; });
    if (focus) b.focus();
    onPick(b.dataset.v);
  };
  w.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) pick(b); });
  w.addEventListener("keydown", (e) => {
    const bs = $$("button", w), i = bs.indexOf(document.activeElement);
    if (i < 0) return;
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (d) { e.preventDefault(); pick(bs[(i + d + bs.length) % bs.length], true); }
  });
  w.set = (v) => { const b = $(`button[data-v="${v}"]`, w); if (b) pick(b); };
  return w;
}

/* ------------------------------------------------------------------ flow */

const FLOW = {
  wide: { vb: [800, 312], nodes: {
      msg: [16, 113, 150, 64, "A message", "“my card was stolen”"], sh: [236, 103, 170, 84, "shad0w", "a small table on your CPU"],
      ans: [604, 22, 180, 66, "Answered", "in microseconds"], llm: [482, 196, 152, 66, "Your LLM", "a full call, seconds"],
      log: [664, 196, 120, 66, "Logged", "the table learns"] },
    paths: { in: "M166 145 L236 145", sure: "M406 128 C 500 128 510 55 604 55", unsure: "M406 162 C 450 162 440 229 482 229",
      log: "M634 229 L664 229", back: "M724 262 L724 290 Q724 300 714 300 L331 300 Q321 300 321 290 L321 189" },
    labels: [["sure", 476, 76, "lab-ok"], ["not sure", 382, 238, "lab-llm"], ["retrain from the log", 420, 292, "lab-dim"]] },
  narrow: { vb: [360, 470], nodes: {
      msg: [90, 6, 150, 58, "A message", "“my card was stolen”"], sh: [81, 104, 168, 76, "shad0w", "a small table on your CPU"],
      ans: [6, 246, 150, 62, "Answered", "in microseconds"], llm: [178, 246, 142, 62, "Your LLM", "a full call, seconds"],
      log: [178, 380, 142, 62, "Logged", "the table learns"] },
    paths: { in: "M165 64 L165 104", sure: "M135 180 C 135 220 81 206 81 246", unsure: "M195 180 C 195 220 249 206 249 246",
      log: "M249 308 L249 380", back: "M320 411 C 352 411 352 142 249 142" },
    labels: [["sure", 64, 222, "lab-ok"], ["not sure", 222, 222, "lab-llm"], ["retrain", 300, 470 - 6, "lab-dim"]] },
};

function vizFlow(el) {
  const share = Math.min(0.95, Math.max(0.05, parseFloat(el.dataset.share) || 0.6));
  const f = frame(el, { kicker: "Figure · request flow", title: "Sure answers stay local. The rest go to your LLM.", controls: reduce ? "" : pauseBtn() });
  let layout = null, svg, P = {}, N = {}, toks = [], spawnAt = 0, served = 0, sent = 0, debt = 0;
  const cap = () => {
    f.cap.innerHTML = reduce
      ? `In this example shad0w is sure about <b class="ok">${pct(share)}</b> of messages and answers them locally; the other <b class="llm">${pct(1 - share)}</b> go to your LLM and are logged.`
      : `<span class="stat ok"><b>${served}</b> answered by shad0w</span><span class="stat llm"><b>${sent}</b> sent to your LLM, then logged</span><span class="stat dim">example: sure about ${pct(share)} of messages</span>`;
  };
  const draw = () => {
    const want = f.stage.clientWidth < 560 ? "narrow" : "wide";
    if (want === layout) return;
    layout = want; toks.forEach((t) => t.c.remove()); toks = [];
    const L = FLOW[layout];
    f.stage.innerHTML = "";
    svg = S("svg", { viewBox: `0 0 ${L.vb[0]} ${L.vb[1]}`, class: "flow-svg", "aria-hidden": "true" }, f.stage);
    const defs = S("defs", {}, svg);
    const mk = S("marker", { id: "fa-" + el.dataset.vid, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    S("path", { d: "M0 1 L9 5 L0 9", class: "arrowhead" }, mk);
    for (const k in L.paths) P[k] = S("path", { d: L.paths[k], class: "edge e-" + k, "marker-end": `url(#fa-${el.dataset.vid})` }, svg);
    for (const [t, x, y, c] of L.labels) S("text", { x, y, class: "elab " + c }, svg).textContent = t;
    for (const k in L.nodes) {
      const [x, y, w, h, t, s] = L.nodes[k];
      const g = S("g", { class: "node n-" + k, transform: `translate(${x} ${y})` }, svg);
      S("rect", { width: w, height: h, rx: 12, class: "nbox" }, g);
      if (k === "sh") S("rect", { width: w, height: h, rx: 12, class: "nshadow", transform: "translate(5 5)" }, g).parentNode.insertBefore(g.lastChild, g.firstChild);
      S("text", { x: 14, y: h / 2 - 3, class: "nt" }, g).textContent = t;
      S("text", { x: 14, y: h / 2 + 15, class: "ns" }, g).textContent = s;
      if (k === "ans") { const us = S("text", { x: w - 12, y: 20, class: "nus", "text-anchor": "end" }, g); us.textContent = "µs"; }
      N[k] = g;
    }
  };
  const flash = (k, cls = "flash", ms = 260) => { const g = N[k]; if (!g) return; g.classList.remove(cls); g.getBoundingClientRect(); g.classList.add(cls); setTimeout(() => g.classList.remove(cls), ms); };
  const spawn = () => {
    debt += share;
    const sure = debt >= 1 - 1e-9 ? (debt -= 1, true) : false;  // exactly `share` of messages over time, evenly spread
    const c = S("circle", { r: 6, class: "tok" }, svg);
    toks.push({ c, seg: "in", t: 0, sure });
  };
  const DUR = { in: 520, unsure: 1900, log: 420 };
  draw(); cap();
  if (reduce) { svg.classList.add("static"); return; }
  const ctl = loop(el, (now, dt) => {
    if (!svg) return;
    dt = Math.min(dt, 60);
    if (now >= spawnAt) { spawn(); spawnAt = now + 1150; }
    toks = toks.filter((k) => {
      k.t += dt;
      const p = P[k.seg], d = DUR[k.seg], e = Math.min(1, k.t / d);
      const pt = p.getPointAtLength(p.getTotalLength() * (k.seg === "unsure" ? ease.crawl(e) : ease.out(e)));
      k.c.setAttribute("cx", pt.x); k.c.setAttribute("cy", pt.y);
      if (e < 1) return true;
      if (k.seg === "in") {
        flash("sh", "pulse", 300);
        if (k.sure) { k.c.remove(); flash("ans", "flash", 320); served++; cap(); return false; }  // a cut, not a tween: that is the point
        k.seg = "unsure"; k.t = 0; k.c.classList.add("slow"); return true;
      }
      if (k.seg === "unsure") { flash("llm", "flash", 320); k.seg = "log"; k.t = 0; return true; }
      k.c.remove(); flash("log", "flash", 320); P.back.classList.remove("run"); P.back.getBoundingClientRect(); P.back.classList.add("run"); sent++; cap(); return false;
    });
  });
  wirePause(el, ctl);
  new ResizeObserver(() => { const old = layout; draw(); if (old !== layout) cap(); }).observe(f.stage);
}

/* ------------------------------------------------------------------ alpha */

async function vizAlpha(el) {
  const f = frame(el, { kicker: "Figure · α, the disagreement budget", title: "Lower α: safer answers, fewer of them." });
  let A, C;
  try {
    [A, C] = await Promise.all([fetch("../demo/alphas.json").then((r) => r.json()), fetch("../demo/alpha-curve.json").then((r) => r.json())]);
  } catch (e) { f.stage.innerHTML = `<p class="muted">Could not load the measurements.</p>`; return; }
  const keys = Object.keys(A).sort((a, b) => a - b);
  let cur = keys.includes(String(D.alpha.toFixed(2))) ? D.alpha.toFixed(2) : keys[Math.floor(keys.length / 2)];
  const segEl = seg("Choose α", keys.map((k) => ({ v: k, label: `α = ${pct(+k)}` })), cur, (v) => { cur = v; update(); });
  f.head.appendChild(segEl);
  f.stage.innerHTML = `<div class="al-tiles">
      <div class="al-tile ok"><span>shad0w answers</span><b data-r="sv">–</b><small>of messages, at microsecond speed</small></div>
      <div class="al-tile"><span>differed from your LLM</span><b data-r="dis">–</b><small data-r="dis2">on the answers it served</small><i class="al-meter"><i data-r="m"></i><em data-r="ma"></em></i></div>
      <div class="al-tile llm"><span>sent to your LLM</span><b data-r="df">–</b><small data-r="thr">–</small></div>
    </div><div class="al-chart"></div>`;
  const R = (k) => $(`[data-r="${k}"]`, f.stage);
  // the curve: the table's confidence on each message, most sure first. The certified line cuts it. Drawn at the
  // container's real pixel width (redrawn on resize) so the labels never squash.
  const n = C.conf.length, box = $(".al-chart", f.stage);
  let G = null;
  const draw = () => {
    const W = Math.max(280, Math.round(box.clientWidth)), H = W < 520 ? 200 : 240;
    if (G && G.W === W) return;
    const pl = 40, pr = 8, pt = 16, pb = 44;
    const X = (i) => pl + (W - pl - pr) * i / (n - 1), Y = (v) => pt + (H - pt - pb) * (1 - v);
    box.innerHTML = "";
    const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "al-svg", "aria-hidden": "true" }, box);
    for (const g of [0, 0.25, 0.5, 0.75, 1]) {
      S("line", { x1: pl, x2: W - pr, y1: Y(g), y2: Y(g), class: "grid" }, svg);
      S("text", { x: pl - 8, y: Y(g) + 4, class: "ax", "text-anchor": "end" }, svg).textContent = pct(g);
    }
    const clip = S("clipPath", { id: "alc-" + el.dataset.vid }, S("defs", {}, svg));
    const clipR = S("rect", { x: pl, y: 0, width: 0, height: H }, clip);
    const step = Math.max(1, Math.floor(n / (W - pl - pr)));  // about one point per pixel
    const pts = C.conf.map((v, i) => [i, v]).filter(([i]) => i % step === 0 || i === n - 1);
    const line = pts.map(([i, v]) => `${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join(" L");
    const area = `M${pl} ${Y(0)} L${line} L${W - pr} ${Y(0)} Z`;
    S("path", { d: area, class: "area-llm" }, svg);
    S("path", { d: area, class: "area-ok", "clip-path": `url(#alc-${el.dataset.vid})` }, svg);
    S("path", { d: "M" + line, class: "curve" }, svg);
    // where the table's answer differed from your LLM: ticks under the axis
    const ticks = S("g", { class: "dticks" }, svg);
    [...C.agree].forEach((a, i) => { if (a === "0") S("line", { x1: X(i), x2: X(i), y1: H - pb + 8, y2: H - pb + 18, "data-i": i }, ticks); });
    S("text", { x: pl, y: H - 6, class: "ax" }, svg).textContent = "← most sure";
    S("text", { x: W - pr, y: H - 6, class: "ax", "text-anchor": "end" }, svg).textContent = "least sure →";
    if (W >= 420) S("text", { x: (pl + W - pr) / 2, y: H - 6, class: "ax", "text-anchor": "middle" }, svg).textContent = "| = differed from your LLM";
    const thrG = S("g", { class: "thr" }, svg);
    S("line", { x1: pl, x2: W - pr, y1: 0, y2: 0 }, thrG);
    const thrT = S("text", { x: pl + 8, y: 15 }, thrG);
    const cutL = S("line", { x1: 0, x2: 0, y1: pt, y2: H - pb, class: "cut" }, svg);
    G = { W, X, Y, clipR, thrG, thrT, cutL, ticks: $$("line", ticks) };
  };
  draw();
  new ResizeObserver(() => { const w = G && G.W; draw(); if (G.W !== w) update(); }).observe(box);
  const src = document.createElement("p");
  src.className = "viz-src";
  src.textContent = `Measured, not simulated: the certifier's output for the playground's 77-intent table at each α, on ${C.n_test.toLocaleString()} held-out test messages (demo/alphas.json). The curve shows ${C.n.toLocaleString()} of those messages; | marks one where the table's answer differed from the LLM's (demo/alpha-curve.json).`;
  el.insertBefore(src, f.cap.nextSibling);
  function update() {
    const a = A[cur], k = +cur;
    const none = !a.served_test;
    const thr = none ? 1 : a.threshold;
    let cut = 0;
    if (!none) while (cut < n && C.conf[cut] >= thr) cut++;
    R("sv").textContent = pct(a.served_test, 1);
    R("df").textContent = pct(1 - a.served_test, 1);
    R("dis").textContent = a.disagreement_test == null ? "–" : pct(a.disagreement_test, 1);
    R("dis2").textContent = none ? "nothing served, nothing to differ" : `allowed: at most ${pct(k)} (α)`;
    R("m").style.width = a.disagreement_test == null ? "0" : Math.min(100, 100 * a.disagreement_test / k) + "%";
    R("ma").textContent = "";
    R("thr").textContent = none ? "no certified line at this α" : `below the certified line of ${pct(thr, 1)} confidence`;
    const { X, Y, clipR, thrG, thrT, cutL } = G;
    clipR.setAttribute("width", cut ? X(cut - 1) - X(0) : 0);
    thrG.setAttribute("transform", `translate(0 ${Y(thr)})`);
    thrT.textContent = none ? "no line passes" : `certified line · ${pct(thr, 1)}`;
    cutL.setAttribute("x1", X(Math.max(cut - 1, 0))); cutL.setAttribute("x2", X(Math.max(cut - 1, 0)));
    cutL.style.opacity = cut ? 1 : 0;
    G.ticks.forEach((t) => t.classList.toggle("in", +t.dataset.i < cut));
    f.cap.innerHTML = none
      ? `At <b>α = ${pct(k)}</b> no confidence line passes the test with this much calibration data, so shad0w answers <b>nothing</b> and every message goes to your LLM. That is the safe outcome. Log more answers to certify a stricter α.`
      : `At <b>α = ${pct(k)}</b> shad0w answers the <b class="ok">${pct(a.served_test)}</b> of messages it is most sure about and sends the rest to your LLM. On the answers it served, it differed from your LLM <b>${pct(a.disagreement_test, 1)}</b> of the time, inside the ${pct(k)} budget.`;
  }
  update();
}

/* ------------------------------------------------------------------ lifecycle */

const STEPS = [
  { k: "log", t: "Log", s: "your LLM answers", d: () => `At first your LLM answers every decision, and shad0w writes each answer to <code>shad0w/&lt;question&gt;/log.jsonl</code>. Nothing changes for your users.`, c: `# your LLM answers; shad0w logs the answer\nintent("my card was stolen")` },
  { k: "train", t: "Train", s: "a small table", d: () => `Once the log holds <code>min_rows</code> answers (default ${D.min_rows.toLocaleString()}), training compiles them into a small table that runs on your CPU.`, c: `shad0w train --question intent\n# or, in Python: intent.train()` },
  { k: "certify", t: "Certify", s: "on unseen answers", d: () => `A slice of the log the table never trained on (<code>cal_fraction</code>, default ${pct(D.cal_fraction)}) sets a confidence line: above it, the table differs from your LLM on at most α (default ${pct(D.alpha)}) of answers, with ${pct(1 - D.delta)} confidence.`, c: `cert = intent.train()\nprint(cert["certified_share_on_calibration"])` },
  { k: "serve", t: "Serve", s: "sure ones in µs", d: () => `Above the line, shad0w answers in microseconds and your LLM is not called. Below it, the message goes to your LLM as before, and that answer is logged too.`, c: `d = intent("lost my card")\nprint(d.source)   # "table" when shad0w answered` },
  { k: "spot-check", t: "Spot-check", s: "and watch drift", d: () => `A random <code>audit_rate</code> share (default ${pct(D.audit_rate)}) is re-asked to your LLM in the background. That measures live agreement, catches drift, and feeds the next training round.`, c: `shad0w report --bundle shad0w/intent/bundle` },
];

function vizLifecycle(el) {
  const want = (el.dataset.step || "").toLowerCase().replace(/[\s_]/g, "-").replace("spotcheck", "spot-check");
  let cur = Math.max(0, STEPS.findIndex((s) => s.k === want));
  const auto = !want && !reduce;
  const f = frame(el, { kicker: "Figure · the loop", title: "Log → train → certify → serve → spot-check, then again.", controls: auto ? pauseBtn() : "" });
  f.stage.innerHTML = `<ol class="lc-steps" role="tablist" aria-label="Steps">${STEPS.map((s, i) => `<li><button type="button" role="tab" id="${el.dataset.vid}-s${i}" aria-selected="false" tabindex="-1" aria-controls="${el.dataset.vid}-p">
      <span class="lc-n">${i + 1}</span><b>${s.t}</b><small>${s.s}</small></button></li>`).join("")}</ol>
    <div class="lc-loop" aria-hidden="true"><span>repeats: every LLM answer and spot check goes back into the log</span></div>
    <div class="lc-panel" role="tabpanel" id="${el.dataset.vid}-p"><p data-r="d"></p><pre><code data-r="c"></code></pre></div>`;
  const bs = $$(".lc-steps button", f.stage);
  const show = (i, focus) => {
    cur = (i + STEPS.length) % STEPS.length;
    bs.forEach((b, k) => { b.setAttribute("aria-selected", k === cur); b.tabIndex = k === cur ? 0 : -1; b.parentElement.classList.toggle("on", k === cur); b.parentElement.classList.toggle("done", k < cur); });
    $(".lc-panel", f.stage).setAttribute("aria-labelledby", bs[cur].id);
    $('[data-r="d"]', f.stage).innerHTML = STEPS[cur].d();
    $('[data-r="c"]', f.stage).textContent = STEPS[cur].c;
    if (focus) bs[cur].focus();
  };
  let ctl = null, acc = 0;
  const stop = () => { if (ctl && !ctl.paused) $(".viz-pause", el).click(); };
  bs.forEach((b, i) => b.addEventListener("click", () => { stop(); show(i); }));
  $(".lc-steps", f.stage).addEventListener("keydown", (e) => {
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (d) { e.preventDefault(); stop(); show(cur + d, true); }
    if (e.key === "Home" || e.key === "End") { e.preventDefault(); stop(); show(e.key === "Home" ? 0 : STEPS.length - 1, true); }
  });
  show(cur);
  f.cap.textContent = "Click a step (or use the arrow keys) to see what happens there.";
  if (auto) {
    ctl = loop(el, (now, dt) => { acc += dt; if (acc > 3200) { acc = 0; show(cur + 1); } });
    wirePause(el, ctl);
  }
}

/* ------------------------------------------------------------------ precedence + keys (a stack of layers) */

function stack(el, { kicker, title, layers, result, note }) {
  const f = frame(el, { kicker, title });
  f.stage.innerHTML = `<div class="st-legend"><span>highest priority</span><span>switch layers on and off</span></div><ol class="st">${layers.map((l, i) => `<li data-i="${i}">
      <button type="button" class="st-row" aria-pressed="${!!l.on}"${l.fixed ? " disabled" : ""}>
        <span class="st-sw" aria-hidden="true"></span>
        <span class="st-name"><b>${l.name}</b><small>${l.where}</small></span>
        <code class="st-code">${esc(l.code)}</code>
        <span class="st-tag"></span>
      </button></li>`).join("")}</ol><div class="st-legend"><span>lowest priority</span><span></span></div>
    <div class="st-out" data-r="out"></div>${note ? `<p class="st-note">${note}</p>` : ""}`;
  const rows = $$(".st > li", f.stage);
  const update = () => {
    const win = layers.findIndex((l) => l.on);
    rows.forEach((li, i) => {
      const l = layers[i], b = $(".st-row", li), tag = $(".st-tag", li);
      li.className = !l.on ? "off" : i === win ? "win" : "lost";
      b.setAttribute("aria-pressed", !!l.on);
      b.setAttribute("aria-label", `${l.name}: ${l.on ? "set" : "not set"}${i === win ? ", wins" : l.on ? ", overridden by a higher layer" : ""}`);
      tag.textContent = !l.on ? "not set" : i === win ? "wins" : "overridden";
    });
    $('[data-r="out"]', f.stage).innerHTML = result(layers[win], win);
    f.cap.textContent = `${layers[win].name} wins: it is the highest layer that is set.`;
  };
  rows.forEach((li, i) => $(".st-row", li).addEventListener("click", () => { if (layers[i].fixed) return; layers[i].on = !layers[i].on; update(); }));
  update();
}

function vizPrecedence(el) {
  const def = D.alpha;
  stack(el, {
    kicker: "Figure · where a setting comes from", title: "The highest layer that sets a value wins.",
    layers: [
      { name: "Keyword in code", where: "one call", code: `shad0w.decision("intent", ..., alpha=0.01)`, v: 0.01 },
      { name: "shad0w.configure()", where: "process-wide, from code", code: `shad0w.configure(alpha=0.02)`, v: 0.02 },
      { name: "Environment", where: "SHAD0W_* variables", code: `SHAD0W_ALPHA=0.03`, v: 0.03, on: true },
      { name: "shad0w.toml, per question", where: "[questions.intent]", code: `[questions.intent]  alpha = 0.04`, v: 0.04 },
      { name: "shad0w.toml, top level", where: "./shad0w.toml or $SHAD0W_CONFIG", code: `alpha = 0.08`, v: 0.08, on: true },
      { name: "Default", where: "built in", code: `alpha = ${def}`, v: def, on: true, fixed: true },
    ],
    result: (l) => `<span>effective</span><code>alpha = ${l.v}</code><span>from <b>${esc(l.name)}</b></span>`,
    note: `Run <code>shad0w config --explain</code> to print every setting with its value and the layer it came from.`,
  });
}

function vizKeys(el) {
  stack(el, {
    kicker: "Figure · where the API key comes from", title: "The first source that is set is the key shad0w uses.",
    layers: [
      { name: "api_key=", where: "a string, or a function called per request", code: `decision(..., api_key=vault.get)`, src: "the api_key= you passed", m: "sk-…3f9a" },
      { name: "api_key_env=", where: "the NAME of a variable", code: `decision(..., api_key_env="TEAM_LLM_KEY")`, src: "TEAM_LLM_KEY", m: "sk-…71c2" },
      { name: "shad0w.configure(api_key=…)", where: "process-wide, from code", code: `shad0w.configure(api_key=key)`, src: "shad0w.configure(api_key=…)", m: "sk-…b5e0" },
      { name: "api_key_env setting", where: "configure(api_key_env=), SHAD0W_API_KEY_ENV or shad0w.toml", code: `api_key_env = "PROD_LLM_KEY"`, src: "PROD_LLM_KEY", m: "sk-…08d4", on: true },
      { name: "Provider variable", where: "used when nothing above is set", code: `OPENAI_API_KEY=sk-...`, src: "OPENAI_API_KEY", m: "sk-…9c1e", on: true, fixed: true },
    ],
    result: (l) => `<span>key</span><code>set via ${esc(l.src)} (${l.m})</code><span>logs, errors and repr() show only <b>${l.m}</b></span>`,
    note: `A key written in <code>shad0w.toml</code> (<code>api_key = ...</code>) is an error: files hold the variable's name, never the key.`,
  });
}

/* ------------------------------------------------------------------ rollout */

async function vizRollout(el) {
  let share = 0.63;
  try { const a = await fetch("../demo/alphas.json").then((r) => r.json()); const k = D.alpha.toFixed(2); if (a[k] && a[k].served_test) share = a[k].served_test; } catch (e) { /* keep the example share */ }
  const f = frame(el, { kicker: "Figure · rolling out", title: "Turn it on in steps. Watch before you serve." });
  const STAGES = {
    off: { code: `mode="off"`, users: "your LLM's answer, always", llm: "every request", log: "your LLM's answers (the table keeps learning)", tip: "The table is not consulted. Use it to switch shad0w off instantly." },
    shadow: { code: `mode="shadow"`, users: "your LLM's answer, always", llm: "every request", log: "your LLM's answers, plus how often the table would have agreed", tip: "The table answers silently, side by side. You see real agreement before a single user gets a table answer." },
    canary: { code: `mode="serve", canary=C`, users: "the table's answer on C of certified requests, else your LLM's", llm: "uncertified requests, plus certified ones held back", log: "every LLM answer, plus spot checks", tip: "Drag the slider: the share of certified answers the table may actually serve." },
    serve: { code: `mode="serve"`, users: "the table's answer whenever it is certified", llm: "uncertified requests, plus spot checks", log: "every LLM answer, plus spot checks", tip: "Full rollout. Every certified answer is served in microseconds." },
  };
  let mode = "shadow", can = 0.25;
  const segEl = seg("Rollout stage", Object.keys(STAGES).map((k, i) => ({ v: k, label: `<span class="ro-i">${i + 1}</span>${k}` })), mode, (v) => { mode = v; update(true); });
  segEl.classList.add("ro-seg");
  f.head.appendChild(segEl);
  const N = 40;
  // a fixed, evenly spread pattern: which of 40 requests are certified, and which certified ones the canary lets through
  const rank = (seed, off) => { const v = Array.from({ length: N }, (_, i) => ((i * seed + off) % 1)); const o = [...v].sort((x, y) => x - y); return v.map((x) => o.indexOf(x)); };
  const cr = rank(0.618034, 0), nCert = Math.round(share * N);
  const cert = cr.map((r) => r < nCert);                     // exactly round(share × N) certified, evenly spread
  const gr = rank(0.7548777, 0.31).map((r, i) => (cert[i] ? r : Infinity));
  const gOrder = gr.map((r, i) => [r, i]).filter((x) => isFinite(x[0])).sort((x, y) => x[0] - y[0]).map((x) => x[1]);
  const gate = Array(N).fill(1);
  gOrder.forEach((i, k) => { gate[i] = (k + 0.5) / gOrder.length; });  // the canary lets through exactly its share of them
  f.stage.innerHTML = `<div class="ro-grid" aria-hidden="true">${cert.map((c) => `<i class="${c ? "c" : ""}"></i>`).join("")}</div>
    <div class="ro-legend"><span><i class="sq ok"></i>table answered</span><span><i class="sq llm"></i>your LLM answered</span><span><i class="sq ring"></i>certified (the table could answer)</span></div>
    <div class="ro-can" hidden><label for="${el.dataset.vid}-c">canary</label><input type="range" id="${el.dataset.vid}-c" min="0" max="100" step="5" value="${can * 100}"><output>${pct(can)}</output></div>
    <dl class="ro-kv"><div><dt>Users get</dt><dd data-r="users"></dd></div><div><dt>Your LLM is called for</dt><dd data-r="llm"></dd></div><div><dt>Logged</dt><dd data-r="log"></dd></div></dl>
    <pre class="ro-code"><code data-r="code"></code></pre>`;
  const cells = $$(".ro-grid i", f.stage), range = $("input", f.stage), out = $("output", f.stage);
  range.addEventListener("input", () => { can = range.value / 100; out.textContent = pct(can); update(false); });
  function update(anim) {
    const st = STAGES[mode];
    let served = 0;
    cells.forEach((c, i) => {
      const t = mode === "serve" ? cert[i] : mode === "canary" ? cert[i] && gate[i] < can : false;
      served += t;
      c.classList.toggle("ok", t);
      c.classList.toggle("ghost", mode === "off");
      c.style.transitionDelay = anim && !reduce ? (i * 12) + "ms" : "0ms";
    });
    $(".ro-can", f.stage).hidden = mode !== "canary";
    $('[data-r="users"]', f.stage).textContent = st.users.replace("C of", pct(can) + " of");
    $('[data-r="llm"]', f.stage).textContent = st.llm;
    $('[data-r="log"]', f.stage).textContent = st.log;
    $('[data-r="code"]', f.stage).textContent = `shad0w.decision("intent", ..., ${st.code.replace("C", String(can))})`;
    f.cap.innerHTML = `${st.tip} <span class="dim">Here: <b class="ok">${served}</b> of ${N} requests answered by the table.</span>`;
  }
  const src = document.createElement("p");
  src.className = "viz-src";
  src.textContent = `Squares: ${N} example requests, ${pct(share)} of them certified, the share the playground's table certifies at α = ${pct(D.alpha)}.`;
  el.insertBefore(src, f.cap.nextSibling);
  update(false);
}

/* ------------------------------------------------------------------ proxy */

function vizProxy(el) {
  const MODES = {
    header: { label: "Chat + header", ep: "POST /v1/chat/completions", req: `client.chat.completions.create(
    model="gpt-6-luna",
    messages=[{"role": "user", "content": text}],
    extra_headers={"X-Shad0w-Question": "intent"})`,
      qs: [["intent", "choice", true]], how: "A chat completion marked with the <code>X-Shad0w-Question</code> header is a decision. The answer comes back as a normal chat completion." },
    decisions: { label: "Decisions API", ep: "POST /v1/decisions", req: `client.decisions.create(
    model="gpt-6-luna", input=text,
    questions=[intent, urgent, tone])`,
      qs: [["intent", "choice", true], ["urgent", "predicate", false], ["tone", "score", null]], how: "Recognised with no marking. Certified questions are answered here; only the rest are forwarded, in one smaller request." },
    systemone: { label: "System One", ep: "POST /v1/systemone", req: `requests.post(proxy + "/v1/systemone", json={
    "model": "kev", "state": text,
    "questions": {"intent": ..., "urgent": ...}})`,
      qs: [["intent", "choice", true], ["urgent", "noul", false]], how: "Jev, Kev, Laya and other System One servers: same treatment as the Decisions API, no marking needed." },
    plain: { label: "Plain chat", ep: "POST /v1/chat/completions", req: `client.chat.completions.create(
    model="gpt-6-luna",
    messages=[{"role": "user", "content": text}])`, qs: [],
      how: "Not a decision: forwarded untouched and returned untouched. Nothing is logged." },
  };
  let mode = "decisions";
  const f = frame(el, { kicker: "Figure · the zero-code proxy", title: "Change one line: your client's base_url." , controls: "" });
  const segEl = seg("Request type", Object.entries(MODES).map(([v, m]) => ({ v, label: m.label })), mode, (v) => { mode = v; update(true); });
  f.head.appendChild(segEl);
  f.stage.innerHTML = `<div class="px-lanes">
      <div class="px-node app"><b>Your app</b><small>openai SDK</small><code>base_url=<wbr>"http://localhost:8010/v1"</code></div>
      <div class="px-wire w1"><i class="px-dot"></i></div>
      <div class="px-node proxy"><b>shad0w proxy</b><small>answers what it is certified on</small><span class="px-log">logged</span></div>
      <div class="px-wire w2"><i class="px-dot"></i></div>
      <div class="px-node up"><b>Upstream LLM</b><small>--upstream</small></div>
    </div>
    <div class="px-body"><div class="px-reqw"><span class="px-h px-ep" data-r="ep"></span><pre class="px-req"><code data-r="req"></code></pre></div><div class="px-qs" data-r="qs"></div></div>`;
  const lanes = $(".px-lanes", f.stage);
  let timer = 0;
  function update(anim) {
    const m = MODES[mode];
    $('[data-r="req"]', f.stage).textContent = m.req;
    $('[data-r="ep"]', f.stage).textContent = m.ep;
    const local = m.qs.filter((q) => q[2] === true), fwd = m.qs.filter((q) => q[2] !== true);
    $('[data-r="qs"]', f.stage).innerHTML = m.qs.length
      ? `<span class="px-h">questions in this request <em>(click to flip)</em></span>` + m.qs.map((q, i) => `<button type="button" class="px-q ${q[2] === true ? "ok" : "fw"}" data-i="${i}" ${q[2] === null ? "disabled" : ""} aria-pressed="${q[2] === true}">
          <b>${q[0]}</b><small>${q[1]}</small><span>${q[2] === true ? "certified → answered here" : q[2] === null ? "score → always forwarded" : "not certified → forwarded, then logged"}</span></button>`).join("")
      : `<span class="px-h">no question: the proxy is a pass-through</span>`;
    lanes.classList.toggle("has-local", local.length > 0);
    lanes.classList.toggle("has-fwd", fwd.length > 0 || !m.qs.length);
    lanes.classList.toggle("plain", !m.qs.length);
    f.cap.innerHTML = `${m.how} ${m.qs.length ? `<span class="dim">Here: <b class="ok">${local.length}</b> answered by the proxy in microseconds, <b class="llm">${fwd.length}</b> forwarded.</span>` : ""}`;
    if (anim && !reduce) { lanes.classList.remove("go"); void lanes.offsetWidth; lanes.classList.add("go"); clearTimeout(timer); timer = setTimeout(() => lanes.classList.remove("go"), 3200); }
  }
  $('[data-r="qs"]', f.stage).addEventListener("click", (e) => {
    const b = e.target.closest(".px-q"); if (!b || b.disabled) return;
    const q = MODES[mode].qs[+b.dataset.i]; q[2] = !q[2]; update(true);
    const nb = $(`.px-q[data-i="${b.dataset.i}"]`, f.stage); if (nb) nb.focus();
  });
  update(false);
  if (!reduce) new IntersectionObserver((es, o) => { if (es[0].isIntersecting) { update(true); o.disconnect(); } }, { threshold: 0.5 }).observe(el);
}

/* ------------------------------------------------------------------ options */

function vizOptions(el) {
  const LEARNED0 = ["refund", "lost_card", "other"];
  const MSGS = [["I want my money back", "refund"], ["my card was stolen", "lost_card"], ["what's the weather like", "other"]];
  let st;
  const reset = () => { st = { learned: [...LEARNED0], add: false, remove: false, rename: false, policy: "defer" }; };
  reset();
  const f = frame(el, { kicker: "Figure · changing your options", title: "Add, remove or rename an option. See what shad0w does." });
  f.stage.innerHTML = `<div class="op-actions">
      <button type="button" class="viz-btn" data-a="add" aria-pressed="false">+ add <code>exchange</code></button>
      <button type="button" class="viz-btn" data-a="remove" aria-pressed="false">− remove <code>other</code></button>
      <button type="button" class="viz-btn" data-a="rename" aria-pressed="false">rename <code>lost_card</code> → <code>card_lost</code></button>
      <span class="op-sep"></span>
      <button type="button" class="viz-btn primary" data-a="retrain">Retrain</button>
      <button type="button" class="viz-btn" data-a="reset">Reset</button></div>
    <div class="op-pol"><span>on_new_option</span></div>
    <div class="op-rows"><div><span class="op-l">your options (code)</span><div class="op-chips" data-r="now"></div></div>
      <div><span class="op-l">what the table learned</span><div class="op-chips" data-r="learned"></div></div></div>
    <div class="op-flag"><span class="op-l">flag</span><code data-r="flag"></code><span data-r="fw"></span></div>
    <ul class="op-msgs" data-r="msgs"></ul><pre class="op-code"><code data-r="code"></code></pre>`;
  const pol = seg("on_new_option", [{ v: "defer", label: "defer" }, { v: "serve", label: "serve" }], "defer", (v) => { st.policy = v; update(); });
  $(".op-pol", f.stage).appendChild(pol);
  const R = (k) => $(`[data-r="${k}"]`, f.stage);
  const current = () => {
    let o = ["refund", "lost_card", "other"];
    if (st.rename) o = o.map((x) => (x === "lost_card" ? "card_lost" : x));
    if (st.remove) o = o.filter((x) => x !== "other");
    if (st.add) o = [...o, "exchange"];
    return o;
  };
  function update() {
    const now = current();
    const ren = st.rename && st.learned.includes("lost_card") ? { lost_card: "card_lost" } : {};
    const known = st.learned.map((x) => ren[x] || x);
    const added = now.filter((x) => !known.includes(x)), removed = known.filter((x) => !now.includes(x));
    const flagFor = (pick) => (added.length && st.policy === "defer" ? "options_changed" : removed.includes(pick) ? "option_removed" : null);
    R("now").innerHTML = now.map((o) => `<span class="op-chip ${added.includes(o) ? "new" : ""}">${o}${added.includes(o) ? "<em>new</em>" : ""}</span>`).join("");
    R("learned").innerHTML = st.learned.map((o) => { const r = ren[o]; const gone = removed.includes(r || o); return `<span class="op-chip ${gone ? "gone" : ""} ${r ? "ren" : ""}">${r ? `${o} → ${r}` : o}${gone ? "<em>removed</em>" : ""}</span>`; }).join("");
    const flags = new Set(MSGS.map(([, p]) => flagFor(ren[p] || p)).filter(Boolean));
    R("flag").textContent = flags.size ? [...flags].join(" · ") : "none";
    R("flag").className = flags.size ? "warn" : "okc";
    R("fw").textContent = flags.has("options_changed") ? "the table never learned some options, so its certificate no longer covers this question" : flags.has("option_removed") ? "answers naming a removed option are never served" : st.rename ? "rename maps the table's labels, no retraining needed" : "certified as usual";
    R("msgs").innerHTML = MSGS.map(([m, p]) => {
      const pick = ren[p] || p, fl = flagFor(pick), table = !fl;
      return `<li class="${table ? "ok" : "llm"}"><span class="op-m">“${m}”</span><span class="op-p">table picks <code>${pick}</code></span><b>${table ? "shad0w answers" : "your LLM answers"}</b></li>`;
    }).join("");
    const opts = JSON.stringify(now).replace(/,/g, ", ");
    R("code").textContent = `shad0w.decision("intent", options=${opts}${st.rename ? `,\n                rename={"lost_card": "card_lost"}` : ""}${added.length ? `,\n                on_new_option="${st.policy}"` : ""})`;
    $$(".op-actions [data-a]", f.stage).forEach((b) => { if (["add", "remove", "rename"].includes(b.dataset.a)) b.setAttribute("aria-pressed", st[b.dataset.a]); });
    $(".op-pol", f.stage).classList.toggle("dim", !added.length);
    f.cap.innerHTML = added.length && st.policy === "defer" ? `Until you retrain, <b class="llm">every</b> decision goes to your LLM (the safe default). Its answers keep being logged, so <b>Retrain</b> picks up the new option.`
      : added.length ? `With <code>on_new_option="serve"</code> the table keeps serving the options it knows; messages that need the new option still reach your LLM when the table is unsure.`
      : removed.length ? `Answers naming a removed option go to your LLM; everything else is served as before.` : `Three example messages; assume the table is certified on each.`;
  }
  $(".op-actions", f.stage).addEventListener("click", (e) => {
    const b = e.target.closest("[data-a]"); if (!b) return;
    const a = b.dataset.a;
    if (a === "reset") { reset(); pol.set("defer"); }
    else if (a === "retrain") st.learned = current();  // the table now knows exactly your options (renames included)
    else st[a] = !st[a];
    update();
  });
  update();
}

/* ------------------------------------------------------------------ try (the live 77-intent table) */

async function vizTry(el) {
  const alt = el.innerHTML.trim();
  el.classList.add("ready");
  el.innerHTML = `<div class="viz-head"><div><span class="viz-k">Live · runs in this tab</span><b class="viz-t">Type a message. A real 77-intent table decides.</b></div></div><div class="viz-try" id="${el.dataset.vid}-t"><p class="muted">Loading the table…</p></div>${alt ? `<div class="sr-only">${alt}</div>` : ""}`;
  try {
    const [{ Bundle }, { mountTry }] = await Promise.all([import("./shad0w.mjs"), import("./tryui.js")]);
    const base = new URL("../demo/", location.href);
    const bundle = await Bundle.load(new URL("bundle", base).href);
    const replay = await (await fetch(new URL("replay.json", base))).json();
    mountTry($(".viz-try", el), { bundle, q: "intent", msgs: replay.messages,
      examples: ["my card got stolen at the airport, please block it", "I was charged twice for the same coffee", "why is my top up still pending?", "is it going to rain tomorrow?"] });
    const more = document.createElement("p");
    more.className = "viz-src";
    more.innerHTML = `The table was trained on a free embedding model's answers (standing in for your LLM), in shadow mode. More in the <a href="../demo/">playground</a>.`;
    el.appendChild(more);
  } catch (e) {
    $(".viz-try", el).innerHTML = `<p class="muted">Could not load the table here. Try the <a href="../demo/">playground</a>.</p>`;
  }
}

/* ------------------------------------------------------------------ params (table generated at build time) */

function vizParams(el) {
  const rows = $$("tbody tr", el);
  const bar = document.createElement("div");
  bar.className = "pm-bar";
  bar.innerHTML = `<input type="search" placeholder="Filter ${rows.length} settings, e.g. alpha, proxy, drift" aria-label="Filter settings"><span aria-live="polite"></span>`;
  el.insertBefore(bar, el.firstChild);
  const inp = $("input", bar), cnt = $("span", bar);
  inp.addEventListener("input", () => {
    const q = inp.value.trim().toLowerCase();
    let n = 0;
    rows.forEach((r) => { const hit = !q || r.textContent.toLowerCase().includes(q); r.hidden = !hit; n += hit; });
    cnt.textContent = q ? `${n} of ${rows.length}` : "";
  });
  el.classList.add("ready");
}

/* ------------------------------------------------------------------ boot */

const VIZ = { flow: vizFlow, alpha: vizAlpha, lifecycle: vizLifecycle, precedence: vizPrecedence, keys: vizKeys, rollout: vizRollout,
  proxy: vizProxy, options: vizOptions, try: vizTry, params: vizParams };

function figures() {
  const els = $$(".viz[data-viz]");
  els.forEach((el, i) => { el.dataset.vid = "viz" + i; });
  const start = (el) => {
    if (el.dataset.started) return;
    el.dataset.started = 1;
    const fn = VIZ[el.dataset.viz];
    if (!fn) return;
    try { const r = fn(el); if (r && r.catch) r.catch((e) => console.warn("figure", el.dataset.viz, e)); } catch (e) { console.warn("figure", el.dataset.viz, e); }
  };
  if (!("IntersectionObserver" in window)) { els.forEach(start); return; }
  const io = new IntersectionObserver((es) => es.forEach((en) => { if (en.isIntersecting) { io.unobserve(en.target); start(en.target); } }), { rootMargin: "400px 0px" });
  els.forEach((el) => io.observe(el));
}

function docsMenu() {
  // phones: the sidebar is a "Docs menu" disclosure. Close it on Escape, outside clicks and link clicks;
  // keep it open whenever the window is wide enough for the sidebar.
  const d = $(".dmenu");
  if (!d) return;
  const wide = matchMedia("(min-width: 901px)");
  const sync = () => { if (wide.matches) d.open = true; };
  wide.addEventListener("change", (e) => { d.open = e.matches; });
  sync();
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && d.open && !wide.matches) { d.open = false; $("summary", d).focus(); } });
  document.addEventListener("click", (e) => { if (!wide.matches && d.open && !e.target.closest(".dmenu")) d.open = false; });
  d.addEventListener("click", (e) => { if (!wide.matches && e.target.closest(".dnav a")) d.open = false; });
}

const boot = () => { docsMenu(); scrollSpy(); tables(); headingLinks(); search(); figures(); };
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
