// One minute of Polymarket, read live in the browser. Trades come from Polymarket's public API (REST history, then the live
// WebSocket); a shad0w table labels each trade's market question (section and topic) in microseconds; labels are checked
// against Polymarket's own event tags, fetched live. Nothing is sent anywhere but Polymarket's public endpoints.
// Everything on the page is clickable: map blocks and the legend filter the page; trades and markets open a details drawer.
import { Bundle } from "./shad0w.mjs";

const $ = (id) => document.getElementById(id);
const [bundle, TAX] = await Promise.all([Bundle.load(new URL("bundle", location.href).href), fetch("taxonomy.json").then((r) => r.json())]);
const SEC = Object.fromEntries(TAX.sections.map((s) => [s.id, s])), TOP = Object.fromEntries(TAX.topics.map((t) => [t.id, t]));
const OTHER = Object.fromEntries(TAX.sections.map((s) => [s.id, TAX.topics.find((t) => t.section === s.id && !t.tags.length).id]));
const DUR = 60, BIG = 1000, LLM_S = 1, HIST = "https://data-api.polymarket.com/trades", WS = "wss://ws-live-data.polymarket.com", GAMMA = "https://gamma-api.polymarket.com/events";
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmtUsd = (v) => v >= 1e6 ? `$${(v / 1e6).toFixed(2)}M` : v >= 1e4 ? `$${Math.round(v / 1e3)}k` : v >= 1e3 ? `$${(v / 1e3).toFixed(1)}k` : `$${v.toFixed(v < 10 ? 2 : 0)}`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x, d = 1) => (x == null || !isFinite(x)) ? "–" : `${(100 * x).toFixed(d)}%`;
const fmtUs = (us) => us == null ? "–" : `${us < 10 ? us.toFixed(1) : Math.round(us)} µs`;
const pmLink = (slug) => `https://polymarket.com/event/${encodeURIComponent(slug || "")}`;
const chip = (sec, top) => `<em style="--c:var(--s-${sec});--ink:var(--i-${sec}, var(--black))">${esc(SEC[sec].name)}</em>${top ? `<em class="t">${esc(TOP[top].name)}</em>` : ""}`;

// Polymarket's tags -> (section, topic): the same rules that built the table's labels (taxonomy.json)
function tagLabel(tags) {
  const low = new Set((tags || []).map((t) => String(t.label ?? t).toLowerCase().trim()));
  const s = TAX.sections.find((x) => x.tags.some((k) => low.has(k)));
  if (!s) return null;
  const t = TAX.topics.find((x) => x.section === s.id && x.tags.some((k) => low.has(k)));
  return { section: s.id, topic: t ? t.id : OTHER[s.id] };
}

function label(title) {   // the hot path: both questions, no probabilities; a topic is shown inside its section
  const a = bundle.decide(title, { probabilities: false }).answers;
  const sec = a.section.choice, top = TOP[a.topic.choice]?.section === sec ? a.topic.choice : OTHER[sec];
  return { sec, top, secCert: a.section.certified, topCert: a.topic.certified && top === a.topic.choice };
}
function timeIt(text, reps = 400) { const t0 = performance.now(); for (let i = 0; i < reps; i++) bundle.decide(text, { probabilities: false }); return (performance.now() - t0) * 1000 / reps; }

// ---------------------------------------------------------------- one session
let S = null, F = null;   // F: the filter, null or {sec, top?}
let usPer = null; const samples = [];   // time per label measured in this tab
function fresh() {
  return { start: performance.now(), end: 0, trades: 0, usd: 0, labels: 0, labelMs: 0, labelN: 0, keys: new Set(),
    bySec: {}, byTop: {}, markets: new Map(), events: new Map(), rows: [], feed: [], buckets: new Map(), minT: Infinity, maxT: 0, liveFrom: Date.now() / 1000,
    ws: null, lastMsg: 0, liveN: 0, chk: { n: 0, sec: 0, top: 0, cert: 0, certOk: 0, usd: 0, usdOk: 0 }, dis: [], biggest: null, timers: [] };
}
const match = (sec, top) => !F || (F.sec === sec && (!F.top || F.top === top));

function ingest(list, live) {   // label a block of trades and add them up; the block is timed as a whole
  const t0 = performance.now(); const out = [];
  for (const tr of list) {
    const key = `${tr.transactionHash}:${tr.asset}:${tr.size}:${tr.side}`;
    if (S.keys.has(key) || !tr.title) continue;
    S.keys.add(key);
    out.push([tr, label(tr.title)]);
  }
  const ms = performance.now() - t0; S.labelMs += ms; S.labelN += out.length; S.labels += out.length;
  for (const [tr, L] of out) add(tr, L, live);
  return { n: out.length, ms };
}

function add(tr, L, live) {
  const usd = (+tr.size || 0) * (+tr.price || 0), ts = +tr.timestamp > 1e12 ? +tr.timestamp / 1000 : +tr.timestamp;
  S.trades++; S.usd += usd; if (live) S.liveN++;
  S.bySec[L.sec] = (S.bySec[L.sec] || 0) + usd;
  const tk = `${L.sec}/${L.top}`; S.byTop[tk] = (S.byTop[tk] || 0) + usd;
  let mk = S.markets.get(tr.title);
  if (!mk) { mk = { title: tr.title, slug: tr.eventSlug, usd: 0, n: 0, ...L }; S.markets.set(tr.title, mk); }
  mk.usd += usd; mk.n++;
  let ev = S.events.get(tr.eventSlug);
  if (!ev) { ev = { slug: tr.eventSlug, usd: 0, titles: new Set(), checked: false }; S.events.set(tr.eventSlug, ev); }
  ev.usd += usd; ev.titles.add(tr.title);
  const b = Math.floor(ts / 15) * 15; let bk = S.buckets.get(b); if (!bk) { bk = {}; S.buckets.set(b, bk); } bk[L.sec] = (bk[L.sec] || 0) + usd;
  S.minT = Math.min(S.minT, ts); S.maxT = Math.max(S.maxT, ts);
  if (!S.biggest || usd > S.biggest.usd) S.biggest = { usd, title: tr.title, slug: tr.eventSlug, outcome: tr.outcome, side: tr.side };
  S.rows.push([ts, usd, tr.side, tr.outcome, tr.price, tr.size, tr.title, tr.eventSlug, L.sec, L.top, L.secCert, L.topCert]);
  S.feed.push({ ts, usd, tr, L, live }); if (S.feed.length > 600) S.feed.splice(0, S.feed.length - 600);
}

// ---------------------------------------------------------------- data in: history, then live
async function history() {
  setPhase("Reading the latest 20,000 trades from Polymarket…");
  const pages = await Promise.all([0, 10000].map((off) => fetch(`${HIST}?limit=10000&offset=${off}`).then((r) => r.ok ? r.json() : []).catch(() => [])));
  if (!S || S.done) return;
  const all = pages.flat(); all.sort((a, b) => a.timestamp - b.timestamp);
  const { n, ms } = ingest(all, false);
  S.histN = n; S.histMs = ms;
  setPhase(n ? `Read and filed ${n.toLocaleString("en-US")} trades from the last ${Math.round((S.maxT - S.minT) / 60)} minutes in ${ms.toFixed(ms < 10 ? 1 : 0)} ms. Now watching live…` : "History unavailable right now; watching live…");
}
function live() {
  if (!S || S.done) return;
  const ws = new WebSocket(WS); S.ws = ws; S.lastMsg = performance.now();
  ws.onopen = () => ws.send(JSON.stringify({ action: "subscribe", subscriptions: [{ topic: "activity", type: "trades" }] }));
  ws.onmessage = (m) => { S.lastMsg = performance.now(); let d; try { d = JSON.parse(m.data); } catch { return; }
    if (d.topic === "activity" && d.payload) pending.push(d.payload); };
  ws.onclose = () => { if (S && !S.done && S.ws === ws) setTimeout(live, 800); };
}
let pending = [];
function pump() { if (!S || !pending.length) return; const list = pending; pending = []; ingest(list, true); }

// ---------------------------------------------------------------- the check: Polymarket's own tags for each event seen
async function fetchTags(slugs) {
  const q = slugs.map((s) => `slug=${encodeURIComponent(s)}`).join("&");
  return fetch(`${GAMMA}?${q}&include_markets=false&limit=${slugs.length}`).then((r) => r.ok ? r.json() : []).catch(() => []);
}
function applyTags(ev) {
  const truth = tagLabel(ev.tags), mine = S.events.get(ev.slug);
  if (!mine) return;
  mine.tags = (ev.tags || []).map((x) => x.label).filter(Boolean); mine.truth = truth;
  if (!truth) return;
  for (const title of mine.titles) {
    const mk = S.markets.get(title); if (!mk || mk.truth) continue;
    mk.truth = truth; const c = S.chk; c.n++;
    const okS = mk.sec === truth.section, okT = mk.top === truth.topic;
    c.sec += okS; c.top += okT; c.usd += mk.usd; c.usdOk += okS ? mk.usd : 0;
    if (mk.secCert) { c.cert++; c.certOk += okS; }
    if (!okS) S.dis.push(mk);
  }
}
async function check() {
  if (!S || S.checking) return;
  const todo = [...S.events.values()].filter((e) => !e.checked && e.slug).sort((a, b) => b.usd - a.usd).slice(0, 40);
  if (!todo.length) return;
  S.checking = true; todo.forEach((e) => { e.checked = true; });
  for (const ev of await fetchTags(todo.map((e) => e.slug))) applyTags(ev);
  S.checking = false;
}

// ---------------------------------------------------------------- filter (map blocks, legend, crumbs)
function setFilter(f) {
  F = f && f.sec ? f : null;
  const c = $("crumbs");
  c.innerHTML = `<span>Showing:</span> <button type="button" class="crumb${F ? "" : " on"}" data-f="">everything</button>` +
    (F ? ` › <button type="button" class="crumb${F.top ? "" : " on"}" data-f="${F.sec}">${esc(SEC[F.sec].name)}</button>` : "") +
    (F?.top ? ` › <button type="button" class="crumb on" data-f="${F.sec}/${F.top}">${esc(TOP[F.top].name)}</button>` : "") +
    (F ? ` <button type="button" class="crumb x" data-f="">✕ clear</button>` : "");
  c.classList.toggle("on", !!F);
  if (S) redraw();
}
$("crumbs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; const [sec, top] = (b.dataset.f || "").split("/"); setFilter(sec ? { sec, top } : null); });

// ---------------------------------------------------------------- drawing
function setPhase(s) { $("phase").textContent = s; }
const secOrder = () => Object.entries(S.bySec).sort((a, b) => b[1] - a[1]);
function redraw() { drawMap(); drawTape(); drawSide(); drawFlow(); }

function squarify(items, x, y, w, h) {   // items: [{id, v}] -> [{id, x, y, w, h}], squarified treemap
  const out = [], total = items.reduce((s, i) => s + i.v, 0); if (!total || w <= 0 || h <= 0) return out;
  let rest = items.filter((i) => i.v > 0).sort((a, b) => b.v - a.v).map((i) => ({ ...i, a: i.v / total * w * h }));
  while (rest.length) {
    const side = Math.min(w, h); let row = [rest[0]], best = worst(row, side), k = 1;
    while (k < rest.length) { const r2 = [...row, rest[k]], w2 = worst(r2, side); if (w2 > best) break; row = r2; best = w2; k++; }
    const sum = row.reduce((s, i) => s + i.a, 0), thick = sum / side; let off = 0;
    for (const i of row) { const len = i.a / thick; out.push(w >= h ? { id: i.id, x, y: y + off, w: thick, h: len } : { id: i.id, x: x + off, y, w: len, h: thick }); off += len; }
    if (w >= h) { x += thick; w -= thick; } else { y += thick; h -= thick; }
    rest = rest.slice(row.length);
  }
  return out;
  function worst(row, side) { const s = row.reduce((t, i) => t + i.a, 0), mx = Math.max(...row.map((i) => i.a)), mn = Math.min(...row.map((i) => i.a));
    return Math.max(side * side * mx / (s * s), s * s / (side * side * mn)); }
}

const tiles = new Map();
function drawMap() {
  const box = $("map"), W = box.clientWidth, H = box.clientHeight;
  if (!S || !S.usd) return;
  if (box.querySelector("p")) box.innerHTML = "";
  const secs = squarify(secOrder().map(([id, v]) => ({ id, v })), 0, 0, W, H), seen = new Set();
  for (const s of secs) {
    const tops = Object.entries(S.byTop).filter(([k]) => k.startsWith(s.id + "/")).map(([k, v]) => ({ id: k, v }));
    const pad = s.w > 60 && s.h > 46 ? 20 : 0;
    for (const t of squarify(tops, s.x + 2, s.y + pad + 2, s.w - 4, s.h - pad - 4)) {
      seen.add(t.id); let el = tiles.get(t.id); const top = t.id.split("/")[1];
      if (!el) { el = document.createElement("button"); el.type = "button"; el.className = "tile"; el.dataset.f = t.id;
        el.style.setProperty("--c", `var(--s-${s.id})`); el.style.setProperty("--ink", `var(--i-${s.id}, var(--black))`); box.appendChild(el); tiles.set(t.id, el); }
      Object.assign(el.style, { left: `${t.x}px`, top: `${t.y}px`, width: `${Math.max(0, t.w - 2)}px`, height: `${Math.max(0, t.h - 2)}px` });
      const v = S.byTop[t.id], big = t.w > 74 && t.h > 34;
      el.innerHTML = big ? `<b>${esc(TOP[top]?.name)}</b><small>${fmtUsd(v)} · ${pct(v / S.usd, 0)}</small>` : "";
      el.title = `${SEC[s.id].name} › ${TOP[top]?.name}: ${fmtUsd(v)} (${pct(v / S.usd, 1)}). Click to focus.`;
      el.setAttribute("aria-label", el.title);
      el.classList.toggle("dim", !match(s.id, top)); el.classList.toggle("sel", !!F?.top && F.top === top);
    }
    const hid = "sec:" + s.id; seen.add(hid); let hd = tiles.get(hid);
    if (!hd) { hd = document.createElement("div"); hd.className = "sechead"; box.appendChild(hd); tiles.set(hid, hd); }
    Object.assign(hd.style, { left: `${s.x}px`, top: `${s.y}px`, width: `${s.w}px`, height: `${s.h}px`, borderColor: `var(--s-${s.id})` });
    hd.innerHTML = pad ? `<button type="button" data-f="${s.id}">${esc(SEC[s.id].name)} <i>${pct(S.bySec[s.id] / S.usd, 0)}</i></button>` : "";
    hd.classList.toggle("dim", !!F && F.sec !== s.id);
  }
  for (const [k, el] of tiles) if (!seen.has(k)) { el.remove(); tiles.delete(k); }
}
$("map").addEventListener("click", (e) => { const b = e.target.closest("[data-f]"); if (!b) return; const [sec, top] = b.dataset.f.split("/");
  setFilter(F && F.sec === sec && (F.top || "") === (top || "") ? null : { sec, top }); });

let flowGeom = null;
function drawFlow() {
  const cv = $("flow"), g = cv.getContext("2d"), dpr = devicePixelRatio || 1, W = cv.clientWidth, H = 120;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  if (!S || !S.buckets.size) return;
  const keys = [...S.buckets.keys()].sort((a, b) => a - b), t0 = keys[0], t1 = Math.max(keys[keys.length - 1], S.liveFrom + DUR);
  const n = Math.max(1, (t1 - t0) / 15 + 1), bw = W / n, order = secOrder().map(([id]) => id), cs = getComputedStyle(document.documentElement);
  let mx = 0; for (const k of keys) mx = Math.max(mx, Object.values(S.buckets.get(k)).reduce((a, b) => a + b, 0) / 15);
  for (const k of keys) { const bk = S.buckets.get(k), x = (k - t0) / 15 * bw; let y = H;
    for (const id of order) { const v = (bk[id] || 0) / 15, h = v / mx * (H - 14);
      g.globalAlpha = !F || F.sec === id ? 1 : 0.18; g.fillStyle = cs.getPropertyValue(`--s-${id}`).trim() || "gray"; g.fillRect(x, y - h, Math.max(1, bw - 1), h); y -= h; } }
  g.globalAlpha = 1;
  const lx = (S.liveFrom - t0) / 15 * bw;
  g.strokeStyle = cs.getPropertyValue("--orange").trim(); g.setLineDash([3, 3]); g.beginPath(); g.moveTo(lx, 0); g.lineTo(lx, H); g.stroke(); g.setLineDash([]);
  g.fillStyle = cs.getPropertyValue("--text-dim").trim(); g.font = "11px ui-monospace, monospace"; g.fillText("live →", Math.min(W - 44, lx + 4), 11);
  g.fillText(`peak ${fmtUsd(mx)}/s`, 2, 11);
  flowGeom = { t0, bw, order };
  $("legend").innerHTML = order.map((id) => `<button type="button" data-f="${id}" class="${F && F.sec !== id ? "dim" : ""}${F?.sec === id ? " on" : ""}"><i style="background:var(--s-${id})"></i>${esc(SEC[id].name)} <small>${pct(S.bySec[id] / S.usd, 0)}</small></button>`).join("");
}
$("legend").addEventListener("click", (e) => { const b = e.target.closest("[data-f]"); if (b) setFilter(F?.sec === b.dataset.f && !F.top ? null : { sec: b.dataset.f }); });
$("flow").addEventListener("mousemove", (e) => {
  if (!S || !flowGeom) return;
  const r = e.currentTarget.getBoundingClientRect(), x = e.clientX - r.left, k = flowGeom.t0 + Math.floor(x / flowGeom.bw) * 15, bk = S.buckets.get(k), tip = $("flowTip");
  if (!bk) { tip.hidden = true; return; }
  const tot = Object.values(bk).reduce((a, b) => a + b, 0);
  tip.innerHTML = `<b>${new Date(k * 1000).toTimeString().slice(0, 5)}–${new Date((k + 15) * 1000).toTimeString().slice(3, 8)} · ${fmtUsd(tot / 15)}/s</b>` +
    flowGeom.order.filter((id) => bk[id]).map((id) => `<span><i style="background:var(--s-${id})"></i>${esc(SEC[id].name)} ${fmtUsd(bk[id])}</span>`).join("");
  tip.hidden = false; tip.style.left = `${Math.min(r.width - 190, Math.max(0, x - 90))}px`;
});
$("flow").addEventListener("mouseleave", () => { $("flowTip").hidden = true; });

const held = new Set();   // lists under the pointer pause, so you can read and click them
function drawTape() {
  if (held.has("tape")) return;
  const list = []; for (let i = S.feed.length - 1; i >= 0 && list.length < 14; i--) { const f = S.feed[i]; if (match(f.L.sec, f.L.top)) list.push(f); }
  $("tapeNote").textContent = F ? `newest first · only ${F.top ? TOP[F.top].name : SEC[F.sec].name} · click one` : "newest first · big trades in orange · click one";
  $("tape").innerHTML = list.map(({ ts, usd, tr, L }) => {
    const tm = new Date(ts * 1000).toTimeString().slice(0, 8);
    return `<li class="${usd >= BIG ? "big" : ""}" data-m="${esc(tr.title)}" tabindex="0" role="button"><span class="tm">${tm}</span><b class="usd">${fmtUsd(usd)}</b><span class="what">${esc(tr.side === "SELL" ? "sold" : "bought")} <i>${esc(tr.outcome)}</i> @ ${Math.round(100 * tr.price)}¢</span>
      <span class="ttl">${esc(tr.title)}</span><span class="chips">${chip(L.sec, L.top)}${L.secCert ? '<u title="certified">✓</u>' : '<u class="no" title="not certified: confident, but below the certified bar">unsure</u>'}</span></li>`;
  }).join("") || `<li class="empty">No trades in this filter yet.</li>`;
}

function drawSide() {
  const top = [...S.markets.values()].filter((m) => match(m.sec, m.top)).sort((a, b) => b.usd - a.usd).slice(0, 8);
  $("topNote").textContent = F ? `in ${F.top ? TOP[F.top].name : SEC[F.sec].name}, this minute · click one` : "by dollars in this minute · click one";
  if (!held.has("top")) $("top").innerHTML = top.map((m) => `<li data-m="${esc(m.title)}" tabindex="0" role="button"><span class="mt">${esc(m.title)}</span><span><b>${fmtUsd(m.usd)}</b> · ${m.n} trades · ${chip(m.sec, m.top)}${m.truth ? (m.truth.section === m.sec ? ' <u title="agrees with Polymarket\'s tags">✓ tags agree</u>' : ' <u class="no">✗ tags differ</u>') : ""}</span></li>`).join("") || `<li class="empty">Nothing here yet.</li>`;
  const c = S.chk;
  $("cN").textContent = c.n ? c.n.toLocaleString("en-US") : "checking…";
  $("cSec").textContent = c.n ? `${pct(c.sec / c.n)} (${c.sec} of ${c.n})` : "–";
  $("cTop").textContent = c.n ? pct(c.top / c.n) : "–";
  $("cCert").textContent = c.cert ? `${pct(c.certOk / c.cert)} of ${c.cert} certified` : "–";
  $("cUsd").textContent = c.usd ? pct(c.usdOk / c.usd) : "–";
  $("kAgree").textContent = c.n ? pct(c.sec / c.n) : "–";
  $("cDisN").textContent = S.dis.length;
  if (!held.has("cDis")) $("cDis").innerHTML = S.dis.slice(-40).reverse().map((m) => `<li data-m="${esc(m.title)}" tabindex="0" role="button"><span class="mt">${esc(m.title)}</span> shad0w: <b>${esc(SEC[m.sec].name)}</b> · tags: <b>${esc(SEC[m.truth.section].name)}</b></li>`).join("");
}

// ---------------------------------------------------------------- the drawer: one market, everything about it
async function openMarket(title) {
  const m = S?.markets.get(title); if (!m) return;
  const d = bundle.decide(title, { probabilities: true }).answers, us = timeIt(title, 300);
  const bars = (probs, names) => Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 3)
    .map(([o, p], i) => `<div class="pbar${i ? "" : " top"}"><i style="width:${Math.max(1, Math.round(p * 100))}%"></i><em>${esc(names[o]?.name || o)}</em><b>${Math.round(p * 100)}%</b></div>`).join("");
  const rows = S.rows.filter((r) => r[6] === title), mins = new Map();
  for (const r of rows) { const k = Math.floor(r[0] / 60) * 60; mins.set(k, (mins.get(k) || 0) + r[1]); }
  const ks = [...mins.keys()].sort((a, b) => a - b), mx = Math.max(...mins.values(), 1);
  const ev = S.events.get(m.slug);
  const tagsHtml = () => ev?.tags ? `<div class="dtags">${ev.tags.map((t) => `<span>${esc(t)}</span>`).join("")}</div>` + (ev.truth
    ? `<p>Those tags map to ${chip(ev.truth.section, ev.truth.topic)} ${ev.truth.section === m.sec ? '<u>✓ the section agrees with shad0w</u>' : '<u class="no">✗ shad0w filed it differently</u>'}</p>` : "<p class=\"muted\">None of these tags map to a section.</p>")
    : `<p class="muted">Fetching Polymarket's tags…</p>`;
  const render = () => {
    $("dBody").innerHTML = `<div class="kicker">market</div><h3>${esc(title)}</h3>
      <p><a class="btn" href="${pmLink(m.slug)}" target="_blank" rel="noopener">Open on Polymarket ↗</a> <button class="btn" type="button" data-f="${m.sec}/${m.top}">Show only ${esc(TOP[m.top].name)}</button></p>
      <dl class="kv"><dt>this minute</dt><dd>${fmtUsd(m.usd)} in ${m.n} trades</dd></dl>
      <div class="dspark">${ks.map((k) => `<i style="height:${Math.max(3, Math.round(mins.get(k) / mx * 100))}%" title="${new Date(k * 1000).toTimeString().slice(0, 5)}: ${fmtUsd(mins.get(k))}"></i>`).join("")}</div>
      <p class="muted" style="font-size:12px;margin:2px 0 14px">dollars a minute in this market, oldest to newest</p>
      <h4>What shad0w read</h4>
      <p class="muted" style="font-size:12.5px">It decided in <b class="o">${fmtUs(us)}</b> in your browser, from the sentence alone.</p>
      <div class="dq"><small>section · ${d.section.certified ? '<u>certified ✓</u>' : '<u class="no">not certified</u>'}</small>${bars(d.section.probabilities, SEC)}</div>
      <div class="dq"><small>topic · ${d.topic.certified ? '<u>certified ✓</u>' : '<u class="no">not certified</u>'}</small>${bars(d.topic.probabilities, TOP)}</div>
      <h4>What Polymarket's own tags say</h4>${tagsHtml()}`;
  };
  render(); $("drawer").hidden = false; $("drawer").classList.add("open");
  if (ev && !ev.tags && ev.slug) { for (const e of await fetchTags([ev.slug])) applyTags(e); if (!ev.tags) ev.tags = []; render(); }
}
function closeDrawer() { $("drawer").classList.remove("open"); $("drawer").hidden = true; }
$("dClose").addEventListener("click", closeDrawer);
addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
$("dBody").addEventListener("click", (e) => { const b = e.target.closest("[data-f]"); if (!b) return; const [sec, top] = b.dataset.f.split("/"); setFilter({ sec, top }); closeDrawer(); $("crumbs").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" }); });
for (const id of ["tape", "top", "cDis"]) {
  $(id).addEventListener("pointerenter", (e) => { if (e.pointerType === "mouse") { held.add(id); $(id).classList.add("held"); } });
  $(id).addEventListener("pointerleave", () => { held.delete(id); $(id).classList.remove("held"); if (S) { drawTape(); drawSide(); } });
  $(id).addEventListener("click", (e) => { const li = e.target.closest("[data-m]"); if (li) openMarket(li.dataset.m); });
  $(id).addEventListener("keydown", (e) => { const li = e.target.closest("[data-m]"); if (li && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openMarket(li.dataset.m); } });
}

// ---------------------------------------------------------------- the pipeline: one trade, read and filed, again and again
const EXAMPLES = ["Will the Fed cut rates in December?", "Counter-Strike: Vitality vs NAVI - Map 1 Winner", "Bitcoin Up or Down - 10:15AM-10:20AM ET",
  "Will Galatasaray SK win on 2026-10-13?", "Will it rain in London tomorrow?", "Who will win the 2028 US Presidential Election?", "Will OpenAI release a new model this month?",
  "Ukraine and Russia ceasefire by December 31?", "Taylor Swift announces a new album by Friday?", "Will Ethereum hit $8,000 in 2026?"];
let exI = 0, lastPiped = null;
function pipe(text, usd, what) {
  const L = label(text), us = usPer ?? timeIt(text, 200);
  const box = $("pipe"); box.classList.remove("go"); void box.offsetWidth; if (!REDUCED) box.classList.add("go");
  $("pIn").textContent = `“${text}”`; $("pUsd").textContent = usd != null ? `${fmtUsd(usd)} · ${what}` : "an example sentence";
  $("pUs").textContent = fmtUs(us); $("pOut").innerHTML = chip(L.sec, L.top); $("pCert").textContent = L.secCert ? "certified ✓" : "not certified: it says so";
}
function pipeTick() {
  if (S && !S.done && S.feed.length) {
    let f = null; for (let i = S.feed.length - 1; i >= 0; i--) if (match(S.feed[i].L.sec, S.feed[i].L.top)) { f = S.feed[i]; break; }
    if (f && f !== lastPiped) { lastPiped = f; pipe(f.tr.title, f.usd, `${f.tr.side === "SELL" ? "sold" : "bought"} ${f.tr.outcome}`); }
  } else if (!S || S.done) { pipe(EXAMPLES[exI++ % EXAMPLES.length]); }
}
setInterval(pipeTick, 1400); pipeTick();

// ---------------------------------------------------------------- shad0w vs an LLM, one call a second
function race() {
  if (!S) return;
  const el = ((S.end || performance.now()) - S.start) / 1000, me = S.trades, llm = Math.min(me, Math.floor(el / LLM_S));
  $("rMe").style.width = me ? "100%" : "0"; $("rLlm").style.width = me ? `${Math.max(0.3, 100 * llm / me)}%` : "0";
  $("rMeN").textContent = me.toLocaleString("en-US"); $("rLlmN").textContent = llm.toLocaleString("en-US");
  if (me > 50) { const left = (me - llm) * LLM_S / 3600;
    $("rNote").textContent = `The LLM, one call a second, would need about ${left >= 1 ? `${left.toFixed(1)} more hours` : `${Math.round(left * 60)} more minutes`} to catch up with what shad0w filed in ${Math.round(el)} seconds. (One call a second is generous: the hosted LLM calls we measured took 1.1–2.4 s.)`; }
}

// ---------------------------------------------------------------- numbers
function measure() {   // time per label, measured here: re-label recent titles in a block and divide (browser clocks are coarse)
  if (!S) return;
  const titles = S.feed.slice(-200).map((x) => x.tr.title).filter(Boolean);
  if (titles.length < 8) return;
  let n = 0; const t0 = performance.now();
  while (performance.now() - t0 < 12) { for (const s of titles) bundle.decide(s, { probabilities: false }); n += titles.length; }
  samples.push((performance.now() - t0) * 1000 / n); if (samples.length > 7) samples.shift();
  usPer = [...samples].sort((a, b) => a - b)[samples.length >> 1];   // the median of the last few blocks: one GC pause doesn't swing it
  cmp();
}
function cmp() {   // the "four ways" table, with this tab's own numbers
  if (usPer == null) return;
  const rate = S && S.liveN > 20 ? S.liveN / Math.max(1, ((S.end || performance.now()) - S.start) / 1000 - 2) : 40;
  $("cmpUs").textContent = fmtUs(usPer);
  $("cmpShare").textContent = `${(rate * usPer / 1e6 * 100).toFixed(3)}%`;
  const back = 20000 * usPer / 1000; $("cmpBack").textContent = back < 1000 ? `${Math.round(back)} ms` : `${(back / 1000).toFixed(1)} s`;
}
function kpis() {
  if (!S) return;
  const el = ((S.end || performance.now()) - S.start) / 1000, left = Math.max(0, DUR - el);
  $("clock").textContent = S.done ? "0" : Math.ceil(left);
  $("kTrades").textContent = S.trades.toLocaleString("en-US");
  $("kUsd").textContent = fmtUsd(S.usd);
  $("kLabels").textContent = (S.labels * 2).toLocaleString("en-US");
  const us = usPer ?? (S.labelN ? S.labelMs * 1000 / S.labelN : null);
  $("kUs").textContent = fmtUs(us);
  $("kShare").textContent = us == null || !el ? "–" : `${(S.labels * us / 1e6 / el * 100).toFixed(3)}%`;
  race();
}

// ---------------------------------------------------------------- try it yourself
const tryIn = $("tryIn");
$("tryEx").innerHTML = EXAMPLES.slice(0, 8).map((e) => `<button type="button" class="chipbtn">${esc(e)}</button>`).join("");
$("tryEx").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { tryIn.value = b.textContent; tryNow(); } });
function tryNow() {
  const text = tryIn.value.trim() || tryIn.placeholder;
  const d = bundle.decide(text, { probabilities: true }).answers, us = timeIt(text, 500);
  const bars = (probs, names) => Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 3)
    .map(([o, p], i) => `<div class="pbar${i ? "" : " top"}"><i style="width:${Math.max(1, Math.round(p * 100))}%"></i><em>${esc(names[o]?.name || o)}</em><b>${Math.round(p * 100)}%</b></div>`).join("");
  $("tryOut").innerHTML = `<div class="dq"><small>section · ${d.section.certified ? '<u>certified ✓</u>' : '<u class="no">not certified: it would hand this one back</u>'}</small>${bars(d.section.probabilities, SEC)}</div>
    <div class="dq"><small>topic · ${d.topic.certified ? '<u>certified ✓</u>' : '<u class="no">not certified</u>'}</small>${bars(d.topic.probabilities, TOP)}</div>
    <p class="muted" style="margin:6px 0 0">Decided in <b class="o">${fmtUs(us)}</b> in your browser (both questions, averaged over 500 runs). An LLM call to do the same: about a second, over the network.</p>`;
}
let tryT = null; tryIn.addEventListener("input", () => { clearTimeout(tryT); tryT = setTimeout(tryNow, 60); }); tryNow();

// ---------------------------------------------------------------- run
async function start() {
  stop(false); S = fresh(); pending = []; tiles.forEach((el) => el.remove()); tiles.clear(); setFilter(null); lastPiped = null;
  $("go").disabled = true; $("go").textContent = "● Running…"; $("report").hidden = true; $("map").innerHTML = '<p class="muted">Reading Polymarket…</p>';
  $("run").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
  const run = S;
  run.timers.push(setInterval(() => { if (run !== S) return; pump(); kpis(); if (performance.now() - run.lastMsg > 6000 && run.ws) { try { run.ws.close(); } catch {} } }, 100));
  run.timers.push(setInterval(() => { if (run !== S) return; drawMap(); drawTape(); drawSide(); }, 500));
  run.timers.push(setInterval(() => { if (run !== S) return; drawFlow(); check(); }, 1000));
  run.timers.push(setInterval(() => { if (run !== S) return; measure(); }, 2000));
  run.timers.push(setInterval(() => { if (run === S && run.ws?.readyState === 1) run.ws.send("PING"); }, 10000));
  run.timers.push(setTimeout(() => { if (run === S) stop(true); }, DUR * 1000));
  live();                        // live first, so nothing is missed while history downloads
  await history();
  measure(); redraw();
}
function stop(report) {
  if (!S) return;
  S.done = true; S.end = performance.now(); S.timers.forEach((t) => { clearInterval(t); clearTimeout(t); });
  try { S.ws?.close(); } catch {}
  $("go").disabled = false; $("go").textContent = "▶ Run another live minute";
  if (!report) return;
  pump(); measure(); kpis(); redraw();
  setPhase("Done. Click anything on the map, the tape or the lists to explore. The report is at the bottom.");
  const secs = secOrder(), c = S.chk, us = usPer ?? S.labelMs * 1000 / Math.max(1, S.labelN);
  const topTopic = (sid) => Object.entries(S.byTop).filter(([k]) => k.startsWith(sid + "/")).sort((a, b) => b[1] - a[1])[0];
  const parts = secs.slice(0, 3).map(([id, v]) => { const tt = topTopic(id); return `${SEC[id].name} ${pct(v / S.usd, 0)} (mostly ${TOP[tt[0].split("/")[1]].name})`; });
  $("rText").innerHTML = `Your browser read <b>${S.trades.toLocaleString("en-US")}</b> trades worth <b>${fmtUsd(S.usd)}</b> across ${S.markets.size.toLocaleString("en-US")} markets. ` +
    `Where the money went: ${parts.join(", ")}. ` + (S.biggest ? `The biggest single trade: ${fmtUsd(S.biggest.usd)} on “${esc(S.biggest.title)}”. ` : "") +
    `shad0w made ${(S.labels * 2).toLocaleString("en-US")} labels at about ${fmtUs(us)} a trade, ${(S.labels * us / 1e6 / DUR * 100).toFixed(3)}% of the minute. ` +
    (c.n ? `Against Polymarket's own tags, its section agreed on ${pct(c.sec / c.n)} of the ${c.n.toLocaleString("en-US")} markets checked (${pct(c.usdOk / c.usd)} of the dollars), and on ${pct(c.certOk / Math.max(1, c.cert))} of the ones it was certified on.` : "");
  $("report").hidden = false;
  window.__pulse = S;
}

function download(name, type, text) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); }
$("dlCsv").addEventListener("click", () => {
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const head = "time_utc,usd,side,outcome,price,shares,market,event_slug,section,topic,section_certified,topic_certified,tags_section,tags_topic";
  const lines = S.rows.map((r) => { const mk = S.markets.get(r[6]); return [new Date(r[0] * 1000).toISOString(), r[1].toFixed(2), r[2], q(r[3]), r[4], r[5], q(r[6]), r[7], r[8], r[9], r[10], r[11], mk?.truth?.section ?? "", mk?.truth?.topic ?? ""].join(","); });
  download("polymarket-minute.csv", "text/csv", [head, ...lines].join("\n"));
});
$("dlJson").addEventListener("click", () => {
  const c = S.chk;
  download("polymarket-minute.json", "application/json", JSON.stringify({ trades: S.trades, usd: +S.usd.toFixed(2), markets: S.markets.size,
    by_section: Object.fromEntries(secOrder().map(([k, v]) => [k, +v.toFixed(2)])), by_topic: Object.fromEntries(Object.entries(S.byTop).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, +v.toFixed(2)])),
    top_markets: [...S.markets.values()].sort((a, b) => b.usd - a.usd).slice(0, 25).map((m) => ({ title: m.title, event_slug: m.slug, usd: +m.usd.toFixed(2), trades: m.n, section: m.sec, topic: m.top })),
    check: { markets: c.n, section_agreement: c.n ? c.sec / c.n : null, topic_agreement: c.n ? c.top / c.n : null, certified_section_agreement: c.cert ? c.certOk / c.cert : null, dollar_agreement: c.usd ? c.usdOk / c.usd : null },
    us_per_trade: usPer }, null, 1));
});
$("go").addEventListener("click", start);
$("again").addEventListener("click", start);
setFilter(null);
if (new URLSearchParams(location.search).has("autostart")) start();
window.__pulseApi = { setFilter, openMarket, get S() { return S; } };
