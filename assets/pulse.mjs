// One minute of Polymarket, read live in the browser. Trades come from Polymarket's public API (REST history, then the live
// WebSocket); a shad0w table labels each trade's market question (section and topic) in microseconds; labels are checked
// against Polymarket's own event tags, fetched live. Nothing is sent anywhere but Polymarket's public endpoints.
import { Bundle } from "./shad0w.mjs";

const $ = (id) => document.getElementById(id);
const [bundle, TAX] = await Promise.all([Bundle.load(new URL("bundle", location.href).href), fetch("taxonomy.json").then((r) => r.json())]);
const SEC = Object.fromEntries(TAX.sections.map((s) => [s.id, s])), TOP = Object.fromEntries(TAX.topics.map((t) => [t.id, t]));
const OTHER = Object.fromEntries(TAX.sections.map((s) => [s.id, TAX.topics.find((t) => t.section === s.id && !t.tags.length).id]));
const DUR = 60, BIG = 1000, HIST = "https://data-api.polymarket.com/trades", WS = "wss://ws-live-data.polymarket.com", GAMMA = "https://gamma-api.polymarket.com/events";
const fmtUsd = (v) => v >= 1e6 ? `$${(v / 1e6).toFixed(2)}M` : v >= 1e4 ? `$${Math.round(v / 1e3)}k` : v >= 1e3 ? `$${(v / 1e3).toFixed(1)}k` : `$${v.toFixed(v < 10 ? 2 : 0)}`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x, d = 1) => (x == null || !isFinite(x)) ? "–" : `${(100 * x).toFixed(d)}%`;

// Polymarket's tags -> (section, topic): the same rules that built the table's labels (taxonomy.json)
function tagLabel(tags) {
  const low = new Set((tags || []).map((t) => String(t.label ?? t).toLowerCase().trim()));
  const s = TAX.sections.find((x) => x.tags.some((k) => low.has(k)));
  if (!s) return null;
  const t = TAX.topics.find((x) => x.section === s.id && x.tags.some((k) => low.has(k)));
  return { section: s.id, topic: t ? t.id : OTHER[s.id] };
}

// ---------------------------------------------------------------- one session
let S = null;
function fresh() {
  return { start: performance.now(), end: 0, phase: "history", trades: 0, usd: 0, labels: 0, labelMs: 0, labelN: 0, keys: new Set(),
    bySec: {}, byTop: {}, markets: new Map(), events: new Map(), rows: [], tape: [], buckets: new Map(), minT: Infinity, maxT: 0, liveFrom: Date.now() / 1000,
    ws: null, lastMsg: 0, checked: 0, chk: { n: 0, sec: 0, top: 0, cert: 0, certOk: 0, usd: 0, usdOk: 0 }, dis: [], biggest: null, timers: [] };
}

function label(title) {   // the hot path: one call, three-question-free fast path; topics are shown inside their section
  const a = bundle.decide(title, { probabilities: false }).answers;
  const sec = a.section.choice, top = TOP[a.topic.choice]?.section === sec ? a.topic.choice : OTHER[sec];
  return { sec, top, secCert: a.section.certified, topCert: a.topic.certified && top === a.topic.choice };
}

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
  S.trades++; S.usd += usd;
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
  if (live) { S.tape.unshift({ ts, usd, tr, L }); if (S.tape.length > 18) S.tape.length = 18; }
}

// ---------------------------------------------------------------- data in: history, then live
async function history() {
  setPhase("Reading the latest 20,000 trades…");
  const pages = await Promise.all([0, 10000].map((off) => fetch(`${HIST}?limit=10000&offset=${off}`).then((r) => r.ok ? r.json() : []).catch(() => [])));
  if (!S || S.done) return;
  const all = pages.flat(); all.sort((a, b) => a.timestamp - b.timestamp);
  const { n, ms } = ingest(all, false);
  S.histN = n; S.histMs = ms;
  S.tape = all.slice(-18).reverse().map((tr) => ({ ts: +tr.timestamp, usd: tr.size * tr.price, tr, L: label(tr.title || "") })).filter((x) => x.tr.title);
  setPhase(n ? `${n.toLocaleString("en-US")} trades from the last ${Math.round((S.maxT - S.minT) / 60)} minutes, labelled in ${ms.toFixed(ms < 10 ? 1 : 0)} ms. Now live…` : "History unavailable right now; going live…");
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
function pump() {   // live trades are labelled in small blocks as they come in
  if (!S || !pending.length) return;
  const list = pending; pending = []; ingest(list, true);
}

// ---------------------------------------------------------------- the check: Polymarket's own tags for each event seen
async function check() {
  if (!S || S.checking) return;
  const todo = [...S.events.values()].filter((e) => !e.checked && e.slug).sort((a, b) => b.usd - a.usd).slice(0, 40);
  if (!todo.length) return;
  S.checking = true; todo.forEach((e) => { e.checked = true; });
  try {
    const q = todo.map((e) => `slug=${encodeURIComponent(e.slug)}`).join("&");
    const evs = await fetch(`${GAMMA}?${q}&include_markets=false&limit=40`).then((r) => r.ok ? r.json() : []);
    for (const ev of evs) {
      const truth = tagLabel(ev.tags), mine = S.events.get(ev.slug);
      if (!truth || !mine) continue;
      for (const title of mine.titles) {
        const mk = S.markets.get(title); if (!mk || mk.truth) continue;
        mk.truth = truth; const c = S.chk; c.n++;
        const okS = mk.sec === truth.section, okT = mk.top === truth.topic;
        c.sec += okS; c.top += okT; c.usd += mk.usd; c.usdOk += okS ? mk.usd : 0;
        if (mk.secCert) { c.cert++; c.certOk += okS; }
        if (!okS) S.dis.push(mk);
      }
    }
  } catch { /* the check is best effort; the labels stand on their own */ }
  S.checking = false;
}

// ---------------------------------------------------------------- drawing
function setPhase(s) { $("phase").textContent = s; }
const secOrder = () => Object.entries(S.bySec).sort((a, b) => b[1] - a[1]);

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
  const secs = squarify(secOrder().map(([id, v]) => ({ id, v })), 0, 0, W, H), live = new Set();
  for (const s of secs) {
    const tops = Object.entries(S.byTop).filter(([k]) => k.startsWith(s.id + "/")).map(([k, v]) => ({ id: k, v }));
    const pad = s.w > 60 && s.h > 46 ? 20 : 0;
    for (const t of squarify(tops, s.x + 2, s.y + pad + 2, s.w - 4, s.h - pad - 4)) {
      live.add(t.id); let el = tiles.get(t.id);
      if (!el) { el = document.createElement("div"); el.className = "tile"; el.style.setProperty("--c", `var(--s-${s.id})`); el.style.setProperty("--ink", `var(--i-${s.id}, var(--black))`); box.appendChild(el); tiles.set(t.id, el); }
      Object.assign(el.style, { left: `${t.x}px`, top: `${t.y}px`, width: `${Math.max(0, t.w - 2)}px`, height: `${Math.max(0, t.h - 2)}px` });
      const v = S.byTop[t.id], big = t.w > 74 && t.h > 34;
      el.innerHTML = big ? `<b>${esc(TOP[t.id.split("/")[1]]?.name)}</b><small>${fmtUsd(v)} · ${pct(v / S.usd, 0)}</small>` : "";
      el.title = `${SEC[s.id].name} › ${TOP[t.id.split("/")[1]]?.name}: ${fmtUsd(v)}`;
    }
    const hid = "sec:" + s.id; live.add(hid); let hd = tiles.get(hid);
    if (!hd) { hd = document.createElement("div"); hd.className = "sechead"; box.appendChild(hd); tiles.set(hid, hd); }
    Object.assign(hd.style, { left: `${s.x}px`, top: `${s.y}px`, width: `${s.w}px`, height: `${s.h}px`, borderColor: `var(--s-${s.id})` });
    hd.innerHTML = pad ? `<span>${esc(SEC[s.id].name)} <i>${pct(S.bySec[s.id] / S.usd, 0)}</i></span>` : "";
  }
  for (const [k, el] of tiles) if (!live.has(k)) { el.remove(); tiles.delete(k); }
}

function drawFlow() {
  const cv = $("flow"), g = cv.getContext("2d"), dpr = devicePixelRatio || 1, W = cv.clientWidth, H = 120;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  if (!S || !S.buckets.size) return;
  const keys = [...S.buckets.keys()].sort((a, b) => a - b), t0 = keys[0], t1 = Math.max(keys[keys.length - 1], S.liveFrom + DUR);
  const n = Math.max(1, (t1 - t0) / 15 + 1), bw = W / n, order = secOrder().map(([id]) => id), cs = getComputedStyle(document.documentElement);
  let mx = 0; for (const k of keys) mx = Math.max(mx, Object.values(S.buckets.get(k)).reduce((a, b) => a + b, 0) / 15);
  for (const k of keys) { const bk = S.buckets.get(k), x = (k - t0) / 15 * bw; let y = H;
    for (const id of order) { const v = (bk[id] || 0) / 15, h = v / mx * (H - 14); g.fillStyle = cs.getPropertyValue(`--s-${id}`).trim() || "gray"; g.fillRect(x, y - h, Math.max(1, bw - 1), h); y -= h; } }
  const lx = (S.liveFrom - t0) / 15 * bw;   // where live begins
  g.strokeStyle = cs.getPropertyValue("--orange").trim(); g.setLineDash([3, 3]); g.beginPath(); g.moveTo(lx, 0); g.lineTo(lx, H); g.stroke(); g.setLineDash([]);
  g.fillStyle = cs.getPropertyValue("--text-dim").trim(); g.font = "11px ui-monospace, monospace"; g.fillText("live →", Math.min(W - 44, lx + 4), 11);
  g.fillText(`peak ${fmtUsd(mx)}/s`, 2, 11);
  $("legend").innerHTML = order.map((id) => `<span><i style="background:var(--s-${id})"></i>${esc(SEC[id].name)}</span>`).join("");
}

function drawTape() {
  $("tape").innerHTML = S.tape.map(({ ts, usd, tr, L }) => {
    const tm = new Date(ts * 1000).toTimeString().slice(0, 8);
    return `<li class="${usd >= BIG ? "big" : ""}"><span class="tm">${tm}</span><b class="usd">${fmtUsd(usd)}</b><span class="what">${esc(tr.side === "SELL" ? "sold" : "bought")} <i>${esc(tr.outcome)}</i> @ ${Math.round(100 * tr.price)}¢</span>
      <a class="ttl" href="https://polymarket.com/event/${encodeURIComponent(tr.eventSlug || "")}" target="_blank" rel="noopener">${esc(tr.title)}</a>
      <span class="chips"><em style="--c:var(--s-${L.sec});--ink:var(--i-${L.sec}, var(--black))">${esc(SEC[L.sec].name)}</em><em class="t">${esc(TOP[L.top].name)}</em>${L.secCert ? '<u title="certified">✓</u>' : '<u class="no" title="not certified">?</u>'}</span></li>`;
  }).join("");
}

function drawSide() {
  const top = [...S.markets.values()].sort((a, b) => b.usd - a.usd).slice(0, 8);
  $("top").innerHTML = top.map((m) => `<li><a href="https://polymarket.com/event/${encodeURIComponent(m.slug || "")}" target="_blank" rel="noopener">${esc(m.title)}</a><span><b>${fmtUsd(m.usd)}</b> · ${m.n} trades · <em style="--c:var(--s-${m.sec});--ink:var(--i-${m.sec}, var(--black))">${esc(TOP[m.top].name)}</em></span></li>`).join("");
  const c = S.chk;
  $("cN").textContent = c.n ? c.n.toLocaleString("en-US") : "checking…";
  $("cSec").textContent = c.n ? `${pct(c.sec / c.n)} (${c.sec} of ${c.n})` : "–";
  $("cTop").textContent = c.n ? pct(c.top / c.n) : "–";
  $("cCert").textContent = c.cert ? `${pct(c.certOk / c.cert)} of ${c.cert} certified` : "–";
  $("cUsd").textContent = c.usd ? pct(c.usdOk / c.usd) : "–";
  $("kAgree").textContent = c.n ? pct(c.sec / c.n) : "–";
  $("cDisN").textContent = S.dis.length;
  $("cDis").innerHTML = S.dis.slice(-30).reverse().map((m) => `<li><a href="https://polymarket.com/event/${encodeURIComponent(m.slug || "")}" target="_blank" rel="noopener">${esc(m.title)}</a> · shad0w: <b>${esc(SEC[m.sec].name)}</b>, tags: <b>${esc(SEC[m.truth.section].name)}</b></li>`).join("");
}

let usPer = null;
function measure() {   // time per label, measured here: re-label recent titles in a block and divide (browser clocks are coarse)
  if (!S) return;
  const titles = S.tape.map((x) => x.tr.title).concat([...S.markets.keys()].slice(-200)).filter(Boolean);
  if (titles.length < 8) return;
  let n = 0; const t0 = performance.now();
  while (performance.now() - t0 < 12) { for (const s of titles) bundle.decide(s, { probabilities: false }); n += titles.length; }
  samples.push((performance.now() - t0) * 1000 / n); if (samples.length > 7) samples.shift();
  usPer = [...samples].sort((a, b) => a - b)[samples.length >> 1];   // the median of the last few blocks: one GC pause or redraw doesn't swing it
}
const samples = [];

function kpis() {
  if (!S) return;
  const el = ((S.end || performance.now()) - S.start) / 1000, left = Math.max(0, DUR - el);
  $("clock").textContent = S.done ? "0" : Math.ceil(left);
  $("kTrades").textContent = S.trades.toLocaleString("en-US");
  $("kUsd").textContent = fmtUsd(S.usd);
  $("kLabels").textContent = (S.labels * 2).toLocaleString("en-US");
  const us = usPer ?? (S.labelN ? S.labelMs * 1000 / S.labelN : null);
  $("kUs").textContent = us == null ? "–" : `${us < 10 ? us.toFixed(1) : Math.round(us)} µs`;
  $("kShare").textContent = us == null || !el ? "–" : `${(S.labels * us / 1e6 / el * 100).toFixed(3)}%`;
}

// ---------------------------------------------------------------- run
async function start() {
  stop(false); S = fresh(); pending = []; tiles.forEach((el) => el.remove()); tiles.clear();
  $("go").disabled = true; $("go").textContent = "Running…"; $("report").hidden = true; $("map").innerHTML = '<p class="muted">Reading Polymarket…</p>';
  const run = S;
  run.timers.push(setInterval(() => { if (run !== S) return; pump(); kpis(); if (performance.now() - run.lastMsg > 6000 && run.ws) { try { run.ws.close(); } catch {} } }, 100));
  run.timers.push(setInterval(() => { if (run !== S) return; drawMap(); drawTape(); drawSide(); }, 400));
  run.timers.push(setInterval(() => { if (run !== S) return; drawFlow(); check(); }, 1000));
  run.timers.push(setInterval(() => { if (run !== S) return; measure(); }, 2000));
  run.timers.push(setInterval(() => { if (run === S && run.ws?.readyState === 1) run.ws.send("PING"); }, 10000));
  run.timers.push(setTimeout(() => { if (run === S) stop(true); }, DUR * 1000));
  live();                        // live first, so nothing is missed while history downloads
  await history();
  drawMap(); drawFlow(); drawTape(); drawSide();
}

function stop(report) {
  if (!S) return;
  S.done = true; S.end = performance.now(); S.timers.forEach((t) => { clearInterval(t); clearTimeout(t); });
  try { S.ws?.close(); } catch {}
  $("go").disabled = false; $("go").textContent = "Start the 60-second analysis →";
  if (!report) return;
  pump(); measure(); kpis(); drawMap(); drawFlow(); drawTape(); drawSide();
  setPhase("Done. The report is below the map.");
  const secs = secOrder(), c = S.chk, us = usPer ?? S.labelMs * 1000 / Math.max(1, S.labelN);
  const topTopic = (sid) => Object.entries(S.byTop).filter(([k]) => k.startsWith(sid + "/")).sort((a, b) => b[1] - a[1])[0];
  const parts = secs.slice(0, 3).map(([id, v]) => { const tt = topTopic(id); return `${SEC[id].name} ${pct(v / S.usd, 0)} (mostly ${TOP[tt[0].split("/")[1]].name})`; });
  $("rText").innerHTML = `Your browser read <b>${S.trades.toLocaleString("en-US")}</b> trades worth <b>${fmtUsd(S.usd)}</b> across ${S.markets.size.toLocaleString("en-US")} markets. ` +
    `Where the money went: ${parts.join(", ")}. ` + (S.biggest ? `The biggest single trade: ${fmtUsd(S.biggest.usd)} on “${esc(S.biggest.title)}”. ` : "") +
    `shad0w made ${(S.labels * 2).toLocaleString("en-US")} labels at about ${us < 10 ? us.toFixed(1) : Math.round(us)} µs a trade, ${(S.labels * us / 1e6 / DUR * 100).toFixed(3)}% of the minute. ` +
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
if (new URLSearchParams(location.search).has("autostart")) start();
