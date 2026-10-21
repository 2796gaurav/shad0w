// Doodle Duel: you draw, a shad0w image table guesses on every stroke, in this tab. The rasteriser below mirrors
// arena/doodle/raster.py line for line, so the table sees exactly the pixels it was trained on.
import { Bundle } from "./shad0w.mjs";
import { SIZE, render, simplify } from "./doodle-raster.mjs";

const $ = (id) => document.getElementById(id);
const [bundle, info] = await Promise.all([Bundle.load(new URL("bundle", location.href).href), fetch("doodle.json").then((r) => r.json())]);
const WORDS = info.words, ICON = { cat: "🐱", fish: "🐟", house: "🏠", tree: "🌳", car: "🚗", sun: "☀️", star: "⭐", apple: "🍎", bicycle: "🚲", umbrella: "☂️", clock: "🕒", airplane: "✈️" };

// ---------------------------------------------------------------- drawing
const cv = $("pad"), g = cv.getContext("2d"); let strokes = [], cur = null, drawing = false;
function resize() { const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1; cv.width = r.width * d; cv.height = r.height * d; g.setTransform(d, 0, 0, d, 0, 0); paint(); }
function paint() {
  const r = cv.getBoundingClientRect(); g.clearRect(0, 0, r.width, r.height);
  g.lineCap = g.lineJoin = "round"; g.lineWidth = 5; g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--white").trim() || "#fff";
  for (const s of [...strokes, ...(cur ? [cur] : [])]) { g.beginPath(); s.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); if (s.length === 1) g.lineTo(s[0][0] + 0.1, s[0][1]); g.stroke(); }
}
const pos = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); drawing = true; cur = [pos(e)]; paint(); if (game && !game.t0) game.t0 = performance.now(); });
cv.addEventListener("pointermove", (e) => { if (!drawing) return; cur.push(pos(e)); paint(); live(); });
const end = () => { if (!drawing) return; drawing = false; if (cur && cur.length) strokes.push(cur); cur = null; paint(); guess(true); };
cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end); cv.addEventListener("pointerleave", end);
$("clear").addEventListener("click", () => { strokes = []; cur = null; paint(); guess(false); });
$("undo").addEventListener("click", () => { strokes.pop(); paint(); guess(false); });
addEventListener("resize", resize);

// ---------------------------------------------------------------- guessing
let lastLive = 0, timing = null;
function live() { const t = performance.now(); if (t - lastLive > 90) { lastLive = t; guess(false, true); } }
function decide(raw) {
  const px = render(simplify(raw));
  const t0 = performance.now(); let n = 0, a;
  do { a = bundle.decideImage(px, SIZE, SIZE, 1, { probabilities: true }).answers.word; n++; } while (performance.now() - t0 < 2 && n < 200);
  return { a, us: (performance.now() - t0) * 1000 / n, px };
}
function guess(strokeEnded, liveOnly = false) {
  const raw = [...strokes, ...(cur && cur.length > 1 ? [cur] : [])];
  if (!raw.length) { $("guess").innerHTML = `<p class="muted">Draw something. It guesses as you go.</p>`; $("thumb").getContext("2d").clearRect(0, 0, 128, 128); return; }
  const { a, us, px } = decide(raw); timing = us;
  const top = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]).slice(0, 3);
  $("guess").innerHTML = `<div class="dd-say ${a.certified ? "sure" : ""}">${a.certified ? `I know! It's ${/^[aeiou]/.test(a.choice) ? "an" : "a"} <b>${a.choice}</b> ${ICON[a.choice] || ""}` : `Hmm… maybe ${/^[aeiou]/.test(a.choice) ? "an" : "a"} <b>${a.choice}</b>?`}</div>` +
    top.map(([o, p], i) => `<div class="pbar${i ? "" : " top"}"><i style="width:${Math.max(1, Math.round(p * 100))}%"></i><em>${ICON[o] || ""} ${o}</em><b>${Math.round(p * 100)}%</b></div>`).join("") +
    `<p class="muted dd-us">${a.certified ? "certified · " : "not certified yet · "}${us < 10 ? us.toFixed(1) : Math.round(us)} µs per guess, in this tab</p>`;
  const tc = $("thumb").getContext("2d"), im = tc.createImageData(SIZE, SIZE);
  for (let i = 0; i < px.length; i++) { const v = px[i]; im.data.set([v, v, v, 255], i * 4); }
  tc.putImageData(im, 0, 0);
  if (game && strokeEnded && !liveOnly) judge(a);
}

// ---------------------------------------------------------------- the game: draw the word, shad0w races to name it
let game = null;
function newRound() {
  const w = WORDS[Math.floor(Math.random() * WORDS.length)];
  game = { word: w, t0: 0, strokes: 0, done: false, round: (game?.round || 0) + 1, score: game?.score || { won: 0, played: 0, strokes: [] } };
  strokes = []; cur = null; paint(); guess(false);
  $("task").innerHTML = `Round ${game.round} · draw ${/^[aeiou]/.test(w) ? "an" : "a"} <b>${w}</b> ${ICON[w] || ""}`;
  $("result").textContent = "";
}
function judge(a) {
  if (game.done) return;
  game.strokes = strokes.length;
  if (a.certified && a.choice === game.word) {
    game.done = true; game.score.won++; game.score.played++; game.score.strokes.push(game.strokes);
    const s = ((performance.now() - game.t0) / 1000).toFixed(1);
    $("result").innerHTML = `✅ shad0w got it after <b>${game.strokes}</b> stroke${game.strokes > 1 ? "s" : ""} (${s} s of drawing). <button type="button" class="btn" id="next">Next word →</button>`;
    $("next").addEventListener("click", newRound); score();
  } else if (a.certified && a.choice !== game.word) {
    $("result").innerHTML = `It's sure it's ${/^[aeiou]/.test(a.choice) ? "an" : "a"} <b>${a.choice}</b>… keep drawing, or <button type="button" class="linkish" id="skip">skip</button>`;
    $("skip").addEventListener("click", () => { game.score.played++; score(); newRound(); });
  }
}
function score() { const s = game.score; $("score").textContent = s.played ? `${s.won} of ${s.played} guessed · ${s.strokes.length ? `${(s.strokes.reduce((a, b) => a + b, 0) / s.strokes.length).toFixed(1)} strokes on average` : ""}` : ""; }
$("play").addEventListener("click", () => { newRound(); $("play").textContent = "New word"; });
$("words").innerHTML = WORDS.map((w) => `<span>${ICON[w] || ""} ${w}</span>`).join("");

// ---------------------------------------------------------------- numbers
const p = (x) => `${(100 * x).toFixed(1)}%`;
$("nums").innerHTML = [["✓", p(info.accuracy), `of ${info.test.toLocaleString("en-US")} unseen finished doodles named correctly (best guess)`],
  ["⚡", p(info.certified_share), `of finished doodles answered with a certificate; ${p(info.certified_accuracy)} of those are right`],
  ["↗", p(info.early.right_when_first_certified), `right when it first says "I know!" mid-drawing, at a median ${Math.round(info.early.median_share_of_strokes * 100)}% of the strokes. Lower than on finished doodles: guessing after every stroke gives it more chances to be wrong`],
  ["0", `${(info.table_kb / 1024).toFixed(1)} MB`, "the whole table. No neural network, no server: pixels in, a guess out."]]
  .map(([i, b, t]) => `<div class="panel"><i>${i}</i><b>${b}</b><p>${t}</p></div>`).join("");
document.querySelectorAll("[data-d]").forEach((e) => { e.textContent = typeof info[e.dataset.d] === "number" ? info[e.dataset.d].toLocaleString("en-US") : info[e.dataset.d]; });
resize(); guess(false);
