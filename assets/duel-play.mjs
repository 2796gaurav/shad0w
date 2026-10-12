// shad0w Duel: the browser side. Rendering (three.js), input, HUD and rounds; the rules live in duel.mjs (shared with
// the training harness) and the bot's decisions come from a shad0w table running in this tab.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js";
import { MAP, MW, MH, H_WALL, H_CRATE, H_BOX, TICK, DT, round, step, describe, visible, eye, SKILL, GUN, spreadDeg, airborne, moving } from "./duel.mjs";
import { Bundle } from "./shad0w.mjs";

const $ = (id) => document.getElementById(id);
const bundle = await Bundle.load(new URL("bundle", location.href).href);
const info = await (await fetch("duel.json")).json().catch(() => ({}));
const view = $("view"), hud = $("hud");

// ---------------- three.js scene ----------------
const renderer = new THREE.WebGLRenderer({ canvas: view, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9cc3de);
scene.fog = new THREE.Fog(0xd9cdb4, 24, 70);
const camera = new THREE.PerspectiveCamera(78, 16 / 9, 0.05, 200);
camera.rotation.order = "YXZ";
scene.add(new THREE.HemisphereLight(0xdfe9f3, 0xb08a58, 0.9));
const sun = new THREE.DirectionalLight(0xfff1d6, 2.1);
sun.position.set(MW * 0.2, 30, -10); sun.target.position.set(MW / 2, 0, MH / 2);
sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -32, right: 32, top: 32, bottom: -32, near: 1, far: 90 });
scene.add(sun, sun.target);

function canvasTex(w, h, paint, repeat = [1, 1]) {
  const c = document.createElement("canvas"); c.width = w; c.height = h; paint(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
}
const rand = (() => { let s = 7; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();
const sandTex = canvasTex(256, 256, (g, w, h) => { g.fillStyle = "#d8c08f"; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 2600; i++) { g.fillStyle = `rgba(${120 + rand() * 60},${90 + rand() * 50},${50 + rand() * 30},${rand() * .18})`; g.fillRect(rand() * w, rand() * h, 2, 2); }
  g.strokeStyle = "rgba(120,90,50,.18)"; g.lineWidth = 2; g.strokeRect(0, 0, w, h); }, [MW / 2, MH / 2]);
const wallTex = canvasTex(256, 256, (g, w, h) => { g.fillStyle = "#cfa86e"; g.fillRect(0, 0, w, h);
  for (let r = 0; r < 8; r++) for (let c = 0; c < 4; c++) { const x = c * 64 + (r % 2) * 32, y = r * 32;
    g.fillStyle = `hsl(${32 + rand() * 6},${38 + rand() * 10}%,${58 + rand() * 9}%)`; g.fillRect(x + 1, y + 1, 62, 30); }
  g.strokeStyle = "rgba(90,62,30,.35)"; g.lineWidth = 2; for (let r = 0; r <= 8; r++) { g.beginPath(); g.moveTo(0, r * 32); g.lineTo(w, r * 32); g.stroke(); } });
const crateTex = canvasTex(128, 128, (g, w, h) => { g.fillStyle = "#8a5a2b"; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 8; i++) { g.fillStyle = `rgba(60,35,15,${.15 + rand() * .2})`; g.fillRect(0, i * 16, w, 2); }
  g.strokeStyle = "#5a3818"; g.lineWidth = 10; g.strokeRect(5, 5, w - 10, h - 10); g.beginPath(); g.moveTo(8, 8); g.lineTo(w - 8, h - 8); g.stroke(); });
const boxTex = canvasTex(128, 128, (g, w, h) => { g.fillStyle = "#b9925a"; g.fillRect(0, 0, w, h); g.strokeStyle = "#7d5a2e"; g.lineWidth = 8; g.strokeRect(4, 4, w - 8, h - 8);
  g.fillStyle = "rgba(80,55,25,.5)"; g.font = "bold 22px monospace"; g.fillText("FRAGILE", 18, 70); });

const floor = new THREE.Mesh(new THREE.PlaneGeometry(MW, MH), new THREE.MeshLambertMaterial({ map: sandTex }));
floor.rotation.x = -Math.PI / 2; floor.position.set(MW / 2, 0, MH / 2); floor.receiveShadow = true; scene.add(floor);
const mats = { [H_WALL]: new THREE.MeshLambertMaterial({ map: wallTex }), [H_CRATE]: new THREE.MeshLambertMaterial({ map: crateTex }), [H_BOX]: new THREE.MeshLambertMaterial({ map: boxTex }) };
for (const hgt of [H_WALL, H_CRATE, H_BOX]) {
  const cells = []; for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) if (Math.abs(MAP[y * MW + x] - hgt) < 1e-3) cells.push([x, y]);
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, hgt, 1), mats[hgt], cells.length);
  const m4 = new THREE.Matrix4(), col = new THREE.Color();
  cells.forEach(([x, y], i) => { m4.makeTranslation(x + .5, hgt / 2, y + .5); mesh.setMatrixAt(i, m4); col.setHSL(0.08, 0.1, 0.86 + ((x * 7 + y * 13) % 5) * 0.035); mesh.setColorAt(i, col); });
  mesh.castShadow = true; mesh.receiveShadow = true; scene.add(mesh);
}

// the bot: dark body, orange visor and trim (shad0w's colours)
const bot = new THREE.Group();
const bodyMat = new THREE.MeshStandardMaterial({ color: 0x33405f, roughness: .55, metalness: .25 }), orangeMat = new THREE.MeshStandardMaterial({ color: 0xfca311, emissive: 0xfca311, emissiveIntensity: 1.4 });
const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.26, 0.75, 6, 12), bodyMat); torso.castShadow = true;
const head = new THREE.Mesh(new THREE.SphereGeometry(0.17, 20, 16), bodyMat); head.castShadow = true;
const visor = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.07, 0.3), orangeMat);
const gun = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.08, 0.08), new THREE.MeshStandardMaterial({ color: 0x111111 }));
const belt = new THREE.Mesh(new THREE.TorusGeometry(0.27, 0.025, 8, 24), orangeMat); belt.rotation.x = Math.PI / 2;
bot.add(torso, head, visor, gun, belt); scene.add(bot);
const flash = new THREE.PointLight(0xffc56b, 0, 6); scene.add(flash);
const tracers = [];
function tracer(x0, y0, z0, x1, y1, z1, color) {
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x0, z0, y0), new THREE.Vector3(x1, z1, y1)]);
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: .85 })); scene.add(l); tracers.push({ l, t: 0.09 });
}
const puffs = [];
function puff(x, y, z) { const s = new THREE.Mesh(new THREE.SphereGeometry(0.05, 6, 6), new THREE.MeshBasicMaterial({ color: 0x3a2a18, transparent: true, opacity: .8 }));
  s.position.set(x, z, y); scene.add(s); puffs.push({ s, t: 0.6 }); }

// ---------------- sound (WebAudio, synthesised) ----------------
let ac = null;
function bang(gain = .25, freq = 900) {
  if (!ac) return; const n = ac.createBufferSource(), b = ac.createBuffer(1, ac.sampleRate * .12, ac.sampleRate), d = b.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 3);
  n.buffer = b; const f = ac.createBiquadFilter(); f.type = "lowpass"; f.frequency.value = freq; const g = ac.createGain(); g.gain.value = gain;
  n.connect(f).connect(g).connect(ac.destination); n.start();
}
function tick(freq = 1800, gain = .15) { if (!ac) return; const o = ac.createOscillator(), g = ac.createGain(); o.frequency.value = freq; g.gain.setValueAtTime(gain, ac.currentTime); g.gain.exponentialRampToValueAtTime(.001, ac.currentTime + .08); o.connect(g).connect(ac.destination); o.start(); o.stop(ac.currentTime + .1); }

// ---------------- input ----------------
const DEBUG = new URLSearchParams(location.search).has("debug");
const keys = new Set(); let mouseDown = false, yaw = 0, pitch = 0, sens = 0.0022, locked = false;
const tapped = new Set();   // R and Space taps are kept until the next tick, so a quick press is never lost
addEventListener("keydown", (e) => { if (!locked && !DEBUG) return; keys.add(e.code); if (e.code === "KeyR" || e.code === "Space") tapped.add(e.code); if (["Space", "ControlLeft", "KeyC"].includes(e.code)) e.preventDefault(); });
addEventListener("keyup", (e) => keys.delete(e.code));
view.addEventListener("mousedown", (e) => { if ((locked || DEBUG) && e.button === 0) mouseDown = true; });
addEventListener("mouseup", (e) => { if (e.button === 0) mouseDown = false; });
addEventListener("mousemove", (e) => { if (!locked) return; yaw += e.movementX * sens; pitch = Math.max(-1.3, Math.min(1.3, pitch - e.movementY * sens)); });
document.addEventListener("pointerlockchange", () => { locked = document.pointerLockElement === view; if (!locked && state === "play" && !DEBUG) pause(); });

// ---------------- match state ----------------
const SET = { skill: { el: "cSkill", vals: ["easy", "normal", "hard", "elite"], label: (v) => v } };
const opts = { skill: "normal" };
let match = null, m = null, state = "menu", acc = 0, last = performance.now(), freezeUntil = 0, roundNo = 0, banner = "", bannerT = 0;
const score = { you: 0, shad0w: 0 }, stats = { shots: 0, hits: 0, heads: 0, decisions: 0, cert: 0, lastDecision: null };
const recent = [];
function newMatch() { score.you = score.shad0w = 0; roundNo = 0; Object.assign(stats, { shots: 0, hits: 0, heads: 0 }); nextRound(); }
function nextRound() {
  roundNo++;
  const botSide = roundNo % 2 ? "T" : "CT";
  m = round((Date.now() + roundNo * 7919) >>> 0, botSide);
  yaw = m.human.yaw; pitch = 0;
  freezeUntil = m.t + 1.6; showBanner(`Round ${roundNo} · you are ${m.human.side}`, 1.6);
}
function showBanner(s, t) { banner = s; bannerT = t; $("banner").textContent = s; $("banner").hidden = false; }

// ---------------- the bot's brain: shad0w, every tick ----------------
function botDecide() {
  const text = describe(m);
  const a = bundle.decide(text, { probabilities: false }).answers;
  stats.decisions++; stats.cert += (a.move.certified && a.stance.certified && a.fire.certified) ? 1 : 0;
  stats.lastDecision = { move: a.move.choice, stance: a.stance.choice, fire: a.fire.choice };
  if (recent.length < 64) recent.push(text); else recent[stats.decisions % 64] = text;
  return stats.lastDecision;
}

function humanAct() {
  const f = Math.cos(yaw), s = Math.sin(yaw);
  let wx = 0, wy = 0;
  if (keys.has("KeyW")) { wx += f; wy += s; } if (keys.has("KeyS")) { wx -= f; wy -= s; }
  if (keys.has("KeyD")) { wx -= s; wy += f; } if (keys.has("KeyA")) { wx += s; wy -= f; }
  const act = { wishX: wx, wishY: wy, crouch: keys.has("ControlLeft") || keys.has("KeyC"), jump: keys.has("Space") || tapped.has("Space"), fire: mouseDown, reload: keys.has("KeyR") || tapped.has("KeyR"), yaw, pitch };
  tapped.clear();
  return act;
}

function simTick() {
  if (m.over) return;
  const frozen = m.t < freezeUntil;
  const b = frozen ? { move: "hold", stance: "stand", fire: "hold_fire" } : botDecide();
  const h = frozen ? { wishX: 0, wishY: 0, crouch: false, jump: false, fire: false, reload: false, yaw, pitch } : humanAct();
  const ev = step(m, b, h, SKILL[opts.skill]);
  for (const e of ev) {
    if (e.kind !== "shot") continue;
    const f = e.who === "bot" ? m.bot : m.human;
    if (e.who === "human") { stats.shots++; bang(.22, 1400); if (e.hit) { stats.hits++; stats.heads += e.head; hitMarker(e.head); tick(e.head ? 2600 : 1700, .2); } }
    else { bang(.3, 700); flash.position.set(f.x, eye(f), f.y); flash.intensity = 6; if (e.hit) damage(); }
    tracer(f.x + Math.cos(f.yaw) * .4, f.y + Math.sin(f.yaw) * .4, eye(f) - .12, e.x, e.y, e.z, e.who === "bot" ? 0xfca311 : 0xfff2c4);
    if (!e.hit) puff(e.x, e.y, e.z);
  }
  if (m.over) endRound();
}
function endRound() {
  const w = m.winner;
  if (w === "human") score.you++; else if (w === "bot") score.shad0w++;
  $("sYou").textContent = score.you; $("sShad").textContent = score.shad0w;
  const done = score.you >= 5 || score.shad0w >= 5;
  showBanner(w === "human" ? "You win the round" : w === "bot" ? "shad0w wins the round" : "Time: draw", 2);
  setTimeout(() => { if (done) finish(); else nextRound(); }, 2000);
}
function finish() {
  state = "over"; document.exitPointerLock?.();
  const acc = stats.shots ? Math.round(100 * stats.hits / stats.shots) : 0;
  $("overTitle").textContent = score.you > score.shad0w ? `You beat shad0w ${score.you}–${score.shad0w} 🏆` : `shad0w wins ${score.shad0w}–${score.you}`;
  $("overSub").innerHTML = `Your accuracy ${acc}% · head shots ${stats.heads}. shad0w made <b>${stats.decisions.toLocaleString("en-US")}</b> decisions this match, every one in this tab.`;
  $("start").textContent = "Play again →"; $("over").hidden = false;
}
function pause() { state = "paused"; $("overTitle").textContent = "Paused"; $("overSub").textContent = "Click to resume."; $("start").textContent = "Resume →"; $("over").hidden = false; }

// ---------------- HUD ----------------
let hitT = 0, dmgT = 0;
function hitMarker(head) { hitT = .18; $("hit").className = head ? "hit head" : "hit"; }
function damage() { dmgT = .35; }
const mini = $("mini"), mg = mini.getContext("2d"), MS = 4;
mini.width = MW * MS; mini.height = MH * MS;
const miniBase = document.createElement("canvas"); miniBase.width = mini.width; miniBase.height = mini.height;
{ const g = miniBase.getContext("2d"); for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) { const h = MAP[y * MW + x];
  g.fillStyle = h === 0 ? "rgba(216,192,143,.75)" : h < 1.2 ? "rgba(138,90,43,.9)" : h < 2 ? "rgba(185,146,90,.9)" : "rgba(30,24,16,.85)"; g.fillRect(x * MS, y * MS, MS, MS); } }
let botSeenT = -9;
function drawHud(dt) {
  const hm = m.human, bt = m.bot;
  $("hp").textContent = Math.max(0, hm.hp); $("ammo").textContent = hm.reloadT > 0 ? "reloading" : `${hm.ammo} / ${GUN.mag}`;
  $("timer").textContent = `${Math.max(0, Math.ceil(75 - m.t))}`;
  const sp = spreadDeg(hm), gap = 6 + sp * 6; $("cross").style.setProperty("--gap", `${gap}px`);
  hitT = Math.max(0, hitT - dt); $("hit").style.opacity = hitT > 0 ? 1 : 0;
  dmgT = Math.max(0, dmgT - dt); $("dmg").style.opacity = dmgT * 1.6;
  if (bannerT > 0) { bannerT -= dt; if (bannerT <= 0) $("banner").hidden = true; }
  // radar: the bot shows while you can see it (and briefly after it fires)
  if (visible(hm, bt)) botSeenT = m.t;
  mg.clearRect(0, 0, mini.width, mini.height); mg.drawImage(miniBase, 0, 0);
  const dot = (f, c, r) => { mg.fillStyle = c; mg.beginPath(); mg.arc(f.x * MS, f.y * MS, r, 0, 7); mg.fill(); };
  if (m.t - botSeenT < 1.2) dot(bt, "#fca311", 4);
  dot(hm, "#ffffff", 4); mg.strokeStyle = "#fff"; mg.lineWidth = 2; mg.beginPath(); mg.moveTo(hm.x * MS, hm.y * MS); mg.lineTo((hm.x + Math.cos(yaw) * 2.2) * MS, (hm.y + Math.sin(yaw) * 2.2) * MS); mg.stroke();
  if (stats.lastDecision) $("brain").innerHTML = `<b>${stats.lastDecision.move.replace("_", " ")}</b> · <b>${stats.lastDecision.stance}</b> · <b>${stats.lastDecision.fire.replace("_", " ")}</b>`;
}
// timing, measured here: decide recent situations in a block, then divide (browser clocks are too coarse for one decision)
setInterval(() => {
  if (recent.length < 8) return;
  let n = 0; const t0 = performance.now();
  while (performance.now() - t0 < 6) { for (const s of recent) bundle.decide(s, { probabilities: false }); n += recent.length; }
  const us = (performance.now() - t0) * 1000 / n;
  $("hUs").textContent = `${us < 10 ? us.toFixed(1) : Math.round(us)} µs`;
  $("hBudget").textContent = `${(us / 16667 * 100).toFixed(2)}%`;
  $("hN").textContent = stats.decisions.toLocaleString("en-US");
  $("hRate").textContent = `${Math.round(1e6 / us).toLocaleString("en-US")}`;
  $("hCert").textContent = stats.decisions ? `${Math.round(100 * stats.cert / stats.decisions)}%` : "–";
}, 1000);

// ---------------- render loop ----------------
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (state === "play" && m) { acc += dt; while (acc >= DT) { simTick(); acc -= DT; } }
  if (m) {
    const hm = m.human, bt = m.bot;
    camera.position.set(hm.x, eye(hm) - (airborne(hm) ? 0 : Math.sin(now / 90) * (moving(hm) ? 0.025 : 0)), hm.y);
    camera.rotation.set(pitch, -yaw - Math.PI / 2, 0);
    const crouch = bt.crouch ? 0.62 : 1;
    bot.position.set(bt.x, bt.z, bt.y); bot.rotation.y = -bt.yaw;
    torso.position.set(0, 0.68 * crouch, 0); torso.scale.y = crouch; belt.position.set(0, 0.55 * crouch, 0);
    head.position.set(0, eye(bt) - bt.z + 0.05, 0); visor.position.set(0.14, eye(bt) - bt.z + 0.07, 0);
    gun.position.set(0.32, eye(bt) - bt.z - 0.22, 0.18);
    bot.visible = bt.hp > 0;
    flash.intensity *= 0.6;
    for (let i = tracers.length - 1; i >= 0; i--) { const t = tracers[i]; t.t -= dt; t.l.material.opacity = Math.max(0, t.t / .09) * .85; if (t.t <= 0) { scene.remove(t.l); t.l.geometry.dispose(); tracers.splice(i, 1); } }
    for (let i = puffs.length - 1; i >= 0; i--) { const p = puffs[i]; p.t -= dt; p.s.material.opacity = p.t; p.s.scale.setScalar(1 + (0.6 - p.t) * 3); if (p.t <= 0) { scene.remove(p.s); puffs.splice(i, 1); } }
    drawHud(dt);
  }
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
function resize() { const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize();

// ---------------- menu ----------------
for (const [k, s] of Object.entries(SET)) {
  $(s.el).innerHTML = s.vals.map((v) => `<button type="button" class="chip${v === opts[k] ? " on" : ""}" data-v="${v}">${s.label(v)}</button>`).join("");
  $(s.el).querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    $(s.el).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); opts[k] = b.dataset.v; state = "menu";
    $("overTitle").textContent = `Difficulty: ${opts[k]}`; $("overSub").textContent = "Its brain is the same at every level; only its hand (reaction time and turn speed) changes."; $("start").textContent = "Play →"; $("over").hidden = false; m = null;
  }));
}
$("start").addEventListener("click", () => {
  if (!ac) { try { ac = new AudioContext(); } catch { ac = null; } }
  if (state !== "paused") newMatch();
  state = "play"; $("over").hidden = true; view.requestPointerLock?.(); last = performance.now(); acc = 0;
});
view.addEventListener("click", () => { if (state === "play" && !locked) view.requestPointerLock?.(); });
if (matchMedia("(pointer: coarse)").matches) $("overSub").textContent = "This one needs a keyboard and a mouse: open it on a desktop.";
newMatch(); state = "menu";   // a live scene behind the menu
if (new URLSearchParams(location.search).has("debug")) {   // test hook: ?debug makes the camera face the bot and starts play
  window.__duel = { get m() { return m; }, get stats() { return stats; }, follow: true };
  const fov = +new URLSearchParams(location.search).get("fov"); if (fov) { camera.fov = fov; camera.updateProjectionMatrix(); }
  setInterval(() => { if (m && window.__duel.follow) { yaw = Math.atan2(m.bot.y - m.human.y, m.bot.x - m.human.x); pitch = -0.05; } }, 16);
}
requestAnimationFrame(frame);
