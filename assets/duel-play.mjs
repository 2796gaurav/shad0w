// shad0w Duel: the browser side. Rendering (three.js), input, sound, HUD and rounds. The rules live in duel.mjs (shared with
// the training harness: CS 1.6's AK-47 numbers, speeds and spray) and the bot's decisions come from a shad0w table in this tab.
// Everything you see and hear is original and made in code: no assets from any commercial game.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js";
import { MAP, MW, MH, H_WALL, H_CRATE, H_BOX, DT, round, step, describe, visible, eye, SKILL, spreadDeg, airborne, moving, callout } from "./duel.mjs";
import { Bundle } from "./shad0w.mjs";

const $ = (id) => document.getElementById(id);
const bundle = await Bundle.load(new URL("bundle", location.href).href);
const view = $("view");
const DEBUG = new URLSearchParams(location.search).has("debug");

// ---------------- scene ----------------
const renderer = new THREE.WebGLRenderer({ canvas: view, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0xe4d6b8, 26, 80);
const camera = new THREE.PerspectiveCamera(78, 16 / 9, 0.03, 300);
camera.rotation.order = "YXZ"; scene.add(camera);
{ // sky: a big sphere with a vertical gradient
  const g = new THREE.SphereGeometry(200, 32, 16), c = [], top = new THREE.Color(0x6aa6d8), hor = new THREE.Color(0xf1e2c0);
  const p = g.attributes.position; for (let i = 0; i < p.count; i++) { const k = Math.max(0, p.getY(i) / 200); const col = hor.clone().lerp(top, Math.pow(k, 0.55)); c.push(col.r, col.g, col.b); }
  g.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
  const sky = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, fog: false })); sky.position.set(MW / 2, 0, MH / 2); scene.add(sky);
}
scene.add(new THREE.HemisphereLight(0xe6eef7, 0xb08a58, 0.85));
const sun = new THREE.DirectionalLight(0xfff0d0, 2.4);
sun.position.set(MW * 0.15, 34, -12); sun.target.position.set(MW / 2, 0, MH / 2);
sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0004;
Object.assign(sun.shadow.camera, { left: -34, right: 34, top: 34, bottom: -34, near: 1, far: 100 });
scene.add(sun, sun.target);

function canvasTex(w, h, paint, repeat = [1, 1]) {
  const c = document.createElement("canvas"); c.width = w; c.height = h; paint(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
}
const rand = (() => { let s = 7; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();
const sandTex = canvasTex(512, 512, (g, w, h) => { g.fillStyle = "#d6bd8c"; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 9000; i++) { g.fillStyle = `rgba(${110 + rand() * 70},${80 + rand() * 60},${40 + rand() * 40},${rand() * .16})`; const s = 1 + rand() * 3; g.fillRect(rand() * w, rand() * h, s, s); }
  for (let i = 0; i < 26; i++) { g.strokeStyle = `rgba(90,65,35,${.06 + rand() * .08})`; g.lineWidth = 1 + rand() * 2; g.beginPath(); const x = rand() * w, y = rand() * h; g.moveTo(x, y); g.bezierCurveTo(x + 40, y + rand() * 30, x + 90, y - 20, x + 140, y + rand() * 20); g.stroke(); } }, [MW / 4, MH / 4]);
const wallTex = canvasTex(512, 512, (g, w, h) => { g.fillStyle = "#c9a26a"; g.fillRect(0, 0, w, h);
  for (let r = 0; r < 16; r++) for (let c = 0; c < 7; c++) { const bw = 86, bh = 32, x = c * bw + (r % 2) * 43 - 43, y = r * bh;
    g.fillStyle = `hsl(${30 + rand() * 8},${34 + rand() * 12}%,${55 + rand() * 12}%)`; g.fillRect(x + 2, y + 2, bw - 4, bh - 4);
    for (let k = 0; k < 14; k++) { g.fillStyle = `rgba(80,55,25,${rand() * .12})`; g.fillRect(x + rand() * bw, y + rand() * bh, 2, 2); } }
  const grd = g.createLinearGradient(0, 0, 0, h); grd.addColorStop(0, "rgba(255,240,210,.08)"); grd.addColorStop(1, "rgba(70,45,20,.22)"); g.fillStyle = grd; g.fillRect(0, 0, w, h); });
const crateTex = canvasTex(256, 256, (g, w, h) => { g.fillStyle = "#8b5c2c"; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 16; i++) { g.fillStyle = `rgba(55,32,12,${.12 + rand() * .2})`; g.fillRect(0, i * 16, w, 2); }
  g.strokeStyle = "#5b3917"; g.lineWidth = 20; g.strokeRect(10, 10, w - 20, h - 20); g.lineWidth = 16; g.beginPath(); g.moveTo(16, 16); g.lineTo(w - 16, h - 16); g.stroke(); });
const boxTex = canvasTex(256, 256, (g, w, h) => { g.fillStyle = "#b48d58"; g.fillRect(0, 0, w, h); g.strokeStyle = "#7a5629"; g.lineWidth = 14; g.strokeRect(7, 7, w - 14, h - 14);
  g.fillStyle = "rgba(70,45,20,.55)"; g.font = "bold 30px monospace"; g.fillText("CARGO", 74, 120); g.font = "bold 18px monospace"; g.fillText("THIS SIDE UP", 66, 160); });
const floor = new THREE.Mesh(new THREE.PlaneGeometry(MW, MH), new THREE.MeshLambertMaterial({ map: sandTex }));
floor.rotation.x = -Math.PI / 2; floor.position.set(MW / 2, 0, MH / 2); floor.receiveShadow = true; scene.add(floor);
const hash = (x, y) => { const v = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return v - Math.floor(v); };
const wallH = (x, y) => H_WALL + Math.floor(hash(Math.floor(x / 3), Math.floor(y / 3)) * 3) * 0.9;   // looks only: every wall is taller than any eye
{
  const walls = [], crates = [], boxes = [];
  for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) { const h = MAP[y * MW + x]; if (h >= H_WALL) walls.push([x, y]); else if (h === H_CRATE) crates.push([x, y]); else if (h === H_BOX) boxes.push([x, y]); }
  const m4 = new THREE.Matrix4(), col = new THREE.Color();
  const inst = (cells, mat, hgt, tint) => { const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(0.999, 1, 0.999), mat, cells.length);
    cells.forEach(([x, y], i) => { const hh = hgt(x, y); m4.makeScale(1, hh, 1).setPosition(x + .5, hh / 2, y + .5); mesh.setMatrixAt(i, m4); col.setHSL(0.08, 0.12, tint(x, y)); mesh.setColorAt(i, col); });
    mesh.castShadow = true; mesh.receiveShadow = true; scene.add(mesh); };
  inst(walls, new THREE.MeshLambertMaterial({ map: wallTex }), wallH, (x, y) => 0.84 + hash(x, y) * 0.12);
  inst(crates, new THREE.MeshLambertMaterial({ map: crateTex }), () => H_CRATE, () => 0.92);
  inst(boxes, new THREE.MeshLambertMaterial({ map: boxTex }), () => H_BOX, () => 0.92);
  const capMat = new THREE.MeshLambertMaterial({ color: 0x9b7a4c });
  const lintel = (x0, y0, x1, y1) => { const b = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0, 0.7, y1 - y0), capMat); b.position.set((x0 + x1) / 2, 3.05, (y0 + y1) / 2); b.castShadow = true; scene.add(b); };
  lintel(2, 17, 6, 19); lintel(19, 12, 25, 13); lintel(22, 15, 25, 16); lintel(31, 16, 34, 17);
  const doorMat = new THREE.MeshLambertMaterial({ color: 0x6b4321 });   // the long doors, left open
  for (const [x, y, r] of [[2.15, 16.3, 0.45], [5.85, 16.3, -0.45]]) { const d = new THREE.Mesh(new THREE.BoxGeometry(0.08, 2.6, 1.1), doorMat); d.position.set(x, 1.3, y); d.rotation.y = r; d.castShadow = true; scene.add(d); }
}

// ---------------- the enemy: a humanoid with shad0w's orange visor ----------------
const mat = (c, o = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: .8, metalness: .05, ...o });
const M = { cloth: mat(0x2b3140), vest: mat(0x3a3f2e), skin: mat(0x7a5a45), mask: mat(0x16181d), boot: mat(0x1a1a1a), metal: mat(0x1d1f22, { metalness: .6, roughness: .4 }),
  wood: mat(0x6a3f1d), visor: new THREE.MeshStandardMaterial({ color: 0xfca311, emissive: 0xfca311, emissiveIntensity: 1.3 }) };
const box = (w, h, d, m) => { const o = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); o.castShadow = true; return o; };
const enemy = new THREE.Group(); scene.add(enemy);
const hips = new THREE.Group(); enemy.add(hips);
const legL = new THREE.Group(), legR = new THREE.Group();
for (const [leg, z] of [[legL, 0.11], [legR, -0.11]]) {
  leg.position.set(0, 0.86, z);
  const thigh = box(0.18, 0.46, 0.17, M.cloth); thigh.position.y = -0.23;
  const knee = new THREE.Group(); knee.position.y = -0.46;
  const shin = box(0.16, 0.38, 0.15, M.cloth); shin.position.y = -0.19; const boot = box(0.26, 0.1, 0.16, M.boot); boot.position.set(0.04, -0.37, 0);
  knee.add(shin, boot); leg.add(thigh, knee); leg.userData.knee = knee; hips.add(leg);
}
const torso = new THREE.Group(); torso.position.y = 0.86; hips.add(torso);
{
  const t = box(0.26, 0.56, 0.42, M.cloth); t.position.y = 0.3;
  const vest = box(0.3, 0.42, 0.46, M.vest); vest.position.y = 0.34;
  const pouch = box(0.08, 0.12, 0.3, M.vest); pouch.position.set(0.18, 0.24, 0);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 18, 14), M.mask); head.position.y = 0.73; head.castShadow = true;
  const visor = box(0.06, 0.06, 0.2, M.visor); visor.position.set(0.12, 0.75, 0);
  const neck = box(0.1, 0.08, 0.1, M.skin); neck.position.y = 0.62;
  const armL = box(0.46, 0.12, 0.12, M.cloth); armL.position.set(0.2, 0.42, 0.22); armL.rotation.z = -0.2;
  const armR = box(0.4, 0.12, 0.12, M.cloth); armR.position.set(0.18, 0.4, -0.2); armR.rotation.z = -0.15;
  const rifle = new THREE.Group(); rifle.position.set(0.35, 0.4, 0.0);
  const rec = box(0.5, 0.08, 0.06, M.metal), barrel = box(0.36, 0.035, 0.035, M.metal), stock = box(0.24, 0.08, 0.05, M.wood), mag = box(0.06, 0.16, 0.05, M.metal), guard = box(0.18, 0.06, 0.07, M.wood);
  barrel.position.x = 0.42; stock.position.x = -0.34; mag.position.set(0.06, -0.1, 0); mag.rotation.z = 0.35; guard.position.x = 0.2;
  rifle.add(rec, barrel, stock, mag, guard);
  torso.add(t, vest, pouch, head, visor, neck, armL, armR, rifle);
}
const flash = new THREE.PointLight(0xffc56b, 0, 7); scene.add(flash);
let enemyDeadT = -1;

// ---------------- your rifle (first person) ----------------
const vm = new THREE.Group(); camera.add(vm); vm.scale.setScalar(0.78);
camera.add(new THREE.PointLight(0xfff0d8, 0.8, 1.2));   // a little fill light on the rifle in your hands
{
  const mt = mat(0x4a4e55, { metalness: .35, roughness: .5, emissive: 0x15171a }), wd = mat(0x7f4d24, { roughness: .65, emissive: 0x1c0d05 });
  const rec = box(0.07, 0.08, 0.42, mt), barrel = box(0.026, 0.026, 0.36, mt), guard = box(0.075, 0.07, 0.2, wd), stock = box(0.06, 0.1, 0.24, wd), grip = box(0.05, 0.12, 0.05, wd);
  const mag = box(0.05, 0.2, 0.08, mt), sight = box(0.012, 0.03, 0.012, mt), hand = box(0.09, 0.07, 0.12, M.cloth);
  barrel.position.z = -0.38; guard.position.set(0, -0.005, -0.25); stock.position.set(0, -0.02, 0.3); grip.position.set(0, -0.09, 0.08); grip.rotation.x = 0.35;
  mag.position.set(0, -0.14, -0.08); mag.rotation.x = -0.35; sight.position.set(0, 0.055, -0.48); hand.position.set(0, -0.06, -0.24);
  vm.add(rec, barrel, guard, stock, grip, mag, sight, hand); vm.userData.mag = mag;
  vm.traverse((o) => { o.castShadow = false; });
}
const muzzle = new THREE.Mesh(new THREE.PlaneGeometry(0.24, 0.24), new THREE.MeshBasicMaterial({ color: 0xffc56b, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
muzzle.position.set(0, 0, -0.6); vm.add(muzzle);
let kick = 0, punch = 0, reloadAnim = 0;

const tracers = [], puffs = [];
function tracer(x0, y0, z0, x1, y1, z1, color) {
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x0, z0, y0), new THREE.Vector3(x1, z1, y1)]);
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: .8 })); scene.add(l); tracers.push({ l, t: 0.06 });
}
function puff(x, y, z) { const s = new THREE.Mesh(new THREE.SphereGeometry(0.06, 6, 6), new THREE.MeshBasicMaterial({ color: 0x8a7350, transparent: true, opacity: .8 }));
  s.position.set(x, z, y); scene.add(s); puffs.push({ s, t: 0.6 }); }

// ---------------- sound: synthesised, positional ----------------
let ac = null;
function out(pan = 0, gain = 1) { const g = ac.createGain(); g.gain.value = gain; const p = ac.createStereoPanner(); p.pan.value = pan; g.connect(p).connect(ac.destination); return g; }
function noise(dur, decay) { const b = ac.createBuffer(1, Math.max(1, ac.sampleRate * dur), ac.sampleRate), d = b.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, decay); const n = ac.createBufferSource(); n.buffer = b; return n; }
function gunshot(pan, gain, near) { if (!ac) return; const n = noise(0.22, 2.2), f = ac.createBiquadFilter(); f.type = "lowpass"; f.frequency.value = near ? 2400 : 900;
  const o = ac.createOscillator(); o.type = "triangle"; o.frequency.setValueAtTime(150, ac.currentTime); o.frequency.exponentialRampToValueAtTime(45, ac.currentTime + .12);
  const og = ac.createGain(); og.gain.setValueAtTime(.6, ac.currentTime); og.gain.exponentialRampToValueAtTime(.001, ac.currentTime + .15);
  const dst = out(pan, gain); n.connect(f).connect(dst); o.connect(og).connect(dst); n.start(); o.start(); o.stop(ac.currentTime + .16); }
function footstep(pan, gain) { if (!ac || gain < .01) return; const n = noise(0.08, 3), f = ac.createBiquadFilter(); f.type = "bandpass"; f.frequency.value = 420 + Math.random() * 160; f.Q.value = .8; n.connect(f).connect(out(pan, gain)); n.start(); }
function ding(freq, gain = .15) { if (!ac) return; const o = ac.createOscillator(), g = ac.createGain(); o.frequency.value = freq; g.gain.setValueAtTime(gain, ac.currentTime); g.gain.exponentialRampToValueAtTime(.001, ac.currentTime + .18); o.connect(g).connect(ac.destination); o.start(); o.stop(ac.currentTime + .2); }
function clack(gain = .2) { if (!ac) return; const n = noise(0.05, 4), f = ac.createBiquadFilter(); f.type = "highpass"; f.frequency.value = 1800; n.connect(f).connect(out(0, gain)); n.start(); }
function heard(x, y) {   // a sound at (x, y) for you: panned by bearing, quieter with distance and through walls
  const hm = m.human, dx = x - hm.x, dy = y - hm.y, d = Math.hypot(dx, dy) || 1;
  const pan = Math.max(-1, Math.min(1, Math.sin(Math.atan2(dy, dx) - yaw)));
  const occl = visible(hm, { ...m.bot, x, y, z: 0, crouch: false }) ? 1 : 0.45;
  return [pan, Math.max(0, 1 - d / 28) * occl];
}

// ---------------- input ----------------
const keys = new Set(); let mouseDown = false, yaw = 0, pitch = 0, locked = false;
const sens = 0.0022, tapped = new Set();   // R and Space taps are kept until the next tick, so a quick press is never lost
addEventListener("keydown", (e) => { if (!locked && !DEBUG) return; keys.add(e.code); if (e.code === "KeyR" || e.code === "Space") tapped.add(e.code);
  if (["Space", "ControlLeft", "KeyC", "ShiftLeft"].includes(e.code)) e.preventDefault(); });
addEventListener("keyup", (e) => keys.delete(e.code));
view.addEventListener("mousedown", (e) => { if ((locked || DEBUG) && e.button === 0) mouseDown = true; });
addEventListener("mouseup", (e) => { if (e.button === 0) mouseDown = false; });
addEventListener("mousemove", (e) => { if (!locked) return; yaw += e.movementX * sens; pitch = Math.max(-1.3, Math.min(1.3, pitch - e.movementY * sens)); });
document.addEventListener("pointerlockchange", () => { locked = document.pointerLockElement === view; if (!locked && state === "play" && !DEBUG) pause(); });

// ---------------- match ----------------
const SET = { skill: { el: "cSkill", vals: ["easy", "normal", "hard", "elite"], label: (v) => v } };
const opts = { skill: "normal" };
let m = null, state = "menu", acc = 0, last = performance.now(), freezeUntil = 0, roundNo = 0, bannerT = 0;
const score = { you: 0, shad0w: 0 }, stats = { shots: 0, hits: 0, heads: 0, decisions: 0, cert: 0, lastDecision: null };
const recent = [];
function newMatch() { score.you = score.shad0w = 0; roundNo = 0; Object.assign(stats, { shots: 0, hits: 0, heads: 0 }); $("sYou").textContent = 0; $("sShad").textContent = 0; $("feed").innerHTML = ""; nextRound(); }
function nextRound() {
  roundNo++;
  m = round((Date.now() + roundNo * 7919) >>> 0, roundNo % 2 ? "T" : "CT");
  yaw = m.human.yaw; pitch = 0; enemyDeadT = -1; enemy.rotation.set(0, 0, 0);
  freezeUntil = m.t + 1.6; showBanner(`Round ${roundNo} · you are ${m.human.side === "CT" ? "Counter-Terrorist" : "Terrorist"}`, 1.8);
}
function showBanner(s, t) { bannerT = t; $("banner").textContent = s; $("banner").hidden = false; }
function feed(killer, head) {
  const li = document.createElement("div"), gun = head ? "AK ✸" : "AK";
  li.innerHTML = killer === "you" ? `<b>you</b> <i>${gun}</i> <span class="o">shad0w</span>` : `<span class="o">shad0w</span> <i>${gun}</i> <b>you</b>`;
  $("feed").prepend(li); setTimeout(() => li.remove(), 6000);
}

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
  const act = { wishX: wx, wishY: wy, crouch: keys.has("ControlLeft") || keys.has("KeyC"), walk: keys.has("ShiftLeft") || keys.has("ShiftRight"),
    jump: keys.has("Space") || tapped.has("Space"), fire: mouseDown, reload: keys.has("KeyR") || tapped.has("KeyR"), yaw, pitch };
  tapped.clear(); return act;
}
function simTick() {
  if (m.over) return;
  const frozen = m.t < freezeUntil;
  const b = frozen ? { move: "hold", stance: "stand", fire: "hold_fire" } : botDecide();
  const h = frozen ? { wishX: 0, wishY: 0, crouch: false, jump: false, fire: false, reload: false, yaw, pitch } : humanAct();
  const wasReloading = m.human.reloadT > 0;
  const ev = step(m, b, h, SKILL[opts.skill]);
  if (!wasReloading && m.human.reloadT > 0) { reloadAnim = 2.5; clack(.25); }
  for (const e of ev) {
    if (e.kind === "step") { if (e.who === "bot") { const [p, g] = heard(e.x, e.y); footstep(p, g * .55); } else footstep(0, .05); continue; }
    if (e.kind !== "shot") continue;
    const f = e.who === "bot" ? m.bot : m.human;
    if (e.who === "human") { stats.shots++; gunshot(0, .32, true); kick = 1; punch = Math.min(punch + 0.012, 0.06); muzzle.material.opacity = 1; muzzle.rotation.z = Math.random() * 3;
      if (e.hit) { stats.hits++; stats.heads += e.head; hitMarker(e.head); if (e.head) ding(2900, .18); } }
    else { const [p, g] = heard(f.x, f.y); gunshot(p, .1 + .35 * g, false); flash.position.set(f.x, eye(f), f.y); flash.intensity = 7; if (e.hit) damage(); }
    tracer(f.x + Math.cos(f.yaw) * .45, f.y + Math.sin(f.yaw) * .45, eye(f) - .14, e.x, e.y, e.z, e.who === "bot" ? 0xffcf7a : 0xfff2c4);
    if (!e.hit) puff(e.x, e.y, e.z);
  }
  if (m.over) endRound(ev);
}
function endRound(ev) {
  const w = m.winner, lastShot = ev.filter((e) => e.kind === "shot" && e.hit).pop();
  if (w === "human") { score.you++; enemyDeadT = 0; feed("you", lastShot?.head); } else if (w === "bot") { score.shad0w++; feed("shad0w", lastShot?.head); }
  $("sYou").textContent = score.you; $("sShad").textContent = score.shad0w;
  const done = score.you >= 5 || score.shad0w >= 5;
  showBanner(w === "human" ? `${m.human.side === "CT" ? "Counter-Terrorists" : "Terrorists"} win · you` : w === "bot" ? "shad0w wins the round" : "Round draw", 2.2);
  setTimeout(() => { if (done) finish(); else nextRound(); }, 2400);
}
function finish() {
  state = "over"; document.exitPointerLock?.();
  const a = stats.shots ? Math.round(100 * stats.hits / stats.shots) : 0;
  $("overTitle").textContent = score.you > score.shad0w ? `You beat shad0w ${score.you}–${score.shad0w} 🏆` : `shad0w wins ${score.shad0w}–${score.you}`;
  $("overSub").innerHTML = `Your accuracy ${a}% · head shots ${stats.heads}. shad0w made <b>${stats.decisions.toLocaleString("en-US")}</b> decisions this match, every one in this tab.`;
  $("start").textContent = "Play again →"; $("over").hidden = false;
}
function pause() { state = "paused"; $("overTitle").textContent = "Paused"; $("overSub").textContent = "Click to resume."; $("start").textContent = "Resume →"; $("over").hidden = false; }

// ---------------- HUD ----------------
let hitT = 0, dmgT = 0;
function hitMarker(head) { hitT = .16; $("hit").className = head ? "hit head" : "hit"; }
function damage() { dmgT = .4; }
const mini = $("mini"), mg = mini.getContext("2d"), MS = 4;
mini.width = MW * MS; mini.height = MH * MS;
const miniBase = document.createElement("canvas"); miniBase.width = mini.width; miniBase.height = mini.height;
{ const g = miniBase.getContext("2d"); for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) { const h = MAP[y * MW + x];
  g.fillStyle = h === 0 ? "rgba(214,189,140,.72)" : h < 1.2 ? "rgba(139,92,44,.9)" : h < 2 ? "rgba(180,141,88,.9)" : "rgba(28,22,14,.85)"; g.fillRect(x * MS, y * MS, MS, MS); } }
function drawHud(dt) {
  const hm = m.human;
  $("hp").textContent = Math.max(0, hm.hp);
  $("ammo").innerHTML = hm.reloadT > 0 ? "<small>reloading</small>" : `${hm.ammo}<small> | ∞</small>`;
  const tl = Math.max(0, Math.ceil(75 - m.t)); $("timer").textContent = `${Math.floor(tl / 60)}:${String(tl % 60).padStart(2, "0")}`;
  $("where").textContent = callout(hm.x, hm.y);
  $("cross").style.setProperty("--gap", `${5 + spreadDeg(hm) * 5}px`);
  hitT = Math.max(0, hitT - dt); $("hit").style.opacity = hitT > 0 ? 1 : 0;
  dmgT = Math.max(0, dmgT - dt); $("dmg").style.opacity = dmgT * 1.6;
  if (bannerT > 0) { bannerT -= dt; if (bannerT <= 0) $("banner").hidden = true; }
  // radar: only you. CS radars show an enemy only when a teammate spots it; in a 1v1 there is nobody to spot it.
  mg.clearRect(0, 0, mini.width, mini.height); mg.drawImage(miniBase, 0, 0);
  mg.fillStyle = "#fff"; mg.beginPath(); mg.arc(hm.x * MS, hm.y * MS, 4, 0, 7); mg.fill();
  mg.strokeStyle = "#fff"; mg.lineWidth = 2; mg.beginPath(); mg.moveTo(hm.x * MS, hm.y * MS); mg.lineTo((hm.x + Math.cos(yaw) * 2.4) * MS, (hm.y + Math.sin(yaw) * 2.4) * MS); mg.stroke();
  if (stats.lastDecision) $("brain").innerHTML = `<b>${stats.lastDecision.move.replace("_", " ")}</b> · <b>${stats.lastDecision.stance}</b> · <b>${stats.lastDecision.fire.replace("_", " ")}</b>`;
}
setInterval(() => {   // timing, measured here: decide recent situations in a block, then divide (browser clocks are too coarse for one decision)
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
let walkPhase = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (state === "play" && m) { acc += dt; while (acc >= DT) { simTick(); acc -= DT; } }
  if (m) {
    const hm = m.human, bt = m.bot, run = moving(hm) && !airborne(hm);
    const bob = run ? Math.sin(now / (hm.walk || hm.crouch ? 140 : 95)) * (hm.crouch ? 0.01 : 0.022) : 0;
    punch = Math.max(0, punch - dt * 0.12); kick = Math.max(0, kick - dt * 9);
    camera.position.set(hm.x, eye(hm) + bob, hm.y);
    camera.rotation.set(pitch + punch, -yaw - Math.PI / 2, hm.hp <= 0 ? 0.6 : 0);
    reloadAnim = Math.max(0, reloadAnim - dt);   // the rifle in your hands: kick, sway, reload dip
    const rk = reloadAnim > 0 ? Math.sin(Math.min(1, (2.5 - reloadAnim) / 2.5) * Math.PI) : 0;
    vm.position.set(0.24 + bob * 0.6, -0.235 - rk * 0.18 + Math.abs(bob) * 0.4, -0.5 + kick * 0.05);
    vm.rotation.set(kick * 0.08 - rk * 0.6, 0, rk * 0.3); vm.userData.mag.position.y = -0.14 - rk * 0.12;
    muzzle.material.opacity *= 0.35;
    const speed = Math.hypot(bt.vx, bt.vy); walkPhase += dt * speed * 2.6;   // the enemy: walk cycle, crouch, death fall
    const crouch = bt.crouch ? 1 : 0, swing = speed > 0.4 ? Math.sin(walkPhase) * 0.55 * (1 - crouch * .5) : 0;
    enemy.position.set(bt.x, bt.z, bt.y); enemy.rotation.y = -bt.yaw;
    legL.rotation.z = swing - crouch * 0.9; legR.rotation.z = -swing - crouch * 0.9;
    legL.userData.knee.rotation.z = legR.userData.knee.rotation.z = crouch * 1.5;
    hips.position.y = -crouch * 0.42; torso.rotation.z = -crouch * 0.15;
    if (enemyDeadT >= 0) { enemyDeadT += dt; enemy.rotation.z = Math.min(1.45, enemyDeadT * 4); enemy.position.y = -Math.min(0.25, enemyDeadT); }
    flash.intensity *= 0.55;
    for (let i = tracers.length - 1; i >= 0; i--) { const t = tracers[i]; t.t -= dt; t.l.material.opacity = Math.max(0, t.t / .06) * .8; if (t.t <= 0) { scene.remove(t.l); t.l.geometry.dispose(); tracers.splice(i, 1); } }
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
    $(s.el).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); opts[k] = b.dataset.v;
    $("overTitle").textContent = `Difficulty: ${opts[k]}`; $("overSub").textContent = "Its brain is the same at every level; only its hand (reaction time and turn speed) changes."; $("start").textContent = "Play →"; $("over").hidden = false;
    newMatch(); state = "menu";
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
requestAnimationFrame(frame);
if (DEBUG) {   // test hooks: ?debug lets keys through without pointer lock; &fov=N zooms; the camera faces the bot until told otherwise
  window.__duel = { get m() { return m; }, get stats() { return stats; }, follow: true };
  const fov = +new URLSearchParams(location.search).get("fov"); if (fov) { camera.fov = fov; camera.updateProjectionMatrix(); }
  setInterval(() => { if (m && window.__duel.follow) { yaw = Math.atan2(m.bot.y - m.human.y, m.bot.x - m.human.x); pitch = -0.05; } }, 16);
}
