// shad0w Duel: the browser side. Rendering (three.js), input, sound, HUD and rounds. The rules live in duel.mjs (shared with
// the training harness: CS 1.6's AK-47 numbers, speeds and spray) and the bot's decisions come from a shad0w table in this tab.
// Everything you see and hear is original and made in code: no assets from any commercial game.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js";
import { MAP, MW, MH, H_WALL, H_CRATE, H_BOX, DT, round, step, describe, visible, eye, HARDEST, spreadDeg, airborne, moving, callout } from "./duel.mjs";
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
  // the tunnels are roofed and dim, lit by a few bare lamps (looks only: the roof is above every eye and every shot)
  const roofMat = new THREE.MeshLambertMaterial({ color: 0x7d6142 }), lampMat = new THREE.MeshBasicMaterial({ color: 0xffd9a0 });
  for (const [x0, y0, x1, y1] of [[38, 9, 42, 15], [38, 18, 42, 26]]) {
    const r = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0 + 1, 0.4, y1 - y0), roofMat); r.position.set((x0 + x1) / 2, 3.1, (y0 + y1) / 2); r.castShadow = true; scene.add(r);
    for (let z = y0 + 1.5; z < y1; z += 3) { const l = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.08, 0.18), lampMat); l.position.set(40, 2.86, z); scene.add(l); }
    const pl = new THREE.PointLight(0xffc98a, 6, 9, 1.6); pl.position.set(40, 2.6, (y0 + y1) / 2); scene.add(pl);
  }
  // palms over the walls by mid and A, and the painted site letters
  const trunkMat = new THREE.MeshLambertMaterial({ color: 0x7a5a3a }), frondMat = new THREE.MeshLambertMaterial({ color: 0x5f7f3a, side: THREE.DoubleSide });
  for (const [x, z, hgt] of [[25.5, 7.6, 5.6], [13.4, 6.5, 6.2], [36.5, 9.4, 5.2]]) {
    const t = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.2, hgt, 8), trunkMat); t.position.set(x, hgt / 2, z); t.rotation.z = 0.06; t.castShadow = true; scene.add(t);
    for (let k = 0; k < 7; k++) { const f = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 0.5), frondMat), a = k / 7 * Math.PI * 2;
      f.position.set(x + Math.cos(a) * 1.05, hgt - 0.25, z + Math.sin(a) * 1.05); f.rotation.set(-Math.PI / 2 + 0.45, 0, -a, "YXZ"); f.rotation.y = -a; f.castShadow = true; scene.add(f); }
  }
  const letter = (ch, x, z, ry) => { const tex = canvasTex(256, 256, (g, w, h) => { g.clearRect(0, 0, w, h); g.fillStyle = "rgba(170,40,30,.85)"; g.font = "bold 200px sans-serif"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(ch, w / 2, h / 2 + 10); });
    const pl = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.5), new THREE.MeshLambertMaterial({ map: tex, transparent: true })); pl.position.set(x, 1.7, z); pl.rotation.y = ry; scene.add(pl); };
  letter("A", 2.02, 5.5, Math.PI / 2); letter("B", 41.98, 4.5, -Math.PI / 2);
}

// ---------------- the enemy: a CT or a T, with shad0w's orange visor as its only mark ----------------
const mat = (c, o = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: .82, metalness: .04, ...o });
const LOOK = {
  CT: { shirt: 0x3a5078, pants: 0x34425e, vest: 0x2a374d, head: 0x3a4048, glove: 0x15171a, boot: 0x141414, helmet: true },
  T: { shirt: 0x9b8460, pants: 0x5e5446, vest: 0x5b4a33, head: 0x3b3127, glove: 0x4a3524, boot: 0x2b2118, helmet: false } };
const visorMat = new THREE.MeshStandardMaterial({ color: 0xfca311, emissive: 0xfca311, emissiveIntensity: 1.3 });
const gunMetal = mat(0x1d1f22, { metalness: .6, roughness: .4 }), gunWood = mat(0x6a3f1d);
const part = (geo, m, x = 0, y = 0, z = 0) => { const o = new THREE.Mesh(geo, m); o.position.set(x, y, z); o.castShadow = true; return o; };
const cap = (r, len) => new THREE.CapsuleGeometry(r, len, 4, 10);
function soldier(side) {
  const L = LOOK[side], lit = (c) => mat(c, { emissive: new THREE.Color(c).multiplyScalar(0.28) });   // a little self-light, so a soldier in shade still reads
  const M = { shirt: lit(L.shirt), pants: lit(L.pants), vest: lit(L.vest), head: lit(L.head), glove: lit(L.glove), boot: lit(L.boot), skin: lit(0x8a6650) };
  const root = new THREE.Group(), hips = new THREE.Group(); root.add(hips);
  const legs = [];
  for (const z of [0.1, -0.1]) {   // hip -> knee -> boot
    const leg = new THREE.Group(); leg.position.set(0, 0.88, z);
    leg.add(part(cap(0.085, 0.3), M.pants, 0, -0.22, 0));
    const knee = new THREE.Group(); knee.position.y = -0.45;
    knee.add(part(cap(0.075, 0.28), M.pants, 0, -0.2, 0), part(new THREE.BoxGeometry(0.27, 0.1, 0.15), M.boot, 0.05, -0.39, 0));
    leg.add(knee); leg.userData.knee = knee; hips.add(leg); legs.push(leg);
  }
  const torso = new THREE.Group(); torso.position.y = 0.9; hips.add(torso);
  hips.add(part(new THREE.BoxGeometry(0.24, 0.2, 0.36), M.pants, 0, 0.9, 0), part(new THREE.BoxGeometry(0.26, 0.05, 0.38), M.vest, 0, 0.98, 0));   // pelvis and belt
  torso.add(part(cap(0.17, 0.32), M.shirt, 0, 0.3, 0));                                        // chest and belly
  torso.add(part(new THREE.BoxGeometry(0.3, 0.38, 0.4), M.vest, 0.01, 0.34, 0));                 // plate carrier
  torso.add(part(new THREE.BoxGeometry(0.06, 0.12, 0.28), M.vest, 0.17, 0.24, 0));               // pouches
  torso.add(part(new THREE.CylinderGeometry(0.05, 0.06, 0.08, 10), M.skin, 0, 0.64, 0));         // neck
  const head = part(new THREE.SphereGeometry(0.125, 18, 14), M.head, 0, 0.77, 0); torso.add(head);
  if (L.helmet) { const h = part(new THREE.SphereGeometry(0.142, 18, 10, 0, Math.PI * 2, 0, Math.PI / 1.9), M.vest, 0, 0.79, 0); torso.add(h); }
  else { torso.add(part(new THREE.TorusGeometry(0.12, 0.035, 8, 18), M.vest, 0, 0.83, 0).rotateX(Math.PI / 2)); }   // head-wrap
  torso.add(part(new THREE.BoxGeometry(0.05, 0.05, 0.19), visorMat, 0.115, 0.775, 0));            // goggles: shad0w's orange
  // arms: shoulder -> elbow -> glove, both hands on the rifle
  const arm = (z, shoulderRot, elbowRot) => { const s = new THREE.Group(); s.position.set(0.02, 0.52, z); s.rotation.set(0, 0, shoulderRot);
    s.add(part(cap(0.06, 0.2), M.shirt, 0, -0.15, 0)); const e = new THREE.Group(); e.position.y = -0.3; e.rotation.z = elbowRot;
    e.add(part(cap(0.055, 0.18), M.shirt, 0, -0.13, 0), part(new THREE.SphereGeometry(0.05, 8, 8), M.glove, 0, -0.27, 0)); s.add(e); return s; };
  torso.add(arm(0.2, 1.15, 0.9), arm(-0.19, 1.3, 0.55));
  const rifle = new THREE.Group(); rifle.position.set(0.36, 0.42, -0.02);
  rifle.add(part(new THREE.BoxGeometry(0.5, 0.08, 0.06), gunMetal), part(new THREE.BoxGeometry(0.36, 0.035, 0.035), gunMetal, 0.42, 0.01, 0),
            part(new THREE.BoxGeometry(0.24, 0.08, 0.05), gunWood, -0.34, -0.02, 0), part(new THREE.BoxGeometry(0.18, 0.06, 0.07), gunWood, 0.2, 0, 0),
            part(new THREE.BoxGeometry(0.06, 0.17, 0.05), gunMetal, 0.06, -0.1, 0).rotateZ(0.35));
  torso.add(rifle);
  root.userData = { hips, torso, legs, rifle };
  root.visible = false; scene.add(root); return root;
}
const SOLDIER = { CT: soldier("CT"), T: soldier("T") };
let enemy = SOLDIER.T;
const flash = new THREE.PointLight(0xffc56b, 0, 7); scene.add(flash);
let enemyDeadT = -1;

// ---------------- your rifle (first person) ----------------
const box = (w, h, d, m) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
const handMat = { glove: mat(0x15171a, { emissive: 0x0b0b0b }), sleeve: mat(0x2c3e5c, { emissive: 0x0d1018 }) };
function dressHands(side) { const L = LOOK[side]; handMat.glove.color.setHex(L.glove); handMat.sleeve.color.setHex(L.shirt); }
const vm = new THREE.Group(); camera.add(vm); vm.scale.setScalar(0.78);
camera.add(new THREE.PointLight(0xfff0d8, 0.8, 1.2));   // a little fill light on the rifle in your hands
{
  const mt = mat(0x4a4e55, { metalness: .35, roughness: .5, emissive: 0x15171a }), wd = mat(0x7f4d24, { roughness: .65, emissive: 0x1c0d05 });
  const rec = box(0.07, 0.08, 0.42, mt), barrel = box(0.026, 0.026, 0.36, mt), guard = box(0.075, 0.07, 0.2, wd), stock = box(0.06, 0.1, 0.24, wd), grip = box(0.05, 0.12, 0.05, wd);
  const mag = box(0.05, 0.2, 0.08, mt), sight = box(0.012, 0.03, 0.012, mt);
  barrel.position.z = -0.38; guard.position.set(0, -0.005, -0.25); stock.position.set(0, -0.02, 0.3); grip.position.set(0, -0.09, 0.08); grip.rotation.x = 0.35;
  mag.position.set(0, -0.14, -0.08); mag.rotation.x = -0.35; sight.position.set(0, 0.055, -0.48);
  vm.add(rec, barrel, guard, stock, grip, mag, sight); vm.userData.mag = mag;
  // your hands: a glove on the handguard and one on the grip, each with a sleeve going back out of view
  for (const [g, d] of [[[-0.04, -0.065, -0.25], [-0.55, -0.45, 0.7]], [[0.02, -0.13, 0.1], [0.3, -0.5, 0.8]]]) {
    const gl = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 10), handMat.glove); gl.scale.set(1.15, 0.9, 1.4); gl.position.set(...g);
    const dir = new THREE.Vector3(...d).normalize(), len = 0.55, sl = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.062, len, 12), handMat.sleeve);
    sl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir); sl.position.set(...g).addScaledVector(dir, len / 2 + 0.03);
    const cuff = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.05, 12), handMat.glove); cuff.quaternion.copy(sl.quaternion); cuff.position.set(...g).addScaledVector(dir, 0.05);
    vm.add(gl, sl, cuff);
  }
  vm.traverse((o) => { o.castShadow = false; });
}
const muzzle = new THREE.Mesh(new THREE.PlaneGeometry(0.24, 0.24), new THREE.MeshBasicMaterial({ color: 0xffc56b, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
muzzle.position.set(0, 0, -0.6); vm.add(muzzle);
let kick = 0, punch = 0, reloadAnim = 0, botKick = 0;

const tracers = [], puffs = [];
function tracer(x0, y0, z0, x1, y1, z1, color) {
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x0, z0, y0), new THREE.Vector3(x1, z1, y1)]);
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: .8 })); scene.add(l); tracers.push({ l, t: 0.06 });
}
function blood(x, y, z, head) { for (let i = 0; i < (head ? 9 : 5); i++) { const s = new THREE.Mesh(new THREE.SphereGeometry(0.03 + Math.random() * 0.04, 5, 5), new THREE.MeshBasicMaterial({ color: 0x7a0f0a, transparent: true, opacity: .9 }));
  s.position.set(x + (Math.random() - .5) * .25, z + (Math.random() - .5) * .25, y + (Math.random() - .5) * .25); scene.add(s); puffs.push({ s, t: 0.35 }); } }
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
function radio(freqs) { if (!ac) return; freqs.forEach((f, i) => { const o = ac.createOscillator(), g = ac.createGain(), t = ac.currentTime + i * .13; o.type = "square"; o.frequency.value = f;
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(.05, t + .01); g.gain.exponentialRampToValueAtTime(.001, t + .11); o.connect(g).connect(ac.destination); o.start(t); o.stop(t + .12); }); }
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
let m = null, state = "menu", acc = 0, last = performance.now(), freezeUntil = 0, roundNo = 0, bannerT = 0;
const score = { you: 0, shad0w: 0 }, stats = { shots: 0, hits: 0, heads: 0, decisions: 0, cert: 0, lastDecision: null, botShots: 0, botHits: 0, botHeads: 0, kills: 0, deaths: 0 };
const recent = [];
function newMatch() { score.you = score.shad0w = 0; roundNo = 0; Object.assign(stats, { shots: 0, hits: 0, heads: 0, botShots: 0, botHits: 0, botHeads: 0, kills: 0, deaths: 0, decisions: 0, cert: 0 }); $("sYou").textContent = 0; $("sShad").textContent = 0; $("feed").innerHTML = ""; nextRound(); }
function nextRound() {
  roundNo++;
  m = round((Date.now() + roundNo * 7919) >>> 0, roundNo % 2 ? "T" : "CT");
  yaw = m.human.yaw; pitch = 0; enemyDeadT = -1;
  for (const k in SOLDIER) SOLDIER[k].visible = false;
  enemy = SOLDIER[m.bot.side]; enemy.visible = true; enemy.rotation.set(0, 0, 0);
  dressHands(m.human.side);
  freezeUntil = m.t + 1.6; setTimeout(() => radio([700, 700, 940]), 900); showBanner(`Round ${roundNo} · you are ${m.human.side === "CT" ? "Counter-Terrorist" : "Terrorist"}`, 1.8);
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
  const ev = step(m, b, h, HARDEST);   // always the hardest hand: 150 ms reactions, fast turns
  if (!wasReloading && m.human.reloadT > 0) { reloadAnim = 2.5; clack(.25); }
  for (const e of ev) {
    if (e.kind === "step") { if (e.who === "bot") { const [p, g] = heard(e.x, e.y); footstep(p, g * .55); } else footstep(0, .05); continue; }
    if (e.kind !== "shot") continue;
    const f = e.who === "bot" ? m.bot : m.human;
    if (e.who === "human") { stats.shots++; gunshot(0, .32, true); kick = 1; punch = Math.min(punch + 0.012, 0.06); muzzle.material.opacity = 1; muzzle.rotation.z = Math.random() * 3;
      if (e.hit) { stats.hits++; stats.heads += e.head; hitMarker(e.head); blood(e.x, e.y, e.z, e.head); if (e.head) ding(2900, .18); } }
    else { botKick = 1; stats.botShots++; const [p, g] = heard(f.x, f.y); gunshot(p, .1 + .35 * g, false); flash.position.set(f.x, eye(f), f.y); flash.intensity = 7;
      if (e.hit) { stats.botHits++; stats.botHeads += e.head; damage(f.x, f.y); } }
    tracer(f.x + Math.cos(f.yaw) * .45, f.y + Math.sin(f.yaw) * .45, eye(f) - .14, e.x, e.y, e.z, e.who === "bot" ? 0xffcf7a : 0xfff2c4);
    if (!e.hit) puff(e.x, e.y, e.z);
  }
  if (m.over) endRound(ev);
}
function endRound(ev) {
  const w = m.winner, lastShot = ev.filter((e) => e.kind === "shot" && e.hit).pop();
  if (w === "human") { score.you++; stats.kills++; enemyDeadT = 0; feed("you", lastShot?.head); radio([660, 880]); } else if (w === "bot") { score.shad0w++; stats.deaths++; feed("shad0w", lastShot?.head); radio([520, 390]); }
  $("sYou").textContent = score.you; $("sShad").textContent = score.shad0w;
  const done = score.you >= 5 || score.shad0w >= 5;
  showBanner(w === "human" ? `${m.human.side === "CT" ? "Counter-Terrorists" : "Terrorists"} win · you` : w === "bot" ? "shad0w wins the round" : "Round draw", 2.2);
  setTimeout(() => { if (done) finish(); else nextRound(); }, 2400);
}
function finish() {
  state = "over"; document.exitPointerLock?.();
  const a = stats.shots ? Math.round(100 * stats.hits / stats.shots) : 0;
  $("overTitle").textContent = score.you > score.shad0w ? `You beat shad0w ${score.you}–${score.shad0w} 🏆` : `shad0w wins ${score.shad0w}–${score.you}`;
  const ba = stats.botShots ? Math.round(100 * stats.botHits / stats.botShots) : 0, hs = (h, k) => k ? Math.round(100 * h / k) : 0;
  $("overSub").innerHTML = `<table class="sb"><tr><th></th><th>rounds</th><th>accuracy</th><th>hits to the head</th><th>shots</th></tr>
    <tr><td><b>you</b></td><td>${score.you}</td><td>${a}%</td><td>${hs(stats.heads, stats.hits)}%</td><td>${stats.shots}</td></tr>
    <tr><td><b class="o">shad0w</b></td><td>${score.shad0w}</td><td>${ba}%</td><td>${hs(stats.botHeads, stats.botHits)}%</td><td>${stats.botShots}</td></tr></table>
    shad0w made <b>${stats.decisions.toLocaleString("en-US")}</b> decisions this match, every one in this tab, ${stats.decisions ? Math.round(100 * stats.cert / stats.decisions) : 0}% of them certified.`;
  $("start").textContent = "Play again →"; $("over").hidden = false;
}
function pause() { state = "paused"; $("overTitle").textContent = "Paused"; $("overSub").textContent = "Click to resume."; $("start").textContent = "Resume →"; $("over").hidden = false; }

// ---------------- HUD ----------------
let hitT = 0, dmgT = 0;
function hitMarker(head) { hitT = .16; $("hit").className = head ? "hit head" : "hit"; }
let dmgDirT = 0;
function damage(x, y) { dmgT = .4; dmgDirT = 1.1;   // which way it came from, relative to where you look (0 = straight ahead)
  const rel = Math.atan2(y - m.human.y, x - m.human.x) - yaw; $("dmgDir").style.setProperty("--a", `${rel * 180 / Math.PI + 90}deg`); }
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
  dmgDirT = Math.max(0, dmgDirT - dt); $("dmgDir").style.opacity = Math.min(1, dmgDirT * 1.5);
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

const NICE = { strafe_left: "strafe left", strafe_right: "strafe right", to_cover: "take cover", hold_fire: "hold fire" };
setInterval(() => {   // the brain panel: the words it reads right now and its three answers with their probabilities
  if (!m || state === "menu" && !DEBUG) return;
  const text = describe(m), r = bundle.decide(text, { probabilities: true }).answers;
  $("bText").textContent = text;
  $("bBars").innerHTML = ["move", "stance", "fire"].map((q) => { const a = r[q], top = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]).slice(0, 3);
    return `<div class="bq"><span class="bl">${q}</span>${top.map(([o, p], i) => `<span class="bo${i ? "" : " top"}"><i style="width:${Math.round(p * 100)}%"></i><em>${NICE[o] || o}</em><b>${Math.round(p * 100)}%</b></span>`).join("")}<span class="bc">${a.certified ? "certified" : "not certified"}</span></div>`; }).join("");
}, 125);

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
    const speed = Math.hypot(bt.vx, bt.vy); walkPhase += dt * speed * 2.6;   // the enemy: walk cycle, crouch, jump, recoil, death fall
    const E = enemy.userData, crouch = bt.crouch ? 1 : 0, air = airborne(bt) ? 1 : 0, swing = speed > 0.4 && !air ? Math.sin(walkPhase) * 0.55 * (1 - crouch * .5) : 0;
    enemy.position.set(bt.x, bt.z, bt.y); enemy.rotation.y = -bt.yaw;
    E.legs[0].rotation.z = swing - crouch * 1.0 - air * 0.5; E.legs[1].rotation.z = -swing - crouch * 1.0 - air * 0.2;
    E.legs[0].userData.knee.rotation.z = E.legs[1].userData.knee.rotation.z = crouch * 1.7 + air * 0.8;
    E.hips.position.y = -crouch * 0.4; E.torso.rotation.z = -crouch * 0.12 + Math.sin(walkPhase * 2) * 0.03 * (speed > 0.4 ? 1 : 0);
    E.rifle.position.x = 0.36 - botKick * 0.05; botKick = Math.max(0, botKick - dt * 10);
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
  window.__duel = { get m() { return m; }, get stats() { return stats; }, follow: true, finish };
  const fov = +new URLSearchParams(location.search).get("fov"); if (fov) { camera.fov = fov; camera.updateProjectionMatrix(); }
  setInterval(() => { if (m && window.__duel.follow) { yaw = Math.atan2(m.bot.y - m.human.y, m.bot.x - m.human.x); pitch = -0.05; } }, 16);
}
