// shad0w Duel: the shared simulation (Node for training and evaluation, the browser for play). No rendering here.
//
// A 1v1 on "Dune", a small three-lane desert map (long, mid with doors, tunnels). One rifle each. Cells are 1 m.
// Heights: walls 3.5 m, crates 1.1 m (crouch behind one and you are hidden), boxes 1.6 m (only a jump lets you see over).
// 60 ticks a second. Every tick the bot answers three questions from words (describe()): move, stance, fire.
// Aiming is mechanical (an aim servo with a reaction delay and a turn-speed limit), like a human hand.

export const TICK = 60, DT = 1 / TICK;
export const H_WALL = 3.5, H_CRATE = 1.1, H_BOX = 1.6;
export const EYE_STAND = 1.6, EYE_CROUCH = 1.05, RADIUS = 0.3;
export const MOVES = ["push", "retreat", "strafe_left", "strafe_right", "hold", "to_cover"];
export const STANCES = ["stand", "crouch", "jump"];
export const FIRES = ["fire", "hold_fire", "reload"];
// Counter-Strike 1.6's AK-47 (Liquipedia): 36 damage, x4 to the head, x1.25 stomach, x0.75 legs; 600 rounds a minute; 30 a magazine
export const GUN = { dmg: 36, head: 4, chest: 1, stomach: 1.25, legs: 0.75, rpm: 600, mag: 30, reload: 2.5, range: 60 };
// CS 1.6 speeds with a rifle: about 215 units a second (4.1 m), walking 0.52x, crouching 0.34x
export const SPEED = { run: 4.1, walk: 2.1, crouch: 1.4 };

// ---------- the map ----------
export const MW = 44, MH = 30;
export function buildMap() {
  const h = new Float32Array(MW * MH).fill(H_WALL);
  const floor = (x0, y0, x1, y1) => { for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) h[y * MW + x] = 0; };
  const put = (x0, y0, x1, y1, v) => { for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) h[y * MW + x] = v; };
  // Dune: CT spawn between A (north-west) and B (north-east); long A down the west, tunnels down the east, mid in the centre.
  // Connectors dog-leg at different rows, so no sight line crosses a whole spawn.
  floor(17, 1, 26, 5);     // CT spawn
  floor(2, 1, 12, 8);      // A site
  floor(12, 2, 17, 3);     // CT to A
  floor(31, 1, 41, 8);     // B site
  floor(26, 4, 31, 5);     // CT to B
  floor(2, 8, 5, 24);      // long A
  floor(2, 23, 15, 25);    // outside long, to T spawn
  floor(15, 23, 28, 28);   // T spawn
  floor(38, 8, 41, 27);    // tunnels
  floor(28, 26, 41, 28);   // upper tunnels, to T spawn
  floor(19, 5, 24, 23);    // mid
  floor(10, 9, 19, 11);    // catwalk: mid to A short
  floor(10, 8, 12, 11);
  floor(24, 15, 38, 17);   // mid to B
  put(2, 17, 2, 18, H_WALL); put(5, 17, 5, 18, H_WALL);           // long doors (a 2 m gap)
  put(19, 12, 21, 12, H_WALL); put(22, 15, 24, 15, H_WALL);       // mid doors, staggered
  put(21, 19, 22, 20, H_BOX);                                    // tall boxes at the bottom of mid
  put(21, 7, 22, 7, H_CRATE);                                    // the crate at the top of mid
  put(4, 4, 5, 5, H_BOX); put(8, 6, 8, 6, H_CRATE); put(3, 12, 3, 12, H_CRATE);   // A site, long
  put(35, 3, 36, 4, H_CRATE); put(39, 6, 40, 6, H_BOX); put(33, 7, 33, 7, H_CRATE); // B site
  put(39, 20, 39, 20, H_CRATE); put(40, 12, 41, 12, H_CRATE);     // tunnels
  put(17, 24, 18, 24, H_CRATE); put(25, 24, 25, 25, H_BOX);       // T spawn (clear of the spawn points)
  put(18, 5, 18, 5, H_CRATE); put(26, 1, 26, 1, H_BOX);           // CT spawn (clear of the spawn points)
  return h;
}
export const MAP = buildMap();
// callouts, as CS players name the classic three-lane desert map's places (first match wins)
export const ZONES = [[21, 7, 22, 8, "Xbox"], [17, 1, 26, 5, "CT Spawn"], [12, 2, 17, 3, "CT Ramp"], [2, 1, 4, 3, "Goose"], [2, 1, 12, 8, "A Site"],
  [26, 4, 31, 5, "CT to B"], [38, 1, 41, 3, "B Platform"], [31, 1, 41, 8, "B Site"], [2, 16, 5, 19, "Long Doors"], [2, 8, 5, 22, "Long A"],
  [2, 23, 15, 25, "Outside Long"], [15, 23, 28, 28, "T Spawn"], [38, 8, 41, 17, "Lower Tunnels"], [38, 18, 41, 25, "Upper Tunnels"], [28, 26, 41, 28, "Outside Tunnels"],
  [19, 5, 24, 10, "Top Mid"], [19, 11, 24, 15, "Mid Doors"], [19, 16, 24, 23, "Lower Mid"], [10, 8, 19, 11, "Short"], [31, 15, 34, 17, "B Doors"], [24, 15, 38, 17, "Mid to B"]];
export function callout(x, y) { for (const [x0, y0, x1, y1, n] of ZONES) if (x >= x0 && x < x1 + 1 && y >= y0 && y < y1 + 1) return n; return ""; }
export const SPAWNS = { T: [[21.5, 27.2], [16.5, 27.5], [27.5, 27.5]], CT: [[21.5, 1.6], [18.5, 2.0], [25.5, 2.0]] };
export const hAt = (x, y) => (x < 0 || y < 0 || x >= MW || y >= MH) ? H_WALL : MAP[(y | 0) * MW + (x | 0)];

// ---------- randomness (deterministic per seed) ----------
export function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

// ---------- line of sight in 3D through the height grid ----------
export function clearLine(x0, y0, z0, x1, y1, z1) {
  const dx = x1 - x0, dy = y1 - y0, n = Math.ceil(Math.hypot(dx, dy) * 4) + 1;
  for (let i = 1; i < n; i++) {
    const t = i / n, x = x0 + dx * t, y = y0 + dy * t, z = z0 + (z1 - z0) * t;
    if (hAt(x, y) > z) return false;
  }
  return true;
}

// ---------- fighters ----------
export function fighter(side, x, y, yaw) {
  return { side, x, y, z: 0, vz: 0, yaw, pitch: 0, hp: 100, ammo: GUN.mag, reloadT: 0, cool: 0, crouch: false, recoil: 0,
           vx: 0, vy: 0, lastSeenT: -99, seenX: null, seenY: null, sawSince: -1, kills: 0, shots: 0, hits: 0, heard: -99, spray: 0, stepT: 0, walk: false };
}
export const eye = (f) => f.z + (f.crouch ? EYE_CROUCH : EYE_STAND);
export const headZ = (f) => eye(f) + 0.05;
export const moving = (f) => Math.hypot(f.vx, f.vy) > 0.6;
export const airborne = (f) => f.z > 0.02;
export function visible(a, b) {   // can a see b's head or chest?
  const ex = eye(a);
  return clearLine(a.x, a.y, ex, b.x, b.y, headZ(b)) || clearLine(a.x, a.y, ex, b.x, b.y, eye(b) - 0.45);
}
export function round(seed, botSide = "T") {
  const r = rng(seed), pick = (side) => SPAWNS[side][Math.floor(r() * 3)];
  const humanSide = botSide === "T" ? "CT" : "T";
  const [bx, by] = pick(botSide), [hx, hy] = pick(humanSide);
  const bot = fighter(botSide, bx, by, Math.atan2(hy - by, hx - bx)), human = fighter(humanSide, hx, hy, Math.atan2(by - hy, bx - hx));
  return { t: 0, r, bot, human, over: false, winner: null, log: [] };
}

// ---------- movement ----------
function blocked(x, y) {
  for (const [ox, oy] of [[RADIUS, 0], [-RADIUS, 0], [0, RADIUS], [0, -RADIUS], [RADIUS * .7, RADIUS * .7], [-RADIUS * .7, RADIUS * .7], [RADIUS * .7, -RADIUS * .7], [-RADIUS * .7, -RADIUS * .7]])
    if (hAt(x + ox, y + oy) > 0) return true;
  return false;
}
export function move(f, wishX, wishY, wantCrouch, wantJump, wantWalk = false) {
  // wishX/wishY: desired direction in the world (any length, normalised here)
  if (wantJump && !airborne(f) && !f.crouch) f.vz = 4.3;
  f.crouch = wantCrouch && !airborne(f);
  f.walk = wantWalk && !f.crouch;
  const speed = f.crouch ? SPEED.crouch : f.walk ? SPEED.walk : SPEED.run, l = Math.hypot(wishX, wishY), ctrl = airborne(f) ? 0.12 : 0.35;
  const tx = l > 1e-6 ? wishX / l * speed : 0, ty = l > 1e-6 ? wishY / l * speed : 0;
  f.vx += (tx - f.vx) * ctrl; f.vy += (ty - f.vy) * ctrl;
  const nx = f.x + f.vx * DT, ny = f.y + f.vy * DT;
  if (!blocked(nx, f.y)) f.x = nx; else f.vx = 0;
  if (!blocked(f.x, ny)) f.y = ny; else f.vy = 0;
  f.vz -= 12 * DT; f.z = Math.max(0, f.z + f.vz * DT); if (f.z === 0) f.vz = Math.max(0, f.vz);
}

// ---------- shooting ----------
export function spreadDeg(f) {   // CS 1.6: the first shot is accurate only when you stand still; running and jumping ruin it
  const v = Math.hypot(f.vx, f.vy);
  let s = f.crouch ? 0.15 : 0.25;
  if (v > 0.6) s += f.crouch ? 0.8 : v > SPEED.walk + 0.3 ? 3.6 : 1.1;
  if (airborne(f)) s += 8;
  return s + f.recoil * 0.6;
}
// the AK's spray: the first shots climb, then it sways left and right
export function sprayKick(n) { return [n < 1 ? 0 : Math.min(7, 0.9 * n) * (n > 9 ? 0.92 : 1), n > 6 ? Math.sin((n - 6) * 0.65) * 1.6 : 0]; }
// fire one round from f along (yaw, pitch) with spread; returns {hit, head, x, y, z} of the impact
export function shoot(f, target, rnd) {
  if (f.cool > 0 || f.reloadT > 0 || f.ammo <= 0) return null;
  f.ammo--; f.shots++; f.cool = 60 / GUN.rpm;
  const sp = spreadDeg(f) * Math.PI / 180, a = rnd() * 2 * Math.PI, m = Math.sqrt(rnd()) * sp, [up, side] = sprayKick(f.spray);
  const yaw = f.yaw + Math.cos(a) * m + side * Math.PI / 180, pitch = f.pitch + Math.sin(a) * m + up * Math.PI / 180 * 0.55;
  f.recoil = Math.min(4, f.recoil + 0.45); f.spray++;
  const dx = Math.cos(yaw) * Math.cos(pitch), dy = Math.sin(yaw) * Math.cos(pitch), dz = Math.sin(pitch);
  const ex = f.x, ey = f.y, ez = eye(f);
  // where does the ray meet a wall? march until the height grid is above the ray
  let wallT = GUN.range;
  for (let t = 0.1; t < GUN.range; t += 0.1) { if (hAt(ex + dx * t, ey + dy * t) > ez + dz * t || ez + dz * t < 0) { wallT = t; break; } }
  // the target: a head sphere and a body cylinder
  const hit = (cx, cy, cz, r) => { const ox = cx - ex, oy = cy - ey, oz = cz - ez, t = ox * dx + oy * dy + oz * dz; if (t < 0) return null;
    const px = ex + dx * t - cx, py = ey + dy * t - cy, pz = ez + dz * t - cz; return px * px + py * py + pz * pz <= r * r ? t : null; };
  const hz = headZ(target), th = hit(target.x, target.y, hz, 0.16);
  let tb = null;
  for (let k = 0; k < 6 && tb === null; k++) { const z = target.z + 0.15 + k * (eye(target) - 0.3 - target.z) / 5; tb = hit(target.x, target.y, z, 0.26); }
  let res = { hit: false, head: false, x: ex + dx * wallT, y: ey + dy * wallT, z: ez + dz * wallT };
  const best = th !== null && (tb === null || th <= tb) ? ["head", th] : tb !== null ? ["body", tb] : null;
  if (best && best[1] < wallT) {
    const head = best[0] === "head", hz = ez + dz * best[1] - target.z, h = eye(target) - target.z;
    const zone = head ? "head" : hz < 0.45 * h ? "legs" : hz < 0.65 * h ? "stomach" : "chest";
    const dmg = Math.round(GUN.dmg * GUN[zone]);
    target.hp -= dmg;
    f.hits++;
    res = { hit: true, head, zone, dmg, x: ex + dx * best[1], y: ey + dy * best[1], z: ez + dz * best[1] };
  }
  return res;
}

// ---------- navigation: distance fields over the floor ----------
const N = MW * MH, NB = [[1, 0], [-1, 0], [0, 1], [0, -1]];
export function field(tx, ty) {
  const d = new Int16Array(N).fill(-1), q = new Int32Array(N);
  let h = 0, t = 0; const s = (ty | 0) * MW + (tx | 0); if (MAP[s] > 0) return d;
  d[s] = 0; q[t++] = s;
  while (h < t) { const c = q[h++], cx = c % MW, cy = (c / MW) | 0;
    for (const [ox, oy] of NB) { const x = cx + ox, y = cy + oy; if (x < 0 || y < 0 || x >= MW || y >= MH) continue; const n = y * MW + x; if (d[n] >= 0 || MAP[n] > 0) continue; d[n] = d[c] + 1; q[t++] = n; } }
  return d;
}
export function downhill(fld, x, y) {   // the direction that lowers the field, or [0,0]
  const cx = x | 0, cy = y | 0, here = fld[cy * MW + cx];
  let best = null, bd = here < 0 ? 1e9 : here;
  for (const [ox, oy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const x2 = cx + ox, y2 = cy + oy; if (x2 < 0 || y2 < 0 || x2 >= MW || y2 >= MH) continue;
    const v = fld[y2 * MW + x2]; if (v < 0) continue;
    if (ox && oy && (MAP[cy * MW + x2] > 0 || MAP[y2 * MW + cx] > 0)) continue;
    if (v < bd) { bd = v; best = [x2 + 0.5 - x, y2 + 0.5 - y]; }
  }
  return best || [0, 0];
}
// the nearest floor cell (within 8 m) that hides you from a point (enemy eye), standing or crouched
export function coverCell(f, ex, ey, ez) {
  let best = null, bd = 1e9;
  for (let y = Math.max(0, (f.y | 0) - 8); y <= Math.min(MH - 1, (f.y | 0) + 8); y++)
    for (let x = Math.max(0, (f.x | 0) - 8); x <= Math.min(MW - 1, (f.x | 0) + 8); x++) {
      if (MAP[y * MW + x] > 0) continue;
      const d = Math.hypot(x + .5 - f.x, y + .5 - f.y); if (d >= bd) continue;
      if (!clearLine(ex, ey, ez, x + .5, y + .5, EYE_CROUCH)) { bd = d; best = [x + .5, y + .5]; }
    }
  return best ? { x: best[0], y: best[1], d: bd } : null;
}

// ---------- the aim servo (the "hand"): reaction delay + turn speed ----------
export const HARDEST = { react: 0.15, turn: 480, jitter: 0.3, aimZ: 0.3 };   // the website's bot: top-1% human reactions (150 ms), tracks fast
export const SKILL = { easy: { react: 0.42, turn: 140, jitter: 1.6 }, normal: { react: 0.30, turn: 220, jitter: 1.0 },
                       hard: { react: 0.22, turn: 320, jitter: 0.6 }, elite: { react: 0.16, turn: 420, jitter: 0.35 } };
export function aim(f, tx, ty, tz, seesNow, t, skill, rnd) {
  const want = Math.atan2(ty - f.y, tx - f.x), wantP = Math.atan2(tz - eye(f), Math.hypot(tx - f.x, ty - f.y));
  if (seesNow && f.sawSince < 0) f.sawSince = t;
  if (!seesNow) f.sawSince = -1;
  const ready = !seesNow || t - f.sawSince >= skill.react;
  if (!ready) return;
  const maxStep = skill.turn * Math.PI / 180 * DT, j = skill.jitter * Math.PI / 180;
  const dy = Math.atan2(Math.sin(want - f.yaw), Math.cos(want - f.yaw)) + (rnd() - .5) * j * 0.3;
  f.yaw += Math.max(-maxStep, Math.min(maxStep, dy));
  f.pitch += Math.max(-maxStep, Math.min(maxStep, wantP - f.pitch));
}
export const aimError = (f, tx, ty, tz) => {
  const want = Math.atan2(ty - f.y, tx - f.x), wantP = Math.atan2(tz - eye(f), Math.hypot(tx - f.x, ty - f.y));
  return Math.hypot(Math.atan2(Math.sin(want - f.yaw), Math.cos(want - f.yaw)), wantP - f.pitch) * 180 / Math.PI;
};

// ---------- the bot's view of the world, in words (the table and the teacher read the same words) ----------
const dBin = (d) => d < 4 ? "very close" : d < 10 ? "close" : d < 20 ? "mid range" : "far";
const hpBin = (h) => h <= 30 ? "critical" : h <= 60 ? "hurt" : h < 100 ? "scratched" : "full";
export function describe(m, exact = false) {
  const b = m.bot, e = m.human, sees = visible(b, e), d = Math.hypot(e.x - b.x, e.y - b.y);
  const parts = [];
  if (sees) {
    const err = aimError(b, e.x, e.y, eye(e) - 0.3);
    const behind = !clearLine(b.x, b.y, eye(b), e.x, e.y, eye(e) - 0.8) ? "behind low cover" : "in the open";
    parts.push(`Enemy visible, ${dBin(d)}, ${behind}, ${e.crouch ? "crouching" : airborne(e) ? "jumping" : "standing"}${moving(e) ? " and moving" : ""}.`);
    parts.push(`Your aim: ${err < 1.5 ? "on target" : err < 5 ? "almost on target" : err < 20 ? "off target" : "far off"}.`);
    if (exact) parts.push(`(distance ${d.toFixed(1)} m, aim error ${err.toFixed(1)} degrees)`);
  } else {
    const ago = m.t - b.lastSeenT;
    parts.push(b.seenX === null ? "Enemy not seen yet." : `Enemy not visible, last seen ${ago < 1.5 ? "just now" : ago < 5 ? "a few seconds ago" : "a while ago"}, ${dBin(Math.hypot(b.seenX - b.x, b.seenY - b.y))} from you.`);
    if (m.t - b.heard < 1) parts.push("You just heard shots nearby.");
  }
  const myCover = !clearLine(e.x, e.y, eye(e), b.x, b.y, EYE_CROUCH);
  parts.push(`You: ${b.crouch ? "crouching" : airborne(b) ? "in the air" : "standing"}, ${moving(b) ? "moving" : "still"}, ${myCover ? "covered when crouched" : "exposed"}.`);
  parts.push(`Health ${hpBin(b.hp)}. Ammo ${b.reloadT > 0 ? "reloading" : b.ammo === 0 ? "empty" : b.ammo <= 8 ? "low" : b.ammo <= 20 ? "half" : "full"}.`);
  if (exact) parts.push(`(health ${b.hp}, ammo ${b.ammo}/${GUN.mag})`);
  return parts.join(" ");
}

// "checking the corner": follow the path towards the goal and aim at the last point of it still in view, which is where the
// enemy would step out. Players call it pre-aiming; it uses only the map and the goal the bot already walks to.
const pathField = new Map();
function checkPoint(m, f, goal) {
  const key = (goal[0] | 0) + "," + (goal[1] | 0); let fld = pathField.get(key);
  if (!fld) { fld = field(goal[0], goal[1]); if (pathField.size > 64) pathField.clear(); pathField.set(key, fld); }
  let x = f.x, y = f.y, last = null;
  if (fld[(f.y | 0) * MW + (f.x | 0)] > 10) return null;   // only within ~10 m of walking: further out, aiming along the path measured better
  for (let k = 0; k < 30; k++) {
    const d = downhill(fld, x, y); if (!d[0] && !d[1]) break;
    x = (x | 0) + 0.5 + Math.sign(Math.round(d[0] * 2)) * 1; y = (y | 0) + 0.5 + Math.sign(Math.round(d[1] * 2)) * 1;
    if (!clearLine(f.x, f.y, eye(f), x, y, EYE_STAND)) break;
    last = [x, y];
  }
  return last && Math.hypot(last[0] - f.x, last[1] - f.y) > 2.5 ? last : null;
}

// ---------- one tick ----------
// botAct: {move, stance, fire}; humanAct: {wishX, wishY, crouch, jump, fire, reload, yaw, pitch} (yaw/pitch absolute) or an AI function
export function step(m, botAct, humanAct, skill = SKILL.normal) {
  if (m.over) return [];
  const b = m.bot, e = m.human, events = [], t = m.t;
  // the bot's movement intent, in the world frame
  const toE = [e.x - b.x, e.y - b.y], tl = Math.hypot(...toE) || 1, fwd = [toE[0] / tl, toE[1] / tl];
  const sees = visible(b, e);
  if (sees) { b.lastSeenT = t; b.seenX = e.x; b.seenY = e.y; }
  let wish = [0, 0];
  const goal = sees || b.seenX !== null ? [sees ? e.x : b.seenX, sees ? e.y : b.seenY] : [e.x, e.y];   // never seen: hunt by map knowledge
  if (botAct.move === "push") { m._f = m._fk === goal.map((v) => v | 0).join() ? m._f : field(goal[0], goal[1]); m._fk = goal.map((v) => v | 0).join(); wish = downhill(m._f, b.x, b.y); }
  else if (botAct.move === "retreat") { wish = [-fwd[0], -fwd[1]]; }
  else if (botAct.move === "strafe_left") { wish = [-fwd[1], fwd[0]]; }   // into a wall this means standing still, which is when a rifle is accurate:
  else if (botAct.move === "strafe_right") { wish = [fwd[1], -fwd[0]]; }   // stepping the other way instead was measured and lost rounds (32% vs 61%)
  else if (botAct.move === "to_cover") { const c = coverCell(b, e.x, e.y, eye(e)); if (c) { if (!m._cf || m._ck !== `${c.x},${c.y}`) { m._cf = field(c.x, c.y); m._ck = `${c.x},${c.y}`; } wish = downhill(m._cf, b.x, b.y); } }
  move(b, wish[0], wish[1], botAct.stance === "crouch", botAct.stance === "jump");
  // aim: at the enemy's chest when seen, else along the path
  if (sees) aim(b, e.x, e.y, eye(e) - (skill.aimZ ?? 0.3), true, t, skill, m.r);
  else { b.sawSince = -1; const c = checkPoint(m, b, goal); if (c) aim(b, c[0], c[1], eye(b), false, t, skill, m.r); else if (Math.hypot(wish[0], wish[1]) > 0.1) aim(b, b.x + wish[0], b.y + wish[1], eye(b), false, t, skill, m.r); }
  // the human (or the scripted opponent)
  const h = typeof humanAct === "function" ? humanAct(m, e, b) : humanAct;
  move(e, h.wishX, h.wishY, h.crouch, h.jump, h.walk);
  if (h.yaw !== undefined) { e.yaw = h.yaw; e.pitch = h.pitch; }
  // guns
  for (const [f, act, other] of [[b, botAct.fire, e], [e, h.fire ? "fire" : h.reload ? "reload" : "hold_fire", b]]) {
    f.cool = Math.max(0, f.cool - DT);
    f.recoil = Math.max(0, f.recoil - (f.cool > 0 ? 0.6 : 3.2) * DT);
    if (act !== "fire" && f.cool === 0) f.spray = Math.max(0, f.spray - 30 * DT);   // the spray resets when you stop
    if (moving(f) && !airborne(f) && !f.crouch && !f.walk) { f.stepT -= DT; if (f.stepT <= 0) { f.stepT = 0.36; events.push({ kind: "step", who: f === b ? "bot" : "human", x: f.x, y: f.y }); } }
    if (f.reloadT > 0) { f.reloadT -= DT; if (f.reloadT <= 0) { f.reloadT = 0; f.ammo = GUN.mag; } }
    if (act === "reload" && f.reloadT === 0 && f.ammo < GUN.mag) { f.reloadT = GUN.reload; events.push({ kind: "reload", who: f === b ? "bot" : "human" }); }
    if (act === "fire" && f.ammo === 0 && f.reloadT === 0) f.reloadT = GUN.reload;
    if (act === "fire") { const s = shoot(f, other, m.r); if (s) { events.push({ kind: "shot", who: f === b ? "bot" : "human", ...s }); other.heard = t; } }
  }
  m.t += DT;
  if (b.hp <= 0 || e.hp <= 0) { m.over = true; m.winner = e.hp <= 0 && b.hp > 0 ? "bot" : b.hp <= 0 && e.hp > 0 ? "human" : "draw"; }
  if (m.t > 75) { m.over = true; m.winner = "draw"; }
  return events;
}

// ---------- scripted opponent (for training data and evaluation; the website's opponent is you) ----------
export function scripted(style = 0.5, skill = SKILL.normal) {
  let target = null, peekT = 0;
  return (m, me, foe) => {
    const sees = visible(me, foe), t = m.t;
    if (sees) { me.lastSeenT = t; me.seenX = foe.x; me.seenY = foe.y; }
    let wish = [0, 0], crouch = false, jump = false, fire = false, reload = false;
    if (sees) {
      aim(me, foe.x, foe.y, eye(foe) - (skill.aimZ ?? 0.3), true, t, skill, m.r);
      const err = aimError(me, foe.x, foe.y, eye(foe) - (skill.aimZ ?? 0.3));
      fire = err < 2.2 && me.ammo > 0;
      crouch = Math.hypot(foe.x - me.x, foe.y - me.y) > 12 && m.r() < style;
      if (m.r() < 0.03) peekT = t + 0.6;
      if (t < peekT) { const d = [foe.x - me.x, foe.y - me.y]; wish = m.r() < .5 ? [-d[1], d[0]] : [d[1], -d[0]]; }
    } else {
      me.sawSince = -1;
      if (!target || Math.hypot(target[0] - me.x, target[1] - me.y) < 1.2 || m.r() < 0.002) {
        const g = me.seenX !== null && m.r() < 0.6 ? [me.seenX, me.seenY] : [foe.x + (m.r() - .5) * 10, foe.y + (m.r() - .5) * 10];
        target = MAP[(Math.max(0, Math.min(MH - 1, g[1] | 0))) * MW + Math.max(0, Math.min(MW - 1, g[0] | 0))] > 0 ? [foe.x, foe.y] : g;
        me._f = field(target[0], target[1]);
      }
      wish = downhill(me._f, me.x, me.y);
      aim(me, me.x + wish[0], me.y + wish[1], eye(me), false, t, skill, m.r);
      reload = me.ammo < 12;
      jump = m.r() < 0.004;
    }
    return { wishX: wish[0], wishY: wish[1], crouch, jump, fire, reload };
  };
}

// a hand-written bot policy (a baseline, and the explorer that finds situations for the teacher)
export function ruleBot(m) {
  const b = m.bot, e = m.human, sees = visible(b, e), d = Math.hypot(e.x - b.x, e.y - b.y);
  if (b.reloadT > 0) return { move: "to_cover", stance: "crouch", fire: "hold_fire" };
  if (!sees) return { move: "push", stance: "stand", fire: b.ammo < 15 ? "reload" : "hold_fire" };
  const err = aimError(b, e.x, e.y, eye(e) - 0.3);
  if (b.hp <= 30) return { move: "to_cover", stance: "crouch", fire: err < 2 ? "fire" : "hold_fire" };
  return { move: d > 15 ? "hold" : "strafe_left", stance: d > 12 ? "crouch" : "stand", fire: err < 2 ? "fire" : "hold_fire" };
}
