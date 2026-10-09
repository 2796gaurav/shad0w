// Pong, deterministic per seed, fixed 60 Hz physics. The same engine labels the training situations (Node),
// evaluates each "brain speed" offline, and runs the playable game on the website.
export const W = 800, H = 500, PW = 12, MARGIN = 26, FPS = 60, WALL = 6;
export const ACTIONS = ["up", "stay", "down"];

export function rng(seed) {  // mulberry32
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** opts: balls (1-6), speed (px per frame at serve), paddle (height px), paddleSpeed (px per frame) */
export function newMatch(seed = 1, opts = {}) {
  const o = { balls: 1, speed: 7, paddle: 90, paddleSpeed: 9, ...opts };
  const m = { o, rand: rng(seed), frame: 0, left: { y: H / 2, score: 0 }, right: { y: H / 2, score: 0 }, balls: [], hits: { left: 0, right: 0 } };
  for (let i = 0; i < o.balls; i++) m.balls.push(serve(m, i % 2 ? -1 : 1, i * 25));
  return m;
}

function serve(m, dir, wait = 40) {
  const a = (m.rand() - 0.5) * 0.9;  // up to ~25 degrees
  return { x: W / 2, y: H * (0.25 + 0.5 * m.rand()), vx: dir * m.o.speed * Math.cos(a), vy: m.o.speed * Math.sin(a), wait };
}

/** Where a ball will cross x = xTarget (bouncing off top and bottom), and in how many frames; null if moving away. */
export function intercept(b, xTarget) {
  if (b.vx === 0 || Math.sign(xTarget - b.x) !== Math.sign(b.vx)) return null;
  const t = (xTarget - b.x) / b.vx;
  // the ball bounces at y = WALL and y = H - WALL (see tick), so reflect inside that band
  const lo = WALL, band = H - 2 * WALL, span = 2 * band;
  let y = (((b.y + b.vy * t - lo) % span) + span) % span;
  if (y > band) y = span - y;
  y += lo;
  return { y, frames: Math.max(0, Math.round(t + (b.wait || 0))) };
}

/** The most urgent ball for the paddle on `side`: the one that reaches it first. */
export function urgent(m, side) {
  const x = side === "right" ? W - MARGIN - PW : MARGIN + PW;
  let best = null;
  for (const b of m.balls) {
    const i = intercept(b, x);
    if (i && (!best || i.frames < best.frames)) best = i;
  }
  return best;
}

const r1 = (v) => Math.round(v * 10) / 10;
/** The situation in plain words for the paddle on `side`: what the LLM, and shad0w, decide on. */
export function describe(m, side = "right") {
  const p = m[side], u = urgent(m, side), ph = m.o.paddle;
  if (!u) {
    const off = r1((p.y - H / 2) / ph);
    return `No ball is coming at you. Your paddle is ${off === 0 ? "at the middle" : `${Math.abs(off)} paddle-heights ${off < 0 ? "above" : "below"} the middle`}.`;
  }
  const off = r1((u.y - p.y) / ph);
  const where = off === 0 ? "level with your paddle's centre" : `${Math.abs(off)} paddle-heights ${off < 0 ? "above" : "below"} your paddle's centre`;
  return `A ball is coming at you and reaches your side in ${u.frames} frames, ${where}.`;
}

export const QUESTION = {
  name: "paddle",
  instructions: "You move a Pong paddle up or down to meet the ball. Pick the move for the next frame: up, stay or down.",
  criteria: { up: "move the paddle up", stay: "keep the paddle still", down: "move the paddle down" },
};

/** One 60 Hz frame. actions: { left: "up"|"stay"|"down", right: ... } */
export function tick(m, actions) {
  m.frame++;
  for (const side of ["left", "right"]) {
    const a = actions[side], p = m[side];
    if (a === "up") p.y -= m.o.paddleSpeed;
    else if (a === "down") p.y += m.o.paddleSpeed;
    else if (typeof a === "number") p.y += Math.max(-m.o.paddleSpeed * 2, Math.min(m.o.paddleSpeed * 2, a - p.y));  // a pointer target
    p.y = Math.max(m.o.paddle / 2, Math.min(H - m.o.paddle / 2, p.y));
  }
  const events = [];
  for (let i = 0; i < m.balls.length; i++) {
    const b = m.balls[i];
    if (b.wait > 0) { b.wait--; continue; }
    b.x += b.vx; b.y += b.vy;
    if (b.y < WALL) { b.y = 2 * WALL - b.y; b.vy = -b.vy; }
    if (b.y > H - WALL) { b.y = 2 * (H - WALL) - b.y; b.vy = -b.vy; }
    for (const side of ["left", "right"]) {
      const face = side === "left" ? MARGIN + PW : W - MARGIN - PW, p = m[side];
      const crossing = side === "left" ? b.vx < 0 && b.x <= face && b.x - b.vx > face : b.vx > 0 && b.x >= face && b.x - b.vx < face;
      if (crossing && Math.abs(b.y - p.y) <= m.o.paddle / 2 + 6) {
        const rel = (b.y - p.y) / (m.o.paddle / 2), sp = Math.min(Math.hypot(b.vx, b.vy) * 1.04, m.o.speed * 2.2);
        const ang = rel * 0.85;
        b.vx = (side === "left" ? 1 : -1) * sp * Math.cos(ang); b.vy = sp * Math.sin(ang);
        b.x = face + (side === "left" ? 1 : -1);
        m.hits[side]++; events.push({ type: "hit", side, ball: i });
      }
    }
    if (b.x < -10 || b.x > W + 10) {
      const scorer = b.x < 0 ? "right" : "left";
      m[scorer].score++; events.push({ type: "point", side: scorer, ball: i });
      m.balls[i] = serve(m, scorer === "left" ? -1 : 1);
    }
  }
  return events;
}

/** A sensible paddle for the other side: follows the most urgent ball with a little reaction delay and error. */
export function trainer(m, side, skill = 0.9) {
  const u = urgent(m, side), p = m[side];
  const target = u ? u.y + (1 - skill) * (m.rand() - 0.5) * m.o.paddle * 2 : H / 2;
  const d = target - p.y;
  return Math.abs(d) < m.o.paddleSpeed * 0.6 ? "stay" : d < 0 ? "up" : "down";
}
