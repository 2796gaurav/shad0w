// Snake, deterministic and seedable: the same engine drives the recorded runs (Node) and the replay on the website.
export const DIRS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
export const OPTIONS = ["up", "down", "left", "right"];
const OPP = { up: "down", down: "up", left: "right", right: "left" };

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

export function newGame(seed, size = 16) {
  const mid = size >> 1;
  const g = { size, seed, rand: rng(seed), snake: [[mid, mid], [mid, mid + 1], [mid, mid + 2]], dir: "up",
              food: null, tick: 0, score: 0, alive: true, cause: null };
  g.food = spawn(g);
  return g;
}

function spawn(g) {
  const busy = new Set(g.snake.map(([x, y]) => x + "," + y));
  for (;;) {
    const f = [Math.floor(g.rand() * g.size), Math.floor(g.rand() * g.size)];
    if (!busy.has(f[0] + "," + f[1])) return f;
  }
}

/** What moving one step in `d` would hit: "wall", "body" or null. The tail cell is free (it moves away). */
export function blocked(g, d) {
  const [dx, dy] = DIRS[d], [hx, hy] = g.snake[0], nx = hx + dx, ny = hy + dy;
  if (nx < 0 || ny < 0 || nx >= g.size || ny >= g.size) return "wall";
  for (let i = 0; i < g.snake.length - 1; i++) if (g.snake[i][0] === nx && g.snake[i][1] === ny) return "body";
  return null;
}

const LEFT = { up: "left", left: "down", down: "right", right: "up" };
const RIGHT = { up: "right", right: "down", down: "left", left: "up" };
/** The absolute direction a relative move ("left" | "straight" | "right") takes from the current heading. */
export const turn = (dir, move) => (move === "left" ? LEFT[dir] : move === "right" ? RIGHT[dir] : dir);

/** One tick. `move` (optional) is relative to the heading: "left", "straight" or "right". */
export function step(g, move) {
  if (!g.alive) return g;
  if (move === "left" || move === "right") g.dir = turn(g.dir, move);
  const hit = blocked(g, g.dir);
  g.tick++;
  if (hit) { g.alive = false; g.cause = hit; return g; }
  const [dx, dy] = DIRS[g.dir], [hx, hy] = g.snake[0], head = [hx + dx, hy + dy];
  g.snake.unshift(head);
  if (head[0] === g.food[0] && head[1] === g.food[1]) {
    g.score++;
    if (g.snake.length >= g.size * g.size) { g.alive = false; g.cause = "won"; return g; }
    g.food = spawn(g);
  } else g.snake.pop();
  return g;
}

export const MOVES = ["left", "straight", "right"];

/** The situation in plain words, from the snake's point of view: what every model (and shad0w) decides on. */
export function describe(g) {
  const [hx, hy] = g.snake[0], [fx, fy] = g.food, [ax, ay] = DIRS[g.dir], [rx, ry] = DIRS[RIGHT[g.dir]];
  const ahead = (fx - hx) * ax + (fy - hy) * ay, right = (fx - hx) * rx + (fy - hy) * ry;
  const food = [ahead > 0 ? `${ahead} ahead` : ahead < 0 ? `${-ahead} behind` : null,
                right > 0 ? `${right} to the right` : right < 0 ? `${-right} to the left` : null].filter(Boolean).join(" and ") || "here";
  const what = (m) => { const b = blocked(g, turn(g.dir, m)); return b ? (b === "wall" ? "a wall" : "your body") : "free"; };
  return `Food is ${food}. Straight is ${what("straight")}, left is ${what("left")}, right is ${what("right")}. Length ${g.snake.length}.`;
}

export const QUESTION = {
  name: "move",
  instructions: "You steer a snake on a grid. Pick the next move: never into a wall or the body, and towards the food.",
  criteria: { left: "turn left", straight: "keep going straight", right: "turn right" },
};

/** A move judged on the board: is it safe, and does it get closer to the food? (for scoring the models' answers) */
export function judge(g, move) {
  const d = turn(g.dir, move);
  if (blocked(g, d)) return { safe: false, closer: false };
  const [hx, hy] = g.snake[0], [dx, dy] = DIRS[d], [fx, fy] = g.food;
  return { safe: true, closer: Math.abs(fx - hx - dx) + Math.abs(fy - hy - dy) < Math.abs(fx - hx) + Math.abs(fy - hy) };
}
