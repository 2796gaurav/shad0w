// Doodle Duel's rasteriser, mirrored line for line from arena/doodle/raster.py (tested pixel for pixel against it).
// ---------------------------------------------------------------- the rasteriser (same as raster.py)
export const SIZE = 128, PAD = 10, R = 3, DISC = [];
for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (dx * dx + dy * dy <= R * R) DISC.push([dx, dy]);
export function fit(strokes) {
  const xs = strokes.flatMap((s) => s[0]), ys = strokes.flatMap((s) => s[1]);
  if (!xs.length) return [];
  const x0 = Math.min(...xs), y0 = Math.min(...ys), w = Math.max(...xs) - x0, h = Math.max(...ys) - y0;
  const k = (SIZE - 2 * PAD) / Math.max(w, h, 1), ox = (SIZE - w * k) / 2, oy = (SIZE - h * k) / 2;
  return strokes.map((s) => [s[0].map((x) => (x - x0) * k + ox), s[1].map((y) => (y - y0) * k + oy)]);
}
export function render(strokes) {
  const img = new Uint8Array(SIZE * SIZE).fill(255);
  for (const s of fit(strokes)) {
    let xs = s[0], ys = s[1];
    if (xs.length === 1) { xs = [xs[0], xs[0]]; ys = [ys[0], ys[0]]; }
    for (let i = 0; i < xs.length - 1; i++) {
      const ax = xs[i], ay = ys[i], bx = xs[i + 1], by = ys[i + 1];
      const n = Math.max(1, Math.ceil(Math.max(Math.abs(bx - ax), Math.abs(by - ay)) * 2));
      for (let j = 0; j <= n; j++) {
        const t = j / n, cx = Math.floor(ax + (bx - ax) * t + 0.5), cy = Math.floor(ay + (by - ay) * t + 0.5);
        for (const [dx, dy] of DISC) { const x = cx + dx, y = cy + dy; if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) img[y * SIZE + x] = 0; }
      }
    }
  }
  return img;
}
// your strokes -> Quick, Draw!'s "simplified" form: 0..255 box, integer points, Ramer-Douglas-Peucker at 2 px
function rdp(pts, eps) {
  if (pts.length < 3) return pts;
  const [a, b] = [pts[0], pts[pts.length - 1]], dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
  let im = 0, dm = 0;
  for (let i = 1; i < pts.length - 1; i++) { const d = Math.abs(dy * pts[i][0] - dx * pts[i][1] + b[0] * a[1] - b[1] * a[0]) / L; if (d > dm) { dm = d; im = i; } }
  return dm > eps ? [...rdp(pts.slice(0, im + 1), eps).slice(0, -1), ...rdp(pts.slice(im), eps)] : [a, b];
}
export function simplify(raw) {
  const all = raw.flat(); if (!all.length) return [];
  const x0 = Math.min(...all.map((p) => p[0])), y0 = Math.min(...all.map((p) => p[1]));
  const k = 255 / Math.max(Math.max(...all.map((p) => p[0])) - x0, Math.max(...all.map((p) => p[1])) - y0, 1);
  return raw.map((s) => { const p = rdp(s.map(([x, y]) => [Math.round((x - x0) * k), Math.round((y - y0) * k)]), 2); return [p.map((q) => q[0]), p.map((q) => q[1])]; });
}

