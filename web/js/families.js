// Closed-form packings of n unit squares in a square (JavaScript port of squarepack/constructions.py).
// A packing is {n, s, squares: Float64Array [x0,y0,t0, x1,y1,t1, ...], method, exact}; the origin is the
// bottom-left corner of the container [0, s]^2, angles are radians counter-clockwise.
export const SQRT2 = Math.SQRT2;
export const EPS = 1e-9;
const Q = Math.PI / 4;

export function isqrt(n) { let k = Math.floor(Math.sqrt(n)); while (k * k > n) k--; while ((k + 1) * (k + 1) <= n) k++; return k; }
export function gridSide(n) { const k = isqrt(n); return k * k < n ? k + 1 : k; }

export function grid(n) {
  const k = gridSide(n);
  const sq = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) { sq[3 * i] = (i % k) + 0.5; sq[3 * i + 1] = Math.floor(i / k) + 0.5; sq[3 * i + 2] = 0; }
  return { n, s: k, squares: sq, method: 'grid', exact: String(k) };
}

// ---- Göbel strip: n = a(a+1) + 2 + b, s = a + 1 + sqrt2/2, b = 1 + floor((a-1) sqrt2) ----
export function gobelStripB(a) { return 1 + Math.floor((a - 1) * SQRT2 + EPS); }
export function gobelStripCapacity(a) { return a * (a + 1) + 2 + gobelStripB(a); }
export function gobelStripMember(a) {
  const k = a + 1, b = gobelStripB(a), s = k + SQRT2 / 2;
  const rows = [];
  for (let i = 0; i < a; i++) for (let j = 0; j < a - i; j++) { rows.push(i + 0.5, s - j - 0.5, 0); rows.push(s - i - 0.5, j + 0.5, 0); }
  rows.push(0.5, 0.5, 0, s - 0.5, s - 0.5, 0);
  const free = s * SQRT2 - 2 * SQRT2, u0 = SQRT2 + (free - b) / 2 + 0.5;
  for (let m = 0; m < b; m++) { const u = u0 + m; rows.push(u / SQRT2, u / SQRT2, Q); }
  return { n: rows.length / 3, s, squares: Float64Array.from(rows), method: 'gobel_strip', exact: `${k} + sqrt(2)/2` };
}

// ---- Göbel square: n = 2a(a+1) + b^2, s = a + 1 + b sqrt2/2, valid iff a-1 < b/sqrt2 < a+1 ----
export function gobelSquareValid(a, b) { return a >= 1 && b >= 1 && (a - 1) < b / SQRT2 - EPS && b / SQRT2 < (a + 1) - EPS; }
export function gobelSquareCapacity(a, b) { return 2 * a * (a + 1) + b * b; }
export function gobelSquareMember(a, b) {
  const s = a + 1 + b * SQRT2 / 2, c = s / 2, h = 1 / SQRT2;
  const rows = [];
  for (let i = 0; i < a; i++) for (let j = 0; j < a - i; j++) {
    rows.push(i + 0.5, j + 0.5, 0, s - i - 0.5, j + 0.5, 0, i + 0.5, s - j - 0.5, 0, s - i - 0.5, s - j - 0.5, 0);
  }
  const tiles = [];
  for (let i = 0; i < b; i++) for (let j = 0; j < b; j++) { const u = i - (b - 1) / 2, v = j - (b - 1) / 2; tiles.push([Math.abs(u) + Math.abs(v), u, v]); }
  tiles.sort((p, q) => (p[0] - q[0]) || (p[1] - q[1]) || (p[2] - q[2]));
  for (const [, u, v] of tiles) rows.push(c + (u + v) * h, c + (u - v) * h, Q);
  return { n: rows.length / 3, s, squares: Float64Array.from(rows), method: 'gobel_square', exact: `${a + 1} + ${b}*sqrt(2)/2` };
}

// ---- portfolio: members described arithmetically, only the winner is built ----
export function memberDescriptors(n) {
  const sGrid = gridSide(n), out = [];
  for (let a = 1; a + 1 + SQRT2 / 2 < sGrid - EPS; a++) out.push({ s: a + 1 + SQRT2 / 2, n: gobelStripCapacity(a), family: 'gobel_strip', build: () => gobelStripMember(a) });
  for (let a = 1; a + 1 + SQRT2 / 2 < sGrid - EPS; a++) {
    for (let b = Math.max(1, Math.floor((a - 1) * SQRT2) + 1); b / SQRT2 < a + 1 - EPS; b++) {
      const s = a + 1 + b * SQRT2 / 2;
      if (gobelSquareValid(a, b) && s < sGrid - EPS) out.push({ s, n: gobelSquareCapacity(a, b), family: 'gobel_square', build: () => gobelSquareMember(a, b) });
    }
  }
  return out;
}

/** number of "L" extensions (row on top + column on the right, side + 1 each) taking (s0, n0) to capacity n below sMax */
export function lChain(s0, n0, n, sMax) {
  let s = s0, m = n0, j = 0;
  while (m < n) {
    if (s + 1 >= sMax - EPS) return null;
    m += Math.floor(s + 1 + EPS) + Math.floor(s + EPS); s += 1; j += 1;
  }
  return [j, s];
}

export function addLs(p, j) {
  if (j <= 0) return p;
  const extra = [];
  for (let i = 0; i < j; i++) {
    const si = p.s + i, top = Math.floor(si + 1 + EPS), right = Math.floor(si + EPS);
    for (let t = 0; t < top; t++) extra.push(t + 0.5, si + 0.5, 0);
    for (let r = 0; r < right; r++) extra.push(si + 0.5, r + 0.5, 0);
  }
  const sq = new Float64Array(p.squares.length + extra.length);
  sq.set(p.squares); sq.set(extra, p.squares.length);
  return { n: sq.length / 3, s: p.s + j, squares: sq, method: p.method + (j === 1 ? '+L' : `+${j}L`), exact: `(${p.exact}) + ${j}` };
}

export function take(p, n) {
  if (n > p.n) throw new Error(`cannot take ${n} of ${p.n}`);
  return { ...p, n, squares: p.squares.slice(0, 3 * n) };
}

/** best closed-form packing for n (grid, Göbel strips/squares with L extensions); O(n) */
export function bestAnalytic(n) {
  const sGrid = gridSide(n);
  let best = null;
  for (const d of memberDescriptors(n)) {
    const r = lChain(d.s, d.n, n, sGrid);
    if (r && (best === null || r[1] < best.s - 1e-12)) best = { s: r[1], j: r[0], d };
  }
  if (best === null) return grid(n);
  return take(addLs(best.d.build(), best.j), n);
}

export function canonicalAngle(t) {
  let out = t - (Math.PI / 2) * Math.floor((t + Q) / (Math.PI / 2));
  if (out <= -Q + 1e-12) out += Math.PI / 2;
  return out;
}
