/**
 * packcore.js -- pure-JavaScript port of the squarepack numerical core.
 *
 * Tightest packings of n unit squares in a square: separating-axis geometry
 * (squarepack/geometry.py), the penalty energy / analytic gradient and the
 * L-BFGS minimiser (squarepack/_fastcore.c) and the "classic" bisection
 * compaction with basin hopping (squarepack/optimize.py).
 *
 * Conventions (identical to the Python package)
 * ---------------------------------------------
 * * A square is the triple (x, y, theta): centre and counter-clockwise rotation
 *   in radians; every small square has side 1.
 * * The container is [0, s] x [0, s]; s is minimised.
 * * Flat packings are Float64Array [x0, y0, t0, x1, y1, t1, ...]; every public
 *   entry point also accepts an array of [x, y, t] rows.
 * * Optimiser state z = [x_0..x_{n-1}, y_0..y_{n-1}, theta_0..theta_{n-1}]
 *   (the layout of the C core).
 * * Penetration depth of two unit squares with relative angle phi and centre
 *   offset d:  p = 1/2 + (|cos phi| + |sin phi|)/2 - max_a |a . d|  over the
 *   four edge directions a of both squares; p > 0 overlap, p <= 0 separated.
 * * Penalty energy at fixed s:
 *       E(z; s) = sum_{i<j} max(0, p_ij)^2 + sum_i sum_{4 sides} max(0, v_i)^2
 *   with v_i the protrusion of square i beyond a container side.
 *
 * The numerics (energy, gradient formulas, L-BFGS structure, pair-list
 * rebuild rule, perturbation moves, compaction schedule) follow the reference
 * implementation line by line; every deliberate deviation is flagged with a
 * "DEVIATION" comment.  The module has no dependencies and runs in browsers
 * (main thread or Web Worker) and in Node.
 */

export const SQRT2 = Math.SQRT2;
export const PI = Math.PI;
export const HALF_PI = Math.PI / 2;
export const QUARTER_PI = Math.PI / 4;

/** Pair-list cutoff used during optimisation (squares may travel ~0.6 in one run). */
export const PAIR_CUTOFF = 2.6;
/** Pair-list cutoff for exact tests: two unit squares overlap only if |dx|, |dy| < sqrt 2. */
export const EXACT_CUTOFF = SQRT2 + 1e-9;
/** L-BFGS iterations per local optimisation (kept short so a worker stays responsive). */
export const LOCAL_MAXITER = 250;

/** Monotonic clock in milliseconds. */
export const now = (typeof performance !== 'undefined' && typeof performance.now === 'function')
  ? () => performance.now()
  : () => Date.now();

// --------------------------------------------------------------------------- //
// basic helpers
// --------------------------------------------------------------------------- //
function sgn(v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

/** (|cos phi| + |sin phi|) / 2: half-width of a unit square projected on an axis at angle phi. */
export function supportHalf(phi) {
  return 0.5 * (Math.abs(Math.cos(phi)) + Math.abs(Math.sin(phi)));
}

/** Derivative of supportHalf (a sub-gradient at the kinks). */
export function dSupportHalf(phi) {
  const c = Math.cos(phi), s = Math.sin(phi);
  return 0.5 * (-sgn(c) * s + sgn(s) * c);
}

/** Reduce an angle to (-pi/4, pi/4] using the 4-fold symmetry of a square (45 degrees -> +pi/4). */
export function canonicalAngle(theta) {
  let out = theta - HALF_PI * Math.floor((theta + QUARTER_PI) / HALF_PI);
  if (out <= -QUARTER_PI + 1e-12) out += HALF_PI;
  return out;
}

/**
 * Normalise a packing to a flat Float64Array [x0, y0, t0, ...].  Accepts a flat
 * typed array / plain array or an array of [x, y, t] rows.  A Float64Array of
 * the right shape is returned as is (no copy).
 */
export function toFlat(squares) {
  if (squares == null) return new Float64Array(0);
  if (squares instanceof Float64Array) {
    if (squares.length % 3 !== 0) throw new TypeError('flat packing length must be a multiple of 3');
    return squares;
  }
  if (!Array.isArray(squares) && !ArrayBuffer.isView(squares)) {
    throw new TypeError('squares must be a flat array or an array of [x, y, theta] rows');
  }
  if (squares.length === 0) return new Float64Array(0);
  const first = squares[0];
  if (Array.isArray(first) || ArrayBuffer.isView(first)) {
    const n = squares.length;
    const out = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      const row = squares[i];
      out[3 * i] = +row[0];
      out[3 * i + 1] = +row[1];
      out[3 * i + 2] = +row[2];
    }
    return out;
  }
  if (squares.length % 3 !== 0) throw new TypeError('flat packing length must be a multiple of 3');
  return Float64Array.from(squares, Number);
}

/** Split a flat packing into separate x, y, theta arrays. */
export function splitFlat(flat) {
  flat = toFlat(flat);
  const n = flat.length / 3;
  const x = new Float64Array(n), y = new Float64Array(n), t = new Float64Array(n);
  for (let i = 0; i < n; i++) { x[i] = flat[3 * i]; y[i] = flat[3 * i + 1]; t[i] = flat[3 * i + 2]; }
  return { x, y, t, n };
}

// --------------------------------------------------------------------------- //
// pair enumeration (port of build_pairs in _fastcore.c)
// --------------------------------------------------------------------------- //
/** Growable pair list with Int32Array storage (reused across rebuilds). */
export class PairList {
  constructor(cap = 1024) {
    this.I = new Int32Array(cap);
    this.J = new Int32Array(cap);
    this.m = 0;
  }
  push(i, j) {
    if (this.m === this.I.length) {
      const cap = this.I.length * 2;
      const I2 = new Int32Array(cap), J2 = new Int32Array(cap);
      I2.set(this.I); J2.set(this.J);
      this.I = I2; this.J = J2;
    }
    if (i < j) { this.I[this.m] = i; this.J[this.m] = j; } else { this.I[this.m] = j; this.J[this.m] = i; }
    this.m++;
  }
}

let gridHead = new Int32Array(0);
let gridNext = new Int32Array(0);
const NDX = [1, 0, 1, 1], NDY = [0, 1, 1, -1];

/**
 * All pairs (i < j) with |dx| < cutoff and |dy| < cutoff, appended to `pl`
 * (which is reset first).  Brute force for n < 64, degenerate spreads or
 * non-finite input; otherwise a uniform cell grid with linked lists, exactly
 * as in the C core.
 */
export function buildPairs(n, x, y, cutoff, pl) {
  pl.m = 0;
  if (n < 2) return pl;
  let xmin = x[0], xmax = x[0], ymin = y[0], ymax = y[0];
  for (let i = 1; i < n; i++) {
    if (x[i] < xmin) xmin = x[i]; if (x[i] > xmax) xmax = x[i];
    if (y[i] < ymin) ymin = y[i]; if (y[i] > ymax) ymax = y[i];
  }
  let bad = false;
  for (let i = 0; i < n; i++) if (!Number.isFinite(x[i]) || !Number.isFinite(y[i])) { bad = true; break; }
  const gx = (xmax - xmin) / cutoff, gy = (ymax - ymin) / cutoff;
  if (bad || n < 64 || !(gx < 1e6) || !(gy < 1e6) || (Math.floor(gx) + 1) * (Math.floor(gy) + 1) > 8 * n + 64) {
    for (let i = 0; i < n; i++) {
      const xi = x[i], yi = y[i];
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(xi - x[j]) < cutoff && Math.abs(yi - y[j]) < cutoff) pl.push(i, j);
      }
    }
    return pl;
  }
  const ncx = Math.floor(gx) + 1, ncy = Math.floor(gy) + 1;
  const ncell = ncx * ncy;
  if (gridHead.length < ncell) gridHead = new Int32Array(Math.max(ncell, 2 * gridHead.length));
  if (gridNext.length < n) gridNext = new Int32Array(Math.max(n, 2 * gridNext.length));
  const head = gridHead, next = gridNext;
  head.fill(-1, 0, ncell);
  for (let i = 0; i < n; i++) {
    const cx = Math.floor((x[i] - xmin) / cutoff), cy = Math.floor((y[i] - ymin) / cutoff);
    const c = cx * ncy + cy;
    next[i] = head[c]; head[c] = i;
  }
  for (let cx = 0; cx < ncx; cx++) {
    for (let cy = 0; cy < ncy; cy++) {
      const c = cx * ncy + cy;
      for (let i = head[c]; i !== -1; i = next[i]) {
        const xi = x[i], yi = y[i];
        for (let j = next[i]; j !== -1; j = next[j]) {
          if (Math.abs(xi - x[j]) < cutoff && Math.abs(yi - y[j]) < cutoff) pl.push(i, j);
        }
        for (let k = 0; k < 4; k++) {
          const nx = cx + NDX[k], ny = cy + NDY[k];
          if (nx < 0 || nx >= ncx || ny < 0 || ny >= ncy) continue;
          for (let j = head[nx * ncy + ny]; j !== -1; j = next[j]) {
            if (Math.abs(xi - x[j]) < cutoff && Math.abs(yi - y[j]) < cutoff) pl.push(i, j);
          }
        }
      }
    }
  }
  return pl;
}

/**
 * Index pairs (i < j) whose centres are within `cutoff` in both x and y.
 * Returns {I, J} as Int32Array views of length m (fresh storage each call).
 */
export function candidatePairs(x, y, cutoff = EXACT_CUTOFF) {
  const n = x.length;
  const pl = buildPairs(n, x, y, cutoff, new PairList(Math.max(16, 8 * n)));
  return { I: pl.I.subarray(0, pl.m), J: pl.J.subarray(0, pl.m), m: pl.m };
}

// --------------------------------------------------------------------------- //
// separating-axis penetration depth
// --------------------------------------------------------------------------- //
/** max_a |a . d| over the four edge directions of squares i and j. */
function projectionMax(dx, dy, ci, si, cj, sj) {
  let M = Math.abs(ci * dx + si * dy), a;
  a = Math.abs(-si * dx + ci * dy); if (a > M) M = a;
  a = Math.abs(cj * dx + sj * dy); if (a > M) M = a;
  a = Math.abs(-sj * dx + cj * dy); if (a > M) M = a;
  return M;
}

/** Penetration depth for each pair (> 0 overlap, <= 0 separated). */
export function pairPenetration(x, y, t, I, J, m = I.length) {
  const out = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const i = I[k], j = J[k];
    const dx = x[i] - x[j], dy = y[i] - y[j];
    const ci = Math.cos(t[i]), si = Math.sin(t[i]), cj = Math.cos(t[j]), sj = Math.sin(t[j]);
    const M = projectionMax(dx, dy, ci, si, cj, sj);
    out[k] = 0.5 + supportHalf(t[j] - t[i]) - M;
  }
  return out;
}

/** Per-square distance by which the bounding box leaves [0, s]^2 (<= 0 inside). */
export function containmentViolation(x, y, t, s) {
  const n = x.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const w = supportHalf(t[i]);
    out[i] = Math.max(w - x[i], x[i] + w - s, w - y[i], y[i] + w - s);
  }
  return out;
}

// --------------------------------------------------------------------------- //
// verification and repair
// --------------------------------------------------------------------------- //
/**
 * Check that `squares` is a valid packing in [0, s]^2.  `tol` is the largest
 * penetration / protrusion accepted.  Non-finite input is never valid
 * (reported with infinite violations).
 */
export function verify(s, squares, tol = 1e-9) {
  const { x, y, t, n } = splitFlat(squares);
  s = +s;
  const report = { ok: true, n, s, maxPenetration: -Infinity, maxOutside: -Infinity, tol, worstPair: null, worstSquare: null };
  if (n === 0) return report;
  let finite = Number.isFinite(s);
  for (let i = 0; i < n && finite; i++) {
    if (!Number.isFinite(x[i]) || !Number.isFinite(y[i]) || !Number.isFinite(t[i])) finite = false;
  }
  if (!finite) {
    report.ok = false; report.maxPenetration = Infinity; report.maxOutside = Infinity;
    return report;
  }
  const cont = containmentViolation(x, y, t, s);
  let worstSq = 0;
  for (let i = 1; i < n; i++) if (cont[i] > cont[worstSq]) worstSq = i;
  report.maxOutside = cont[worstSq];
  report.worstSquare = worstSq;
  const { I, J, m } = candidatePairs(x, y, EXACT_CUTOFF);
  if (m > 0) {
    const pen = pairPenetration(x, y, t, I, J, m);
    let w = 0;
    for (let k = 1; k < m; k++) if (pen[k] > pen[w]) w = k;
    report.maxPenetration = pen[w];
    report.worstPair = [I[w], J[w]];
  }
  report.ok = report.maxOutside <= tol && report.maxPenetration <= tol;
  return report;
}

/**
 * Smallest factor lam >= 1 such that scaling s, x and y by lam (angles fixed)
 * removes every overlap and protrusion; Infinity when impossible (coincident
 * centres or a centre on / outside the boundary).
 */
export function repairScale(s, x, y, t) {
  const n = x.length;
  let lam = 1.0;
  const { I, J, m } = candidatePairs(x, y, SQRT2 * 1.5);
  for (let k = 0; k < m; k++) {
    const i = I[k], j = J[k];
    const dx = x[i] - x[j], dy = y[i] - y[j];
    const ci = Math.cos(t[i]), si = Math.sin(t[i]), cj = Math.cos(t[j]), sj = Math.sin(t[j]);
    const M = projectionMax(dx, dy, ci, si, cj, sj);
    const need = 0.5 + supportHalf(t[j] - t[i]);
    if (need > M) {
      if (M <= 0) return Infinity;
      const r = need / M;
      if (r > lam) lam = r;
    }
  }
  for (let i = 0; i < n; i++) {
    const w = supportHalf(t[i]);
    const vals = [x[i], s - x[i], y[i], s - y[i]];
    for (let q = 0; q < 4; q++) {
      const val = vals[q];
      if (val < w) {
        if (val <= 0) return Infinity;
        const r = w / val;
        if (r > lam) lam = r;
      }
    }
  }
  return lam;
}

/**
 * Scale the packing by the minimal factor (times 1 + margin) that makes it
 * exactly valid.  Returns {s, squares} with `squares` a flat Float64Array
 * (angles unchanged).  Throws when no scaling can repair it.
 */
export function repair(s, squares, margin = 1e-12) {
  const { x, y, t, n } = splitFlat(squares);
  let lam = repairScale(+s, x, y, t);
  if (!Number.isFinite(lam)) throw new Error('packing cannot be repaired by scaling (coincident centres)');
  lam *= (1.0 + margin);
  const out = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) { out[3 * i] = x[i] * lam; out[3 * i + 1] = y[i] * lam; out[3 * i + 2] = t[i]; }
  return { s: s * lam, squares: out };
}

// --------------------------------------------------------------------------- //
// penalty energy and gradient (port of energy_grad_c)
// --------------------------------------------------------------------------- //
let trigC = new Float64Array(0);
let trigS = new Float64Array(0);

/** Any pair with max projection above this is separated whatever the angles (R <= sqrt2/2). */
const FAR_M = 0.5 + SQRT2 / 2 + 1e-4;

/**
 * Penalty energy E (returned) and, when `grad` is given, its gradient written
 * into grad[0..3n).  `z = [x..., y..., t...]`; (I, J) is the pair list (first
 * `m` entries, default all).  Same formulas as energy_grad_c in _fastcore.c.
 */
export function energyGrad(z, n, s, I, J, grad, m = I.length) {
  const n2 = 2 * n;
  if (grad) grad.fill(0, 0, 3 * n);
  if (trigC.length < n) { trigC = new Float64Array(n); trigS = new Float64Array(n); }
  const C = trigC, S = trigS;
  // cos/sin per square (the C code evaluates them per pair; identical values)
  for (let i = 0; i < n; i++) { C[i] = Math.cos(z[n2 + i]); S[i] = Math.sin(z[n2 + i]); }
  let E = 0.0;
  for (let k = 0; k < m; k++) {
    const i = I[k], j = J[k];
    const dx = z[i] - z[j], dy = z[n + i] - z[n + j];
    const ci = C[i], si = S[i], cj = C[j], sj = S[j];
    const A0 = ci * dx + si * dy, A1 = -si * dx + ci * dy, A2 = cj * dx + sj * dy, A3 = -sj * dx + cj * dy;
    let kk = 0, M = Math.abs(A0), a;
    a = Math.abs(A1); if (a > M) { M = a; kk = 1; }
    a = Math.abs(A2); if (a > M) { M = a; kk = 2; }
    a = Math.abs(A3); if (a > M) { M = a; kk = 3; }
    // exact early-out: p = 1/2 + R - M with R <= sqrt2/2, so such a pair contributes nothing
    if (M > FAR_M) continue;
    const phi = z[n2 + j] - z[n2 + i];
    const cphi = Math.cos(phi), sphi = Math.sin(phi);
    const R = 0.5 * (Math.abs(cphi) + Math.abs(sphi));
    const p = 0.5 + R - M;
    if (p <= 0) continue;
    E += p * p;
    if (!grad) continue;
    let Ak, ax, ay;
    if (kk === 0) { Ak = A0; ax = ci; ay = si; }
    else if (kk === 1) { Ak = A1; ax = -si; ay = ci; }
    else if (kk === 2) { Ak = A2; ax = cj; ay = sj; }
    else { Ak = A3; ax = -sj; ay = cj; }
    const sg = sgn(Ak);
    const g = 2.0 * p;
    const gxi = -g * sg * ax, gyi = -g * sg * ay;
    grad[i] += gxi; grad[j] -= gxi; grad[n + i] += gyi; grad[n + j] -= gyi;
    const dMi = (kk === 0) ? sg * A1 : (kk === 1) ? -sg * A0 : 0.0;
    const dMj = (kk === 2) ? sg * A3 : (kk === 3) ? -sg * A2 : 0.0;
    const dR = 0.5 * (-sgn(cphi) * sphi + sgn(sphi) * cphi);
    grad[n2 + i] += g * (-dR - dMi);
    grad[n2 + j] += g * (dR - dMj);
  }
  for (let i = 0; i < n; i++) {
    const c = C[i], sn = S[i];
    const w = 0.5 * (Math.abs(c) + Math.abs(sn));
    let v0 = w - z[i], v1 = z[i] + w - s, v2 = w - z[n + i], v3 = z[n + i] + w - s;
    if (v0 < 0) v0 = 0; if (v1 < 0) v1 = 0; if (v2 < 0) v2 = 0; if (v3 < 0) v3 = 0;
    E += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    if (grad) {
      grad[i] += 2.0 * (v1 - v0);
      grad[n + i] += 2.0 * (v3 - v2);
      grad[n2 + i] += 2.0 * (v0 + v1 + v2 + v3) * 0.5 * (-sgn(c) * sn + sgn(sn) * c);
    }
  }
  return E;
}

const scratchPl = new PairList(1024);

/** Largest penetration / protrusion of z at side s (exact pair test); Infinity for non-finite z. */
export function maxViolation(z, n, s) {
  for (let i = 0; i < 3 * n; i++) if (!Number.isFinite(z[i])) return Infinity;
  const x = z.subarray(0, n), y = z.subarray(n, 2 * n);
  const pl = buildPairs(n, x, y, EXACT_CUTOFF, scratchPl);
  const n2 = 2 * n;
  let worst = -1e300;
  for (let k = 0; k < pl.m; k++) {
    const i = pl.I[k], j = pl.J[k];
    const dx = z[i] - z[j], dy = z[n + i] - z[n + j];
    const ci = Math.cos(z[n2 + i]), si = Math.sin(z[n2 + i]), cj = Math.cos(z[n2 + j]), sj = Math.sin(z[n2 + j]);
    const M = projectionMax(dx, dy, ci, si, cj, sj);
    const phi = z[n2 + j] - z[n2 + i];
    const p = 0.5 + 0.5 * (Math.abs(Math.cos(phi)) + Math.abs(Math.sin(phi))) - M;
    if (p > worst) worst = p;
  }
  for (let i = 0; i < n; i++) {
    const w = 0.5 * (Math.abs(Math.cos(z[n2 + i])) + Math.abs(Math.sin(z[n2 + i])));
    let v = w - z[i]; if (v > worst) worst = v;
    v = z[i] + w - s; if (v > worst) worst = v;
    v = w - z[n + i]; if (v > worst) worst = v;
    v = z[n + i] + w - s; if (v > worst) worst = v;
  }
  return worst;
}

// --------------------------------------------------------------------------- //
// L-BFGS with internal pair-list rebuild (port of lbfgs_c)
// --------------------------------------------------------------------------- //
const LBFGS_H = 10;
let ws = null;   // workspace reused between calls of the same size

function workspace(N) {
  if (ws === null || ws.N !== N) {
    ws = {
      N,
      zref: new Float64Array(N), g: new Float64Array(N), gn: new Float64Array(N),
      d: new Float64Array(N), zn: new Float64Array(N),
      S: new Float64Array(N * LBFGS_H), Y: new Float64Array(N * LBFGS_H),
      rho: new Float64Array(LBFGS_H), alpha: new Float64Array(LBFGS_H),
      pl: new PairList(1024),
    };
  }
  return ws;
}

function dot(N, a, ao, b, bo) {
  let r = 0;
  for (let i = 0; i < N; i++) r += a[ao + i] * b[bo + i];
  return r;
}

function needsRebuild(n, z, zref, D) {
  for (let i = 0; i < 2 * n; i++) if (Math.abs(z[i] - zref[i]) > D) return true;
  return false;
}

/**
 * L-BFGS minimisation of E(z; s) at fixed s, in place.  Returns {E, iters}
 * (E w.r.t. the final pair list).  Armijo backtracking, displacement cap of
 * 0.5 per step, pair list (cutoff) rebuilt whenever a square moved more than
 * (cutoff - sqrt2)/2 from the point of the last build.
 */
export function lbfgsInPlace(z, n, s, maxiter = LOCAL_MAXITER, gtol = 1e-11, ftol = 1e-18, cutoff = PAIR_CUTOFF) {
  const N = 3 * n, H = LBFGS_H;
  const D = 0.5 * (cutoff - SQRT2) - 1e-9;
  const w = workspace(N);
  const zref = w.zref, g = w.g, gn = w.gn, d = w.d, zn = w.zn, S = w.S, Y = w.Y, rho = w.rho, alpha = w.alpha, pl = w.pl;
  const x = z.subarray(0, n), y = z.subarray(n, 2 * n);
  const xn = zn.subarray(0, n), yn = zn.subarray(n, 2 * n);
  let hist = 0, head = 0;
  zref.set(z);
  buildPairs(n, x, y, cutoff, pl);
  let E = energyGrad(z, n, s, pl.I, pl.J, g, pl.m);
  let it = 0, stall = 0;
  for (it = 0; it < maxiter; it++) {
    if (E < 1e-26) break;
    const gnorm = Math.sqrt(dot(N, g, 0, g, 0));
    if (gnorm < gtol) break;
    // two-loop recursion
    for (let i = 0; i < N; i++) d[i] = -g[i];
    for (let k = 0; k < hist; k++) {
      const idx = (head - 1 - k + H) % H, off = idx * N;
      const a = rho[idx] * dot(N, S, off, d, 0);
      alpha[idx] = a;
      for (let i = 0; i < N; i++) d[i] -= a * Y[off + i];
    }
    if (hist > 0) {
      const idx = (head - 1 + H) % H, off = idx * N;
      const gamma = dot(N, S, off, Y, off) / dot(N, Y, off, Y, off);
      for (let i = 0; i < N; i++) d[i] *= gamma;
    }
    for (let k = hist - 1; k >= 0; k--) {
      const idx = (head - 1 - k + H) % H, off = idx * N;
      const beta = rho[idx] * dot(N, Y, off, d, 0);
      const c = alpha[idx] - beta;
      for (let i = 0; i < N; i++) d[i] += c * S[off + i];
    }
    let dg = dot(N, d, 0, g, 0);
    if (dg >= 0) { for (let i = 0; i < N; i++) d[i] = -g[i]; dg = -gnorm * gnorm; hist = 0; }
    // cap the step so that no square jumps more than ~0.5 in one go
    let dmax = 0;
    for (let i = 0; i < N; i++) { const a = Math.abs(d[i]); if (a > dmax) dmax = a; }
    let step = (hist === 0) ? Math.min(1.0, 0.1 / (dmax + 1e-300)) : 1.0;
    if (step * dmax > 0.5) step = 0.5 / dmax;
    let En = 0, accepted = false;
    for (let ls = 0; ls < 40; ls++) {
      for (let i = 0; i < N; i++) zn[i] = z[i] + step * d[i];
      if (needsRebuild(n, zn, zref, D)) {
        zref.set(zn);
        buildPairs(n, xn, yn, cutoff, pl);
        E = energyGrad(z, n, s, pl.I, pl.J, g, pl.m);   // re-evaluate the base point
        dg = dot(N, d, 0, g, 0);
        if (dg >= 0) {
          for (let i = 0; i < N; i++) d[i] = -g[i];
          dg = -dot(N, g, 0, g, 0); hist = 0;
          step = Math.min(step, 0.1 / (dmax + 1e-300));   // dmax deliberately stale, as in the C code
          continue;
        }
      }
      En = energyGrad(zn, n, s, pl.I, pl.J, gn, pl.m);
      if (En <= E + 1e-4 * step * dg) { accepted = true; break; }
      step *= 0.5;
    }
    if (!accepted) break;
    // history update (written into slot `head` unconditionally, exactly like the C code:
    // when the curvature test fails the slot keeps the pair but head is not advanced)
    const off = head * N;
    for (let i = 0; i < N; i++) { S[off + i] = zn[i] - z[i]; Y[off + i] = gn[i] - g[i]; }
    const ys = dot(N, S, off, Y, off);
    if (ys > 1e-14) { rho[head] = 1.0 / ys; head = (head + 1) % H; if (hist < H) hist++; }
    const dec = E - En;
    z.set(zn); g.set(gn); E = En;
    if (dec <= ftol * Math.max(E, 1e-300)) { if (++stall >= 8) break; } else stall = 0;
  }
  return { E, iters: it };
}

/**
 * L-BFGS on a copy of z.  Returns {z, E, iters}.
 * opts: {maxiter = 250, gtol = 1e-11, ftol = 1e-18, cutoff = 2.6}.
 */
export function lbfgs(z, n, s, opts = {}) {
  const { maxiter = LOCAL_MAXITER, gtol = 1e-11, ftol = 1e-18, cutoff = PAIR_CUTOFF } = opts;
  const zc = Float64Array.from(z);
  const r = lbfgsInPlace(zc, n, s, maxiter, gtol, ftol, cutoff);
  return { z: zc, E: r.E, iters: r.iters };
}

// --------------------------------------------------------------------------- //
// seedable random numbers (sfc32 seeded through splitmix32)
// --------------------------------------------------------------------------- //
/** Deterministic PRNG with the sampling helpers the search needs. */
export function makeRng(seed = 0) {
  let h = Math.floor(Number(seed));
  if (!Number.isFinite(h)) h = 0;
  h = (h % 4294967296 + 4294967296) % 4294967296 >>> 0;
  const splitmix = () => {
    h = (h + 0x9E3779B9) | 0;
    let z = h;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
    return (z ^ (z >>> 15)) >>> 0;
  };
  let a = splitmix(), b = splitmix(), c = splitmix(), d = splitmix();
  const next = () => {
    a |= 0; b |= 0; c |= 0; d |= 0;
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 12; i++) next();
  let spare = null;
  const rng = {
    /** uniform in [0, 1) */
    random: next,
    uniform(lo, hi) { return lo + (hi - lo) * next(); },
    /** integer in [lo, hi) */
    integers(lo, hi) { return lo + Math.floor(next() * (hi - lo)); },
    normal(mu = 0, sigma = 1) {
      if (spare !== null) { const v = spare; spare = null; return mu + sigma * v; }
      let u, v, q;
      do { u = 2 * next() - 1; v = 2 * next() - 1; q = u * u + v * v; } while (q >= 1 || q === 0);
      const f = Math.sqrt(-2 * Math.log(q) / q);
      spare = v * f;
      return mu + sigma * u * f;
    },
    pick(arr) { return arr[Math.floor(next() * arr.length)]; },
    /** index drawn with probability proportional to weights[i] */
    choiceWeighted(weights) {
      let tot = 0;
      for (let i = 0; i < weights.length; i++) tot += weights[i];
      let r = next() * tot;
      for (let i = 0; i < weights.length; i++) { r -= weights[i]; if (r < 0) return i; }
      return weights.length - 1;
    },
    /** m distinct indices from [0, k), optionally with probabilities `prob` (sequential weighted draws) */
    sample(k, m, prob = null) {
      const out = [];
      if (prob === null) {
        const idx = new Int32Array(k);
        for (let i = 0; i < k; i++) idx[i] = i;
        for (let i = 0; i < m; i++) {
          const j = i + Math.floor(next() * (k - i));
          const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
          out.push(idx[i]);
        }
        return out;
      }
      const wts = Float64Array.from(prob);
      for (let i = 0; i < m; i++) {
        const j = rng.choiceWeighted(wts);
        out.push(j);
        wts[j] = 0;
      }
      return out;
    },
  };
  return rng;
}

// --------------------------------------------------------------------------- //
// configurations
// --------------------------------------------------------------------------- //
/** A packing under optimisation: side s and separate x, y, theta arrays. */
export class Config {
  constructor(s, x, y, t) { this.s = s; this.x = x; this.y = y; this.t = t; }
  get n() { return this.x.length; }
  copy() { return new Config(this.s, Float64Array.from(this.x), Float64Array.from(this.y), Float64Array.from(this.t)); }
  scaled(sNew) {
    const f = sNew / this.s, n = this.n;
    const x = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = this.x[i] * f; y[i] = this.y[i] * f; }
    return new Config(sNew, x, y, Float64Array.from(this.t));
  }
  /** Flat packing with canonical angles. */
  squares() {
    const n = this.n, out = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) { out[3 * i] = this.x[i]; out[3 * i + 1] = this.y[i]; out[3 * i + 2] = canonicalAngle(this.t[i]); }
    return out;
  }
  /** Optimiser layout [x..., y..., t...]. */
  z() {
    const n = this.n, z = new Float64Array(3 * n);
    z.set(this.x, 0); z.set(this.y, n); z.set(this.t, 2 * n);
    return z;
  }
  maxViolation() { return maxViolation(this.z(), this.n, this.s); }
  static fromZ(s, z, n) {
    return new Config(s, Float64Array.from(z.subarray(0, n)), Float64Array.from(z.subarray(n, 2 * n)), Float64Array.from(z.subarray(2 * n, 3 * n)));
  }
  static fromSquares(s, squares) {
    const { x, y, t } = splitFlat(squares);
    return new Config(+s, x, y, t);
  }
}

/** The configuration scaled by the minimal factor that makes it exactly valid. */
export function repaired(cfg, margin = 1e-12) {
  const r = repair(cfg.s, cfg.squares(), margin);
  return Config.fromSquares(r.s, r.squares);
}

/** Axis-aligned k x k grid holding n squares (k = ceil sqrt n); valid at s = k. */
export function exactGrid(n) {
  let k = Math.floor(Math.sqrt(n));
  if (k * k < n) k += 1;
  const x = new Float64Array(n), y = new Float64Array(n), t = new Float64Array(n);
  for (let i = 0; i < n; i++) { x[i] = (i % k) + 0.5; y[i] = Math.floor(i / k) + 0.5; }
  return new Config(k, x, y, t);
}

// --------------------------------------------------------------------------- //
// local optimisation (fixed s)
// --------------------------------------------------------------------------- //
/** Minimise E at fixed s from cfg; returns {cfg, E} with E w.r.t. the exact pair list. */
export function localOpt(cfg, maxiter = LOCAL_MAXITER, gtol = 1e-11, ftol = 1e-18) {
  const n = cfg.n, s = cfg.s;
  const z = cfg.z();
  lbfgsInPlace(z, n, s, maxiter, gtol, ftol, PAIR_CUTOFF);
  const pl = buildPairs(n, z.subarray(0, n), z.subarray(n, 2 * n), EXACT_CUTOFF, scratchPl);
  const E = energyGrad(z, n, s, pl.I, pl.J, null, pl.m);
  return { cfg: Config.fromZ(s, z, n), E };
}

export function isFeasible(cfg, ptol = 1e-8) {
  return cfg.maxViolation() <= ptol;
}

// --------------------------------------------------------------------------- //
// perturbations for basin hopping (the "classic" move set)
// --------------------------------------------------------------------------- //
export function randomAngle(rng) {
  const r = rng.random();
  if (r < 0.4) return 0.0;
  if (r < 0.7) return QUARTER_PI;
  return rng.uniform(-QUARTER_PI, QUARTER_PI);
}

/** Penalty energy attributed to each square (pair terms counted on both members). */
function pairEnergyPerSquare(cfg) {
  const n = cfg.n, e = new Float64Array(n);
  const { I, J, m } = candidatePairs(cfg.x, cfg.y, EXACT_CUTOFF);
  if (m > 0) {
    const pen = pairPenetration(cfg.x, cfg.y, cfg.t, I, J, m);
    for (let k = 0; k < m; k++) {
      const p = pen[k] > 0 ? pen[k] * pen[k] : 0;
      e[I[k]] += p; e[J[k]] += p;
    }
  }
  const cont = containmentViolation(cfg.x, cfg.y, cfg.t, cfg.s);
  for (let i = 0; i < n; i++) if (cont[i] > 0) e[i] += cont[i] * cont[i];
  return e;
}

/** A point far (in L-inf) from all centres, found by random sampling. */
function emptiestPoint(cfg, rng, samples = 400) {
  const w = 0.5, n = cfg.n;
  let bestX = cfg.s / 2, bestY = cfg.s / 2, bestClear = -Infinity;
  for (let k = 0; k < samples; k++) {
    const px = rng.uniform(w, cfg.s - w), py = rng.uniform(w, cfg.s - w);
    let clear = Infinity;
    for (let i = 0; i < n; i++) {
      const dd = Math.max(Math.abs(px - cfg.x[i]), Math.abs(py - cfg.y[i]));
      if (dd < clear) clear = dd;
    }
    if (clear > bestClear) { bestClear = clear; bestX = px; bestY = py; }
  }
  return [bestX, bestY];
}

export const PERTURB_KINDS = ['jitter', 'kick', 'rotate', 'swap', 'hole', 'shake'];
export const PERTURB_WEIGHTS = [0.25, 0.25, 0.15, 0.1, 0.15, 0.1];

/** A perturbed copy of cfg (used when the local optimiser is stuck). */
export function perturb(cfg, rng, kind = null) {
  const c = cfg.copy();
  const n = c.n;
  if (kind === null) kind = PERTURB_KINDS[rng.choiceWeighted(PERTURB_WEIGHTS)];
  if (kind === 'jitter') {
    const sig = rng.pick([0.03, 0.1, 0.25]);
    for (let i = 0; i < n; i++) c.x[i] += rng.normal(0, sig);
    for (let i = 0; i < n; i++) c.y[i] += rng.normal(0, sig);
    for (let i = 0; i < n; i++) c.t[i] += rng.normal(0, sig);
  } else if (kind === 'shake') {
    for (let i = 0; i < n; i++) c.x[i] += rng.normal(0, 0.5);
    for (let i = 0; i < n; i++) c.y[i] += rng.normal(0, 0.5);
    for (let i = 0; i < n; i++) c.t[i] += rng.normal(0, 0.3);
  } else if (kind === 'kick') {
    const e = pairEnergyPerSquare(c);
    const m = rng.integers(1, Math.min(4, n) + 1);
    const prob = new Float64Array(n);
    for (let i = 0; i < n; i++) prob[i] = e[i] + 1e-9;
    for (const i of rng.sample(n, m, prob)) {
      c.x[i] = rng.uniform(0.5, c.s - 0.5);
      c.y[i] = rng.uniform(0.5, c.s - 0.5);
      c.t[i] = randomAngle(rng);
    }
  } else if (kind === 'rotate') {
    const m = rng.integers(1, Math.min(4, n) + 1);
    for (const i of rng.sample(n, m)) {
      c.t[i] = rng.random() < 0.7 ? randomAngle(rng) : c.t[i] + rng.normal(0, 0.4);
    }
  } else if (kind === 'swap') {
    if (n >= 2) {
      const [i, j] = rng.sample(n, 2);
      let tmp = c.x[i]; c.x[i] = c.x[j]; c.x[j] = tmp;
      tmp = c.y[i]; c.y[i] = c.y[j]; c.y[j] = tmp;
    }
  } else if (kind === 'hole') {
    const e = pairEnergyPerSquare(c);
    let i = 0, emax = 0;
    for (let q = 0; q < n; q++) if (e[q] > emax) { emax = e[q]; i = q; }
    if (!(emax > 0)) i = rng.integers(0, n);
    const [hx, hy] = emptiestPoint(c, rng);
    c.x[i] = hx; c.y[i] = hy; c.t[i] = randomAngle(rng);
  }
  for (let i = 0; i < n; i++) {
    if (c.x[i] < 0.3) c.x[i] = 0.3; else if (c.x[i] > c.s - 0.3) c.x[i] = c.s - 0.3;
    if (c.y[i] < 0.3) c.y[i] = 0.3; else if (c.y[i] > c.s - 0.3) c.y[i] = c.s - 0.3;
  }
  return c;
}

// --------------------------------------------------------------------------- //
// seeds
// --------------------------------------------------------------------------- //
export function seedRandom(n, s, rng) {
  const x = new Float64Array(n), y = new Float64Array(n), t = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = rng.uniform(0.5, s - 0.5);
  for (let i = 0; i < n; i++) y[i] = rng.uniform(0.5, s - 0.5);
  for (let i = 0; i < n; i++) t[i] = randomAngle(rng);
  return new Config(s, x, y, t);
}

/** Random cells of the ceil(s) x ceil(s) grid, jittered; 15% of the angles random. */
export function seedGridJitter(n, s, rng) {
  const k = Math.ceil(s - 1e-9);
  const x = new Float64Array(n), y = new Float64Array(n), t = new Float64Array(n);
  const ncell = Math.min(n, k * k);
  const cells = rng.sample(k * k, ncell);
  for (let i = 0; i < ncell; i++) {
    x[i] = ((cells[i] % k) + 0.5) * (s / k);
    y[i] = (Math.floor(cells[i] / k) + 0.5) * (s / k);
  }
  for (let i = ncell; i < n; i++) x[i] = rng.uniform(0.5, s - 0.5);
  for (let i = ncell; i < n; i++) y[i] = rng.uniform(0.5, s - 0.5);
  for (let i = 0; i < n; i++) x[i] += rng.normal(0, 0.05);
  for (let i = 0; i < n; i++) y[i] += rng.normal(0, 0.05);
  const rnd = new Float64Array(n);
  for (let i = 0; i < n; i++) rnd[i] = rng.random();
  const angles = new Float64Array(n);
  for (let i = 0; i < n; i++) angles[i] = randomAngle(rng);
  for (let i = 0; i < n; i++) t[i] = rnd[i] < 0.15 ? angles[i] : 0.0;
  return new Config(s, x, y, t);
}

// --------------------------------------------------------------------------- //
// search context, feasibility and compaction (generators: one `yield` per local
// optimisation so that a Web Worker can interleave message handling)
// --------------------------------------------------------------------------- //
export function newStats() {
  return { localOpts: 0, improvements: 0, seeds: 0, feasibleFailures: 0 };
}

/**
 * Shared state of one search: rng, tolerances, deadline, stop callback and
 * statistics.  `timeUp(until)` is polled between local optimisations.
 */
export function makeContext(opts = {}) {
  const ctx = {
    rng: opts.rng || makeRng(opts.seed ?? 0),
    ptol: opts.ptol ?? 1e-8,
    hops: opts.hops ?? 4,
    maxiter: opts.maxiter ?? LOCAL_MAXITER,
    deadline: opts.deadline ?? Infinity,
    shouldStop: opts.shouldStop || null,
    maxLocalOpts: opts.maxLocalOpts ?? Infinity,
    stats: opts.stats || newStats(),
    stopped: false,
  };
  ctx.timeUp = (until = ctx.deadline) => {
    if (ctx.stopped) return true;
    if (ctx.stats.localOpts >= ctx.maxLocalOpts) { ctx.stopped = true; return true; }
    if (now() >= Math.min(until, ctx.deadline)) return true;
    if (ctx.shouldStop && ctx.shouldStop()) { ctx.stopped = true; return true; }
    return false;
  };
  ctx.localOpt = (cfg) => {
    ctx.stats.localOpts++;
    return localOpt(cfg, ctx.maxiter);
  };
  return ctx;
}

/**
 * Locally optimise; on failure perturb, then enlarge s until a valid packing
 * appears.  Returns a repaired Config or null.
 * DEVIATION: the reference never checks the clock here; this port gives up
 * (returns null) once the deadline / stop flag is hit.
 */
export function* makeFeasibleGen(cfg, ctx, hops = 3, grow = 1.03, maxGrow = 12) {
  let cur = cfg;
  for (let round = 0; round < maxGrow; round++) {
    if (ctx.timeUp()) return null;
    let { cfg: res, E } = ctx.localOpt(cur);
    yield;
    if (isFeasible(res, ctx.ptol)) return repaired(res);
    for (let h = 0; h < hops; h++) {
      if (ctx.timeUp()) return null;
      const r2 = ctx.localOpt(perturb(res, ctx.rng));
      yield;
      if (isFeasible(r2.cfg, ctx.ptol)) return repaired(r2.cfg);
      if (r2.E < E) { res = r2.cfg; E = r2.E; }   // keep the better of the two as the base
    }
    cur = res.scaled(res.s * grow);
  }
  ctx.stats.feasibleFailures++;
  return null;
}

/**
 * Shrink a valid packing as far as the local optimiser (plus basin hopping)
 * allows: bisection on s, hops only while the step is coarse.  `onImprove(cfg)`
 * receives every (repaired) improvement.  Returns the best valid Config.
 */
export function* compactGen(cfg, ctx, until = ctx.deadline, onImprove = null, sched = {}) {
  const step0 = sched.step0 ?? 0.01, minStep = sched.minStep ?? 1e-7, hopMinStep = sched.hopMinStep ?? 5e-4;
  const hops = sched.hops ?? ctx.hops;
  let best = repaired(cfg);
  let step = step0 * best.s;
  while (step > minStep * best.s && !ctx.timeUp(until)) {
    const sTry = best.s - step;
    const trial = best.scaled(sTry);
    let { cfg: res, E } = ctx.localOpt(trial);
    yield;
    let ok = isFeasible(res, ctx.ptol);
    let h = 0;
    let base = res;
    // basin hopping only while the step is coarse; the fine bisection is pure local descent
    const hopsNow = step > hopMinStep * best.s ? hops : 0;
    while (!ok && h < hopsNow && !ctx.timeUp(until)) {
      const r2 = ctx.localOpt(perturb(base, ctx.rng));
      yield;
      h++;
      if (isFeasible(r2.cfg, ctx.ptol)) { res = r2.cfg; ok = true; }
      else if (r2.E < E) { base = r2.cfg; E = r2.E; }
    }
    if (ok) {
      best = repaired(res);
      ctx.stats.improvements++;
      if (onImprove) onImprove(best);
      step = Math.min(step * 1.5, 0.02 * best.s);
    } else {
      step *= 0.5;
    }
  }
  return best;
}

function drain(gen) {
  let r;
  while (!(r = gen.next()).done) { /* one local optimisation per step */ }
  return r.value;
}

/**
 * Compact a valid packing.  Returns {s, squares, stats}.
 * opts: {budgetMs = 10000, seed = 0, ptol = 1e-8, hops = 4, maxiter = 250,
 *        step0, minStep, hopMinStep, onImprove(s, squares), shouldStop(), maxLocalOpts, stats}.
 * onImprove only sees packings that pass verify after repair.
 */
export function compact(s, squares, opts = {}) {
  const budgetMs = opts.budgetMs ?? 10000;
  const ctx = makeContext({ ...opts, deadline: now() + budgetMs });
  const cfg = Config.fromSquares(s, squares);
  const cb = opts.onImprove ? (c) => {
    const sq = c.squares();
    if (verify(c.s, sq, 1e-9).ok) opts.onImprove(c.s, sq);
  } : null;
  const best = drain(compactGen(cfg, ctx, ctx.deadline, cb, opts));
  return { s: best.s, squares: best.squares(), stats: ctx.stats };
}

// --------------------------------------------------------------------------- //
// global driver (the "bisect" strategy of optimize.search with grid / random seeds)
// --------------------------------------------------------------------------- //
/**
 * Generator form of `search` (one yield per local optimisation).  The incumbent
 * is `squares0` at `s0` when given (repaired / made feasible if slightly
 * invalid), otherwise a jittered grid at s0 (default ceil sqrt n) made
 * feasible; an exact k x k grid is always kept as a valid fallback.  Seeds
 * (jittered grid, random) are then compacted in turn until the budget is
 * spent.  Returns {s, squares, stats, history}.
 *
 * DEVIATION: the reference cycles analytic / dropped-family / block seeds too;
 * those need the constructions module and are not ported.
 */
export function* searchGen(n, opts = {}) {
  const {
    s0 = null, squares0 = null, budgetMs = 10000, onImprove = null,
    incumbentFraction = 0.25, seedMix = ['grid', 'random'],
  } = opts;
  n = n | 0;
  const t0 = now();
  const ctx = makeContext({ ...opts, deadline: t0 + budgetMs });
  const rng = ctx.rng, stats = ctx.stats;
  const history = [];
  let best = null;
  const improve = (c) => {
    if (c === null) return false;
    if (best !== null && !(c.s < best.s - 1e-12)) return false;
    const sq = c.squares();
    if (!verify(c.s, sq, 1e-9).ok) return false;
    best = c.copy();
    history.push([now() - t0, c.s]);
    if (onImprove) onImprove(c.s, sq);
    return true;
  };
  const finish = () => ({ s: best.s, squares: best.squares(), stats, history });

  if (n <= 0) { best = new Config(0, new Float64Array(0), new Float64Array(0), new Float64Array(0)); return finish(); }
  if (n === 1) { best = new Config(1, Float64Array.of(0.5), Float64Array.of(0.5), Float64Array.of(0)); return finish(); }

  const grid = exactGrid(n);
  const sStart = s0 != null ? +s0 : grid.s;
  let incumbent = null, fromUser = false;
  if (squares0 != null) {
    const cfg = Config.fromSquares(sStart, squares0);
    try {
      const rep = repaired(cfg);
      if (verify(rep.s, rep.squares(), 1e-9).ok) { incumbent = rep; fromUser = true; }
    } catch (e) { /* unrepairable: treat it as a seed below */ }
    if (incumbent === null) incumbent = yield* makeFeasibleGen(cfg, ctx);
  } else {
    incumbent = yield* makeFeasibleGen(seedGridJitter(n, sStart, rng), ctx);
  }
  // the starting point is not an "improvement": the user's packing is adopted silently and the
  // exact grid is a silent valid fallback; onImprove only fires for something strictly tighter
  if (fromUser) {
    best = incumbent.copy();
    history.push([now() - t0, best.s]);
    improve(grid);
  } else {
    best = grid.copy();
    history.push([now() - t0, best.s]);
    if (incumbent !== null) improve(incumbent);
  }
  if (incumbent !== null) {
    // like the reference, cap the compaction of a supplied incumbent so fresh seeds get their turn;
    // a jittered-grid start is itself the first seed and gets the whole budget
    const until = fromUser ? Math.min(ctx.deadline, t0 + incumbentFraction * budgetMs) : ctx.deadline;
    const res = yield* compactGen(incumbent, ctx, until, improve);
    improve(res);
  }

  let k = 0;
  while (!ctx.timeUp()) {
    const kind = seedMix[k % seedMix.length];
    k++;
    stats.seeds++;
    const sSeed = best.s * 1.02;
    const cfg = kind === 'grid' ? seedGridJitter(n, sSeed, rng) : seedRandom(n, sSeed, rng);
    const feas = yield* makeFeasibleGen(cfg, ctx);
    if (feas === null) continue;
    improve(feas);
    const res = yield* compactGen(feas, ctx, ctx.deadline, improve);
    improve(res);
  }
  return finish();
}

/**
 * Numerical search for a packing of n unit squares.  Deterministic for a given
 * `seed` up to where the time budget cuts the trial sequence (pass
 * `maxLocalOpts` instead of / in addition to `budgetMs` for a fully
 * reproducible run).
 * opts: {s0, squares0, budgetMs = 10000, seed = 0, onImprove(s, squares), shouldStop(),
 *        ptol = 1e-8, hops = 4, maxiter = 250, incumbentFraction = 0.25, maxLocalOpts, stats}.
 * Returns {s, squares, stats, history}; onImprove only sees verified packings.
 */
export function search(n, opts = {}) {
  return drain(searchGen(n, opts));
}
