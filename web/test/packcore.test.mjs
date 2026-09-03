// Tests for web/js/packcore.js and web/js/optimizer.worker.js.
// Run with:  node web/test/packcore.test.mjs      (no test framework; ~15 s)
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import {
  SQRT2, QUARTER_PI, PAIR_CUTOFF, canonicalAngle, candidatePairs, energyGrad, verify, repair,
  lbfgs, search, compact, makeRng, now,
} from '../js/packcore.js';

const results = [];
function section(name) { console.log(`\n== ${name}`); }
function ok(msg) { console.log(`   ok  ${msg}`); }

// --------------------------------------------------------------------------- //
section('canonical angle');
{
  assert.ok(Math.abs(canonicalAngle(Math.PI / 2) - 0) < 1e-15);
  assert.ok(Math.abs(canonicalAngle(-QUARTER_PI) - QUARTER_PI) < 1e-15);      // -pi/4 maps to +pi/4
  assert.ok(Math.abs(canonicalAngle(QUARTER_PI) - QUARTER_PI) < 1e-15);
  assert.ok(Math.abs(canonicalAngle(3 * Math.PI + 0.1) - 0.1) < 1e-12);
  assert.ok(Math.abs(canonicalAngle(-0.2) + 0.2) < 1e-15);
  ok('reduction to (-pi/4, pi/4]');
}

// --------------------------------------------------------------------------- //
section('candidatePairs: cell grid agrees with brute force');
{
  const rng = makeRng(99);
  for (const [n, spread] of [[200, 14], [300, 1.0], [64, 8]]) {
    const x = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = rng.uniform(0, spread); y[i] = rng.uniform(0, spread); }
    for (const cutoff of [SQRT2 + 1e-9, PAIR_CUTOFF]) {
      const { I, J, m } = candidatePairs(x, y, cutoff);
      const seen = new Set();
      for (let k = 0; k < m; k++) {
        assert.ok(I[k] < J[k]);
        assert.ok(Math.abs(x[I[k]] - x[J[k]]) < cutoff && Math.abs(y[I[k]] - y[J[k]]) < cutoff);
        seen.add(I[k] * n + J[k]);
      }
      assert.equal(seen.size, m, 'no duplicate pairs');
      let brute = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        if (Math.abs(x[i] - x[j]) < cutoff && Math.abs(y[i] - y[j]) < cutoff) { brute++; assert.ok(seen.has(i * n + j)); }
      }
      assert.equal(m, brute);
    }
  }
  ok('n = 64, 200, 300 at both cutoffs: same pair sets');
}

// --------------------------------------------------------------------------- //
section('energyGrad vs central finite differences');
{
  const rng = makeRng(20240903);
  let worst = 0, trials = 0;
  for (let trial = 0; trial < 25; trial++) {
    const n = 3 + rng.integers(0, 12);
    const s = Math.sqrt(n) * rng.uniform(0.75, 1.05);      // too small: many overlaps and protrusions
    const z = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      z[i] = rng.uniform(0.2, s - 0.2);
      z[n + i] = rng.uniform(0.2, s - 0.2);
      z[2 * n + i] = rng.uniform(-1.2, 1.2);
    }
    const { I, J } = candidatePairs(z.subarray(0, n), z.subarray(n, 2 * n), PAIR_CUTOFF);
    const grad = new Float64Array(3 * n);
    const E = energyGrad(z, n, s, I, J, grad);
    assert.ok(E > 0, 'configuration should overlap');
    assert.ok(Math.abs(energyGrad(z, n, s, I, J, null) - E) === 0, 'energy-only call agrees');
    const h = 1e-6;
    const zp = Float64Array.from(z), zm = Float64Array.from(z);
    let num = 0, den = 0;
    for (let k = 0; k < 3 * n; k++) {
      zp[k] = z[k] + h; zm[k] = z[k] - h;
      const fd = (energyGrad(zp, n, s, I, J, null) - energyGrad(zm, n, s, I, J, null)) / (2 * h);
      zp[k] = z[k]; zm[k] = z[k];
      num += (fd - grad[k]) ** 2;
      den += grad[k] ** 2;
    }
    const rel = Math.sqrt(num) / Math.max(Math.sqrt(den), 1e-12);
    worst = Math.max(worst, rel);
    trials++;
    assert.ok(rel < 1e-6, `trial ${trial}: n=${n} relative gradient error ${rel}`);
  }
  ok(`${trials} random overlapping configurations, worst relative error ${worst.toExponential(2)}`);
}

// --------------------------------------------------------------------------- //
section('verify / repair on the exact n = 5 packing');
const s5 = 2 + 1 / SQRT2;
const five = [
  [0.5, 0.5, 0], [s5 - 0.5, 0.5, 0], [0.5, s5 - 0.5, 0], [s5 - 0.5, s5 - 0.5, 0],
  [s5 / 2, s5 / 2, QUARTER_PI],
];
{
  const rep = verify(s5, five);
  assert.ok(rep.ok, `exact packing should verify: ${JSON.stringify(rep)}`);
  assert.ok(Math.abs(rep.maxPenetration) < 1e-12 && Math.abs(rep.maxOutside) < 1e-12, 'touching squares');
  const flat = new Float64Array(five.flat());
  assert.ok(verify(s5, flat).ok, 'flat Float64Array input');
  assert.ok(verify(s5, Array.from(flat)).ok, 'flat plain-array input');
  const f = (s5 - 1e-4) / s5;
  const shrunk = five.map(([x, y, t]) => [x * f, y * f, t]);
  const bad = verify(s5 - 1e-4, shrunk);
  assert.ok(!bad.ok && bad.maxPenetration > 1e-6, 'shrunk packing must be rejected');
  assert.ok(!verify(3, [[0.5, 0.5, 0], [NaN, 1.5, 0]]).ok, 'NaN rejected');
  assert.ok(!verify(1.9, [[0.5, 0.5, 0], [1.5, 1.5, 0]]).ok, 'protrusion rejected');
  assert.ok(verify(2, [[0.5, 0.5, 0], [1.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]]).ok, 'touching grid accepted');
  const r = repair(s5 - 1e-4, shrunk);
  assert.ok(verify(r.s, r.squares).ok, 'repaired packing verifies');
  assert.ok(Math.abs(r.s - s5) < 1e-9, `repair restores s: ${r.s} vs ${s5}`);
  assert.ok(r.squares instanceof Float64Array && r.squares.length === 15);
  ok(`accepts s = ${s5.toFixed(12)}, rejects s - 1e-4 (penetration ${bad.maxPenetration.toExponential(2)}), repair -> ${r.s.toFixed(12)}`);
}

// --------------------------------------------------------------------------- //
section('lbfgs resolves a mildly overlapping 3 x 3 arrangement');
{
  const n = 9, s = 3.2;
  const z = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    z[i] = 0.5 + 0.95 * (i % 3);
    z[n + i] = 0.5 + 0.95 * Math.floor(i / 3);
    z[2 * n + i] = 0.0;
  }
  const { I, J } = candidatePairs(z.subarray(0, n), z.subarray(n, 2 * n), PAIR_CUTOFF);
  const E0 = energyGrad(z, n, s, I, J, null);
  assert.ok(E0 > 1e-3, 'initial overlap');
  const r = lbfgs(z, n, s, { maxiter: 250, gtol: 1e-11, ftol: 1e-18, cutoff: PAIR_CUTOFF });
  assert.ok(r.E < 1e-12, `final energy ${r.E}`);
  assert.ok(z[0] === 0.5, 'input z untouched');
  const flat = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) { flat[3 * i] = r.z[i]; flat[3 * i + 1] = r.z[n + i]; flat[3 * i + 2] = r.z[2 * n + i]; }
  const rep = verify(s, flat, 1e-6);
  assert.ok(rep.ok, `result is a packing up to sqrt(E): ${JSON.stringify(rep)}`);
  ok(`E ${E0.toExponential(2)} -> ${r.E.toExponential(2)} in ${r.iters} iterations`);
}

// --------------------------------------------------------------------------- //
section('lbfgs timing at n = 100 (250 iterations)');
{
  const n = 100, s = 9.6;
  const rng = makeRng(7);
  const z = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) { z[i] = rng.uniform(0.5, s - 0.5); z[n + i] = rng.uniform(0.5, s - 0.5); z[2 * n + i] = rng.uniform(-0.5, 0.5); }
  let best = Infinity, iters = 0, E = 0;
  for (let rep = 0; rep < 7; rep++) {
    const t = now();
    const r = lbfgs(z, n, s, { maxiter: 250 });
    const dt = now() - t;
    if (dt < best) best = dt;
    iters = r.iters; E = r.E;
  }
  assert.ok(iters === 250, `should not converge early on a random start (iters=${iters})`);
  assert.ok(best < 150, `250 iterations took ${best.toFixed(1)} ms`);
  results.push(`lbfgs n=100, 250 iterations: ${best.toFixed(2)} ms (best of 7), final E=${E.toExponential(2)}`);
  ok(results[results.length - 1]);
}

// --------------------------------------------------------------------------- //
section('search is deterministic for a fixed seed');
{
  const a = search(7, { s0: 3, seed: 5, maxLocalOpts: 60, budgetMs: 60000 });
  const b = search(7, { s0: 3, seed: 5, maxLocalOpts: 60, budgetMs: 60000 });
  assert.equal(a.s, b.s);
  assert.deepEqual(Array.from(a.squares), Array.from(b.squares));
  assert.equal(a.stats.localOpts, 60);
  assert.ok(verify(a.s, a.squares).ok);
  ok(`two runs agree: s = ${a.s} after ${a.stats.localOpts} local optimisations`);
}

// --------------------------------------------------------------------------- //
section('compact() shrinks a valid packing');
{
  // five squares of the 3 x 3 grid at s = 3 -> the 2 + 1/sqrt2 packing is reachable
  const start = [[0.5, 0.5, 0], [1.5, 0.5, 0], [2.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]];
  let calls = 0;
  const r = compact(3, start, {
    budgetMs: 1500, seed: 4,
    onImprove(s, squares) { calls++; assert.ok(verify(s, squares, 1e-9).ok); },
  });
  assert.ok(verify(r.s, r.squares, 1e-9).ok);
  assert.ok(r.s < 2.9, `compaction should make progress from s = 3: ${r.s}`);
  assert.ok(calls > 0 && r.stats.localOpts > 0);
  ok(`n = 5 from s = 3: s = ${r.s.toFixed(10)} (record ${s5.toFixed(10)}), ${calls} improvements`);
}

// --------------------------------------------------------------------------- //
section('search(10) from a jittered grid at s = 4 (8 s budget)');
{
  const t = now();
  let improvements = 0, lastS = Infinity;
  const r = search(10, {
    s0: 4, budgetMs: 8000, seed: 1,
    onImprove(s, squares) {
      improvements++;
      assert.ok(s < lastS, 'improvements must be monotone');
      assert.ok(verify(s, squares, 1e-9).ok, 'onImprove must only see verified packings');
      lastS = s;
    },
  });
  const dt = now() - t;
  assert.ok(verify(r.s, r.squares, 1e-9).ok, 'result verifies');
  assert.ok(r.s <= 3.75, `n = 10 should reach s <= 3.75, got ${r.s}`);
  assert.ok(improvements > 0 && Math.abs(lastS - r.s) < 1e-12);
  results.push(`search n=10, 8 s, seed 1: s = ${r.s.toFixed(10)} (record 3.7071067812), ${r.stats.localOpts} local opts, ${improvements} improvements, ${(dt / 1000).toFixed(1)} s`);
  ok(results[results.length - 1]);
}

// --------------------------------------------------------------------------- //
section('worker protocol (Node worker_threads)');
{
  const url = new URL('../js/optimizer.worker.js', import.meta.url);
  const runWorker = (msg, stopAfterMs = 0) => new Promise((resolve, reject) => {
    const w = new Worker(url);
    const got = { improve: 0, progress: 0, done: null, error: null };
    const timer = setTimeout(() => { w.terminate(); reject(new Error('worker timed out')); }, 20000);
    w.on('message', (m) => {
      if (m.type === 'improve') { got.improve++; assert.ok(verify(m.s, m.squares).ok); assert.ok(Array.isArray(m.squares)); }
      else if (m.type === 'progress') { got.progress++; assert.ok(typeof m.elapsedMs === 'number' && typeof m.localOpts === 'number' && typeof m.bestS === 'number'); }
      else if (m.type === 'done') { got.done = m; clearTimeout(timer); w.terminate().then(() => resolve(got)); }
      else if (m.type === 'error') { got.error = m; clearTimeout(timer); w.terminate().then(() => reject(new Error(m.message))); }
    });
    w.on('error', (e) => { clearTimeout(timer); reject(e); });
    w.postMessage(msg);
    if (stopAfterMs > 0) setTimeout(() => w.postMessage({ type: 'stop' }), stopAfterMs);
  });

  // full run without an initial packing (jittered grid) and a short budget
  const t1 = now();
  const g1 = await runWorker({ type: 'start', n: 6, s: 3, budgetMs: 1500, seed: 3 });
  assert.ok(g1.done && verify(g1.done.s, g1.done.squares).ok, 'done packing verifies');
  assert.ok(g1.done.s <= 3 + 1e-9, `n = 6 must not be worse than the grid: ${g1.done.s}`);
  assert.ok(g1.progress >= 3, `expected progress messages, got ${g1.progress}`);
  assert.ok(g1.done.stats.localOpts > 0);
  ok(`n = 6, 1.5 s: s = ${g1.done.s.toFixed(8)}, ${g1.improve} improve, ${g1.progress} progress, ${g1.done.stats.localOpts} local opts, wall ${(now() - t1).toFixed(0)} ms`);

  // start from a supplied valid packing (the exact n = 5 packing), then stop early
  const t2 = now();
  const g2 = await runWorker({ type: 'start', n: 5, s: s5, squares: five.flat(), budgetMs: 15000, seed: 2 }, 600);
  const wall = now() - t2;
  assert.ok(g2.done && g2.done.stats.stopped === true, 'stop must be acknowledged');
  assert.ok(wall < 3000, `stop should take effect quickly (took ${wall.toFixed(0)} ms)`);
  assert.ok(verify(g2.done.s, g2.done.squares).ok && g2.done.s <= s5 + 1e-9, 'never worse than the input');
  ok(`n = 5 from the exact packing, stopped after 600 ms: s = ${g2.done.s.toFixed(10)}, wall ${wall.toFixed(0)} ms`);
}

console.log('\nAll tests passed.');
for (const line of results) console.log('  ' + line);
