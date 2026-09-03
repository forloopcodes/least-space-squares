/**
 * optimizer.worker.js -- module Web Worker running the packcore search on the
 * client's CPU.  Create it with `new Worker(url, { type: 'module' })`.
 *
 * Protocol (main thread -> worker)
 *   { type: 'start', n, s, squares, budgetMs, seed }
 *       `squares` is a flat array [x0, y0, t0, ...] (or [x, y, t] rows) that
 *       must be valid at side `s`; when it is missing the search starts from a
 *       jittered grid at `s` (default ceil sqrt n).  A slightly invalid packing
 *       is repaired / made feasible first.
 *   { type: 'stop' }
 *       Cooperative abort: the flag is checked between local optimisations
 *       (each at most 250 L-BFGS iterations) and the worker yields to its event
 *       loop every ~40 ms so the message can actually arrive.  The best packing
 *       so far is then reported with a 'done' message.
 *
 * Protocol (worker -> main thread)
 *   { type: 'improve', s, squares }               every verified improvement (flat array)
 *   { type: 'progress', elapsedMs, localOpts, bestS }   about 4 times per second
 *   { type: 'done', s, squares, stats }           at the end (budget spent or stopped)
 *   { type: 'error', message }                    on a bad request or an exception
 *
 * The file also runs under Node's worker_threads (used by the test-suite).
 */
import { searchGen, newStats, now } from './packcore.js';

let stopFlag = false;
let running = false;
let post = () => {};

const PROGRESS_MS = 250;   // progress cadence
const YIELD_MS = 40;       // how long to compute before letting messages through

const yieldToEventLoop = () => new Promise((resolve) => setTimeout(resolve, 0));

function handle(msg) {
  if (!msg || typeof msg !== 'object') return;
  if (msg.type === 'stop') { stopFlag = true; return; }
  if (msg.type !== 'start') return;
  if (running) {
    post({ type: 'error', message: 'a search is already running; send {type: "stop"} first' });
    return;
  }
  running = true;
  stopFlag = false;
  run(msg)
    .catch((e) => post({ type: 'error', message: String((e && e.message) || e) }))
    .finally(() => { running = false; });
}

async function run(msg) {
  const n = msg.n | 0;
  if (!(n >= 1)) throw new Error('n must be a positive integer');
  const budgetMs = Number.isFinite(+msg.budgetMs) ? +msg.budgetMs : 10000;
  const seed = msg.seed == null ? 1 : msg.seed;
  const s0 = Number.isFinite(+msg.s) ? +msg.s : null;
  const squares0 = msg.squares == null ? null : msg.squares;
  const stats = newStats();
  const t0 = now();
  let bestS = s0 != null ? s0 : Math.ceil(Math.sqrt(n) - 1e-9);
  const gen = searchGen(n, {
    s0, squares0, budgetMs, seed, stats,
    onImprove(s, squares) {
      bestS = s;
      post({ type: 'improve', s, squares: Array.from(squares) });
    },
    shouldStop: () => stopFlag,
  });
  let lastProgress = t0, lastYield = t0;
  let result;
  for (;;) {
    const r = gen.next();        // one local optimisation per step
    if (r.done) { result = r.value; break; }
    const t = now();
    if (t - lastProgress >= PROGRESS_MS) {
      post({ type: 'progress', elapsedMs: t - t0, localOpts: stats.localOpts, bestS });
      lastProgress = t;
    }
    if (t - lastYield >= YIELD_MS) {
      await yieldToEventLoop();
      lastYield = now();
    }
  }
  post({
    type: 'done',
    s: result.s,
    squares: Array.from(result.squares),
    stats: { ...stats, elapsedMs: now() - t0, stopped: stopFlag, history: result.history },
  });
}

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  // browser: register synchronously so no early message is lost
  post = (m) => self.postMessage(m);
  self.onmessage = (ev) => handle(ev.data);
} else {
  // Node worker_threads (tests); never reached in a browser
  import('node:worker_threads').then(({ parentPort }) => {
    post = (m) => parentPort.postMessage(m);
    parentPort.on('message', handle);
  });
}
