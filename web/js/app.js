import { bestAnalytic, grid, gridSide, canonicalAngle } from './families.js';
import { bestKnown, recordEntry } from './records.js';
import { CACHE } from './cache.js';
import { Renderer } from './render.js';

let core = null;   // packcore.js (verify/repair + optimizer) is optional: the page works without it
try { core = await import('./packcore.js'); } catch (e) { console.warn('packcore.js unavailable, verification/refine disabled', e); }

const METHODS = [
  ['best', 'best available (cache, then closed forms)'],
  ['closed', 'closed forms only (Göbel strips / squares + L)'],
  ['grid', 'grid ⌈√n⌉'],
];

function fmt(v, d = 10) { return v === null || v === undefined ? '—' : Number(v).toFixed(d).replace(/\.?0+$/, ''); }

function fromCache(n) {
  const c = CACHE[n]; if (!c) return null;
  return { n, s: c.s, squares: Float64Array.from(c.squares), method: c.method, exact: c.exact };
}
function lookAheadCache(n) {   // s(n) <= s(m) for m > n: a cached packing of m squares minus m - n of them
  let best = fromCache(n);
  for (let m = n + 1; m <= n + 12; m++) { const c = fromCache(m); if (c && (!best || c.s < best.s - 1e-12)) best = { ...c, n, squares: c.squares.slice(0, 3 * n), method: c.method + ` (n=${m} minus ${m - n})` }; }
  return best;
}

function compute(n, method) {
  const t0 = performance.now();
  let p;
  if (method === 'grid') p = grid(n);
  else if (method === 'closed') p = bestAnalytic(n);
  else { const a = bestAnalytic(n), c = n <= 400 ? lookAheadCache(n) : null; p = (c && c.s < a.s - 1e-12) ? c : a; }
  p.ms = performance.now() - t0;
  return p;
}

class Panel {
  constructor(root, title, method) {
    this.root = root; this.method = method; this.worker = null;
    root.innerHTML = `
      <div class="bar">
        <strong>${title}</strong>
        <select class="method">${METHODS.map(([k, v]) => `<option value="${k}" ${k === method ? 'selected' : ''}>${v}</option>`).join('')}</select>
        <label>n <input class="n" type="number" min="1" max="5000000" value="26"></label>
        <button class="run">Compute</button>
        <button class="refine" ${core ? '' : 'disabled title="packcore.js not loaded"'}>Refine on CPU</button>
        <label>budget <select class="budget"><option value="5">5 s</option><option value="15" selected>15 s</option><option value="60">60 s</option><option value="300">5 min</option></select></label>
        <button class="stop" disabled>Stop</button>
        <button class="fit">Fit</button>
        <button class="dl">JSON</button>
      </div>
      <canvas></canvas>
      <div class="prog"><div></div></div>
      <div class="stats"></div>`;
    this.$ = sel => root.querySelector(sel);
    this.renderer = new Renderer(this.$('canvas'));
    this.$('.run').onclick = () => this.compute();
    this.$('.method').onchange = () => { this.method = this.$('.method').value; this.compute(); };
    this.$('.n').onkeydown = e => { if (e.key === 'Enter') this.compute(); };
    this.$('.refine').onclick = () => this.refine();
    this.$('.stop').onclick = () => this.stop();
    this.$('.fit').onclick = () => this.renderer.fit();
    this.$('.dl').onclick = () => this.download();
  }
  get n() { return Math.max(1, Math.min(5000000, Math.floor(Number(this.$('.n').value) || 1))); }
  set n(v) { this.$('.n').value = v; }
  compute() {
    this.stop();
    const n = this.n;
    this.p = compute(n, this.method);
    this.show('closed form');
  }
  show(source) {
    const p = this.p, n = p.n;
    let ver = null;
    if (core && n <= 20000) { const t = performance.now(); ver = core.verify(p.s, p.squares, 1e-9); ver.ms = performance.now() - t; }
    this.renderer.setPacking(p.s, p.squares);
    const bk = bestKnown(n), rec = recordEntry(n), gap = bk === null ? null : p.s - bk;
    const rows = [
      ['n', n.toLocaleString()],
      ['s', `<span class="s-value">${fmt(p.s, 12)}</span>`],
      ['method', `${p.method}${p.exact ? ' · ' + p.exact : ''}`],
      ['lower bound √n', fmt(Math.sqrt(n), 6)],
      ['density n/s²', fmt(n / (p.s * p.s), 5)],
      ['best known', bk === null ? '— (table ends at 324)' : `${fmt(bk, 10)}${rec ? ' · ' + rec.exact : ''}`],
      ['gap', gap === null ? '—' : `<span class="${gap <= 1e-9 ? 'gap-ok' : 'gap-bad'}">${gap <= 1e-9 ? 'matches the record' : '+' + fmt(gap, 6)}</span>`],
      ['time', `${fmt(p.ms, 1)} ms (${source})`],
      ['verified', ver ? (ver.ok ? `<span class="gap-ok">valid</span> · max penetration ${ver.maxPenetration.toExponential(1)} · ${fmt(ver.ms, 1)} ms` : `<span class="gap-bad">INVALID</span>`) : (core ? 'skipped (n > 20000)' : 'n/a')],
      ['renderer', `${this.renderer.mode}, ${n.toLocaleString()} instances`],
    ];
    this.$('.stats').innerHTML = rows.map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join('');
  }
  refine() {
    if (!core || !this.p) return;
    if (this.p.n > 400) { alert('The numerical refinement is meant for n ≤ 400 (it is O(n) per step but needs many steps).'); return; }
    this.stop();
    const budgetMs = Number(this.$('.budget').value) * 1000, t0 = performance.now();
    this.worker = new Worker(new URL('./optimizer.worker.js', import.meta.url), { type: 'module' });
    this.$('.stop').disabled = false; this.$('.refine').disabled = true;
    const bar = this.$('.prog > div');
    this.worker.onmessage = ({ data }) => {
      if (data.type === 'improve') {
        const sq = Float64Array.from(data.squares);
        const v = core.verify(data.s, sq, 1e-9);
        if (v.ok && data.s < this.p.s - 1e-12) { this.p = { n: this.p.n, s: data.s, squares: sq, method: 'numeric refinement', exact: '', ms: performance.now() - t0 }; this.show('worker'); }
      } else if (data.type === 'progress') {
        bar.style.width = `${Math.min(100, 100 * data.elapsedMs / budgetMs)}%`;
      } else if (data.type === 'done') {
        this.stop();
      }
    };
    this.worker.onerror = e => { console.error(e); this.stop(); };
    for (let i = 0; i < this.p.squares.length; i += 3) this.p.squares[i + 2] = canonicalAngle(this.p.squares[i + 2]);
    this.worker.postMessage({ type: 'start', n: this.p.n, s: this.p.s, squares: Array.from(this.p.squares), budgetMs, seed: (Date.now() % 100000) });
  }
  stop() {
    if (this.worker) { this.worker.postMessage({ type: 'stop' }); this.worker.terminate(); this.worker = null; }
    this.$('.stop').disabled = true; this.$('.refine').disabled = !core; this.$('.prog > div').style.width = '0';
  }
  download() {
    if (!this.p) return;
    const rows = []; for (let i = 0; i < this.p.n; i++) rows.push([this.p.squares[3 * i], this.p.squares[3 * i + 1], this.p.squares[3 * i + 2]]);
    const blob = new Blob([JSON.stringify({ n: this.p.n, s: this.p.s, method: this.p.method, exact: this.p.exact, angle_unit: 'radians', squares: rows })], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `packing-${this.p.n}.json`; a.click();
  }
}

const left = new Panel(document.getElementById('left'), 'A', 'best');
const right = new Panel(document.getElementById('right'), 'B', 'closed');
const nInput = document.getElementById('n'), sync = document.getElementById('sync');
function runBoth() {
  const n = Math.max(1, Math.min(5000000, Math.floor(Number(nInput.value) || 1)));
  nInput.value = n;
  if (sync.checked) { left.n = n; right.n = n; }
  left.compute(); right.compute();
}
document.getElementById('go').onclick = runBoth;
nInput.addEventListener('keydown', e => { if (e.key === 'Enter') runBoth(); });
document.getElementById('env').textContent = `${left.renderer.mode === 'webgl2' ? 'GPU: WebGL2 instancing' : 'GPU: unavailable, Canvas2D fallback'} · CPU: ${navigator.hardwareConcurrency || '?'} threads · optimizer ${core ? 'ready' : 'not loaded'}`;
const q = new URLSearchParams(location.search);
if (q.get('n')) nInput.value = q.get('n');
runBoth();
window.squarepack = { left, right, bestAnalytic, grid, gridSide, bestKnown, CACHE };
