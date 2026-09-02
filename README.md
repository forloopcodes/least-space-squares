# squarepack — tightest packings of *n* unit squares in a square

**Input:** `n` (number of unit squares) → **intermediate:** `s` (side of the smallest
enclosing square found) → **output:** `[[x_center, y_center, angle], …]` for all `n` squares.

```python
from squarepack import pack
s, squares = pack(11)            # s = 3.8770836…, squares = [[x, y, angle_rad], ...] (11 rows)
s, squares = pack(11, degrees=True)
s, squares = pack(41, time_budget=60)   # spend 60 s of numerical search on top of the analytic result
```

```text
$ python -m squarepack 26
n = 26
s = 5.621320343560   (tilted 3x3 block at 45deg, offset (+0,+0.5))  best known: 5.6213203436  lower bound sqrt(n) = 5.099020
   0  x=0.500000000000  y=0.500000000000  angle(rad)=0.000000000000
   ...
$ python -m squarepack 41 --budget 120 --json --svg 41.svg --save
```

Coordinates live in the container `[0, s] × [0, s]` (origin at the bottom-left corner),
angles are counter-clockwise, reduced to `[-π/4, π/4)` by the 4-fold symmetry of a square.
Every packing returned by the library has passed `squarepack.verify` (no overlaps, nothing
outside the container, tolerance 1e-9) — the verifier is an exact separating-axis test, not a
sampling test.

Install: `pip install numpy scipy` (Python ≥ 3.9). A C compiler (gcc/clang) is optional but makes
the numerical search ~25× faster; the pure-numpy path is used automatically otherwise.
Tests: `python -m pytest -q tests`.

---

## 1. The problem and what "optimal" can mean

`s(n)` = side of the smallest square that contains `n` non-overlapping unit squares (rotation
allowed). Known facts used by the solver:

* **Lower bound** `s(n) ≥ √n` (area). Together with monotonicity this settles `s(k²) = k`.
* `s(k²−1) = s(k²−2) = k` for all `k` (Nagamochi 2005), and `s(n) = ⌈√n⌉` is proven for
  n ≤ 10, 13–16, 22–25, 33–36, 46–49, … (Göbel, Friedman, Kearney–Shiu, Stromquist, Bentz).
* `s(5) = 2 + 1/√2` and `s(10) = 3 + 1/√2` are the only proven non-trivial optima.
* Deciding the general problem is NP-hard (it contains the square packing decision problem), and
  the best known packings for most `n` come from decades of human insight plus large computer
  searches (Friedman's survey; records maintained by D. Ellsworth). Wasted area for the best
  constructions grows like `O(s^0.6)` (Erdős–Graham / Chung–Graham) and must grow at least like
  `Ω(√(s·|s − round(s)|))` (Roth–Vaughan), so no construction can be perfect for all `n`.

Therefore "the most optimized algorithm" is a **portfolio**: closed-form constructions that are
exact and linear-time where a structure is known, a fast structural search that discovers most of
the record structures itself, and a numerical optimiser as the general fallback — every candidate
verified, the smallest side wins. The literature table (`squarepack.known`) is used only to
*measure* the solver, never as a source of coordinates.

## 2. Methods (iterated and benchmarked against each other)

| # | method | idea | cost | where it wins |
|---|--------|------|------|---------------|
| 1 | **grid** | `s = ⌈√n⌉` | O(n) | n = k², k²−1, k²−2 and many small n (proven optimal there) |
| 2 | **Göbel strip** | 45° strip of `b = 1+⌊(a−1)√2⌋` squares on the diagonal, two corner squares, two staircases of `a` steps: `n = a(a+1)+2+b`, `s = a+1+√2/2` | O(n) | 27, 38, 52, 67, 84, 104, 125, 149, 174, 201, 231, 262, … (all records) |
| 3 | **Göbel square** | `b×b` block at 45° in the centre, four staircases of `a` steps, valid iff `(a−1)√2 < b < (a+1)√2`: `n = 2a(a+1)+b²`, `s = a+1+b√2/2` | O(n) | 5, 40, 65, 89, … |
| 4 | **L extension** | any packing of side `s` → side `s+1` with `⌊s+1⌋+⌊s⌋` more squares | O(n) | 82, 101, 122, 145, … (records "add L's to s(65)") |
| 5 | **tilted block + row fill** | a `p×q` block at 45° at offset `(dx,dy)` (multiples of ¼) from the centre; the rest filled by unit rows anchored to the bottom or top wall with an independent seam on each side of the block; exact `s` by vectorised bisection | O(s · #offsets · log 1/ε) ≈ 0.1–10 s | reproduces 2–4 **and** the Friedman/Stenlund records 26 (`7/2+3√2/2`), 66 (`3+4√2`), 85 (`11/2+3√2`), 150, 202; near-records for 18, 19, 37, 50, 53, 54, 102, 123 |
| 6 | **numerical compaction** | penalty energy `E = Σ max(0,p_ij)² + Σ max(0,v_i)²` (separating-axis penetration depth `p_ij`, protrusion `v_i`), analytic gradient, L-BFGS in C; shrink `s` by bisection, basin hopping when stuck, exact repair at the end | budgeted (thousands of local optimisations per minute) | 11 (Trump's record from random seeds in < 1 min), 18, 19, 29, 41, … |
| 6b | **penalty continuation** | `s` as a variable, minimise `s + μE` for `μ → ∞` | budgeted | faster descent, less precise; used as a variant of 6 |
| 7 | **√7 family** (`families.py`) | staircases anchored at opposite corners plus a band of 1×2 dominoes at angle `arctan((4−√7)/3)`, each squeezed between two staircase corners exactly 2 apart: `s = m − 1/2 + √7/2` | O(n) | 18, 53, 86, 127, 151, 176, 204, 234, 299 (all records of this form) |
| 8 | **DeVincentis family** | `s = 7 − √2/2 + √(1+√2) + 3k`, `n = 9k² + 44k + 54` | O(n) | 54, 107, 178, 267, … |
| 9 | **special exact packings** | Wainwright's 19 (`3 + 4√2/3`), Schadt's 50 (`7 + 4/7`) and 171 (`13 + 4/7`), obtained by transcribing and generalising the published drawings (`scripts/svg_to_packing.py` extracts exact coordinates from the record SVGs) | O(n) | 19, 50, 171, 198 |

### 2.1 Geometry kernel (`squarepack/geometry.py`)

Two unit squares with centres `c_i, c_j` and angles `θ_i, θ_j` are separated iff on one of the
four edge directions `a ∈ {e1_i, e2_i, e1_j, e2_j}` the projected centre distance exceeds the
sum of half-extents. Because a square's edge normals are its own edge directions, that sum is
the same on all four axes, `½ + (|cos φ| + |sin φ|)/2`, `φ = θ_j − θ_i`, so the whole test is

    p_ij = ½ + (|cos φ| + |sin φ|)/2 − max_a |a · (c_i − c_j)|      (overlap ⇔ p_ij > 0).

`verify` evaluates this for every candidate pair (cell grid, expected O(n)) and the container
constraint `w(θ) ≤ x ≤ s − w(θ)` with `w(θ) = (|cos θ| + |sin θ|)/2`. `repair` computes the
*minimal* scale factor `λ = max_ij (½ + R_ij)/M_ij` (and the container analogue) that removes
every overlap, which turns a numerically converged configuration into an exactly valid packing
with a certified `s`.

### 2.2 Closed-form families (`constructions.py`)

Both Göbel families and the L extension are generated directly from their formulas; the
constructions are ordered so that removing the last squares (for `n` below the family capacity)
keeps the structure. All family members up to `s ≈ 40` are checked by the verifier in the tests.

### 2.3 Tilted-block search (`blocks.py`)

The block is a convex polygon; clipping it to a horizontal unit band gives an interval
`[xl, xr]`, and the band then holds `⌊xl⌋` squares from the left wall and `⌊s − xr⌋` from the
right wall. Bands entirely above/below the block are full rows. Bottom-anchored bands `[j, j+1]`
and top-anchored bands `[s−j−1, s−j]` meet at a seam; because the block separates the two sides,
the left and right sides may choose different seams — this is exactly what makes Göbel's strips
(staircases anchored at *opposite* corners) expressible, and what produces the "L" fillers of
the 26/66/85 records. The count is exact for the chosen structure and is evaluated for all
offsets at once in numpy; the smallest `s` per shape/offset comes from a 42-step bisection.
Every result is passed through `repair` and verified.

The engine was then generalised (`results/blocks_ext.md`): the obstacle may be a *union* of convex
pieces (two half-turn-symmetric blocks, S/W-bent strips with corner pivots, row-shifted "sheared"
blocks, chains of parallel rows with per-row tilt kept in exact contact), the block angle is free,
and all candidates of a family are bisected at once (batched branch-and-bound, ~10× faster: every
n ≤ 300 in under 10 s). Fed with the exact tilted squares of the published drawings the fill
engine reproduces the fills of the 18, 19, 37, 50, 54, 70, 87, 88 and 123 records, but rigid
approximations of those structures are never better than the plain 45° block, so the search
itself did not gain new record matches — the missing records need several coordinated continuous
moves, which is what the numerical search is for.

### 2.4 Numerical search (`optimize.py`, `_fastcore.c`)

* Energy and gradient are piecewise smooth; the gradient is derived analytically (checked
  against finite differences to 1e-9 in the tests) and implemented both in numpy and in C.
* `local_opt` = L-BFGS (history 10, Armijo backtracking, per-step displacement cap) with a
  cell-list pair rebuild whenever a square has moved more than `(cutoff − √2)/2`.
* `compact` = shrink `s` by a step, rescale, re-optimise; success ⇒ keep and enlarge the step,
  failure ⇒ up to `hops` basin-hopping perturbations (jitter, kick the most-overlapping squares to
  random places, rotate, swap, move the worst square into the emptiest spot, global shake), then
  halve the step; the fine phase below 5e-4·s is a pure bisection.
* `search` = compact the incumbent (¼ of the budget), then cycle through seed types
  (analytic member, analytic member with random squares dropped, jittered grid, uniform random,
  and *block seeds*: tilted-block members for n..n+2 with random squares dropped and re-inserted
  into the largest holes), each made feasible by enlarging `s` if necessary and then compacted.
  Improvements are accepted only if they pass `verify` after `repair`.
* Variants that were implemented and benchmarked against each other (`results/search_tuning.md`):
  a C simulated-annealing kernel (`strategy="anneal"` / `"anneal-hop"`, single-square Metropolis
  moves with container shrink, ~2 M moves/s), an extended basin-hopping move set
  (`perturb_mix="extended"`: reinsert-into-hole, swap+rotate, row/column shifts, 45° cluster
  rotation) and an angle-snapping polish (`snap=True`). At 30–60 s budgets only the block seeds
  are a consistent win (mean gap to the records on the hard set 0.037 → 0.027), so they are the
  default; the others stay available for long runs.

## 3. Complexity

* Closed-form constructions: Θ(n) — optimal, the output has n triples. Selecting the best
  family member for `n` is O(√n) arithmetic.
* Tilted-block search: O(#shapes · #offsets · ⌊s⌋ · 42) vectorised; ≈ 0.1 s (n ≈ 10) to ≈ 10 s
  (n ≈ 300) on one core.
* Verification: O(n) expected (cell grid), O(n²) worst case for pathological inputs.
* Numerical search: one local optimisation costs O(iterations · (pairs + n)), pairs = O(n) with
  the cell list; the number of local optimisations is set by the time budget. The problem being
  NP-hard, no polynomial-time exact algorithm is expected; the budgeted search is the general
  fallback and the cache (`data/best_packings.json`) makes repeated queries O(n).

`pack(n)` therefore costs O(n) for cached / closed-form `n`, a few seconds when the block search
runs, and exactly the requested budget when a numerical search is requested.

## 4. Results

See `results/benchmark.md` (generated by `scripts/benchmark.py`) for the per-`n` table comparing
every method with the best known value from the literature, and `results/svg/index.html`
(`scripts/gallery.py`) for drawings. Summary numbers are reproduced in the last section of this
README after each benchmark run.

## 5. Reproducing / extending

```bash
python -m pytest -q tests                                    # unit tests (geometry, families, optimiser)
python scripts/benchmark.py --n-max 100 --budget 60 --save   # full comparison, caches improvements
python -m squarepack 41 --budget 300 --save                  # longer search for one n
python scripts/gallery.py 11 18 26 41                        # SVG drawings
```

## 6. References

* E. Friedman, *Packing Unit Squares in Squares: A Survey and New Results*, Electron. J. Combin.,
  Dynamic Survey DS7. Records maintained by D. Ellsworth at
  <https://kingbird.myphotos.cc/packing/squares_in_squares.html> (source of `squarepack/known.py`,
  used for benchmarking only).
* F. Göbel (1979) — the strip and square families; H. Nagamochi (2005) — `s(k²−1) = s(k²−2) = k`;
  P. Erdős & R. Graham (1975), F. Chung & R. Graham (2020), K. F. Roth & R. C. Vaughan (1978) — wasted-area bounds.
* The compaction/basin-hopping scheme follows the standard approach for dense packings (cf.
  Gensane & Ryckelynck 2005 for squares, and Packomania-style circle packing).

## 7. Benchmark summary (n = 1..100, 60 s numerical budget per n, 2 workers)

Generated by `scripts/benchmark.py`; full table in `results/benchmark.md`.

| stage of the portfolio | values of n matching the best known side |
|---|---:|
| grid only | 64 / 100 |
| + closed-form families (Göbel strips/squares, L extensions, √7, DeVincentis, 19, 50) | 81 / 100 |
| + tilted-block search | 84 / 100 |
| + numerical compaction (60 s) | 84 / 100 |

* mean gap to the best known side over all 100 values: 0.00352
* every returned packing passes the exact verifier (max penetration ≤ 1e-9)
* time: closed-form families 2.2 s total, tilted-block search 204 s total, numerical search 60 s for each non-proven n
* the numerical search alone rediscovers Trump's n = 11 record (3.8770836) from random starts, lands within
  4·10⁻⁵ of the n = 18 record (7/2 + √7/2, now also produced exactly by the √7 family) and reaches
  4.6782 for n = 17 (Bidwell's record 4.6755 uses three distinct angles)

Values of n (≤ 100) still above the literature record after this budget, with the gap:

| n | found | best known | gap | note |
|---:|---:|---:|---:|:--|
| 11 | 3.877090 | 3.877084 | +0.000006 | numeric:bisect:drop |
| 17 | 4.678228 | 4.675530 | +0.002698 | numeric:bisect:random |
| 28 | 5.828427 | 5.824445 | +0.003983 | gobel_square |
| 29 | 6.000000 | 5.933833 | +0.066167 | grid |
| 37 | 6.621320 | 6.598620 | +0.022701 | tilted_block |
| 39 | 6.822876 | 6.810722 | +0.012154 | sqrt7 |
| 41 | 6.949747 | 6.926693 | +0.023054 | tilted_block |
| 51 | 7.707107 | 7.700799 | +0.006308 | gobel_strip+L+L |
| 55 | 8.000000 | 7.945771 | +0.054229 | grid |
| 68 | 8.822876 | 8.803460 | +0.019416 | sqrt7+L |
| 69 | 8.846667 | 8.827212 | +0.019455 | devincentis+L |
| 70 | 8.904291 | 8.881667 | +0.022624 | numeric:bisect:block |
| 71 | 9.000000 | 8.944072 | +0.055928 | grid |
| 83 | 9.656854 | 9.634826 | +0.022029 | tilted_block |
| 87 | 9.852139 | 9.838817 | +0.013322 | numeric:tilted_block |
| 88 | 9.895934 | 9.888153 | +0.007781 | numeric:numeric:tilted_block |

Over n = 1..324 the analytic part alone (closed forms + tilted-block search, no numerical search)
matches 244 of the 324 record values — 235 from the closed-form families, 9 more from the block
engine (`results/block_sweep.md`; most n are settled by the closed forms in milliseconds, the block
search takes 1–7 s for n ≤ 250 and up to 33 s near n = 300, on one core). The 80
remaining records are irregular packings found by multi-day simulated-annealing runs (Trump's 11,
Bidwell's 17, Schadt's and Ellsworth's 2024–2026 records) whose structure no closed form captures.
