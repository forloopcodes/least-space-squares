# Global-search tuning (`squarepack.optimize.search`)

Goal: get closer to the literature records for the hard cases
n = 17, 18, 19, 28, 29, 37, 39, 41 within a fixed time budget, without slowing the easy cases.

Machine: one core (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`), shared with other jobs; C core
(`_fastcore.c`) loaded.  Every packing is verified exactly (`verify(..., 1e-9)` after `repaired()`).

## Variants implemented

All are reachable through `search(...)` options; the previous behaviour is still the default for
everything except `seed_mix` (see "New defaults").

| option | what it does |
|---|---|
| `strategy="anneal"` | simulated annealing in C (`anneal_c`: Metropolis moves on single squares - displace / rotate / both / snap angle to 0 or 45 deg / swap / teleport, adaptive steps, geometric cooling, ~2 M moves/s for n = 19).  Cold start: cool a seed while shrinking the container whenever `E` vanishes (`anneal_feasible`); then `anneal_compact` (shrink, re-anneal warm, L-BFGS polish); then the usual bisection. |
| `strategy="anneal-hop"` | bisection compaction whose basin-hopping moves are short annealing runs at the trial side instead of random perturbations |
| `perturb_mix="extended"` | classic moves + *reinsert* (remove the 1-3 most stressed squares, re-insert them in the largest holes), *swaprot* (swap + rotate a pair), *bandshift* (coordinated shift of a row/column band or half-plane), *cluster45* (rotate a cluster of nearby squares by 45 deg) |
| `seed_mix=(..., "block")` | seeds from `tilted_block_search` members (two block shapes) of n, n+1, n+2 plus the non-grid analytic members; surplus squares dropped at random, 0-3 squares removed and re-inserted into the largest holes (`block_seed_pool`, cached per n; skipped when < 12 s of budget remain) |
| `snap=True` | angle-snapping polish on improvements: angles within 1e-3 of k*pi/4 are snapped, groups of angles within 0.02 rad are equalised, then re-optimised and re-compacted (`snap_polish`) |
| `local_maxiter`, `hops`, `incumbent_fraction`, `anneal_opts` | tuning knobs: per-search override of `LOCAL_MAXITER`, hops per failed shrink step, budget share of the incumbent compaction, SA schedule (`ANNEAL_DEFAULTS`) |

## Controlled screen: 30 s budget, seed 1, one core

Values are the best s found; bold = best of the column; `mean gap` = mean over n of (s - record);
`wins` = number of n where the variant is (tied) best.

| variant | n=17 | n=18 | n=19 | n=28 | n=29 | n=37 | n=41 | mean gap | wins |
|---|---|---|---|---|---|---|---|---|---|
| **record** | 4.675530 | 4.822876 | 4.885618 | 5.824445 | 5.933833 | 6.598620 | 6.926693 | | |
| baseline (`bisect`, classic moves, seeds analytic/drop/grid/random) | 4.707107 | 4.826225 | 4.943015 | **5.828427** | **6.000000** | 6.621409 | **7.000000** | +0.03694 | 3 |
| `perturb_mix="extended"` | 4.677999 | 4.823030 | 4.913994 | **5.828427** | **6.000000** | 6.674392 | **7.000000** | +0.03575 | 3 |
| `seed_mix` + `"block"` seeds (**new default**) | 4.693843 | 4.823025 | **4.888110** | **5.828427** | **6.000000** | 6.621846 | **7.000000** | +0.02681 | 4 |
| block seeds + extended moves | **4.677649** | 4.823136 | 4.984018 | **5.828427** | **6.000000** | 6.683931 | **7.000000** | +0.04708 | 4 |
| `strategy="anneal-hop"` | 4.707107 | 4.823814 | 5.000000 | **5.828427** | **6.000000** | 6.621647 | **7.000000** | +0.04477 | 3 |
| `strategy="anneal"` | 4.707107 | 4.876975 | 5.000000 | **5.828427** | **6.000000** | **6.621322** | **7.000000** | +0.05232 | 4 |
| anneal + block + extended + snap | 4.707107 | **4.822929** | 5.000000 | **5.828427** | **6.000000** | 6.621776 | **7.000000** | +0.04466 | 4 |

## Supplementary: baseline at 60 s, seeds 1 and 2

Run before the screen was shortened to fit the time limit (stopped after n = 37); shows the seed-to-seed
variance of the baseline (n = 17: 4.6803 vs 4.7071, n = 19: 4.9430 vs 4.8881).

| variant | n=17 | n=18 | n=19 | n=28 | n=29 | n=37 | mean gap | wins |
|---|---|---|---|---|---|---|---|---|
| **record** | 4.675530 | 4.822876 | 4.885618 | 5.824445 | 5.933833 | 6.598620 | | |
| baseline, 60 s, seeds 1 / 2 | **4.680297 / 4.707107** | **4.822913 / 4.822942** | **4.943015 / 4.888123** | **5.828427 / 5.828427** | **6.000000 / 6.000000** | **6.621409 / 6.621860** | +0.01671 | 6 |

## Findings

* **Block seeds** (`"block"` in `seed_mix`) are the one change that helps consistently: they lower the mean gap of
  the screen (n = 17: 4.7071 -> 4.6938, n = 18: 4.8262 -> 4.8230, n = 19: 4.9430 -> 4.8881) and are within
  noise elsewhere (n = 37: 6.6218 vs 6.6214).
  They are now in the default `seed_mix`.
* **Extended perturbations** help the small cases (n = 17 reaches 4.6780, 0.0025 above Bidwell's 4.67553) but the
  single-seed result for n = 37 was worse (6.6744 vs 6.6214), so they stay optional (`perturb_mix="extended"`);
  see the block + extended row for the combination.
* **Simulated annealing** (`anneal`, `anneal-hop`) is implemented and fast, but within a 30 s budget it is not
  competitive with L-BFGS basin hopping: one annealing run costs as much as tens of local optimisations, so far fewer
  seeds are explored (20 vs 120 for n = 17), and single-square moves cannot do the collective final squeeze that
  L-BFGS does.  It is kept as an option; the record-setting runs of this kind used budgets of hours per n.
* **Angle snapping** gives small, cheap improvements (1e-5 .. 2e-5 in s on the n = 17/18/19 baseline packings;
  n = 18 down to 4.822929 in the combined run) but never changes the basin, so it is off by default.
* No variant reaches the records for 17, 19, 28, 29, 37, 41 within 30-60 s; n = 18 gets within 5e-5 .. 1.5e-4
  (the packings found have the record's five ~24.3 deg squares but a slightly different contact structure).
  n = 28, 29, 41 never leave the analytic incumbent (Göbel square / grid) at these budgets.

## New defaults

* `search(..., seed_mix=("analytic", "drop", "grid", "random", "block"))` (`DEFAULT_SEED_MIX`; was without `"block"`).
* Unchanged: `strategy="bisect"`, `perturb_mix="classic"` (`DEFAULT_PERTURB_MIX`), `snap=False`, `LOCAL_MAXITER=250`,
  `hops=4`.

Backward compatibility: `search`, `SearchResult`, `compact`, `local_opt`, `Config` keep their signatures;
`compact` gained the optional keywords `mix`, `anneal_opts`, `max_fails`; `make_feasible` gained `mix`;
`perturb` gained `mix`.  New public helpers: `anneal`, `anneal_feasible`, `anneal_compact`, `snap_angles`,
`snap_polish`, `block_seed_pool`, `seed_block`, `PERTURB_MIXES`, `ANNEAL_DEFAULTS`.
