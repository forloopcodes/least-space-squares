# Adversarial correctness review (workflow: 4 reviewers → 2 independent skeptics per finding)

Reviewed at commit 416e8c7; fixes landed in 20f8767. Reviewers were told to reproduce every claim
by running code; skeptics were told to refute (default refuted when not reproducible or without
user-visible effect).

| # | file | finding | verdict | action |
|---|------|---------|---------|--------|
| 1 | geometry.py `candidate_pairs` | int64 cell-key overflow in the n > 512 grid path drops a neighbouring-cell pair, so `verify` could accept an overlap for coordinates ~2^32 | confirmed (2/2) | cells keyed by `(cx, cy)` tuples; non-finite input routed to the dense path; test |
| 2 | _fastcore.c `build_pairs` | signed-overflow of the cell-count guard for coordinate spreads ~4·10⁹; float→long cast of non-finite values (UB, heap corruption) | confirmed (2/2) / refuted as unreachable | guard evaluated in double, non-finite or oversized extents take the brute-force path |
| 3 | _fastcore.c `max_violation_c` | NaN coordinates reported as feasible (numpy verifier rejects them) | refuted (no user-visible path) | hardened anyway: non-finite input returns +inf |
| 4 | blocks.py `spec_polygons` | hull of a sheared piece (h > 1, u ≠ 0) omits tile corners → overlapping packings from that family | confirmed (2/2) | along-row half-extent enlarged by \|u\|/2 (tightest slanted parallelogram containing the staircase); randomized validity test |
| 5 | blocks.py `tilted_block_search` | s_max gate used the pre-repair bisection side, so e.g. `tilted_block_search(6)` returned a 3.0000000005 "block" instead of None | confirmed (2/2) | gate on the repaired side; test |
| 6 | blocks.py `_Candidate.build` | `take(n)` could drop every rotated tile, returning a mislabelled grid | confirmed (2/2) | candidates with no rotated tile left are rejected |
| 7 | optimize.py `block_seed_pool` | default seed mix overran the time budget by up to ~10 s (pool built without a deadline) | confirmed (2/2) | deadline threaded into the pool; incomplete pools are not cached |
| 8 | optimize.py `anneal` / `penalty` strategies | non-default strategies ignore the deadline for one seed pass | confirmed | no seed pass is started within 1 s of the deadline; `penalty_descent` checks the deadline between continuation stages (an SA pass remains un-interruptible; documented) |
| 9 | solver.py | angles were not reduced as documented (`+π/4`, 1.99 rad from the √7 family) | confirmed by re-run at HEAD~1 | all output angles reduced to (−π/4, π/4]; test |
| 10 | solver.py `_load_cache` | a damaged `data/best_packings.json` made every call raise | confirmed | per-entry validation, damaged files ignored with a warning, caches keyed by path; test |
| 11 | solver.py `solve` | the block search ran even when a closed form or the cache already matched the record | confirmed | skipped when settled |
| 12 | README | complexity claims did not cover the √7/DeVincentis engines (best_analytic(900) ≈ 15 s); stale counts | confirmed | wording corrected, counts regenerated |
| 13 | scripts/benchmark.py | results JSON contained `-Infinity` | confirmed | non-finite values written as null; test |
| 14 | known.py | grouped "194, 195 … Proved" entries lost the smaller n | confirmed | smaller member of each proven pair inherits the entry |

The verify stage re-run after the fixes (skeptics working on 20f8767) reported the remaining
optimiser/solver/README claims as "describes the previous revision, not HEAD".
