# Generalised tilted-block model (`squarepack/blocks.py`)

Machine: 4 shared cores, single-threaded numpy (`OPENBLAS_NUM_THREADS=1`).
"Before" = the original single-convex-block model (commit `238958c`), "after" = this branch.
Times are wall-clock seconds of one `tilted_block_search(n)` call with default settings.

## What was generalised

1. **Obstacle = union of convex pieces.**  A candidate is a `(K, 6)` *spec* of pieces
   `(w, h, ox, oy, theta, u)`; the row model clips every piece to every band and takes the
   union extents (`_band_extents`, one vectorised pass over `P*K` polygons).  Since a
   non-convex union no longer separates the left and right fills in every band, the two
   seams `(J_L, J_R)` are checked **exactly** against the O(B) pairs of bottom/top bands
   that overlap in `y` (only `j' = B-1-j` and `j' = B-j` can) - the count is exact for
   any obstacle, and the check is skipped for a single convex piece (provably vacuous).
   Pieces are also checked pairwise for overlap (vectorised separating-axis test).
2. **Batched branch-and-bound bisection** (`min_side_batch`): all candidates of a family
   (all shapes x offsets x row/column model, up to ~10^5) are bisected at once; a
   candidate whose lower bound exceeds the best upper bound (+ margin) is dropped.  This
   is what makes the extra families affordable: stage 1 alone went from 10 s to 1 s for
   n = 89 and from 22 s to 1.1 s for n = 150.
3. **Families** (`FAMILIES`, all verified valid on random parameters in the tests):
   * `single`: one `p x q` block at **any angle** (grid of 45 deg plus 20/30/40/50/60/70 deg);
   * `two`: two equal blocks placed half-turn symmetrically at `centre +- e`;
   * `strip` / `strip_w`: 45 deg strips whose end segments are rotated by a bend angle
     about the joint corner (S-shaped `[q1 | q2 | q1]` and W-shaped
     `[q1 | 1 | q2 | 1 | q1]` chains, `chain_specs`);
   * `shear`: rows of a block shifted against each other (each row its own piece, so the
     staircase outline is exact);
   * `RowChain`: `R` rows of `p` parallel squares with per-row tilt, row direction and
     lateral slide, consecutive rows held in exact contact (`_row_contact`, separating-axis
     over all tile pairs), half-turn or mirror symmetric.  This is the family the record
     structures actually belong to (see "Expressiveness").
4. **Continuous refinement** (`_refine`): a batched pattern search over the continuous
   parameters (offsets to 1e-7, angle, bend, shear, slides) of the best candidates of each
   family, i.e. finer-than-1/4 offsets and a golden-section-like angle refinement; and a
   relaxation of the best rigid blocks into `RowChain`s.
5. Mixed row/column anchoring per side was **not** added: the "L" fillers of the record
   families (82 = 65 + 2L, 101, 122, ...) are already exact in the row model because a
   wall-adjacent column is the `+1` of every band's right part, and the baseline already
   reproduces them.

The old API (`block_capacity`, `row_fill`, `min_side_vec`, `build_block_packing`,
`block_polygons`, `block_tiles`, `tilted_block_search(n, s_max, offsets, shapes, angle)`)
is unchanged; `row_fill`/`block_capacity` agree with the old implementation on 3000 random
single blocks (checked during development), and the old tests pass.

## Expressiveness (what the fill model can now represent)

Feeding the **exact tilted squares of the record packings** (parsed from the reference
SVGs) to the union row model as pieces reproduces the records' axis-aligned fill:

| n | record s | row-model capacity at s + 1e-9 | note |
|---|---|---|---|
| 18 | 4.822876 | 18 | two kinked 1x4 chains at 24.3 deg |
| 19 | 4.885618 | 19 | 2x4 strip with slid rows (exact chain parameters in `test_row_chain_expresses_record_19`) |
| 37 | 6.598620 | 37 | 3x5 strip, end rows tilted 2.9 deg |
| 50 | 7.571429 | 50 | sheared lattice band at 36.87 deg |
| 54 | 7.846667 | 54 | two staggered 8-chains |
| 70 | 8.881667 | 70 | 8-row sheared band at 22.4 deg (`test_union_row_model_reproduces_record_fill`) |
| 87 | 9.838817 | 87 | sheared domino row + two 3-chains |
| 88 | 9.888153 | 88 | 2x11 strip with tilted end rows (`test_union_row_model_reproduces_record_fill`) |
| 123 | 11.601400 | 123 | 3x12 strip with tilted end rows |
| 102 | 10.611388 | 100 | the only one the row model cannot complete (2 squares short) |

So the fill side is no longer the limitation; the *tilted* side is.  All these records are
`RowChain`s (or unions of two), but their optima need several coordinated parameter moves
(e.g. n = 19: an inner-row gap of (3 sqrt2 - 4)/3 plus two slides of (3 sqrt2 - 4)/6 at a
strip offset where the rigid strip is not competitive), which a coordinate pattern search
started from a straight strip does not find - see the table below.

## Before / after

### Measurement set

| n | record | before s | before t (s) | after s | after t (s) | after - record | after structure |
|---|---|---|---|---|---|---|---|
| 18 | 4.822876 | 4.828427 | 2.0 | 4.828427 | 1.1 | +0.0056 | tilted 1x4 block at 45deg, offset (+0,+0) |
| 19 | 4.885618 | 4.914214 | 1.9 | 4.914214 | 1.5 | +0.0286 | tilted 2x4 block at 45deg, offset (-0.25,-0.25) |
| 28 | 5.824445 | 5.828427 | 3.2 | 5.828427 | 1.7 | +0.0040 | tilted 4x4 block at 45deg, offset (+0,+0) |
| 37 | 6.598620 | 6.621320 | 1.5 | 6.621320 | 2.1 | +0.0227 | tilted 3x3 block at 45deg, offset (-1,-0.5) |
| 50 | 7.571429 | 7.621320 | 2.0 | 7.621320 | 2.6 | +0.0499 | tilted 3x3 block at 45deg, offset (-1,-0.5) |
| 53 | 7.822876 | 7.828427 | 6.7 | 7.828427 | 3.6 | +0.0056 | tilted 4x4 block at 45deg, offset (-0.5,-0.5) |
| 54 | 7.846667 | 7.914214 | 5.7 | 7.914214 | 4.0 | +0.0675 | tilted 2x8 block at 45deg, offset (-0.25,-0.25) |
| 68 | 8.803460 | 8.828427 | 8.4 | 8.828427 | 4.2 | +0.0250 | tilted 4x4 block at 45deg, offset (-1,-1) |
| 69 | 8.827212 | 8.914214 | 7.9 | 8.914214 | 4.3 | +0.0870 | tilted 2x8 block at 45deg, offset (-0.75,-0.75) |
| 70 | 8.881667 | 8.914214 | 7.8 | 8.914214 | 7.0 | +0.0325 | tilted 2x9 block at 45deg, offset (-0.5,+0) |
| 87 | 9.838817 | 9.863961 | 9.7 | 9.863961 | 4.6 | +0.0251 | tilted 4x9 block at 45deg, offset (-0.25,+0.25) |
| 88 | 9.888153 | 9.914214 | 8.8 | 9.914214 | 5.5 | +0.0261 | tilted 2x11 block at 45deg, offset (-0.25,-0.25) |
| 102 | 10.611388 | 10.621320 | 4.3 | 10.621320 | 6.6 | +0.0099 | tilted 3x10 block at 45deg, offset (-0.5,+0) |
| 123 | 11.601400 | 11.621320 | 5.1 | 11.621320 | 5.9 | +0.0199 | tilted 3x10 block at 45deg, offset (-1,-0.5) |

### Regression set (records must be hit exactly)

| n | record | before s | before t (s) | after s | after t (s) | after - record | after structure |
|---|---|---|---|---|---|---|---|
| 26 | 5.621320 | 5.621320 | 1.0 | 5.621320 | 1.8 | +0.0000 | tilted 3x3 block at 45deg, offset (-0.5,+0) |
| 27 | 5.707107 | 5.707107 | 1.8 | 5.707107 | 2.3 | +0.0000 | tilted 1x5 block at 45deg, offset (+0,+0) |
| 40 | 6.828427 | 6.828427 | 5.5 | 6.828427 | 1.9 | +0.0000 | tilted 4x4 block at 45deg, offset (+0,+0) |
| 52 | 7.707107 | 7.707107 | 4.2 | 7.707107 | 2.9 | +0.0000 | tilted 1x8 block at 45deg, offset (+0,+0) |
| 65 | 8.535534 | 8.535534 | 3.1 | 8.535534 | 3.4 | +0.0000 | tilted 5x5 block at 45deg, offset (+0,+0) |
| 66 | 8.656854 | 8.656854 | 3.4 | 8.656854 | 3.9 | +0.0000 | tilted 3x8 block at 45deg, offset (-0.25,-0.25) |
| 85 | 9.742641 | 9.742641 | 8.5 | 9.742641 | 6.6 | +0.0000 | tilted 6x6 block at 45deg, offset (-0.5,+0) |
| 89 | 9.949747 | 9.949747 | 10.2 | 9.949747 | 3.5 | +0.0000 | tilted 7x7 block at 45deg, offset (+0,+0) |
| 150 | 12.778175 | 12.778175 | 22.3 | 12.778175 | 2.5 | +0.0000 | tilted 6x11 block at 45deg, offset (-0.25,-0.25) |
| 202 | 14.727922 | 14.727922 | 20.3 | 14.727922 | 6.0 | +0.0000 | tilted 1x18 block at 45deg, offset (+0,+0) |

### Large n (timing)

| n | record | before s | before t (s) | after s | after t (s) | after - record | after structure |
|---|---|---|---|---|---|---|---|
| 250 | 16.000000 | - | - | 16.000000 | 4.1 | +0.0000 | tilted 1x1 block at 45deg, offset (-1.25,-1.25) |
| 300 | 17.824123 | - | - | 17.849242 | 6.9 | +0.0251 | tilted 4x21 block at 45deg, offset (+0,+0) |

## Reading the table

* Every regression n still hits its literature record exactly (to 1e-9), and every
  measurement n returns a verified packing.
* Timing: stage 1 (the old model's search space) is ~10x faster than before
  (n = 89: 10.2 s -> 0.9 s, n = 150: 22.3 s -> 1.1 s); the whole pipeline (six grid
  stages, refinement, chain relaxation) takes 1-7 s for every n <= 300 (before: 1-22 s),
  with n = 300 at 6.9 s (the secondary grids are coarsened above n = 120).
* Side lengths on the measurement set are unchanged: none of the new *rigid* families
  beats the 45-degree block there.  This is a genuine finding, not a search failure - for
  each of 18/19/37/70/88/102 the exact record structure was evaluated in the new model and
  its rigid approximations (rigid two-chain, rigid bent strip with corner pivots, rigid
  sheared block) all need a larger side than the plain 45-degree block, e.g. two rigid 1x4
  chains at 26 deg give 4.859 for n = 18 (record 4.8229, block 4.8284), rigid 1-degree S-bends
  of the 2x11 strip give 9.925 for n = 88 (block 9.914).  The records are obtained only with
  the non-rigid row chains (per-row tilt with the row direction kept, slides, gaps).
* The `RowChain` family is therefore exposed with `RowChain.from_block` and full
  parameter access; `tilted_block_search(..., relax=True)` relaxes the best rigid blocks
  into it (time-boxed), but a global optimiser over its parameters (annealing / contact
  solving) is needed to reach the records from scratch.
