"""Tests for the generalised tilted-block model (union-of-pieces row model, families, search)."""
import math

import numpy as np
import pytest

from squarepack import blocks as B
from squarepack.blocks import (FAMILIES, RowChain, block_capacity, block_polygons, block_tiles, build_block_packing,
                               build_spec_packing, min_side_vec, row_fill, spec_capacity, spec_polygons,
                               tilted_block_search)
from squarepack.geometry import QUARTER_PI, SQRT2, as_arrays, candidate_pairs, pair_penetration, verify

# --------------------------------------------------------------------------- #
# regression: literature records reproduced exactly by the search
# --------------------------------------------------------------------------- #
RECORDS = {26: 3.5 + 1.5 * SQRT2, 27: 5 + SQRT2 / 2, 40: 4 + 2 * SQRT2, 52: 7 + SQRT2 / 2, 65: 5 + 2.5 * SQRT2,
           66: 3 + 4 * SQRT2, 85: 5.5 + 3 * SQRT2, 89: 5 + 3.5 * SQRT2, 150: 5 + 5.5 * SQRT2, 202: 2 + 9 * SQRT2}


@pytest.mark.parametrize("n", sorted(RECORDS))
def test_search_hits_records(n):
    p = tilted_block_search(n)
    assert p is not None and p.n == n
    assert abs(p.s - RECORDS[n]) < 1e-9
    assert verify(p.s, p.squares, 1e-12).ok


# --------------------------------------------------------------------------- #
# old API
# --------------------------------------------------------------------------- #
def test_old_api_still_works():
    assert block_capacity(5 + SQRT2 / 2 + 1e-9, 1, 5, 0.0, 0.0)[0] == 27
    assert block_capacity(np.array([5.0, 5.8]), 1, 5, 0.0, 0.0)[1] == 27
    dx = np.array([0.0, 0.0, 0.5, -0.5, 0.25])
    dy = np.array([0.0, 0.5, 0.0, 0.0, 0.25])
    sides = min_side_vec(26, 3, 3, dx, dy, math.sqrt(26), 6.0, False)
    assert np.isfinite(sides).any() and abs(sides.min() - (3.5 + 1.5 * SQRT2)) < 1e-9
    pk = build_block_packing(3.5 + 1.5 * SQRT2 + 1e-9, 3, 3, 0.0, 0.5)
    assert pk is not None and pk.n >= 26 and verify(pk.s, pk.squares).ok
    polys = block_polygons(2, 3, np.array([1.0, 2.0]), np.array([1.0, 1.5]), QUARTER_PI)
    assert polys.shape == (2, 4, 2)
    assert abs(np.ptp(polys[0, :, 0]) - 5 / SQRT2) < 1e-12
    tiles = block_tiles(2, 3, 3.0, 3.0, QUARTER_PI)
    assert tiles.shape == (6, 3) and np.allclose(tiles[0, :2], (3.0, 3.0), atol=0.51)


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def test_quads_overlap_touching_is_allowed():
    sq = lambda x, y, th: spec_polygons(np.zeros(1), np.array([[[1, 1, x, y, th, 0.0]]]))[0, 0]
    assert not B._quads_overlap(sq(0, 0, 0)[None], sq(1, 0, 0)[None])[0]
    assert B._quads_overlap(sq(0, 0, 0)[None], sq(0.99, 0, 0)[None])[0]
    d = 0.5 + 1 / SQRT2
    assert not B._quads_overlap(sq(0, 0, 0)[None], sq(d, 0, QUARTER_PI)[None])[0]
    assert B._quads_overlap(sq(0, 0, 0)[None], sq(d - 1e-3, 0, QUARTER_PI)[None])[0]


def test_row_fill_single_piece_uses_independent_seams():
    rng = np.random.default_rng(0)
    s = rng.uniform(6, 10, 200)
    polys = block_polygons(3, 4, s / 2 + rng.uniform(-1, 1, 200), s / 2 + rng.uniform(-1, 1, 200), QUARTER_PI)
    total, JL, JR, Fb, Ft, lb, rb, lt, rt = row_fill(s, polys)
    Bp = np.floor(s + B.EPS).astype(int)
    for i in range(200):
        best = -1
        for jl in range(Fb[i], Bp[i] - Ft[i] + 1):
            for jr in range(Fb[i], Bp[i] - Ft[i] + 1):
                tot = Bp[i] * (Fb[i] + Ft[i]) + lb[i, Fb[i]:jl].sum() + lt[i, Ft[i]:Bp[i] - jl].sum() \
                    + rb[i, Fb[i]:jr].sum() + rt[i, Ft[i]:Bp[i] - jr].sum()
                best = max(best, tot)
        assert total[i] == best


def _check_family(fam, gen, rng, trials):
    built = 0
    for _ in range(trials):
        params, s, tr = gen(rng)
        spec = fam.specs(np.asarray(params, float)[None])
        cap = spec_capacity(np.array([s]), spec, np.array([tr]))[0]
        if cap < 0:
            continue
        pk = build_spec_packing(s, spec[0], tr)
        assert pk is not None and pk.n == cap, "capacity must equal the number of squares built"
        assert verify(pk.s, pk.squares, 1e-9).ok, (fam.name, params, s, tr)
        built += 1
    assert built >= trials // 4


def test_single_blocks_any_angle_build_valid():
    rng = np.random.default_rng(1)
    _check_family(FAMILIES["single"], lambda r: ([int(r.integers(1, 6)), int(r.integers(1, 8)), r.uniform(-1.5, 1.5),
                                                  r.uniform(-1.5, 1.5), r.uniform(0, math.pi / 2), 0.0],
                                                 r.uniform(6, 11), bool(r.integers(0, 2))), rng, 60)


def test_two_blocks_build_valid():
    rng = np.random.default_rng(2)
    _check_family(FAMILIES["two"], lambda r: ([int(r.integers(1, 4)), int(r.integers(1, 6)), r.uniform(0, 3), r.uniform(0, 3),
                                               r.uniform(0, math.pi / 2)], r.uniform(6, 12), bool(r.integers(0, 2))), rng, 80)


@pytest.mark.parametrize("kind,name", [(0, "strip"), (1, "strip_w")])
def test_bent_strips_build_valid(kind, name):
    rng = np.random.default_rng(3 + kind)
    _check_family(FAMILIES[name], lambda r: ([int(r.integers(1, 4)), int(r.integers(1, 4)), int(r.integers(1, 5)),
                                              r.uniform(0, 0.15), r.uniform(-1, 1), r.uniform(-1, 1), kind],
                                             r.uniform(7, 13), bool(r.integers(0, 2))), rng, 60)


def test_sheared_blocks_build_valid():
    rng = np.random.default_rng(4)
    _check_family(FAMILIES["shear"], lambda r: ([int(r.integers(1, 6)), int(r.integers(2, 6)), r.uniform(-1, 1), r.uniform(-1, 1),
                                                 r.uniform(0, math.pi / 2), r.uniform(-0.8, 0.8)], r.uniform(7, 12),
                                                bool(r.integers(0, 2))), rng, 80)


def _random_chain(r):
    p, R = int(r.integers(1, 5)), int(r.integers(1, 7))
    fam, params = RowChain.from_block(p, R, r.uniform(0, math.pi / 2), r.uniform(-1, 1), r.uniform(-1, 1),
                                      "rot" if r.integers(0, 2) else "mirror")
    H = fam.H
    params[fam.ith:fam.ith + H] += r.uniform(-0.15, 0.15, H)
    params[fam.iph:fam.iph + H] += r.uniform(-0.3, 0.3, H)
    params[fam.it:fam.it + H] = r.uniform(-0.6, 0.6, H)
    params[fam.ig:fam.ig + H] = np.where(r.uniform(size=H) < 0.5, 0.0, r.uniform(0, 0.2, H))
    return fam, params


def test_row_chains_build_valid():
    rng = np.random.default_rng(5)
    built = 0
    for _ in range(80):
        fam, params = _random_chain(rng)
        s, tr = rng.uniform(7, 13), bool(rng.integers(0, 2))
        spec = fam.specs(params[None])
        cap = spec_capacity(np.array([s]), spec, np.array([tr]))[0]
        if cap < 0:
            continue
        pk = build_spec_packing(s, spec[0], tr)
        assert pk.n == cap and verify(pk.s, pk.squares, 1e-9).ok
        built += 1
    assert built >= 20


def test_row_chain_rows_touch_without_overlap():
    rng = np.random.default_rng(6)
    for _ in range(60):
        fam, params = _random_chain(rng)
        params[fam.ig:fam.ig + fam.H] = 0.0
        tiles = np.vstack([B.spec_tiles(8.0, pc) for pc in fam.specs(params[None])[0]])
        x, y, t = as_arrays(tiles)
        I, J = candidate_pairs(x, y)
        if len(I) == 0:
            continue
        pen = pair_penetration(x, y, t, I, J)
        assert pen.max() <= 1e-9                      # never overlapping
        if fam.R >= 2:
            assert pen.max() >= -1e-9                 # consecutive rows are in contact


# --------------------------------------------------------------------------- #
# expressiveness: record structures the old (single convex block) model could not represent
# --------------------------------------------------------------------------- #
def test_row_chain_expresses_record_19():
    # Wainwright's n = 19 (s = 3 + 4 sqrt2 / 3): a 2x4 strip at 45 degrees whose two inner rows are
    # (3 sqrt2 - 4)/3 apart and slid by (3 sqrt2 - 4)/6 from the strip centre, and whose outer rows are
    # slid back by the same amount - a mirror-symmetric row chain.
    fam, params = RowChain.from_block(2, 4, -QUARTER_PI, -0.25, 0.25, "mirror")
    params[fam.ig + 0] = (3 * SQRT2 - 4) / 3
    params[fam.it + 0] = (3 * SQRT2 - 4) / 6
    params[fam.it + 1] = (4 - 3 * SQRT2) / 6
    s_rec = 3 + 4 * SQRT2 / 3
    spec = fam.specs(params[None])
    assert spec_capacity(np.array([s_rec + 1e-9]), spec, np.array([False]))[0] >= 19
    assert spec_capacity(np.array([s_rec - 1e-6]), spec, np.array([False]))[0] < 19
    pk = build_spec_packing(s_rec + 1e-9, spec[0], False).take(19)
    assert verify(pk.s, pk.squares, 1e-9).ok
    # the single-block model needs 7/2 + sqrt2
    assert abs(tilted_block_search(19, families=(), refine=False, relax=False).s - (3.5 + SQRT2)) < 1e-9


# tilted squares (x, y, angle) of the record packings, y up (Friedman/Ellsworth, kingbird.myphotos.cc)
TILES_70 = [  # DeVincentis 2014, s = root of 23x^4 - 742x^3 + 8848x^2 - 45876x + 86229
    (6.3997456125219, 7.5397470635533, -0.3916952579571), (7.3142014244688, 7.1342466852320, -0.3916952579571),
    (8.2286572364158, 6.7287463069107, -0.3916952579571), (5.5165701648219, 6.8225892060084, -0.3916952579571),
    (6.4310259767689, 6.4170888276870, -0.3916952579571), (7.3454817887159, 6.0115884493657, -0.3916952579571),
    (4.6333947171220, 6.1054313484634, -0.3916952579571), (5.5478505290690, 5.6999309701421, -0.3916952579571),
    (6.4623063410159, 5.2944305918207, -0.3916952579571), (4.6548676796334, 4.9590285617797, -0.3916952579571),
    (5.5791308933160, 4.5772727342758, -0.3916952579571), (3.6842916911969, 5.2286572364158, -0.3916952579571),
    (2.4819211444871, 1.3419196934557, 2.7498973956327), (1.5674653325402, 1.7474200717770, 2.7498973956327),
    (0.6530095205932, 2.1529204500983, 2.7498973956327), (3.3650965921871, 2.0590775510006, 2.7498973956327),
    (2.4506407802401, 2.4645779293220, 2.7498973956327), (1.5361849682931, 2.8700783076433, 2.7498973956327),
    (4.2482720398870, 2.7762354085456, 2.7498973956327), (3.3338162279400, 3.1817357868669, 2.7498973956327),
    (2.4193604159931, 3.5872361651883, 2.7498973956327), (4.2267990773756, 3.9226381952293, 2.7498973956327),
    (3.3025358636930, 4.3043940227332, 2.7498973956327), (5.1973750658121, 3.6530095205932, 2.7498973956327),
]
S_70 = 8.881666757009004
TILES_88 = [  # Ellsworth 2025 (bent 2x11 strip), s = 9.88815305375857
    (4.3535533905933, 5.5345996631653, -0.7853981633974), (5.0606601717798, 4.8274928819787, -0.7853981633974),
    (3.6120870547062, 4.8229861991335, -0.7288016373830), (4.3203278398133, 4.1147454140264, -0.7288016373830),
    (2.8764196840447, 4.1103889342038, -0.7778775146364), (3.5835381480102, 3.4032537147682, -0.7778775146364),
    (2.1643451033365, 3.3646085445482, -0.8419946894119), (2.8725858884436, 2.6563677594411, -0.8419946894119),
    (1.4716480811559, 2.6389568845130, -0.8419946894119), (2.1798888662630, 1.9307160994059, -0.8419946894119),
    (0.7256750624236, 1.9729807172792, -0.8419946894119), (1.4339158475307, 1.2647399321721, -0.8419946894119),
    (5.0651668546250, 6.2760659990524, -0.8419946894119), (5.7734076397322, 5.5678252139452, -0.8419946894119),
    (5.7777641195548, 7.0117333697139, -0.7929188121585), (6.4848993389904, 6.3046149057483, -0.7929188121585),
    (6.5235445092103, 7.7238079504221, -0.7288016373830), (7.2317852943175, 7.0155671653150, -0.7288016373830),
    (7.2491961692455, 8.4165049726027, -0.7288016373830), (7.9574369543527, 7.7082641874955, -0.7288016373830),
    (7.9151723364794, 9.1624779913350, -0.7288016373830), (8.6234131215865, 8.4542372062278, -0.7288016373830),
]
S_88 = 9.888153053758572


@pytest.mark.parametrize("n,s_rec,tiles", [(70, S_70, TILES_70), (88, S_88, TILES_88)])
def test_union_row_model_reproduces_record_fill(n, s_rec, tiles):
    """Given the record's tilted squares as pieces (a non-convex union), the exact row model with the
    cross-side seam check places enough axis-aligned squares to complete the record packing."""
    spec = np.array([[1.0, 1.0, x - s_rec / 2, y - s_rec / 2, a, 0.0] for x, y, a in tiles])
    s = s_rec + 1e-9
    cap = spec_capacity(np.array([s]), spec[None], np.array([False]))[0]
    assert cap >= n
    pk = build_spec_packing(s, spec, False).take(n)
    assert verify(pk.s, pk.squares, 1e-8).ok
    # the single rigid block cannot do this well
    single = tilted_block_search(n, families=(), refine=False, relax=False)
    assert single.s > s_rec + 1e-3


def test_search_returns_verified_packing_for_small_n():
    for n in (18, 19, 37):
        p = tilted_block_search(n)
        assert p is not None and p.n == n and verify(p.s, p.squares, 1e-12).ok
        assert p.s < math.ceil(math.sqrt(n))
