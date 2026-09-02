import math

import numpy as np
import pytest

from squarepack import pack, solve, verify, repair, best_analytic, analytic_candidates, best_known
from squarepack.blocks import tilted_block_search, block_capacity
from squarepack.constructions import (gobel_square_member, gobel_square_valid, gobel_strip_member,
                                      gobel_strip_capacity, add_L, grid)
from squarepack.geometry import pair_penetration, candidate_pairs, SQRT2
from squarepack.optimize import energy_grad, search, Config, local_opt


def test_verify_accepts_touching_and_rejects_overlap():
    assert verify(2, [[0.5, 0.5, 0], [1.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]]).ok
    assert not verify(2, [[0.5, 0.5, 0], [1.4, 0.5, 0]]).ok
    assert not verify(1.9, [[0.5, 0.5, 0], [1.5, 1.5, 0]]).ok  # sticks out
    # rotated square touching the corner square of the n=5 packing
    s = 2 + 1 / SQRT2
    assert verify(s, [[0.5, 0.5, 0], [s / 2, s / 2, math.pi / 4]]).ok
    assert not verify(s - 1e-3, [[0.5, 0.5, 0], [(s - 1e-3) / 2, (s - 1e-3) / 2, math.pi / 4]]).ok


def test_penetration_matches_geometry():
    # two axis-aligned squares 0.9 apart overlap by 0.1
    x = np.array([0.0, 0.9]); y = np.zeros(2); t = np.zeros(2)
    p = pair_penetration(x, y, t, np.array([0]), np.array([1]))
    assert abs(p[0] - 0.1) < 1e-12
    # a diamond next to a square: separated when centre distance is 0.5 + 1/sqrt2
    x = np.array([0.0, 0.5 + 1 / SQRT2]); t = np.array([0.0, math.pi / 4])
    p = pair_penetration(x, y, t, np.array([0]), np.array([1]))
    assert abs(p[0]) < 1e-12


def test_candidate_pairs_dense_and_grid_agree():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 40, 900); y = rng.uniform(0, 40, 900)
    I1, J1 = candidate_pairs(x, y, dense_limit=10 ** 6)
    I2, J2 = candidate_pairs(x, y, dense_limit=10)
    a = set(zip(I1.tolist(), J1.tolist())); b = set(zip(I2.tolist(), J2.tolist()))
    assert a == b and len(a) > 0


def test_repair_makes_valid():
    s = 2 + 1 / SQRT2
    sq = np.array([[0.5, 0.5, 0], [s - 0.5, 0.5, 0], [0.5, s - 0.5, 0], [s - 0.5, s - 0.5, 0], [s / 2, s / 2, math.pi / 4]])
    f = (s - 1e-4) / s
    s2, sq2 = repair(s - 1e-4, sq * np.array([f, f, 1]))
    assert verify(s2, sq2).ok and abs(s2 - s) < 1e-9


@pytest.mark.parametrize("a", range(1, 15))
def test_gobel_strip_members_valid(a):
    p = gobel_strip_member(a)
    assert p.n == gobel_strip_capacity(a)
    assert verify(p.s, p.squares).ok


def test_gobel_square_members_valid():
    for a in range(1, 12):
        for b in range(1, 30):
            if gobel_square_valid(a, b):
                p = gobel_square_member(a, b)
                assert verify(p.s, p.squares).ok and p.n == 2 * a * (a + 1) + b * b


def test_add_L_valid():
    p = add_L(add_L(gobel_square_member(4, 5)))
    assert verify(p.s, p.squares).ok and p.n == 101 and abs(p.s - (7 + 2.5 * SQRT2)) < 1e-12


@pytest.mark.parametrize("n,expected", [(1, 1), (2, 2), (4, 2), (5, 2 + SQRT2 / 2), (9, 3), (10, 3 + SQRT2 / 2),
                                        (27, 5 + SQRT2 / 2), (40, 4 + 2 * SQRT2), (52, 7 + SQRT2 / 2),
                                        (65, 5 + 2.5 * SQRT2), (82, 6 + 2.5 * SQRT2), (100, 10)])
def test_analytic_matches_known_closed_forms(n, expected):
    p = best_analytic(n)
    assert abs(p.s - expected) < 1e-12 and p.n == n and verify(p.s, p.squares).ok


def test_analytic_never_beats_literature_and_is_valid():
    for n in range(1, 200):
        p = best_analytic(n)
        assert verify(p.s, p.squares).ok
        assert p.s >= best_known(n) - 1e-9


@pytest.mark.parametrize("n,expected", [(26, 3.5 + 1.5 * SQRT2), (66, 3 + 4 * SQRT2), (85, 5.5 + 3 * SQRT2)])
def test_tilted_block_reproduces_records(n, expected):
    p = tilted_block_search(n)
    assert p is not None and abs(p.s - expected) < 1e-9 and verify(p.s, p.squares).ok


def test_block_capacity_gobel_strip():
    assert block_capacity(5 + SQRT2 / 2 + 1e-9, 1, 5, 0.0, 0.0)[0] == 27


def test_energy_gradient_finite_difference():
    rng = np.random.default_rng(1)
    n, s = 7, 2.4
    z = np.concatenate([rng.uniform(0.3, s - 0.3, n), rng.uniform(0.3, s - 0.3, n), rng.uniform(-1, 1, n)])
    I, J = np.triu_indices(n, 1)
    E, g = energy_grad(z, n, s, I, J)
    assert E > 0
    h = 1e-6
    for i in range(len(z)):
        zp = z.copy(); zp[i] += h; zm = z.copy(); zm[i] -= h
        num = (energy_grad(zp, n, s, I, J)[0] - energy_grad(zm, n, s, I, J)[0]) / (2 * h)
        assert abs(num - g[i]) < 1e-5 * (1 + abs(g[i]))


def test_local_opt_reaches_zero_energy_for_easy_case():
    rng = np.random.default_rng(2)
    cfg = Config(3.2, rng.uniform(0.5, 2.7, 9), rng.uniform(0.5, 2.7, 9), np.zeros(9))
    res, E = local_opt(cfg, maxiter=1000)
    assert E < 1e-12


def test_search_returns_valid_and_not_worse_than_analytic():
    r = search(7, time_budget=2.0, seed=0)
    assert verify(r.packing.s, r.packing.squares).ok and r.packing.s <= 3 + 1e-12


def test_pack_api_shapes():
    s, arr = pack(12)
    assert abs(s - 4) < 1e-12 and len(arr) == 12 and all(len(row) == 3 for row in arr)
    s, arr = pack(5, degrees=True)
    assert any(abs(abs(row[2]) - 45) < 1e-9 for row in arr)
    sol = solve(26, use_cache=False)
    assert abs(sol.s - (3.5 + 1.5 * SQRT2)) < 1e-9 and sol.best_known is not None
