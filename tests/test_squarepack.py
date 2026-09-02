import math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import pytest

from squarepack import pack, solve, verify, repair, best_analytic, analytic_candidates, best_known
from squarepack.blocks import tilted_block_search, block_capacity
from squarepack.constructions import (gobel_square_member, gobel_square_valid, gobel_strip_member,
                                      gobel_strip_capacity, add_L, grid)
from squarepack.geometry import pair_penetration, candidate_pairs, SQRT2
from squarepack.optimize import energy_grad, search, Config, local_opt
from squarepack import fastcore


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


def test_exact_form_recognition():
    from squarepack.exact import exact_form
    assert exact_form(3.5 + math.sqrt(7) / 2) == "(7 + sqrt(7))/2"
    assert exact_form(5 + SQRT2 / 2) == "(10 + sqrt(2))/2"
    assert exact_form(3 + 4 * SQRT2 / 3) == "(9 + 4*sqrt(2))/3"
    assert exact_form(7 + 4 / 7) == "53/7"
    assert exact_form(3.8770835900228) is None


def test_cli_runs(tmp_path):
    import subprocess, sys, json
    out = subprocess.run([sys.executable, "-m", "squarepack", "10", "--json", "--degrees", "--svg", str(tmp_path / "p.svg")],
                         capture_output=True, text=True, check=True, cwd=str(ROOT))
    d = json.loads(out.stdout)
    assert d["n"] == 10 and abs(d["s"] - (3 + SQRT2 / 2)) < 1e-9 and len(d["squares"]) == 10
    assert (tmp_path / "p.svg").read_text().startswith("<svg")
def test_perturb_kinds_keep_shape():
    from squarepack.optimize import PERTURB_MIXES, perturb, seed_random
    rng = np.random.default_rng(3)
    cfg = seed_random(12, 4.0, rng)
    for kind in PERTURB_MIXES["extended"]:
        c = perturb(cfg, rng, kind=kind)
        assert c.n == 12 and np.all(c.x >= 0.3) and np.all(c.x <= 3.7) and np.all(c.y >= 0.3) and np.all(c.y <= 3.7)
    c = perturb(cfg, rng, mix="extended")
    assert c.n == 12


def test_snap_angles_snaps_and_clusters():
    from squarepack.optimize import snap_angles
    t = np.array([1e-4, -2e-4, math.pi / 4 - 1e-4, 0.4, 0.41, 0.405, -0.3])
    out, changed = snap_angles(t, tol=1e-3, cluster_tol=0.02)
    assert changed
    assert out[0] == 0.0 and out[1] == 0.0 and abs(abs(out[2]) - math.pi / 4) < 1e-15
    assert out[3] == out[4] == out[5] and abs(out[3] - 0.405) < 1e-12
    assert out[6] == -0.3


def test_anneal_energy_consistent_with_numpy():
    from squarepack import fastcore
    from squarepack.optimize import anneal, energy_only, seed_random
    if fastcore.load() is None:
        pytest.skip("C core not available")
    rng = np.random.default_rng(4)
    cfg = seed_random(9, 3.1, rng)
    feas, final, E = anneal(cfg, rng, sweeps=200, T0=1e-2, T1=1e-6)
    assert final.n == 9 and abs(E - energy_only(final.x, final.y, final.t, final.s)) < 1e-9 * (1 + E)
    feas, final, E = anneal(cfg, rng, sweeps=2000, T0=1e-2, T1=1e-7, shrink=0.01, etol=1e-10)
    if feas is not None:
        assert feas.s <= cfg.s and feas.max_violation() < 2e-5


@pytest.mark.parametrize("kw", [dict(strategy="anneal"), dict(strategy="anneal-hop"),
                                dict(perturb_mix="extended", snap=True, seed_mix=("block", "random"))])
def test_search_variants_return_valid_packings(kw):
    r = search(7, time_budget=1.0, seed=0, **kw)
    assert verify(r.packing.s, r.packing.squares).ok and r.packing.s <= 3 + 1e-12


def test_block_seed_pool_and_block_seeds_run():
    from squarepack.optimize import block_seed_pool, _block_shape
    pool = block_seed_pool(26)
    assert pool and all(p.n >= 26 for p in pool)
    assert _block_shape(pool[0]) is not None or pool[0].method != "tilted_block"
    r = search(26, time_budget=6.0, seed=0, seed_mix=("block",))
    assert verify(r.packing.s, r.packing.squares).ok and r.packing.s <= best_analytic(26).s + 1e-9


def test_candidate_pairs_huge_coordinates_no_overflow():
    # two overlapping squares far from the origin in the cell-grid path (n > 512): the pair must be found
    n = 600
    x = np.arange(n, dtype=float) * 3.0 + 2.0 ** 32
    y = np.zeros(n) + 2.0 ** 32
    x[1] = x[0] + 0.9
    s = float(x.max() + 5)
    rep = verify(s, np.column_stack([x, y, np.zeros(n)]), 1e-9)
    assert not rep.ok and rep.worst_pair == (0, 1)
    if fastcore.load() is not None:
        assert fastcore.max_violation(np.concatenate([x, y, np.zeros(n)]), n, s) > 0.09
        z = np.concatenate([x, y, np.zeros(n)]); z[5] = float("inf")
        assert fastcore.max_violation(z, n, s) == float("inf")


def test_verify_rejects_non_finite():
    assert not verify(3, [[0.5, 0.5, 0], [float("nan"), 1.5, 0]]).ok
    assert not verify(3, [[0.5, 0.5, float("nan")]]).ok


def test_sheared_block_pieces_are_valid():
    from squarepack.blocks import build_spec_packing
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(40):
        w, h = int(rng.integers(1, 5)), int(rng.integers(2, 5))
        u = float(rng.choice([-0.5, -0.25, 0.25, 0.5, 1.0]))
        s = float(rng.uniform(6, 9))
        spec = np.array([[w, h, 0.0, 0.0, math.pi / 4, u]])
        pk = build_spec_packing(s, spec, False, "x", "", {})
        if pk is not None:
            checked += 1
            assert verify(pk.s, pk.squares, 1e-9).ok, (w, h, u, s)
    assert checked > 10


def test_tilted_block_search_respects_s_max():
    for n in (6, 12, 20):
        assert tilted_block_search(n) is None


def test_solve_angles_reduced_and_blocks_skipped_when_settled():
    for n in (27, 39, 86):
        sol = solve(n, use_cache=False)
        assert np.all(sol.squares[:, 2] > -math.pi / 4 - 1e-12) and np.all(sol.squares[:, 2] <= math.pi / 4 + 1e-12)
        assert verify(sol.s, sol.squares).ok


def test_corrupt_cache_is_ignored(tmp_path):
    import squarepack.solver as S
    bad = tmp_path / "cache.json"
    bad.write_text('{"5": {"s": 1.0, "squares": [[0,0,0]]}, "x": 3, "7": {"s": "no"}, "9": {"s": 3.0, "squares": [[0.5,0.5,0]]}}')
    assert S._load_cache(bad) == {}
    bad.write_text("not json at all")
    assert S._load_cache(tmp_path / "other.json") == {}
    p = grid(4)
    assert S.update_cache(p, path=tmp_path / "new.json")
    assert S.cached(4, path=tmp_path / "new.json").n == 4


def test_benchmark_json_is_strict():
    import json as _json
    txt = (ROOT / "results" / "benchmark.json").read_text()
    assert "Infinity" not in txt and "NaN" not in txt
    _json.loads(txt)


def test_grouped_proved_entries():
    from squarepack.known import is_proved, record_entry
    assert is_proved(194) and is_proved(195) and record_entry(194)[3] == record_entry(195)[3]
    assert not is_proved(29)
