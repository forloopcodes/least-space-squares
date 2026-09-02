import math

import numpy as np
import pytest

from squarepack import analytic_candidates, best_analytic, best_known, solve, verify
from squarepack.families import (A7, F7, S19, S50, S54, Frame, devincentis_capacity, devincentis_member,
                                 devincentis_side, family_devincentis, family_schadt, family_sqrt7,
                                 family_wainwright, schadt_member, sqrt7_member, sqrt7_side, wainwright_member)

SQRT2 = math.sqrt(2.0)
SQRT7 = math.sqrt(7.0)
S7 = lambda m: (2 * m - 1) / 2 + SQRT7 / 2  # noqa: E731
S54K = lambda k: 7 + 3 * k - SQRT2 / 2 + math.sqrt(1 + SQRT2)  # noqa: E731


def _valid(p):
    rep = verify(p.s, p.squares, 1e-9)
    assert rep.ok, rep
    return True


# --------------------------------------------------------------------------- #
# exact record values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,expected", [
    (18, S7(4)), (53, S7(7)), (86, S7(9)), (127, S7(11)),
    (19, 3 + 4 * SQRT2 / 3),
    (54, S54K(0)), (107, S54K(1)),
])
def test_records_exact(n, expected):
    p = best_analytic(n)
    assert p.n == n and abs(p.s - expected) < 1e-9
    assert abs(p.s - best_known(n)) < 1e-9
    _valid(p)


@pytest.mark.parametrize("n,expected", [(18, S7(4)), (53, S7(7)), (86, S7(9)), (127, S7(11)),
                                        (151, S7(12)), (176, S7(13)), (204, S7(14)), (234, S7(15)), (299, S7(17))])
def test_sqrt7_family_function(n, expected):
    p = family_sqrt7(n)
    assert p is not None and p.n == n and abs(p.s - expected) < 1e-9 and p.method == "sqrt7"
    _valid(p)


@pytest.mark.parametrize("n,expected", [(54, S54K(0)), (107, S54K(1)), (178, S54K(2)), (267, S54K(3))])
def test_devincentis_family_function(n, expected):
    p = family_devincentis(n)
    assert p is not None and p.n == n and abs(p.s - expected) < 1e-9
    _valid(p)


def test_wainwright_and_schadt():
    p = family_wainwright(19)
    assert p is not None and p.n == 19 and abs(p.s - S19) < 1e-9 and _valid(p)
    assert family_wainwright(20) is None
    p = family_schadt(50)
    assert p is not None and abs(p.s - 7 + 0 - 4 / 7) < 1e-9 and abs(p.s - S50) < 1e-12 and _valid(p)
    p = family_schadt(171)
    assert p is not None and abs(p.s - (13 + 4 / 7)) < 1e-9 and _valid(p)
    assert best_analytic(198).s <= 14 + 4 / 7 + 1e-9    # 171 + one L
    assert family_schadt(172) is None


# --------------------------------------------------------------------------- #
# extension sizes are valid and match the capacity rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m", range(4, 21))
def test_sqrt7_members_valid(m):
    p = sqrt7_member(m)
    assert abs(p.s - sqrt7_side(m)) < 1e-12 and abs(p.s - S7(m)) < 1e-12
    _valid(p)
    # every member holds at least the number of squares of the plain grid of side floor(s)
    assert p.n >= m * m
    # sub-packings stay valid (the ends of the band are removed first)
    _valid(p.take(p.n - 3))


def test_sqrt7_record_capacities():
    caps = {4: 18, 7: 53, 9: 86, 11: 127, 12: 151, 13: 176, 14: 204, 15: 234, 17: 299}
    for m, n in caps.items():
        assert sqrt7_member(m).n >= n, (m, sqrt7_member(m).n)


def test_sqrt7_angle_and_side():
    assert abs(A7 - math.atan((4 - SQRT7) / 3)) < 1e-15
    assert abs(F7 - (SQRT7 - 1) / 2) < 1e-15
    # the domino diagonal fits exactly between the staircases: (f, 1 + f) has length 2
    assert abs(F7 ** 2 + (1 + F7) ** 2 - 4) < 1e-12


@pytest.mark.parametrize("k", range(0, 8))
def test_devincentis_members_valid(k):
    p = devincentis_member(k)
    assert p.n == devincentis_capacity(k) == 9 * k * k + 44 * k + 54
    assert abs(p.s - devincentis_side(k)) < 1e-12 and abs(p.s - (S54 + 3 * k)) < 1e-12
    _valid(p)
    _valid(p.take(p.n - 5))


def test_other_members_valid():
    _valid(wainwright_member())
    assert wainwright_member().n == 19
    assert schadt_member(0).n == 50 and _valid(schadt_member(0))
    assert schadt_member(1).n == 171 and _valid(schadt_member(1))


# --------------------------------------------------------------------------- #
# the sliding-square kernel
# --------------------------------------------------------------------------- #
def test_frame_first_and_last_touch_exactly():
    a = math.pi / 6
    fr = Frame([-math.sin(a), math.cos(a)], [math.cos(a), math.sin(a)])
    s = 6.0
    rects = np.array([[3.0, 6.0, 0.0, 1.0]])   # a row on the bottom right
    v0 = np.array([2.0])
    u = fr.first(v0, rects, s)[0]
    sq = fr.square(u, v0[0])
    rep = verify(s, [sq] + [[x + 0.5, 0.5, 0.0] for x in range(3, 6)], 1e-9)
    assert rep.ok and abs(rep.max_penetration) < 1e-9          # touching, not overlapping
    # pushing a hair further overlaps
    sq2 = fr.square(u - 1e-3, v0[0])
    assert not verify(s, [sq2] + [[x + 0.5, 0.5, 0.0] for x in range(3, 6)], 1e-9).ok
    u2 = fr.last(v0, rects, s)[0]
    assert u2 > u and verify(s, [fr.square(u2, v0[0])], 1e-9).ok


# --------------------------------------------------------------------------- #
# portfolio integration
# --------------------------------------------------------------------------- #
def test_portfolio_never_beats_literature():
    for n in list(range(1, 130)) + [151, 176, 178, 204, 234, 267, 299]:
        p = best_analytic(n)
        assert p.n == n and p.s >= best_known(n) - 1e-9
        _valid(p)


def test_cli_reports_sqrt7_for_18():
    sol = solve(18, use_cache=False, use_blocks=False)
    assert abs(sol.s - S7(4)) < 1e-9 and sol.n == 18
    assert any(p.method == "sqrt7" for p in analytic_candidates(18))
