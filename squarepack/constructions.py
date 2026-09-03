"""Closed-form ("analytic") packings of n unit squares in a square.

Every construction here is a *family* of packings indexed by one or two integer
parameters, with the side length ``s`` given in closed form.  For a requested
``n`` the family member with the smallest ``s`` whose capacity is at least
``n`` is returned, with the surplus squares removed (removing squares from a
valid packing keeps it valid).  All constructions are O(n) time and space -
linear in the size of the output - and produce packings that pass
:func:`squarepack.geometry.verify` with tolerance ~1e-12.

Families
--------
grid
    The trivial packing, ``s = ceil(sqrt(n))``.  Optimal for n = k^2, k^2-1,
    k^2-2 (Nagamochi 2005) and for a handful of other small n.
gobel_strip
    Göbel (1979), generalised by Friedman / Ellsworth.  A 45-degree strip of
    ``b = 1 + floor((a-1) sqrt 2)`` squares along the main diagonal, flanked by
    an unrotated square in each of the two diagonal corners and sandwiched by
    two unrotated staircases of ``a`` steps.  ``n = a(a+1) + 2 + b``,
    ``s = a + 1 + sqrt(2)/2``.  Best known for 27, 38, 52, 67, 84, 104, ...
gobel_square
    Göbel (1979).  A ``b x b`` block of squares rotated 45 degrees in the
    centre, surrounded by four unrotated staircases of ``a`` steps; valid when
    ``a - 1 < b/sqrt 2 < a + 1``.  ``n = 2a(a+1) + b^2``,
    ``s = a + 1 + b sqrt(2)/2``.  Best known for 5, 40, 65, 89, ...  (and the
    record families 82, 101, 122, ... obtained by adding "L"s to 65).
add_L
    A packing of side ``s`` becomes a packing of side ``s + 1`` holding
    ``floor(s + 1) + floor(s)`` more squares (a row on top and a column on the
    right).  Used to extend any family member to larger ``n``.

The single-angle record families of :mod:`squarepack.families` (the sqrt(7)
family of Hämäläinen/Friedman: 18, 53, 86, 127, ...; DeVincentis' family:
54, 107, 178, 267, ...; Wainwright's 19; Schadt's 50 and 171) are part of the
portfolio of :func:`analytic_candidates` as well.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from .geometry import QUARTER_PI, SQRT2

EPS = 1e-9


@dataclass
class Packing:
    """A packing of ``n`` unit squares in ``[0, s]^2``.

    ``squares`` has shape ``(n, 3)`` with rows ``(x_center, y_center, angle_radians)``.
    """
    n: int
    s: float
    squares: np.ndarray
    method: str
    exact: str = ""
    meta: Dict = field(default_factory=dict)

    def take(self, n: int) -> "Packing":
        """Keep the first ``n`` squares (the constructions order the most dispensable last)."""
        if n > self.n:
            raise ValueError(f"cannot take {n} squares from a packing of {self.n}")
        return replace(self, n=n, squares=np.array(self.squares[:n], dtype=float, copy=True))

    def as_list(self, degrees: bool = False) -> List[List[float]]:
        out = np.array(self.squares, dtype=float)
        if degrees:
            out[:, 2] = np.degrees(out[:, 2])
        return out.tolist()


def _rows(n_side: int, n: int):
    """Row-major centres of the first ``n`` cells of an ``n_side x n_side`` grid."""
    idx = np.arange(n)
    return np.column_stack([(idx % n_side) + 0.5, (idx // n_side) + 0.5, np.zeros(n)])


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
def grid(n: int) -> Packing:
    k = math.isqrt(n)
    if k * k < n:
        k += 1
    return Packing(n, float(k), _rows(k, n), "grid", exact=str(k), meta={"k": k})


# --------------------------------------------------------------------------- #
# Göbel strip
# --------------------------------------------------------------------------- #
def gobel_strip_b(a: int) -> int:
    """Number of 45-degree squares in the strip of the Göbel strip with ``a`` steps."""
    return 1 + math.floor((a - 1) * SQRT2 + EPS)


def gobel_strip_capacity(a: int) -> int:
    return a * (a + 1) + 2 + gobel_strip_b(a)


def gobel_strip_member(a: int) -> Packing:
    """The full Göbel strip packing with ``a`` staircase steps (a >= 1)."""
    if a < 1:
        raise ValueError("a must be >= 1")
    k = a + 1
    b = gobel_strip_b(a)
    s = k + SQRT2 / 2
    sq: List[List[float]] = []
    # two staircases: cells (i, j) with i + j <= a - 1, anchored at the top-left and bottom-right corners
    for i in range(a):
        for j in range(a - i):
            sq.append([i + 0.5, s - j - 0.5, 0.0])
            sq.append([s - i - 0.5, j + 0.5, 0.0])
    # corner squares on the diagonal
    sq.append([0.5, 0.5, 0.0])
    sq.append([s - 0.5, s - 0.5, 0.0])
    # the 45-degree strip: in diagonal coordinates u = (x + y)/sqrt2 the free
    # segment is [sqrt2, s*sqrt2 - sqrt2]; centre the b unit-long diamonds in it.
    free = s * SQRT2 - 2 * SQRT2
    u0 = SQRT2 + (free - b) / 2.0 + 0.5
    for m in range(b):
        u = u0 + m
        sq.append([u / SQRT2, u / SQRT2, QUARTER_PI])
    arr = np.array(sq, dtype=float)
    return Packing(len(arr), s, arr, "gobel_strip",
                   exact=f"{k} + sqrt(2)/2", meta={"a": a, "b": b})


def gobel_strip(n: int) -> Optional[Packing]:
    """Smallest Göbel strip holding ``n`` squares (None when the trivial grid is at least as good)."""
    a = 1
    while gobel_strip_capacity(a) < n:
        a += 1
    p = gobel_strip_member(a)
    if p.s >= grid(n).s - EPS:
        return None
    return p.take(n)


# --------------------------------------------------------------------------- #
# Göbel square
# --------------------------------------------------------------------------- #
def gobel_square_valid(a: int, b: int) -> bool:
    return a >= 1 and b >= 1 and (a - 1) < b / SQRT2 - EPS and b / SQRT2 < (a + 1) - EPS


def gobel_square_capacity(a: int, b: int) -> int:
    return 2 * a * (a + 1) + b * b


def gobel_square_member(a: int, b: int) -> Packing:
    """The Göbel square with ``a`` staircase steps and a central ``b x b`` tilted block."""
    if not gobel_square_valid(a, b):
        raise ValueError(f"(a, b) = ({a}, {b}) violates a - 1 < b/sqrt2 < a + 1")
    s = a + 1 + b * SQRT2 / 2
    c = s / 2.0
    sq: List[List[float]] = []
    for i in range(a):
        for j in range(a - i):
            sq.append([i + 0.5, j + 0.5, 0.0])
            sq.append([s - i - 0.5, j + 0.5, 0.0])
            sq.append([i + 0.5, s - j - 0.5, 0.0])
            sq.append([s - i - 0.5, s - j - 0.5, 0.0])
    h = 1.0 / SQRT2
    # tilted block: order from the centre outwards so that the outermost (corner) tiles are dropped first
    tiles = []
    for i in range(b):
        for j in range(b):
            u = i - (b - 1) / 2.0
            v = j - (b - 1) / 2.0
            tiles.append((abs(u) + abs(v), u, v))
    tiles.sort()
    for _, u, v in tiles:
        sq.append([c + (u + v) * h, c + (u - v) * h, QUARTER_PI])
    arr = np.array(sq, dtype=float)
    return Packing(len(arr), s, arr, "gobel_square",
                   exact=f"{a + 1} + {b}*sqrt(2)/2", meta={"a": a, "b": b})


def gobel_square_members(n: int) -> List[Packing]:
    """All Göbel squares that hold ``n`` squares and beat the grid, cheapest first."""
    s_grid = grid(n).s
    out: List[Packing] = []
    a = 1
    while True:
        # smallest s for this a uses the smallest admissible b
        b_lo = max(1, math.floor((a - 1) * SQRT2) + 1)
        if a + 1 + b_lo * SQRT2 / 2 >= s_grid - EPS and 2 * a * (a + 1) >= n:
            break
        if a > 2 * math.isqrt(n) + 4:
            break
        b = b_lo
        while b / SQRT2 < a + 1 - EPS:
            if gobel_square_valid(a, b) and gobel_square_capacity(a, b) >= n:
                s = a + 1 + b * SQRT2 / 2
                if s < s_grid - EPS:
                    out.append(gobel_square_member(a, b).take(n))
                break  # larger b only increases s
            b += 1
        a += 1
    out.sort(key=lambda p: p.s)
    return out


def gobel_square(n: int) -> Optional[Packing]:
    members = gobel_square_members(n)
    return members[0] if members else None


# --------------------------------------------------------------------------- #
# L extension
# --------------------------------------------------------------------------- #
def add_L(p: Packing) -> Packing:
    """Extend a packing of side ``s`` to side ``s + 1`` by a row of ``floor(s+1)`` squares on
    top and a column of ``floor(s)`` squares on the right."""
    s = p.s
    top = math.floor(s + 1 + EPS)
    right = math.floor(s + EPS)
    extra = [[i + 0.5, s + 0.5, 0.0] for i in range(top)]
    extra += [[s + 0.5, j + 0.5, 0.0] for j in range(right)]
    arr = np.vstack([p.squares, np.array(extra, dtype=float)])
    return Packing(len(arr), s + 1.0, arr, p.method + "+L",
                   exact=f"({p.exact}) + 1" if p.exact else "", meta=dict(p.meta, base_n=p.n))


# --------------------------------------------------------------------------- #
# portfolio of analytic candidates
# --------------------------------------------------------------------------- #
@dataclass
class Member:
    """A family member described arithmetically: side, capacity and a builder that
    constructs the packing only when the member is selected (keeps the portfolio O(n))."""
    s: float
    n: int
    family: str
    build: Callable[[], Packing]


def member_descriptors(n: int) -> List[Member]:
    """Every family member with side below the grid side for ``n``, without building it."""
    from .families import member_descriptors as family_descriptors   # lazy: families imports this module

    s_grid = grid(n).s
    out: List[Member] = []
    a = 1
    while a + 1 + SQRT2 / 2 < s_grid - EPS:
        out.append(Member(a + 1 + SQRT2 / 2, gobel_strip_capacity(a), "gobel_strip", lambda a=a: gobel_strip_member(a)))
        a += 1
    a = 1
    while a + 1 + SQRT2 / 2 < s_grid - EPS:
        b = max(1, math.floor((a - 1) * SQRT2) + 1)
        while b / SQRT2 < a + 1 - EPS:
            s = a + 1 + b * SQRT2 / 2
            if gobel_square_valid(a, b) and s < s_grid - EPS:
                out.append(Member(s, gobel_square_capacity(a, b), "gobel_square",
                                  lambda a=a, b=b: gobel_square_member(a, b)))
            b += 1
        a += 1
    out.extend(family_descriptors(s_grid, n))
    return out


def l_chain(s0: float, n0: int, n: int, s_max: float):
    """Number of "L" extensions that take a member ``(s0, n0)`` to capacity ``n`` while staying
    below ``s_max``; returns ``(j, s0 + j)`` or ``None``.  Pure arithmetic (mirrors :func:`add_L`)."""
    s, m, j = s0, n0, 0
    while m < n:
        if s + 1 >= s_max - EPS:
            return None
        m += math.floor(s + 1 + EPS) + math.floor(s + EPS)
        s += 1.0
        j += 1
    return j, s


def add_Ls(p: Packing, j: int) -> Packing:
    """``add_L`` applied ``j`` times, generating all the extra squares at once (O(n) total)."""
    if j <= 0:
        return p
    s = p.s
    extra: List[List[float]] = []
    for i in range(j):
        si = s + i
        top = math.floor(si + 1 + EPS)
        right = math.floor(si + EPS)
        extra += [[t + 0.5, si + 0.5, 0.0] for t in range(top)]
        extra += [[si + 0.5, r + 0.5, 0.0] for r in range(right)]
    arr = np.vstack([p.squares, np.array(extra, dtype=float).reshape(-1, 3)])
    return Packing(len(arr), s + j, arr, p.method + "+L" * j,
                   exact=f"({p.exact}) + {j}" if p.exact else "", meta=dict(p.meta, base_n=p.n))


def analytic_candidates(n: int, top: int = 6) -> List[Packing]:
    """The best analytic packings for ``n`` (grid plus the ``top`` best family members, each
    extended by "L"s as needed), best first.

    Selection is arithmetic: every member contributes its closed-form side and capacity, the
    number of L extensions is computed from the capacity recurrence, and only the winners are
    built, so the whole portfolio costs O(n) plus O(sqrt(n)) descriptors.
    """
    s_grid = grid(n).s
    scored = []
    for d in member_descriptors(n):
        r = l_chain(d.s, d.n, n, s_grid)
        if r is not None:
            scored.append((r[1], r[0], d))
    scored.sort(key=lambda t: t[0])
    cands: List[Packing] = [grid(n)]
    for s_j, j, d in scored[:top]:
        p = add_Ls(d.build(), j).take(n)
        assert abs(p.s - s_j) < 1e-9 and p.n == n
        cands.append(p)
    cands.sort(key=lambda q: (q.s, q.method != "grid"))
    return cands


def best_analytic(n: int) -> Packing:
    return analytic_candidates(n)[0]


def _family(name: str) -> Callable[[int], Optional[Packing]]:
    def f(n: int) -> Optional[Packing]:
        from . import families
        return families.FAMILIES[name](n)
    f.__name__ = f"family_{name}"
    f.__doc__ = f"Smallest member of the {name} family (squarepack.families) holding n squares."
    return f


CONSTRUCTIONS: Dict[str, Callable[[int], Optional[Packing]]] = {
    "grid": grid,
    "gobel_strip": gobel_strip,
    "gobel_square": gobel_square,
    "sqrt7": _family("sqrt7"),
    "devincentis": _family("devincentis"),
    "wainwright": _family("wainwright"),
    "schadt": _family("schadt"),
}
