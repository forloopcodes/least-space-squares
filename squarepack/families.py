"""Closed-form record families with a single rotation angle.

The record packings implemented here all share one mechanism: a diagonal
channel between two axis-aligned staircases (plus a few corner cells) is
filled by unit squares that all have the *same* rotation angle ``a``.  In
the frame rotated by ``a`` those squares are axis aligned, so they are
non-overlapping as soon as they occupy disjoint unit strips ("lanes") or are
consecutive along a chain.  Their positions are fixed by *contacts* with the
staircases, which we compute exactly with a separating-axis "sliding square"
kernel (:class:`Frame`): the set of positions of a unit square sliding along
one frame axis that overlap an axis-aligned rectangle is an open interval,
so the first admissible position is the end of a union of intervals.

Families
--------
sqrt7 (Hämäläinen 1980, Gustafsson 1981, Friedman 1997, Cantrell, Ellsworth)
    ``s = m - 1/2 + sqrt(7)/2``, ``m = floor(s) >= 4``; angle
    ``a = arctan((4 - sqrt 7)/3)``.  Two staircases of ``m-2`` and ``m-1``
    rows, corner cells and a band of 1x2 dominoes tilted by ``a``, stacked
    side by side in lanes with half-unit lateral shifts: each domino is
    squeezed end-to-end between a corner of one staircase and a corner of
    the other, which are exactly 2 apart in the tilted frame.  Members
    ``m = 4, 9, 11, 13, 14, 17`` are the records 18, 86, 127, 176, 204, 299;
    ``m = 7`` (53) uses the 2024/26 refinement of the drawing with two extra
    algebraic angles (see :func:`sqrt7_member_53`).
devincentis (Cantrell 2005, DeVincentis 2014, Hajba 2024, Ellsworth)
    ``s = 7 - sqrt2/2 + sqrt(1 + sqrt2) + 3k``; angle
    ``a = pi/2 - arctan((3 + sqrt2 + 2 sqrt(1 + 5 sqrt2))/7)``.  Two rows of
    squares chained along the tilted axis with a per-step drift of
    ``cos a - sin a``; the pattern repeats every 3 units along the diagonal,
    ``n = 9k^2 + 44k + 54`` (54, 107, 178, 267, ...).
wainwright (Wainwright 1979)
    ``n = 19``, ``s = 3 + 4 sqrt2/3``: two pairs of 45-degree dominoes with
    the two middle columns shifted by half the diagonal slack.
schadt (Schadt 2025, Ellsworth 2025)
    ``n = 50``, ``s = 7 + 4/7`` with the 3-4-5 angle ``arctan(3/4)`` (all
    coordinates rational) and ``n = 171``, ``s = 13 + 4/7`` (two copies).

All members are verified with :func:`squarepack.geometry.verify` at 1e-9
before being returned; construction is O(n).
"""
from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constructions import EPS, Packing, grid
from .geometry import verify

SQRT2 = math.sqrt(2.0)
SQRT7 = math.sqrt(7.0)
TOUCH = 1e-10   # two squares closer than this are considered touching, not overlapping

Rect = Tuple[float, float, float, float]   # x0, x1, y0, y1 (axis-aligned obstacle)


# --------------------------------------------------------------------------- #
# exact sliding-square kernel
# --------------------------------------------------------------------------- #
class Frame:
    """Orthonormal frame ``(d, e)``: unit squares occupy ``[u, u+1] x [v, v+1]``
    in frame coordinates (``u`` along ``d``, ``v`` along ``e``) and are moved
    along ``d`` only.  The frame's squares all have the same world angle."""

    def __init__(self, d: Sequence[float], e: Sequence[float]):
        self.d = np.asarray(d, float)
        self.e = np.asarray(e, float)
        # half side of the axis-aligned bounding box of a unit square of this frame
        self.h = 0.5 * (abs(self.d[0]) + abs(self.e[0]))
        self.angle = math.atan2(self.d[1], self.d[0])

    def intervals(self, v0: np.ndarray, rects: np.ndarray, s: float):
        """Forbidden open intervals ``(lo, hi)`` (shape ``(P, C)``) of ``u`` for squares
        in strips ``[v0, v0+1]`` against rectangles ``rects`` (``(C, 4)``), and the
        wall interval ``(wlo, whi)`` (shape ``(P,)``).  Dropped intervals are ``(inf, -inf)``."""
        d, e, h = self.d, self.e, self.h
        v0 = np.asarray(v0, float).reshape(-1)
        rects = np.asarray(rects, float).reshape(-1, 4)
        cv = (v0 + 0.5)[:, None]
        P, C = len(v0), len(rects)
        lo = np.full((P, C), -np.inf)
        hi = np.full((P, C), np.inf)
        wlo = np.full(P, -np.inf)
        whi = np.full(P, np.inf)
        dead = np.zeros(P, bool)
        for ax in (0, 1):
            x0 = rects[None, :, 2 * ax]
            x1 = rects[None, :, 2 * ax + 1]
            if abs(d[ax]) < 1e-14:   # projection independent of u
                c = cv * e[ax]
                sep = (c + h <= x0 + TOUCH) | (c - h >= x1 - TOUCH)
                lo = np.where(sep, np.inf, lo)
                hi = np.where(sep, -np.inf, hi)
                cw = (v0 + 0.5) * e[ax]
                dead |= (cw < h - TOUCH) | (cw > s - h + TOUCH)
                continue
            t1 = (x0 - h - cv * e[ax]) / d[ax] - 0.5   # square entirely below the rectangle on this axis
            t2 = (x1 + h - cv * e[ax]) / d[ax] - 0.5   # square entirely above
            lo = np.maximum(lo, np.minimum(t1, t2))
            hi = np.minimum(hi, np.maximum(t1, t2))
            ta = (h - (v0 + 0.5) * e[ax]) / d[ax] - 0.5
            tb = (s - h - (v0 + 0.5) * e[ax]) / d[ax] - 0.5
            wlo = np.maximum(wlo, np.minimum(ta, tb))
            whi = np.minimum(whi, np.maximum(ta, tb))
        X = rects[:, [0, 0, 1, 1]]
        Y = rects[:, [2, 3, 2, 3]]
        pd = X * d[0] + Y * d[1]
        pe = X * e[0] + Y * e[1]
        lo = np.maximum(lo, (pd.min(1) - 1.0)[None, :])
        hi = np.minimum(hi, pd.max(1)[None, :])
        sep_e = (v0[:, None] + 1.0 <= pe.min(1)[None, :] + TOUCH) | (v0[:, None] >= pe.max(1)[None, :] - TOUCH)
        drop = sep_e | (lo >= hi)
        lo = np.where(drop, np.inf, lo)
        hi = np.where(drop, -np.inf, hi)
        whi = np.where(dead, -np.inf, whi)
        return lo, hi, wlo, whi

    def first(self, v0, rects, s, L=None) -> np.ndarray:
        """Smallest admissible ``u >= L`` for each strip (``nan`` when none)."""
        lo, hi, wlo, whi = self.intervals(v0, rects, s)
        P = len(lo)
        Lo = np.full(P, -np.inf) if L is None else np.asarray(L, float).reshape(-1)
        start = np.maximum(Lo, wlo)
        cand = np.concatenate([start[:, None], hi], axis=1)            # (P, C+1)
        inside = (lo[:, None, :] + TOUCH < cand[:, :, None]) & (cand[:, :, None] < hi[:, None, :] - TOUCH)
        ok = ~inside.any(axis=2) & (cand >= start[:, None] - TOUCH) & (cand <= whi[:, None] + TOUCH)
        ok &= np.isfinite(cand)
        cand = np.where(ok, cand, np.inf)
        u = cand.min(axis=1)
        return np.where(np.isfinite(u), u, np.nan)

    def last(self, v0, rects, s, U=None) -> np.ndarray:
        """Largest admissible ``u <= U`` for each strip (``nan`` when none)."""
        lo, hi, wlo, whi = self.intervals(v0, rects, s)
        P = len(lo)
        Up = np.full(P, np.inf) if U is None else np.asarray(U, float).reshape(-1)
        start = np.minimum(Up, whi)
        cand = np.concatenate([start[:, None], lo], axis=1)
        inside = (lo[:, None, :] + TOUCH < cand[:, :, None]) & (cand[:, :, None] < hi[:, None, :] - TOUCH)
        ok = ~inside.any(axis=2) & (cand <= start[:, None] + TOUCH) & (cand >= wlo[:, None] - TOUCH)
        ok &= np.isfinite(cand)
        cand = np.where(ok, cand, -np.inf)
        u = cand.max(axis=1)
        return np.where(np.isfinite(u), u, np.nan)

    def square(self, u: float, v0: float) -> List[float]:
        c = (u + 0.5) * self.d + (v0 + 0.5) * self.e
        return [float(c[0]), float(c[1]), self.angle]

    def squares(self, u: np.ndarray, v0: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float); v0 = np.asarray(v0, float)
        c = (u + 0.5)[:, None] * self.d[None, :] + (v0 + 0.5)[:, None] * self.e[None, :]
        return np.column_stack([c, np.full(len(u), self.angle)])


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def rect_cells(x0: float, x1: float, y0: float, y1: float) -> List[List[float]]:
    """Unit cells of an integer-sized axis-aligned rectangle (row major from the corner)."""
    nx, ny = int(round(x1 - x0)), int(round(y1 - y0))
    return [[x0 + i + 0.5, y0 + j + 0.5, 0.0] for j in range(ny) for i in range(nx)]


def rects_cells(rects: Sequence[Rect]) -> List[List[float]]:
    out: List[List[float]] = []
    for r in rects:
        out += rect_cells(*r)
    return out


def rect_area(r: Rect) -> int:
    return int(round((r[1] - r[0]) * (r[3] - r[2])))


def rects_overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[1] - TOUCH and b[0] < a[1] - TOUCH and a[2] < b[3] - TOUCH and b[2] < a[3] - TOUCH


def cells_clear(cells: Sequence[Rect], squares: np.ndarray) -> bool:
    """True when the axis-aligned unit ``cells`` overlap none of the rotated ``squares``."""
    if len(squares) == 0 or len(cells) == 0:
        return True
    sq = np.asarray(squares, float)
    cx = np.array([(c[0] + c[1]) / 2 for c in cells])
    cy = np.array([(c[2] + c[3]) / 2 for c in cells])
    dx = sq[:, 0][:, None] - cx[None, :]
    dy = sq[:, 1][:, None] - cy[None, :]
    t = sq[:, 2][:, None]
    c, s_ = np.cos(t), np.sin(t)
    proj = np.maximum.reduce([np.abs(dx), np.abs(dy), np.abs(c * dx + s_ * dy), np.abs(-s_ * dx + c * dy)])
    need = 0.5 + 0.5 * (np.abs(c) + np.abs(s_))
    return bool(np.all(proj >= need - TOUCH))


def _finish(n_target: Optional[int], s: float, squares: np.ndarray, method: str, exact: str, meta: Dict) -> Packing:
    arr = np.asarray(squares, float).reshape(-1, 3)
    rep = verify(s, arr, 1e-9)
    if not rep.ok:   # pragma: no cover - construction bug guard
        raise RuntimeError(f"{method}: construction is invalid: {rep}")
    p = Packing(len(arr), float(s), arr, method, exact=exact, meta=dict(meta))
    return p if n_target is None else p.take(n_target)


# --------------------------------------------------------------------------- #
# sqrt(7) family
# --------------------------------------------------------------------------- #
F7 = (SQRT7 - 1.0) / 2.0                    # fractional part of s
A7 = math.atan((4.0 - SQRT7) / 3.0)         # 24.295 degrees


def sqrt7_side(m: int) -> float:
    return m + F7


def _sqrt7_staircases(m: int, s: float) -> Tuple[List[Rect], List[Rect], Rect]:
    A = [(0.0, float(m - 2 - i), s - 1 - i, s - i) for i in range(m - 2)]              # top-left, rows m-2 .. 1
    B = [(s - (m - 1) + j, s, float(j), float(j + 1)) for j in range(m - 2)]           # bottom-right, rows m-1 .. 2
    rowcell = (s - 1, s, float(m - 2), float(m - 1))                                    # the "row 1" cell of B
    return A, B, rowcell


def _sqrt7_band(fr: Frame, s: float, rects: List[Rect], phases: np.ndarray, deltas: Sequence[float]):
    """Domino band for every phase: returns per phase a list of ``(k, [u0, u1] or [u0])``."""
    R = np.asarray(rects, float)
    vmax = s * (abs(fr.e[0]) + abs(fr.e[1]))
    ks = np.arange(-1, int(math.ceil(vmax)) + 1)
    P, K = len(phases), len(ks)
    V0 = (phases[:, None] + ks[None, :]).reshape(-1)
    u0 = fr.first(V0, R, s).reshape(P, K)
    results = []
    for delta in deltas:
        V1 = V0 + delta
        # lower bound for column 1: clear of column-0 squares of the overlapping lanes
        L = np.where(np.isnan(u0), -np.inf, u0) + 1.0
        if delta > TOUCH:
            nb = np.concatenate([L[:, 1:], np.full((P, 1), -np.inf)], axis=1)
            L = np.maximum(L, nb)
        elif delta < -TOUCH:
            nb = np.concatenate([np.full((P, 1), -np.inf), L[:, :-1]], axis=1)
            L = np.maximum(L, nb)
        u1 = fr.first(V1, R, s, L=L.reshape(-1)).reshape(P, K)
        if abs(delta) > TOUCH:
            u1 = np.where(np.isnan(u0), np.nan, u1)   # a lone column-1 square would need its own checks
        results.append((delta, u0, u1, ks))
    return results


@lru_cache(maxsize=None)
def sqrt7_member(m: int) -> Packing:
    """The sqrt(7) family member with ``floor(s) = m`` (``s = m - 1/2 + sqrt7/2``), ``m >= 4``."""
    if m < 4:
        raise ValueError("m must be >= 4")
    if m == 4:
        return sqrt7_member_18()
    if m == 7:
        return sqrt7_member_53()
    s = sqrt7_side(m)
    a = A7
    ev = np.array([math.cos(a), math.sin(a)])
    eu = np.array([-math.sin(a), math.cos(a)])
    fr = Frame(eu, ev)
    A, B, rowcell = _sqrt7_staircases(m, s)
    nphase = 64
    phases = np.arange(nphase) / nphase
    deltas = (0.0, 0.03, 0.06, -0.03, -0.06)
    best = None
    for rc in (0, 1):
        rects = A + B + ([rowcell] if rc else [])
        n_static = sum(rect_area(r) for r in rects)
        for delta, u0, u1, ks in _sqrt7_band(fr, s, rects, phases, deltas):
            for ip, phi in enumerate(phases):
                lanes = []
                for ik, k in enumerate(ks):
                    a0, a1 = u0[ip, ik], u1[ip, ik]
                    if np.isnan(a0):
                        continue
                    lanes.append((k, a0, None if np.isnan(a1) else a1))
                if not lanes:
                    continue
                band = np.array([fr.square(a0, phi + k) for k, a0, _ in lanes]
                                + [fr.square(a1, phi + k + delta) for k, _, a1 in lanes if a1 is not None])
                fills = _sqrt7_corner_fill(s, rects, band)
                total = n_static + len(band) + sum(rect_area(c) for c in fills)
                key = (total, -abs(delta))
                if best is None or key > best[0]:
                    best = (key, rc, phi, delta, lanes, fills)
    (total, _), rc, phi, delta, lanes, fills = best
    rects = A + B + ([rowcell] if rc else [])
    squares: List[List[float]] = rects_cells(A) + rects_cells(B) + (rect_cells(*rowcell) if rc else [])
    squares += rects_cells(fills)
    # band lanes from the middle outwards (the ends are the most dispensable), column 0 before column 1
    mid = (lanes[0][0] + lanes[-1][0]) / 2.0
    for k, a0, a1 in sorted(lanes, key=lambda t: (abs(t[0] - mid), t[0])):
        squares.append(fr.square(a0, phi + k))
        if a1 is not None:
            squares.append(fr.square(a1, phi + k + delta))
    return _finish(None, s, np.array(squares), "sqrt7", exact=f"{2 * m - 1}/2 + sqrt(7)/2",
                   meta={"m": m, "phase": float(phi), "delta": float(delta), "rowcell": rc,
                         "lanes": len(lanes), "corner_cells": sum(rect_area(c) for c in fills)})


def _sqrt7_corner_fill(s: float, static: List[Rect], band: np.ndarray) -> List[Rect]:
    """Largest of a few corner blocks that fit at the bottom-left and top-right corners."""
    out: List[Rect] = []
    bl_opts = [[(0.0, 1.0, 0.0, 3.0)], [(0.0, 1.0, 0.0, 2.0)], [(0.0, 2.0, 0.0, 1.0)], [(0.0, 1.0, 0.0, 1.0)]]
    tr_opts = [[(s - 2, s, s - 2, s)], [(s - 2, s, s - 1, s)], [(s - 1, s, s - 2, s)], [(s - 1, s, s - 1, s)]]
    for opts in (bl_opts, tr_opts):
        for blocks in opts:
            cells = [c for b in blocks for c in
                     [(b[0] + i, b[0] + i + 1, b[2] + j, b[2] + j + 1)
                      for i in range(rect_area((b[0], b[1], 0, 1))) for j in range(rect_area((0, 1, b[2], b[3])))]]
            if any(rects_overlap(c, r) for c in cells for r in static + out):
                continue
            if cells_clear(cells, band):
                out += blocks
                break
    return out


# -- the two special members ------------------------------------------------- #
class _Xf:
    """Tiny SVG-like transform stack (3x3 homogeneous matrices) used to transcribe drawings."""

    def __init__(self, M=None):
        self.M = np.eye(3) if M is None else M

    def translate(self, tx: float, ty: float = 0.0) -> "_Xf":
        T = np.eye(3); T[0, 2], T[1, 2] = tx, ty
        return _Xf(self.M @ T)

    def rotate(self, rad: float) -> "_Xf":
        c, s_ = math.cos(rad), math.sin(rad)
        return _Xf(self.M @ np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]))

    def scale(self, sx: float, sy: float) -> "_Xf":
        T = np.eye(3); T[0, 0], T[1, 1] = sx, sy
        return _Xf(self.M @ T)

    def square(self, cx: float, cy: float) -> List[float]:
        """World centre/angle of the unit square whose local centre is ``(cx, cy)``."""
        p = self.M @ np.array([cx, cy, 1.0])
        ex = self.M[:2, :2] @ np.array([1.0, 0.0])
        return [float(p[0]), float(p[1]), math.atan2(ex[1], ex[0])]

    def rect(self, x0: float, y0: float, w: int, h: int) -> List[List[float]]:
        return [self.square(x0 + i + 0.5, y0 + j + 0.5) for j in range(h) for i in range(w)]


def _flip_y(s: float, squares: Sequence[Sequence[float]]) -> np.ndarray:
    """SVG drawings use y pointing down; convert to y up (angles change sign)."""
    arr = np.asarray(squares, float).reshape(-1, 3).copy()
    arr[:, 1] = s - arr[:, 1]
    arr[:, 2] = -arr[:, 2]
    arr[:, 2] = (arr[:, 2] + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return arr


def _rot(theta: float, v: Sequence[float]) -> np.ndarray:
    c, s_ = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])


def sqrt7_member_18() -> Packing:
    """Hämäläinen's n = 18, s = 7/2 + sqrt7/2 (Gustafsson's form with 8 rotated squares)."""
    s = sqrt7_side(4)
    a = A7
    ca, sa, ta = math.cos(a), math.sin(a), math.tan(a)
    x1 = 2.5 + SQRT7 / 2 - ca + 0.75 * sa
    y1 = 1 - sa + 0.75 * sa * ta
    x2 = 1.5 * sa * ca - 2 * sa + SQRT7 / 2 * sa * ca + 2.5 - ca ** 2 + SQRT7 / 2 - ca
    y2 = 2.5 - 1.5 * ca ** 2 + 2 * ca + SQRT7 / 2 - SQRT7 / 2 * ca ** 2 - sa * ca - sa
    side = []
    side += _Xf().rect(0, 0, 1, 3) + _Xf().rect(1, 0, 1, 1)          # L in the corner
    side += _Xf().translate(0, s).scale(1, -1).rect(0, 0, 1, 1)        # square in the opposite corner
    for (x, y) in ((x1, y1), (x2, y2)):
        side += _Xf().translate(x, y).rotate(a).rect(0, 0, 1, 2)       # two tilted dominoes
    other = [[s - x, s - y, t] for x, y, t in side]                    # 180-degree copy
    order = side[:5] + other[:5] + side[5:] + other[5:]
    arr = np.array(order, float)
    arr[:, 2] = (arr[:, 2] + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return _finish(None, s, arr, "sqrt7", exact="7/2 + sqrt(7)/2", meta={"m": 4})


def sqrt7_member_53() -> Packing:
    """n = 53, s = 13/2 + sqrt7/2 (Cantrell 2002/2024, Ellsworth 2026).

    The band uses the sqrt7 angle ``a`` for two dominoes, and two extra
    angles ``b`` and ``c`` for the others; ``tan b`` and ``tan c`` are roots of
    integer polynomials of degree 24 and 12 (given in the drawing) - we use
    their 50-digit values.  Every other length is closed-form in ``s, a, b, c``
    (the derivation comments of the drawing), so the construction is exact
    up to the two algebraic angles.
    """
    s = sqrt7_side(7)
    a = A7
    b = math.radians(26.598224749160228085259276387718886680278104214204)
    c = math.radians(32.012452423751345634083231604618271105356637647095)
    P = np.array([1.0, 5.0])
    base = P + _rot(-a, [2 - (s - 7) * math.sin(a), -(math.cos(a) - math.sin(a))])
    r1 = (s - 7) * math.sin(a)
    v2 = math.cos(a) - math.sin(a)
    r3 = ((base + _rot(-b, [1, 0]))[1] - 3) / math.cos(b)
    v5 = _rot(b, np.array([3.0, 3.0]) - (base + _rot(-b, [1, -r3])))[1]
    v6 = math.cos(b) - math.sin(b)
    u7 = ((base + _rot(-b, [2, 2 - r3 + v5 - v6]))[1] - (s - 4)) / math.sin(b)
    Q = np.array([5.0, 1.0])
    r8 = _rot(b, (Q + _rot(-a, [0, 1])) - (base + _rot(-b, [3, -r3 + v5 - v6])))[0] / math.cos(b - a)
    u9 = r8 - _rot(b, (Q + _rot(-a, [0, 2])) - (base + _rot(-b, [3 + u7, -r3 + v5 - v6])))[0] / math.cos(b - a)
    rA = (s - 7) * math.sin(c)
    uB = -(np.array([s - 1, s - 6]) + _rot(-c, [rA, -2]))[1] / math.sin(c)
    sq: List[List[float]] = []
    # staircases (SVG coordinates, y down): rows of widths 5..1 and 6..1
    for r in range(5):
        sq += _Xf().rect(0, r, 5 - r, 1)
    for r in range(6):
        sq += _Xf().translate(s, s).scale(-1, -1).rect(0, r, 6 - r, 1)
    sq += _Xf().translate(s, 0).scale(-1, 1).rect(0, 0, 1, 1)
    sq += _Xf().translate(0, s).scale(1, -1).rect(0, 0, 1, 2)
    g = _Xf().translate(1, 5).rotate(-a).translate(-r1, 0)
    sq += g.rect(0, 0, 1, 2)
    g = g.translate(1, -v2)
    sq += g.rect(0, 0, 1, 2)
    g = g.translate(1, 0).rotate(a).rotate(-b).translate(0, -r3)
    sq += g.rect(0, 0, 1, 2)
    g = g.translate(1, v5)
    sq += g.rect(0, 0, 1, 2)
    g = g.translate(1, -v6)
    sq += g.rect(0, 0, 1, 1) + g.translate(u7, 1).rect(0, 0, 1, 1)
    g = _Xf().translate(5, 1).rotate(-a).translate(-r8, 0)
    sq += g.rect(0, 0, 1, 1) + g.translate(u9, 1).rect(0, 0, 1, 1)
    g = _Xf().translate(s, s).translate(-1, -6).rotate(-c).translate(rA, -1).translate(-1, 0)
    sq += g.rect(0, 0, 1, 1) + g.translate(-uB, -1).rect(0, 0, 1, 1)
    return _finish(None, s, _flip_y(s, sq), "sqrt7", exact="13/2 + sqrt(7)/2",
                   meta={"m": 7, "extra_angles_deg": (26.598224749160228, 32.012452423751346)})


SQRT7_CAPACITY = {4: 18, 5: 27, 6: 39, 7: 53, 8: 68, 9: 86, 10: 105, 11: 127, 12: 151, 13: 176, 14: 204, 15: 234, 16: 265, 17: 299, 18: 334, 19: 372, 20: 412, 21: 453, 22: 497, 23: 543, 24: 590, 25: 640, 26: 692, 27: 745, 28: 801, 29: 859, 30: 918, 31: 980, 32: 1044, 33: 1109, 34: 1177, 35: 1246, 36: 1318, 37: 1392, 38: 1467, 39: 1545, 40: 1625, 41: 1706, 42: 1790, 43: 1876, 44: 1963, 45: 2053, 46: 2145, 47: 2238, 48: 2334, 49: 2431, 50: 2531, 51: 2633, 52: 2736, 53: 2842, 54: 2950, 55: 3059, 56: 3171, 57: 3285, 58: 3400, 59: 3518, 60: 3638, 61: 3759, 62: 3883, 63: 4009, 64: 4136, 65: 4266, 66: 4397, 67: 4531, 68: 4667, 69: 4804, 70: 4944, 71: 5086, 72: 5229, 73: 5375, 74: 5523, 75: 5672, 76: 5824, 77: 5978, 78: 6133, 79: 6291, 80: 6450}


def sqrt7_capacity(m: int) -> Optional[int]:
    """Capacity of the sqrt7 member ``m`` (tabulated for m <= 80; None beyond the table, where the
    member engine would cost ~0.4 s per m and the family is left out of the O(n) portfolio)."""
    return SQRT7_CAPACITY.get(m)


def family_sqrt7(n: int) -> Optional[Packing]:
    """Smallest sqrt7 family member holding ``n`` squares (None when the grid is at least as good).

    Capacities: m = 4: 18, 5: 27, 6: 39, 7: 53, 8: 68, 9: 86, 10: 105, 11: 127, 12: 151, 13: 176,
    14: 204, 15: 234, 16: 265, 17: 299, ...  (every literature record of the form
    ``(2m-1)/2 + sqrt7/2`` is reproduced)."""
    s_grid = grid(n).s
    m = 4
    while sqrt7_side(m) < s_grid - EPS:
        p = sqrt7_member(m)
        if p.n >= n:
            return p.take(n)
        m += 1
    return None


# --------------------------------------------------------------------------- #
# DeVincentis family  (54, 107, 178, 267, ...)
# --------------------------------------------------------------------------- #
S54 = 7 - SQRT2 / 2 + math.sqrt(1 + SQRT2)
A54 = math.pi / 2 - math.atan((3 + SQRT2 + 2 * math.sqrt(1 + 5 * SQRT2)) / 7)   # 34.7349 degrees


def devincentis_side(k: int) -> float:
    return S54 + 3 * k


def devincentis_capacity(k: int) -> int:
    return 9 * k * k + 44 * k + 54


def _addon54(s1: float) -> List[List[float]]:
    """DeVincentis' block of 8 tilted squares (SVG coordinates of the n=54 drawing, y down)."""
    a = math.pi / 2 - A54          # the drawing's angle (55.265 degrees)
    ca, sa, ta = math.cos(a), math.sin(a), math.tan(a)
    sec, cot, csc = 1 / ca, 1 / ta, 1 / sa
    y2 = s1 - 1 - (1 - (s1 - 7) * sa) * ca
    x3 = ca + sa * (2 - (s1 - 6) * ca - (s1 - 7) * sa)
    y3 = s1 - 1 - 2 * sec - sa - ca * ((s1 - 6) * ca + (s1 - 7) * sa - 2) - 6 * ta + s1 * ta
    x4 = 1 + sa + ca + 6 * ca * sa - ca * sa * s1
    y4 = s1 - 1 + ca - sa + 6 * ca ** 2 - ca ** 2 * s1
    x5 = 2 * ca + 2 * sa * (2 - (s1 - 6) * ca - (s1 - 7) * sa)
    y5 = 2 * (s1 - 4 - 2 * sec - sa - ca * ((s1 - 6) * ca + (s1 - 7) * sa - 2) - 6 * ta + s1 * ta)
    r1 = ca * (2 * (s1 - 7) + (s1 - 6) * cot) + (6 - s1 + 2 * csc) * (1 + 2 * csc * sec) * sa - 6 * cot
    out: List[List[float]] = []
    for (x, y) in ((0.0, 6.0), (0.75, y2), (x3, y3), (x4, y4)):
        out += _Xf().translate(x, y).rotate(-a).rect(0, 0, 1, 1)
    g = _Xf().translate(x5, y5).rotate(-a)
    out += g.rect(0, 0, 2, 1) + g.translate(-r1, 1).rect(0, 0, 2, 1)
    return out


def _devincentis_even(k: int) -> np.ndarray:
    """180-degree symmetric members (k even): jogged staircases and k/2 + 1 addon copies per side."""
    s = devincentis_side(k)
    m = 7 + 3 * k
    J = 2 + 3 * (k // 2)
    side: List[List[float]] = []
    side += _Xf().translate(s, 0).scale(-1, 1).rect(0, 0, 1, 1)
    for r in range(m - 2):
        w = (m - 1 - r) if r <= J else (m - 2 - r)
        side += _Xf().rect(0, r, w, 1)
    addon = _addon54(S54)
    for cpy in range(k // 2 + 1):
        g = _Xf().translate(3 * cpy, 3 * k - 3 * cpy)
        side += [g.square(x, y)[:2] + [t] for x, y, t in addon]
    other = [[s - x, s - y, t] for x, y, t in side]
    return _flip_y(s, side + other)


def _hajba_odd(k: int) -> np.ndarray:
    """Mirror-symmetric members (k odd, Hajba's form): regular staircases, two-row chains."""
    s = devincentis_side(k)
    m = 7 + 3 * k
    a = A54
    v1 = math.cos(a) - math.sin(a)
    s1 = devincentis_side(1)
    u2 = 6 + (2 + 4 * v1) / math.tan(a) - (s1 - 5) / math.sin(a)
    sq: List[List[float]] = []
    for r in range(m - 2):
        sq += _Xf().rect(0, r, m - 2 - r, 1)
    for r in range(m - 1):
        sq += _Xf().translate(s, s).scale(-1, -1).rect(0, r, m - 1 - r, 1)
    sq += _Xf().translate(s, 0).scale(-1, 1).rect(0, 0, 1, 1)
    sq += _Xf().translate(0, s).scale(1, -1).rect(0, 0, 1, 1)
    side: List[List[float]] = []
    j = (k + 1) // 2          # number of 4-square blocks per row

    def chain(g: _Xf, positions: Sequence[Tuple[float, float, int]]) -> List[List[float]]:
        out: List[List[float]] = []
        for (x, y, w) in positions:
            out += g.translate(x, y).rect(0, 0, w, 1)
        return out

    # The chain of 6 ("one, one, two, one, one": drifts d, d, 0, d, d) sits at the far end of
    # the half-band, anchored 5 units left of the right wall and 3k+2 above the bottom (y up);
    # its copy shifted by (-u2, -1) is the second row.  Every extra block is a copy of the
    # first four squares of both rows shifted by -(3, 3) along the diagonal (period of the pattern).
    six = [(0.0, 0.0, 1), (1.0, -v1, 1), (2.0, -2 * v1, 2), (4.0, -3 * v1, 1), (5.0, -4 * v1, 1)]
    four = six[:3]
    for b in range(j):
        g0 = _Xf().translate(-3.0 * b, 3.0 * b).translate(s, s).translate(-5.0, -(3.0 * k + 2.0)) \
            .rotate(-a).translate(0, -1)
        for g in (g0, g0.translate(-u2, -1)):
            side += chain(g, six if b == 0 else four)
    mirror = [[y, x, (math.pi / 2 - t)] for x, y, t in side]    # rotate(90) scale(1 -1): (x, y) -> (y, x)
    return _flip_y(s, sq + side + mirror)


@lru_cache(maxsize=None)
def devincentis_member(k: int) -> Packing:
    """Member ``k`` of the DeVincentis family: ``s = 7 - sqrt2/2 + sqrt(1+sqrt2) + 3k``."""
    if k < 0:
        raise ValueError("k must be >= 0")
    s = devincentis_side(k)
    arr = _devincentis_even(k) if k % 2 == 0 else _hajba_odd(k)
    return _finish(None, s, arr, "devincentis", exact=f"{7 + 3 * k} - sqrt(2)/2 + sqrt(1 + sqrt(2))",
                   meta={"k": k})


def family_devincentis(n: int) -> Optional[Packing]:
    s_grid = grid(n).s
    k = 0
    while devincentis_side(k) < s_grid - EPS:
        if devincentis_capacity(k) >= n:
            p = devincentis_member(k)
            if p.n >= n:
                return p.take(n)
        k += 1
    return None


# --------------------------------------------------------------------------- #
# Wainwright's n = 19
# --------------------------------------------------------------------------- #
S19 = 3 + 4 * SQRT2 / 3


@lru_cache(maxsize=None)
def wainwright_member() -> Packing:
    """n = 19, s = 3 + 4 sqrt2 / 3: staircases of 3 and 6 cells, two corner cells and
    two pairs of 45-degree dominoes; the inner pair is shifted by half the diagonal slack."""
    s = S19
    sq: List[List[float]] = []
    sq += _Xf().rect(0, 0, 2, 1) + _Xf().rect(0, 1, 1, 1)                       # 3-cell staircase
    g = _Xf().translate(s, s).scale(-1, -1)
    sq += g.rect(0, 0, 3, 1) + g.rect(0, 1, 2, 1) + g.rect(0, 2, 1, 1)           # 6-cell staircase
    side: List[List[float]] = []
    side += _Xf().translate(s, 0).scale(-1, 1).rect(0, 0, 1, 1)                  # corner cell
    g = _Xf().translate(s, 0).scale(0.5, 0.5).translate(s, s).scale(2, 2).translate(-3, -1).rotate(math.radians(135))
    side += g.rect(0, 0, 1, 2)
    g2 = g.translate(1, -2).rotate(math.radians(-45)).translate(-s, s).translate(3.5, -3.5).rotate(math.radians(45))
    side += g2.rect(0, 0, 1, 2)
    other = [[y, x, math.pi / 2 - t] for x, y, t in side]     # scale(1 -1) rotate(-90): (x, y) -> (y, x)
    arr = _flip_y(s, sq + side + other)
    return _finish(None, s, arr, "wainwright", exact="3 + 4*sqrt(2)/3", meta={})


def family_wainwright(n: int) -> Optional[Packing]:
    if n > 19 or S19 >= grid(n).s - EPS:
        return None
    return wainwright_member().take(n)


# --------------------------------------------------------------------------- #
# Schadt's n = 50 (s = 7 + 4/7) and Ellsworth's n = 171 (s = 13 + 4/7)
# --------------------------------------------------------------------------- #
S50 = 7 + 4.0 / 7.0
A50 = math.atan2(3.0, 4.0)      # the 3-4-5 angle: cos = 4/5, sin = 3/5


def _schadt_core(s1: float, trimmed: bool = False) -> List[List[float]]:
    """The 34 + 16 squares of the n = 50 drawing (SVG coordinates, y down, side s1).

    With ``trimmed`` the leftmost column of the staircases is omitted (the form used
    twice inside the n = 171 packing)."""
    a = A50
    r1 = 1 - (s1 - 7) * math.cos(a)          # 19/35
    strip = [(0.0, 0.0), (-1.0, 0.2), (-2.0, 0.2)]
    side: List[List[float]] = []
    x0 = 1 if trimmed else 0
    side += _Xf().rect(x0, 0, 3 - x0, 1) + _Xf().rect(x0, 1, 2 - x0, 1) + (_Xf().rect(0, 2, 1, 1) if not trimmed else [])
    g = _Xf().translate(0, s1).scale(1, -1)
    side += g.rect(x0, 0, 4 - x0, 1) + g.rect(x0, 1, 3 - x0, 1) + g.rect(x0, 2, 2 - x0, 2)
    g = _Xf().translate(3, 1).rotate(-a).translate(-r1, 0)
    for (dx, dy) in ((0.0, 0.0), (0.2, 1.0)):
        for (x, y) in strip:
            side += g.translate(dx + x, dy + y).rect(0, 0, 1, 1)
    extra: List[List[float]] = []
    g = _Xf().translate(3, 1).rotate(-a).translate(-r1, 0).translate(0.2, 1).translate(0.3, 1)
    for (x, y) in ((0.0, 0.0), (-1.0, 0.2), (-2.0, 0.2), (-3.0, 0.4)):
        extra += g.translate(x, y).rect(0, 0, 1, 1)
    other = [[s1 - x, s1 - y, t] for x, y, t in side]
    return side + other + extra


@lru_cache(maxsize=None)
def schadt_member(k: int) -> Packing:
    """k = 0: n = 50, s = 7 + 4/7; k = 1: n = 171, s = 13 + 4/7 (two copies of the core)."""
    if k == 0:
        s = S50
        return _finish(None, s, _flip_y(s, _schadt_core(s)), "schadt", exact="7 + 4/7", meta={"k": 0})
    if k == 1:
        s1 = S50
        s = s1 + 6
        first = [_Xf().translate(7, 6).square(x, y)[:2] + [t] for x, y, t in _schadt_core(s1, trimmed=True)]
        # translate(s s) rotate(90) scale(-1 1): (x, y) -> (s - y, s - x)
        second = [[s - y, s - x, -math.pi / 2 - t] for x, y, t in first]
        rest: List[List[float]] = []
        rest += _Xf().translate(s - 6, 0).rect(0, 0, 6, 6)
        g = _Xf().translate(0, s).scale(1, -1)
        rest += g.rect(0, 0, 8, 7) + g.rect(0, 7, 7, 1)
        return _finish(None, s, _flip_y(s, first + second + rest), "schadt", exact="13 + 4/7", meta={"k": 1})
    raise ValueError("k must be 0 or 1")


def family_schadt(n: int) -> Optional[Packing]:
    s_grid = grid(n).s
    for k, cap in ((0, 50), (1, 171)):
        if schadt_member(k).s >= s_grid - EPS:
            return None
        if cap >= n:
            return schadt_member(k).take(n)
    return None


# --------------------------------------------------------------------------- #
# portfolio hooks
# --------------------------------------------------------------------------- #
FAMILIES = {
    "sqrt7": family_sqrt7,
    "devincentis": family_devincentis,
    "wainwright": family_wainwright,
    "schadt": family_schadt,
}


def member_descriptors(s_max: float, n: int) -> list:
    """Lightweight ``(s, capacity, family, builder)`` descriptors of the family members with side
    below ``s_max`` (each family stops at its first member holding ``n``); nothing is built here."""
    from .constructions import Member
    out = []
    m = 4
    while sqrt7_side(m) < s_max - EPS:
        cap = sqrt7_capacity(m)
        if cap is None:
            break
        out.append(Member(sqrt7_side(m), cap, "sqrt7", lambda m=m: sqrt7_member(m)))
        if cap >= n:
            break
        m += 1
    k = 0
    while devincentis_side(k) < s_max - EPS:
        cap = devincentis_capacity(k)
        out.append(Member(devincentis_side(k), cap, "devincentis", lambda k=k: devincentis_member(k)))
        if cap >= n:
            break
        k += 1
    if S19 < s_max - EPS:
        out.append(Member(S19, 19, "wainwright", wainwright_member))
    for k, (side, cap) in enumerate(((7 + 4 / 7, 50), (13 + 4 / 7, 171))):
        if side < s_max - EPS:
            out.append(Member(side, cap, "schadt", lambda k=k: schadt_member(k)))
            if cap >= n:
                break
    return out


def family_members_below(s_max: float, n: Optional[int] = None) -> List[Packing]:
    """Family members (full, not truncated) with side below ``s_max`` for the analytic portfolio.

    With ``n`` given, each family stops at its first member holding ``n`` squares
    (an "L" added to a member never beats the next member of the same family)."""
    out: List[Packing] = []
    n = 0 if n is None else n
    m = 4
    while sqrt7_side(m) < s_max - EPS:
        out.append(sqrt7_member(m))
        if out[-1].n >= n:
            break
        m += 1
    k = 0
    while devincentis_side(k) < s_max - EPS:
        out.append(devincentis_member(k))
        if out[-1].n >= n:
            break
        k += 1
    if S19 < s_max - EPS:
        out.append(wainwright_member())
    for k in (0, 1):
        if schadt_member(k).s < s_max - EPS:
            out.append(schadt_member(k))
            if out[-1].n >= n:
                break
    return [replace(p, squares=np.array(p.squares, dtype=float, copy=True)) for p in out]
