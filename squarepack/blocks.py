"""Tilted-block constructions: a ``p x q`` block of unit squares rotated 45 degrees
plus a greedy fill of axis-aligned squares.

This single mechanism generalises Göbel's squares (p = q, centred), Göbel's
strips (p = 1), and the record packings of Friedman and Stenlund such as
n = 26 (3x3 block, s = 7/2 + 3 sqrt2 / 2), n = 66 (3x8 block, s = 3 + 4 sqrt2)
and n = 85 (6x6 block, s = 11/2 + 3 sqrt2).

Row model
---------
For a container side ``s`` and a convex tilted block, the axis-aligned squares
are placed in horizontal unit bands: the lowest ``J`` bands are anchored to the
bottom wall (``[j, j+1]``) and the remaining ``floor(s) - J`` bands to the top
wall (``[s-j-1, s-j]``).  In a band that meets the block on the x-interval
``[xl, xr]`` we place ``floor(xl)`` squares from the left wall and
``floor(s - xr)`` from the right wall; a band that misses the block holds
``floor(s)`` squares.  Because the block is convex the interval is obtained by
clipping the block polygon to the band.  The seam ``J`` and the transposed
(column) model are both tried.  The count is exact for the chosen structure
and costs ``O(s)`` to evaluate; all block offsets are evaluated at once with
numpy, and the smallest ``s`` admitting ``n`` squares for each block shape and
offset is found by a vectorised bisection.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .constructions import Packing, grid
from .geometry import QUARTER_PI, SQRT2, repair, verify

EPS = 1e-10  # slack for floor() of exactly-touching positions (results are made exact by repair())


def block_polygons(p: int, q: int, cx: np.ndarray, cy: np.ndarray, angle: float) -> np.ndarray:
    """Vertices ``(P, 4, 2)`` of ``p x q`` rectangles centred at ``(cx, cy)`` rotated by ``angle``."""
    c = math.cos(angle)
    s = math.sin(angle)
    e1 = np.array([c, s]) * (p / 2.0)
    e2 = np.array([-s, c]) * (q / 2.0)
    ctr = np.stack([cx, cy], axis=-1)[:, None, :]
    corners = np.array([e1 + e2, -e1 + e2, -e1 - e2, e1 - e2])[None, :, :]
    return ctr + corners


def block_tiles(p: int, q: int, cx: float, cy: float, angle: float) -> np.ndarray:
    """Centres/angles of the ``p*q`` unit squares of the block, centre-most first."""
    c = math.cos(angle)
    s = math.sin(angle)
    e1 = np.array([c, s])
    e2 = np.array([-s, c])
    tiles = []
    for i in range(p):
        for j in range(q):
            u = i - (p - 1) / 2.0
            v = j - (q - 1) / 2.0
            pos = np.array([cx, cy]) + u * e1 + v * e2
            tiles.append((abs(u) + abs(v), pos[0], pos[1]))
    tiles.sort()
    return np.array([[x, y, angle] for _, x, y in tiles])


def _band_intervals(polys: np.ndarray, y0: np.ndarray, y1: np.ndarray):
    """x-extent of convex polygons ``(P,4,2)`` clipped to bands ``(P,B)``; returns ``(hit, xl, xr)``."""
    E0 = polys
    E1 = np.roll(polys, -1, axis=1)
    P, B = y0.shape
    xl = np.full((P, B), np.inf)
    xr = np.full((P, B), -np.inf)
    for yy in (y0, y1):
        for e in range(4):
            xa = E0[:, e, 0][:, None]
            ya = E0[:, e, 1][:, None]
            xb = E1[:, e, 0][:, None]
            yb = E1[:, e, 1][:, None]
            dyy = yb - ya
            horiz = np.abs(dyy) < 1e-14
            tpar = (yy - ya) / np.where(horiz, 1.0, dyy)
            inside = (~horiz) & (tpar >= -1e-12) & (tpar <= 1 + 1e-12)
            xs = xa + tpar * (xb - xa)
            xl = np.minimum(xl, np.where(inside, xs, np.inf))
            xr = np.maximum(xr, np.where(inside, xs, -np.inf))
            insh = horiz & (np.abs(yy - ya) <= 1e-12)
            xl = np.minimum(xl, np.where(insh, np.minimum(xa, xb), np.inf))
            xr = np.maximum(xr, np.where(insh, np.maximum(xa, xb), -np.inf))
    for v in range(4):
        vx = polys[:, v, 0][:, None]
        vy = polys[:, v, 1][:, None]
        inside = (vy >= y0 - 1e-12) & (vy <= y1 + 1e-12)
        xl = np.minimum(xl, np.where(inside, vx, np.inf))
        xr = np.maximum(xr, np.where(inside, vx, -np.inf))
    return np.isfinite(xl), xl, xr


def row_fill(s: np.ndarray, polys: np.ndarray):
    """Row-model fill for containers ``s (P,)`` and blocks ``polys (P,4,2)``.

    Bands that lie entirely below/above the block are full rows of ``floor(s)``
    squares.  Bands that meet the block are split by it into a left and a
    right part, and each side chooses its own seam (``J_L``, ``J_R``) between
    bottom- and top-anchored bands: the block separates the two sides, so the
    choices are independent (this is what makes Göbel's strips - staircases
    anchored at opposite corners - representable).

    Returns ``(total, JL, JR, Fb, Ft, lb, rb, lt, rt)``.
    """
    s = np.asarray(s, float)
    P = len(s)
    Bp = np.floor(s + EPS).astype(int)
    Bmax = int(Bp.max()) if P else 0
    if Bmax == 0:
        z = np.zeros((P, 0), int)
        zi = np.zeros(P, int)
        return zi, zi, zi, zi, zi, z, z, z, z
    j = np.arange(Bmax)[None, :].astype(float)
    valid = j < Bp[:, None]
    sc = s[:, None]
    ymin = polys[:, :, 1].min(axis=1)
    ymax = polys[:, :, 1].max(axis=1)
    Fb = np.clip(np.floor(ymin + EPS).astype(int), 0, Bp)
    Ft = np.clip(np.floor(s - ymax + EPS).astype(int), 0, Bp - Fb)
    hit, xl, xr = _band_intervals(polys, np.broadcast_to(j, (P, Bmax)), np.broadcast_to(j + 1, (P, Bmax)))
    lb = np.where(hit, np.floor(np.clip(xl, 0, sc) + EPS), 0).astype(int)
    rb = np.where(hit, np.floor(np.clip(sc - xr, 0, sc) + EPS), 0).astype(int)
    hit, xl, xr = _band_intervals(polys, sc - j - 1, sc - j)
    lt = np.where(hit, np.floor(np.clip(xl, 0, sc) + EPS), 0).astype(int)
    rt = np.where(hit, np.floor(np.clip(sc - xr, 0, sc) + EPS), 0).astype(int)
    # only the middle bands take part in the per-side seam choice
    mid_b = valid & (j >= Fb[:, None])
    mid_t = valid & (j >= Ft[:, None])
    lb[~mid_b] = 0
    rb[~mid_b] = 0
    lt[~mid_t] = 0
    rt[~mid_t] = 0
    zero = np.zeros((P, 1), int)
    Jgrid = np.arange(Bmax + 1)[None, :]
    topidx = np.clip(Bp[:, None] - Jgrid, 0, Bmax)
    allowed = (Jgrid >= Fb[:, None]) & (Jgrid <= (Bp - Ft)[:, None])
    ar = np.arange(P)

    def best_seam(bot, top):
        cb = np.concatenate([zero, np.cumsum(bot, axis=1)], axis=1)
        ct = np.concatenate([zero, np.cumsum(top, axis=1)], axis=1)
        tot = cb + np.take_along_axis(ct, topidx, axis=1)
        tot = np.where(allowed, tot, -1)
        J = np.argmax(tot, axis=1)
        return tot[ar, J], J

    totL, JL = best_seam(lb, lt)
    totR, JR = best_seam(rb, rt)
    total = Bp * (Fb + Ft) + totL + totR
    return total, JL, JR, Fb, Ft, lb, rb, lt, rt


def block_capacity(s, p: int, q: int, dx, dy, angle: float = QUARTER_PI, transpose: bool = False) -> np.ndarray:
    """Capacity (block + row fill) for arrays ``s, dx, dy``; ``-1`` where the block does not fit."""
    s = np.atleast_1d(np.asarray(s, float))
    dx = np.atleast_1d(np.asarray(dx, float))
    dy = np.atleast_1d(np.asarray(dy, float))
    s, dx, dy = np.broadcast_arrays(s, dx, dy)
    polys = block_polygons(p, q, s / 2 + dx, s / 2 + dy, angle)
    inside = np.all(polys >= -EPS, axis=(1, 2)) & np.all(polys <= s[:, None, None] + EPS, axis=(1, 2))
    if transpose:
        polys = polys[:, :, ::-1]
    total = row_fill(s, polys)[0]
    cap = total + p * q
    cap[~inside] = -1
    return cap


def build_block_packing(s: float, p: int, q: int, dx: float, dy: float, angle: float = QUARTER_PI,
                        transpose: bool = False) -> Optional[Packing]:
    """Construct the packing explicitly (``None`` if the block does not fit)."""
    cx, cy = s / 2.0 + dx, s / 2.0 + dy
    polys = block_polygons(p, q, np.array([cx]), np.array([cy]), angle)
    if not (np.all(polys >= -EPS) and np.all(polys <= s + EPS)):
        return None
    tiles = block_tiles(p, q, cx, cy, angle)
    if transpose:
        polys = polys[:, :, ::-1]
    total, JL, JR, Fb, Ft, lb, rb, lt, rt = row_fill(np.array([s]), polys)
    JL, JR, Fb, Ft = int(JL[0]), int(JR[0]), int(Fb[0]), int(Ft[0])
    B = int(math.floor(s + EPS))
    rows: List[List[float]] = []
    for j in range(Fb):                       # full rows below the block
        rows += [[i + 0.5, j + 0.5, 0.0] for i in range(B)]
    for j in range(Ft):                       # full rows above the block
        rows += [[i + 0.5, s - j - 0.5, 0.0] for i in range(B)]
    for j in range(Fb, JL):                   # left side, bottom-anchored
        rows += [[i + 0.5, j + 0.5, 0.0] for i in range(lb[0, j])]
    for j in range(Ft, B - JL):               # left side, top-anchored
        rows += [[i + 0.5, s - j - 0.5, 0.0] for i in range(lt[0, j])]
    for j in range(Fb, JR):                   # right side, bottom-anchored
        rows += [[s - i - 0.5, j + 0.5, 0.0] for i in range(rb[0, j])]
    for j in range(Ft, B - JR):               # right side, top-anchored
        rows += [[s - i - 0.5, s - j - 0.5, 0.0] for i in range(rt[0, j])]
    arr = np.array(rows, dtype=float).reshape(-1, 3)
    if transpose:
        arr = arr[:, [1, 0, 2]]
    sq = np.vstack([arr, tiles]) if len(arr) else tiles
    exact = f"tilted {p}x{q} block at 45deg, offset ({dx:+g},{dy:+g})" + (" transposed" if transpose else "")
    return Packing(len(sq), float(s), sq, "tilted_block", exact=exact,
                   meta={"p": p, "q": q, "dx": dx, "dy": dy, "angle": angle, "transpose": transpose})


def min_side_vec(n: int, p: int, q: int, dx: np.ndarray, dy: np.ndarray, s_lo: float, s_hi: float,
                 transpose: bool, angle: float = QUARTER_PI, iters: int = 42) -> np.ndarray:
    """Per-offset smallest ``s`` in ``[s_lo, s_hi]`` with capacity >= n (``inf`` where none)."""
    P = len(dx)
    cap = block_capacity(np.full(P, s_hi), p, q, dx, dy, angle, transpose)
    active = cap >= n
    lo = np.full(P, s_lo)
    hi = np.full(P, s_hi)
    out = np.full(P, np.inf)
    if not np.any(active):
        return out
    idx = np.nonzero(active)[0]
    dxa, dya = dx[idx], dy[idx]
    lo, hi = lo[idx], hi[idx]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = block_capacity(mid, p, q, dxa, dya, angle, transpose) >= n
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    out[idx] = hi
    return out


DEFAULT_OFFSETS = tuple(k / 4.0 for k in range(-5, 6))


def tilted_block_search(n: int, s_max: Optional[float] = None,
                        offsets: Sequence[float] = DEFAULT_OFFSETS,
                        shapes: Optional[Iterable[Tuple[int, int]]] = None,
                        angle: float = QUARTER_PI, verbose: bool = False) -> Optional[Packing]:
    """Best tilted-block packing for ``n`` with side below ``s_max`` (default: grid side).

    Searches block shapes ``p <= q`` and centre offsets in ``offsets`` (both
    row and column models), bisecting on ``s`` for each; returns the verified
    packing with the smallest ``s`` or ``None`` if nothing beats ``s_max``.
    """
    if s_max is None:
        s_max = grid(n).s
    s_lo = math.sqrt(n)
    best: Optional[Packing] = None
    best_s = s_max - 1e-12
    if shapes is None:
        pmax = int(s_max * SQRT2) + 1
        shapes = [(p, q) for p in range(1, pmax + 1) for q in range(p, pmax + 1)
                  if p * q <= n and (p + q) / SQRT2 <= s_max + 1e-9]
    DX, DY = np.meshgrid(np.asarray(offsets, float), np.asarray(offsets, float), indexing="ij")
    DX = DX.ravel()
    DY = DY.ravel()
    for p, q in shapes:
        for transpose in (False, True):
            if transpose and p == q:
                continue
            sides = min_side_vec(n, p, q, DX, DY, s_lo, best_s, transpose, angle)
            order = np.argsort(sides)
            for k in order:
                s = float(sides[k])
                if not np.isfinite(s) or s >= best_s - 1e-12:
                    break
                pk = build_block_packing(s, p, q, float(DX[k]), float(DY[k]), angle, transpose)
                if pk is None or pk.n < n:
                    continue
                pk = pk.take(n)
                # the bisection stops within ~EPS of the exact threshold; scaling by the
                # minimal repair factor makes the packing exactly valid (strict check)
                s_fix, sq_fix = repair(pk.s, pk.squares)
                pk = Packing(pk.n, s_fix, sq_fix, pk.method, pk.exact, pk.meta)
                if not verify(pk.s, pk.squares, 1e-12).ok or pk.s >= best_s - 1e-12:
                    continue
                best, best_s = pk, pk.s
                if verbose:
                    print(f"  n={n}: s={pk.s:.10f} {pk.exact}")
                break
    return best
