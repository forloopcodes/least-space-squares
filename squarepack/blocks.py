"""Tilted-block constructions: rotated blocks of unit squares plus an exact row fill
of axis-aligned squares.

This single mechanism generalises Göbel's squares (p = q, centred), Göbel's
strips (p = 1), and the record packings of Friedman and Stenlund such as
n = 26 (3x3 block, s = 7/2 + 3 sqrt2 / 2), n = 66 (3x8 block, s = 3 + 4 sqrt2)
and n = 85 (6x6 block, s = 11/2 + 3 sqrt2).

Obstacles
---------
The tilted part of a packing is a *union of convex pieces*; every piece is a
``w x h`` block of unit squares (a parallelogram when the rows are sheared)
described by the six numbers ``(w, h, ox, oy, theta, u)``: tiles along the
two directions, centre offset from the container centre, rotation angle and
shear (each row is shifted by ``u`` along the row relative to the previous
one).  A candidate is a ``(K, 6)`` *spec*; the families below produce them:

single
    one ``p x q`` block at any angle (Göbel squares/strips, 26, 66, 85, ...);
two
    two equal blocks placed symmetrically about the container centre
    (``centre +- e``), the rigid version of the two parallel 1x4 chains of
    n = 18;
strip
    a 45-degree strip of width ``p`` whose end segments are rotated by a small
    angle about the joint corner ("bent" strips: S-shaped as in n = 37 and 88,
    W-shaped as in n = 102 and 123);
shear
    a single block whose rows are shifted against each other (a staircase of
    row pieces, as in n = 70);
chain (:class:`RowChain`)
    the most general group: ``R`` rows of ``p`` parallel unit squares with a
    tilt, a row direction and a lateral slide per row, consecutive rows kept
    in exact contact, and the whole chain symmetric under a half turn or a
    reflection.  Every record of the "bent strip" type (19, 37, 88, 102, 123)
    and the kinked chains of 18 and 70 are members; the search only reaches
    them by a local pattern search from a straight strip, which is why the
    family is exposed (``RowChain.from_block``) rather than gridded.

Row model
---------
For a container side ``s`` the axis-aligned squares are placed in horizontal
unit bands: the lowest ``J`` bands are anchored to the bottom wall
(``[j, j+1]``) and the remaining ``floor(s) - J`` bands to the top wall
(``[s-j-1, s-j]``).  In a band that meets the obstacle on the x-interval
``[xl, xr]`` (the union of the clipped pieces) we place ``floor(xl)`` squares
from the left wall and ``floor(s - xr)`` from the right wall; a band that
misses the obstacle holds ``floor(s)`` squares.  Each side chooses its own
seam ``J_L``, ``J_R``; because a non-convex union no longer separates the two
sides in every band, the seam pair is checked exactly against the O(B) pairs
of bands that overlap in ``y`` (the check is vacuous for a single convex
piece).  The transposed (column) model is also tried.  The count is exact for
the chosen structure and costs ``O(s^2)`` per candidate; all candidates of a
family are evaluated at once with numpy, and the smallest ``s`` admitting
``n`` squares is found by a vectorised bisection that discards candidates as
soon as they cannot beat the best one found so far.  A pattern search over
the continuous parameters (offsets, angle, shear, bend) polishes the best
grid candidates.  Every returned packing is verified.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .constructions import Packing, grid
from .geometry import QUARTER_PI, SQRT2, repair, verify

EPS = 1e-10  # slack for floor() of exactly-touching positions (results are made exact by repair())
TOL = 1e-9   # touching tolerance for piece/piece and row/row conflicts

SPEC_W, SPEC_H, SPEC_OX, SPEC_OY, SPEC_TH, SPEC_U = range(6)


# --------------------------------------------------------------------------- #
# piece geometry
# --------------------------------------------------------------------------- #
def spec_polygons(s: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """Vertices ``(P, K, 4, 2)`` (counter-clockwise) of the pieces of ``spec (P, K, 6)`` for sides ``s (P,)``."""
    s = np.asarray(s, float)[:, None]
    w, h, ox, oy, th, u = (spec[..., i] for i in range(6))
    c = np.cos(th)
    sn = np.sin(th)
    cx = 0.5 * s + ox
    cy = 0.5 * s + oy
    out = np.empty(spec.shape[:2] + (4, 2))
    for m, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        a = 0.5 * (sx * w + u * sy * h)   # along e1 (rows are shifted by u per unit of height)
        b = 0.5 * sy * h                  # along e2
        out[..., m, 0] = cx + a * c - b * sn
        out[..., m, 1] = cy + a * sn + b * c
    return out


def spec_tiles(s: float, piece: Sequence[float]) -> np.ndarray:
    """Centres/angles ``(w*h, 3)`` of the unit squares of one piece, centre-most first."""
    w, h, ox, oy, th, u = piece
    w, h = int(round(w)), int(round(h))
    c, sn = math.cos(th), math.sin(th)
    cx, cy = 0.5 * s + ox, 0.5 * s + oy
    I, J = np.meshgrid(np.arange(w) - (w - 1) / 2.0, np.arange(h) - (h - 1) / 2.0, indexing="ij")
    a = (I + u * J).ravel()
    b = J.ravel()
    order = np.argsort(np.abs(a) + np.abs(b), kind="stable")
    a, b = a[order], b[order]
    return np.column_stack([cx + a * c - b * sn, cy + a * sn + b * c, np.full(len(a), float(th))])


def block_polygons(p: int, q: int, cx: np.ndarray, cy: np.ndarray, angle: float) -> np.ndarray:
    """Vertices ``(P, 4, 2)`` of ``p x q`` rectangles centred at ``(cx, cy)`` rotated by ``angle``."""
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)
    spec = np.zeros((len(cx), 1, 6))
    spec[:, 0, SPEC_W] = p
    spec[:, 0, SPEC_H] = q
    spec[:, 0, SPEC_OX] = cx
    spec[:, 0, SPEC_OY] = cy
    spec[:, 0, SPEC_TH] = angle
    return spec_polygons(np.zeros(len(cx)), spec)[:, 0]


def block_tiles(p: int, q: int, cx: float, cy: float, angle: float) -> np.ndarray:
    """Centres/angles of the ``p*q`` unit squares of the block, centre-most first."""
    return spec_tiles(0.0, (p, q, cx, cy, angle, 0.0))


def _quads_overlap(A: np.ndarray, B: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Separating-axis test for convex quadrilaterals ``A, B (P, 4, 2)``; touching is not an overlap."""
    overlap = np.ones(len(A), bool)
    for Q in (A, B):
        E = np.roll(Q, -1, axis=1) - Q          # edges (P,4,2)
        nx, ny = E[..., 1], -E[..., 0]          # edge normals (P,4)
        pa = A[:, None, :, 0] * nx[:, :, None] + A[:, None, :, 1] * ny[:, :, None]  # (P,4 axes,4 verts)
        pb = B[:, None, :, 0] * nx[:, :, None] + B[:, None, :, 1] * ny[:, :, None]
        scale = np.hypot(nx, ny) + 1e-300
        sep = (pa.max(axis=2) <= pb.min(axis=2) + tol * scale) | (pb.max(axis=2) <= pa.min(axis=2) + tol * scale)
        overlap &= ~sep.any(axis=1)
    return overlap


# --------------------------------------------------------------------------- #
# row model
# --------------------------------------------------------------------------- #
def _band_intervals(polys: np.ndarray, y0: np.ndarray, y1: np.ndarray):
    """x-extent of convex polygons ``(P,4,2)`` clipped to bands ``(P,B)``; returns ``(hit, xl, xr)``."""
    P, B = y0.shape
    xa = polys[:, :, 0][:, :, None]
    ya = polys[:, :, 1][:, :, None]
    xb = np.roll(polys[:, :, 0], -1, axis=1)[:, :, None]
    yb = np.roll(polys[:, :, 1], -1, axis=1)[:, :, None]
    dyy = yb - ya
    horiz = np.abs(dyy) < 1e-14
    inv = 1.0 / np.where(horiz, 1.0, dyy)
    dxx = xb - xa
    xl = np.full((P, B), np.inf)
    xr = np.full((P, B), -np.inf)
    for yy in (y0, y1):
        yy = yy[:, None, :]
        tpar = (yy - ya) * inv
        inside = (~horiz) & (tpar >= -1e-12) & (tpar <= 1 + 1e-12)
        xs = xa + tpar * dxx
        xl = np.minimum(xl, np.where(inside, xs, np.inf).min(axis=1))
        xr = np.maximum(xr, np.where(inside, xs, -np.inf).max(axis=1))
        insh = horiz & (np.abs(yy - ya) <= 1e-12)
        xl = np.minimum(xl, np.where(insh, np.minimum(xa, xb), np.inf).min(axis=1))
        xr = np.maximum(xr, np.where(insh, np.maximum(xa, xb), -np.inf).max(axis=1))
    inside = (ya >= y0[:, None, :] - 1e-12) & (ya <= y1[:, None, :] + 1e-12)
    xl = np.minimum(xl, np.where(inside, xa, np.inf).min(axis=1))
    xr = np.maximum(xr, np.where(inside, xa, -np.inf).max(axis=1))
    return np.isfinite(xl), xl, xr


def _band_extents(polys: np.ndarray, y0: np.ndarray, y1: np.ndarray):
    """Union x-extent of the pieces ``(P,K,4,2)`` clipped to bands ``(P,B)``."""
    P, K = polys.shape[:2]
    if K == 1:
        return _band_intervals(polys[:, 0], y0, y1)
    B = y0.shape[1]
    flat = polys.reshape(P * K, polys.shape[2], 2)
    y0r = np.repeat(y0, K, axis=0)
    y1r = np.repeat(y1, K, axis=0)
    _, l, r = _band_intervals(flat, y0r, y1r)
    xl = l.reshape(P, K, B).min(axis=1)
    xr = r.reshape(P, K, B).max(axis=1)
    return np.isfinite(xl), xl, xr


def _cummax_exclusive(m: np.ndarray) -> np.ndarray:
    """``out[:, k] = max(m[:, :k])`` (0 for k = 0), shape ``(P, B+1)``."""
    return np.concatenate([np.zeros((len(m), 1), m.dtype), np.maximum.accumulate(m, axis=1)], axis=1)


def row_fill(s: np.ndarray, polys: np.ndarray):
    """Row-model fill for containers ``s (P,)`` and obstacles ``polys (P,4,2)`` or ``(P,K,4,2)``.

    Bands that lie entirely below/above the obstacle are full rows of
    ``floor(s)`` squares.  Bands that meet it are split into a left and a
    right part, and each side chooses its own seam (``J_L``, ``J_R``) between
    bottom- and top-anchored bands (this is what makes Göbel's strips -
    staircases anchored at opposite corners - representable).  For a single
    convex obstacle the two choices are independent; for a union of pieces
    the seam pair is checked against every pair of bottom/top bands that
    overlap in ``y`` so that the count is exact for any obstacle.

    Returns ``(total, JL, JR, Fb, Ft, lb, rb, lt, rt)``.
    """
    s = np.asarray(s, float)
    polys = np.asarray(polys, float)
    if polys.ndim == 3:
        polys = polys[:, None]
    P = len(s)
    Bp = np.floor(s + EPS).astype(int)
    Bmax = int(Bp.max()) if P else 0
    if Bmax == 0:
        z = np.zeros((P, 0), int)
        zi = np.zeros(P, int)
        return zi, zi, zi, zi, zi, z, z, z, z
    j = np.arange(Bmax, dtype=float)[None, :]
    valid = j < Bp[:, None]
    sc = s[:, None]
    ymin = polys[..., 1].min(axis=(1, 2))
    ymax = polys[..., 1].max(axis=(1, 2))
    Fb = np.clip(np.floor(ymin + EPS).astype(int), 0, Bp)
    Ft = np.clip(np.floor(s - ymax + EPS).astype(int), 0, Bp - Fb)
    jb = np.broadcast_to(j, (P, Bmax))
    hit, xl, xr = _band_extents(polys, jb, jb + 1)
    lb = np.where(hit, np.floor(np.clip(xl, 0, sc) + EPS), 0).astype(int)
    rb = np.where(hit, np.floor(np.clip(sc - xr, 0, sc) + EPS), 0).astype(int)
    hit, xl, xr = _band_extents(polys, sc - j - 1, sc - j)
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

    def seam_totals(bot, top):
        cb = np.concatenate([zero, np.cumsum(bot, axis=1)], axis=1)
        ct = np.concatenate([zero, np.cumsum(top, axis=1)], axis=1)
        return cb + np.take_along_axis(ct, topidx, axis=1)

    totL = seam_totals(lb, lt)   # (P, Bmax+1) indexed by JL
    totR = seam_totals(rb, rt)   # indexed by JR
    ar = np.arange(P)
    if polys.shape[1] == 1:
        # a single convex obstacle separates the two sides in every pair of overlapping
        # bands, so the seams are independent
        tL = np.where(allowed, totL, -1)
        tR = np.where(allowed, totR, -1)
        JL = tL.argmax(axis=1)
        JR = tR.argmax(axis=1)
        total = Bp * (Fb + Ft) + tL[ar, JL] + tR[ar, JR]
        return total, JL, JR, Fb, Ft, lb, rb, lt, rt
    # exact cross-side validity: bottom band j and top band j' overlap in y iff s-2 < j+j' < s,
    # i.e. j' = B-1-j always and j' = B-j when s is not an integer
    jt = np.stack([Bp[:, None] - 1 - j, Bp[:, None] - j], axis=2).astype(int)     # (P, B, 2) candidate j'
    jok = (jt >= 0) & (jt < Bmax)
    jok[:, :, 1] &= (Bp < s - TOL)[:, None]
    jc = np.clip(jt, 0, Bmax - 1)
    Bj = np.where(jok, Bp[:, None, None] - jt, 0)                                    # B - j'

    def need(bot, top):
        top_at = np.take_along_axis(np.broadcast_to(top[:, None, :], (P, Bmax, Bmax)), jc, axis=2)  # top[j']
        bad = jok & ((bot[:, :, None] + top_at) > sc[:, :, None] + TOL)
        return _cummax_exclusive(np.where(bad, Bj, 0).max(axis=2))                  # (P, B+1)

    # left-bottom j (used iff j < JL) against right-top j' (used iff j' < B - JR): need JR >= B - j'
    needJR = need(lb, rt)                                                            # indexed by JL
    # right-bottom j (used iff j < JR) against left-top j' (used iff j' < B - JL): need JL >= B - j'
    needJL = need(rb, lt)                                                            # indexed by JR
    ok = (Jgrid[:, None, :] >= needJR[:, :, None]) & (Jgrid[:, :, None] >= needJL[:, None, :])
    ok &= allowed[:, :, None] & allowed[:, None, :]
    tot = np.where(ok, totL[:, :, None] + totR[:, None, :], -1)
    flat = tot.reshape(P, -1).argmax(axis=1)
    JL = flat // (Bmax + 1)
    JR = flat % (Bmax + 1)
    total = Bp * (Fb + Ft) + tot.reshape(P, -1)[ar, flat]
    return total, JL, JR, Fb, Ft, lb, rb, lt, rt


# --------------------------------------------------------------------------- #
# capacity of a spec
# --------------------------------------------------------------------------- #
def spec_capacity(s, spec: np.ndarray, transpose=None) -> np.ndarray:
    """Capacity (tiles + row fill) of specs ``(P, K, 6)`` for sides ``s (P,)``; ``-1`` where a
    piece leaves the container or two pieces overlap.  ``transpose`` selects the column model per candidate."""
    s = np.atleast_1d(np.asarray(s, float))
    spec = np.asarray(spec, float)
    if spec.ndim == 2:
        spec = spec[None]
    if len(spec) == 1 and len(s) > 1:
        spec = np.broadcast_to(spec, (len(s),) + spec.shape[1:])
    polys = spec_polygons(s, spec)
    ok = np.all(polys >= -EPS, axis=(1, 2, 3)) & np.all(polys <= s[:, None, None, None] + EPS, axis=(1, 2, 3))
    K = spec.shape[1]
    if K > 1:
        I, J = np.triu_indices(K, 1)
        P = len(s)
        ov = _quads_overlap(polys[:, I].reshape(P * len(I), 4, 2), polys[:, J].reshape(P * len(J), 4, 2))
        ok &= ~ov.reshape(P, len(I)).any(axis=1)
    if transpose is not None:
        tr = np.broadcast_to(np.asarray(transpose, bool), (len(s),))
        if tr.any():
            polys = np.where(tr[:, None, None, None], polys[..., ::-1], polys)
    total = row_fill(s, polys)[0]
    cap = total + np.rint(spec[:, :, SPEC_W] * spec[:, :, SPEC_H]).sum(axis=1).astype(int)
    cap[~ok] = -1
    return cap


def block_capacity(s, p: int, q: int, dx, dy, angle: float = QUARTER_PI, transpose: bool = False) -> np.ndarray:
    """Capacity (block + row fill) for arrays ``s, dx, dy``; ``-1`` where the block does not fit."""
    s = np.atleast_1d(np.asarray(s, float))
    dx = np.atleast_1d(np.asarray(dx, float))
    dy = np.atleast_1d(np.asarray(dy, float))
    s, dx, dy = np.broadcast_arrays(s, dx, dy)
    spec = np.zeros((len(s), 1, 6))
    spec[:, 0, SPEC_W] = p
    spec[:, 0, SPEC_H] = q
    spec[:, 0, SPEC_OX] = dx
    spec[:, 0, SPEC_OY] = dy
    spec[:, 0, SPEC_TH] = angle
    return spec_capacity(s, spec, np.full(len(s), bool(transpose)))


# --------------------------------------------------------------------------- #
# explicit construction
# --------------------------------------------------------------------------- #
def build_spec_packing(s: float, spec: np.ndarray, transpose: bool = False, method: str = "tilted_block",
                       exact: str = "", meta: Optional[Dict] = None) -> Optional[Packing]:
    """Construct the packing of ``spec (K, 6)`` explicitly (``None`` if it does not fit)."""
    spec = np.asarray(spec, float).reshape(-1, 6)
    if spec_capacity(np.array([s]), spec[None], np.array([transpose]))[0] < 0:
        return None
    polys = spec_polygons(np.array([s]), spec[None])
    if transpose:
        polys = polys[..., ::-1]
    total, JL, JR, Fb, Ft, lb, rb, lt, rt = row_fill(np.array([s]), polys)
    JL, JR, Fb, Ft = int(JL[0]), int(JR[0]), int(Fb[0]), int(Ft[0])
    B = int(math.floor(s + EPS))
    rows: List[List[float]] = []
    for j in range(Fb):                       # full rows below the obstacle
        rows += [[i + 0.5, j + 0.5, 0.0] for i in range(B)]
    for j in range(Ft):                       # full rows above the obstacle
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
    tiles = np.vstack([spec_tiles(s, piece) for piece in spec])
    sq = np.vstack([arr, tiles]) if len(arr) else tiles
    meta = dict(meta or {})
    meta.setdefault("spec", spec.tolist())
    meta.setdefault("transpose", bool(transpose))
    return Packing(len(sq), float(s), sq, method, exact=exact, meta=meta)


def build_block_packing(s: float, p: int, q: int, dx: float, dy: float, angle: float = QUARTER_PI,
                        transpose: bool = False) -> Optional[Packing]:
    """Construct the single-block packing explicitly (``None`` if the block does not fit)."""
    spec = np.array([[p, q, dx, dy, angle, 0.0]])
    exact = f"tilted {p}x{q} block at {math.degrees(angle):g}deg, offset ({dx:+g},{dy:+g})" + (" transposed" if transpose else "")
    return build_spec_packing(s, spec, transpose, "tilted_block", exact,
                              {"p": p, "q": q, "dx": dx, "dy": dy, "angle": angle, "transpose": transpose})


# --------------------------------------------------------------------------- #
# vectorised minimal side
# --------------------------------------------------------------------------- #
def min_side_batch(n: int, spec: np.ndarray, transpose: np.ndarray, s_lo: float, s_hi: float,
                   iters: int = 44, margin: float = np.inf) -> np.ndarray:
    """Per-candidate smallest ``s`` in ``[s_lo, s_hi]`` with capacity >= n (``inf`` where none).

    Bisection on all candidates at once; a candidate whose lower bound exceeds
    the best upper bound found so far by more than ``margin`` is dropped.
    """
    P = len(spec)
    out = np.full(P, np.inf)
    if P == 0 or s_hi <= s_lo:
        return out
    transpose = np.broadcast_to(np.asarray(transpose, bool), (P,))
    cap = spec_capacity(np.full(P, s_hi), spec, transpose)
    idx = np.nonzero(cap >= n)[0]
    if not len(idx):
        return out
    lo = np.full(len(idx), s_lo)
    hi = np.full(len(idx), s_hi)
    best = s_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = spec_capacity(mid, spec[idx], transpose[idx]) >= n
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
        best = min(best, float(hi.min()))
        keep = lo <= best + margin
        if not keep.all():
            idx, lo, hi = idx[keep], lo[keep], hi[keep]
    out[idx] = hi
    return out


def min_side_vec(n: int, p: int, q: int, dx: np.ndarray, dy: np.ndarray, s_lo: float, s_hi: float,
                 transpose: bool, angle: float = QUARTER_PI, iters: int = 42) -> np.ndarray:
    """Per-offset smallest ``s`` in ``[s_lo, s_hi]`` with capacity >= n (``inf`` where none)."""
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    spec = np.zeros((len(dx), 1, 6))
    spec[:, 0, SPEC_W] = p
    spec[:, 0, SPEC_H] = q
    spec[:, 0, SPEC_OX] = dx
    spec[:, 0, SPEC_OY] = dy
    spec[:, 0, SPEC_TH] = angle
    return min_side_batch(n, spec, np.full(len(dx), bool(transpose)), s_lo, s_hi, iters, margin=np.inf)


# --------------------------------------------------------------------------- #
# families: parameter vectors -> specs
# --------------------------------------------------------------------------- #
class Family:
    """A parametrised family of specs.  ``params`` rows are ``(P, m)``; ``cont`` lists the
    continuous parameters (index, initial step) used by the pattern search; ``rank`` orders
    families by simplicity (ties in ``s`` are reported with the simplest structure)."""
    name = ""
    rank = 0
    cont: Tuple[Tuple[int, float], ...] = ()

    def specs(self, params: np.ndarray) -> np.ndarray:  # (P, m) -> (P, K, 6)
        raise NotImplementedError

    def label(self, params: np.ndarray, transpose: bool) -> str:
        raise NotImplementedError

    def meta(self, params: np.ndarray, transpose: bool) -> Dict:
        return {"family": self.name, "params": [float(v) for v in params], "transpose": bool(transpose)}

    @staticmethod
    def _deg(a: float) -> str:
        return f"{math.degrees(a):.6g}deg"


class SingleBlock(Family):
    """``[p, q, dx, dy, theta, u]``: one block (``u`` = row shear, 0 for a rectangle)."""
    name = "single"
    cont = ((2, 0.125), (3, 0.125), (4, math.radians(2.0)))

    def specs(self, params):
        return np.asarray(params, float)[:, None, :]

    def label(self, params, transpose):
        p, q, dx, dy, th, u = params
        sh = f", shear {u:+.6g}" if abs(u) > 1e-12 else ""
        return (f"tilted {int(round(p))}x{int(round(q))} block at {self._deg(th)}, offset ({dx:+.6g},{dy:+.6g}){sh}"
                + (" transposed" if transpose else ""))


class ShearBlock(SingleBlock):
    rank = 1
    """``[p, q, dx, dy, theta, u]``: ``q`` rows of ``p`` squares, row ``j`` shifted by ``u * j`` along
    the row.  The union of the rows is a staircase, so every row is its own convex piece
    (all candidates of a batch must have the same ``q``)."""
    name = "shear"
    cont = ((2, 0.125), (3, 0.125), (4, math.radians(2.0)), (5, 0.1))

    def specs(self, params):
        params = np.asarray(params, float)
        q = int(round(params[0, 1]))
        if not np.all(np.rint(params[:, 1]) == q):
            raise ValueError("all sheared blocks of a batch must have the same number of rows")
        P = len(params)
        p, dx, dy, th, u = params[:, 0], params[:, 2], params[:, 3], params[:, 4], params[:, 5]
        c, sn = np.cos(th), np.sin(th)
        spec = np.zeros((P, q, 6))
        for k in range(q):
            j = k - (q - 1) / 2.0
            a = u * j          # along e1
            spec[:, k, SPEC_W] = p
            spec[:, k, SPEC_H] = 1
            spec[:, k, SPEC_OX] = dx + a * c - j * sn
            spec[:, k, SPEC_OY] = dy + a * sn + j * c
            spec[:, k, SPEC_TH] = th
        return spec


class TwoBlocks(Family):
    rank = 2
    """``[p, q, ex, ey, theta]``: two ``p x q`` blocks at angle ``theta`` centred at ``centre +- (ex, ey)``."""
    name = "two"
    cont = ((2, 0.125), (3, 0.125), (4, math.radians(2.0)))

    def specs(self, params):
        params = np.asarray(params, float)
        P = len(params)
        spec = np.zeros((P, 2, 6))
        for k, sgn in enumerate((1.0, -1.0)):
            spec[:, k, SPEC_W] = params[:, 0]
            spec[:, k, SPEC_H] = params[:, 1]
            spec[:, k, SPEC_OX] = sgn * params[:, 2]
            spec[:, k, SPEC_OY] = sgn * params[:, 3]
            spec[:, k, SPEC_TH] = params[:, 4]
        return spec

    def label(self, params, transpose):
        p, q, ex, ey, th = params
        return (f"two {int(round(p))}x{int(round(q))} blocks at {self._deg(th)}, centres +-({ex:.6g},{ey:.6g})"
                + (" transposed" if transpose else ""))


def chain_specs(p: np.ndarray, lengths: np.ndarray, angles: np.ndarray, centre_seg: int) -> np.ndarray:
    """Specs ``(P, S, 6)`` of a chain of ``S`` segments of width ``p`` (tiles across) and lengths
    ``lengths (P, S)`` along directions ``angles (P, S)``.  Consecutive segments touch at a
    corner: the next segment pivots about the corner of the previous segment's end face on
    the side it turns towards.  Segment ``centre_seg`` is centred at the origin."""
    p = np.asarray(p, float)
    lengths = np.asarray(lengths, float)
    angles = np.asarray(angles, float)
    P, S = lengths.shape
    cx = np.zeros((P, S))
    cy = np.zeros((P, S))
    for k in range(1, S):
        a0, a1 = angles[:, k - 1], angles[:, k]
        d0 = np.stack([np.cos(a0), np.sin(a0)], -1)
        n0 = np.stack([-np.sin(a0), np.cos(a0)], -1)
        d1 = np.stack([np.cos(a1), np.sin(a1)], -1)
        n1 = np.stack([-np.sin(a1), np.cos(a1)], -1)
        delta = a1 - a0
        sig = np.where(delta >= 0, 1.0, -1.0)[:, None]
        c0 = np.stack([cx[:, k - 1], cy[:, k - 1]], -1)
        pivot = c0 + 0.5 * lengths[:, k - 1:k] * d0 + sig * 0.5 * p[:, None] * n0
        c1 = pivot + 0.5 * lengths[:, k:k + 1] * d1 - sig * 0.5 * p[:, None] * n1
        cx[:, k], cy[:, k] = c1[:, 0], c1[:, 1]
    cx -= cx[:, centre_seg:centre_seg + 1]
    cy -= cy[:, centre_seg:centre_seg + 1]
    spec = np.zeros((P, S, 6))
    spec[:, :, SPEC_W] = lengths
    spec[:, :, SPEC_H] = p[:, None]
    spec[:, :, SPEC_OX] = cx
    spec[:, :, SPEC_OY] = cy
    spec[:, :, SPEC_TH] = angles
    return spec


class BentStrip(Family):
    rank = 3
    """``[p, q1, q2, a, dx, dy, kind]``: a 45-degree strip of width ``p`` bent by the angle ``a``.

    ``kind 0`` (S): segments ``[q1 @ +a] [q2 @ 0] [q1 @ +a]``;
    ``kind 1`` (W): segments ``[q1 @ -a] [1 @ 0] [q2 @ +a] [1 @ 0] [q1 @ -a]``.
    Both are symmetric under a half turn about the middle segment's centre,
    which sits at ``centre + (dx, dy)``.
    """
    name = "strip"
    cont = ((3, math.radians(1.0)), (4, 0.125), (5, 0.125))

    def __init__(self, kind: int = 0):
        self.kind = int(kind)      # one instance per kind: the two kinds have different piece counts

    def _segments(self, params):
        params = np.asarray(params, float)
        p, q1, q2, a = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        if not np.all(np.rint(params[:, 6]) == self.kind):
            raise ValueError("strip kind does not match the family instance")
        one = np.ones_like(q1)
        if self.kind == 0:
            L = np.stack([q1, q2, q1], -1)
            A = np.stack([a, 0 * a, a], -1)
            c = 1
        else:
            L = np.stack([q1, one, q2, one, q1], -1)
            A = np.stack([-a, 0 * a, a, 0 * a, -a], -1)
            c = 2
        return p, L, QUARTER_PI + A, c

    def specs(self, params):
        params = np.asarray(params, float)
        p, L, A, c = self._segments(params)
        spec = chain_specs(p, L, A, c)
        spec[:, :, SPEC_OX] += params[:, 4:5]
        spec[:, :, SPEC_OY] += params[:, 5:6]
        return spec

    def label(self, params, transpose):
        p, q1, q2, a, dx, dy, kind = params
        p, q1, q2 = int(round(p)), int(round(q1)), int(round(q2))
        if int(round(kind)) == 0:
            shape = f"S-bent {p}x{2 * q1 + q2} strip [{q1}|{q2}|{q1}]"
        else:
            shape = f"W-bent {p}x{2 * q1 + q2 + 2} strip [{q1}|1|{q2}|1|{q1}]"
        return (f"{shape} at 45deg, bend {math.degrees(a):.6g}deg, offset ({dx:+.6g},{dy:+.6g})"
                + (" transposed" if transpose else ""))


def _row_contact(theta_a, v_a, theta_b, v_b, t, p: int) -> np.ndarray:
    """Smallest along-distance ``D`` (measured along row ``a``'s y axis, between the row centres)
    at which no tile of row ``b`` overlaps a tile of row ``a``; row ``b``'s centre sits at
    ``(t, D)`` in row ``a``'s tile frame.  Rows are ``p`` parallel unit squares at tilt
    ``theta`` with in-row drift ``v`` (tile ``i`` at ``R(theta) (i, v i)`` from the centre).
    Exact separating-axis computation over all tile pairs and the four edge normals;
    ``+inf`` when no along-shift separates a pair."""
    theta_a, v_a, theta_b, v_b, t = np.broadcast_arrays(*[np.asarray(x, float) for x in (theta_a, v_a, theta_b, v_b, t)])
    P = theta_a.shape
    i = np.arange(p) - (p - 1) / 2.0
    d = theta_b - theta_a                                       # relative tilt (in a's frame b is at angle d)
    cd, sd = np.cos(d), np.sin(d)
    # tile offsets in a's frame: a: (i, v_a i); b: R(d) (i', v_b i') + (t, D)
    ax = i[None, :, None] * np.ones_like(theta_a)[..., None, None]
    ay = i[None, :, None] * v_a[..., None, None]
    bx = (cd[..., None, None] * i[None, None, :] - sd[..., None, None] * v_b[..., None, None] * i[None, None, :]) + t[..., None, None]
    by = (sd[..., None, None] * i[None, None, :] + cd[..., None, None] * v_b[..., None, None] * i[None, None, :])
    d0x = bx - ax                                               # (..., p, p) D-independent offset
    d0y = by - ay
    half = 0.5 + 0.5 * (np.abs(cd) + np.abs(sd))                # half-extent sum on any edge normal of a or b
    half = half[..., None, None]
    best = np.full(d0x.shape, np.inf)
    # axes: a's x, a's y, b's x, b's y (unit vectors in a's frame)
    axes = ((np.ones_like(cd), np.zeros_like(cd)), (np.zeros_like(cd), np.ones_like(cd)), (cd, sd), (-sd, cd))
    for nx, ny in axes:
        nx = nx[..., None, None]
        ny = ny[..., None, None]
        proj0 = nx * d0x + ny * d0y
        with np.errstate(divide="ignore", invalid="ignore"):
            Dn = np.where(np.abs(ny) < 1e-12,
                          np.where(np.abs(proj0) >= half - 1e-12, -np.inf, np.inf),
                          (np.sign(ny) * half - proj0) / np.where(np.abs(ny) < 1e-12, 1.0, ny))
        best = np.minimum(best, Dn)
    return np.maximum(best.max(axis=(-1, -2)), 0.0)


class RowChain(Family):
    rank = 4
    """A strip of ``R`` rows of ``p`` parallel unit squares, symmetric under a half turn about
    its centre (which sits at ``centre + (dx, dy)``).

    ``params = [p, R, dx, dy, theta_0..theta_{H-1}, phi_0..phi_{H-1}, t_0..t_{H-1}, g_0..g_{H-1}]``
    with ``H = ceil(R / 2)`` rows in the upper half (index 0 nearest the centre).  Row ``k`` has
    tile tilt ``theta_k`` and row direction ``phi_k`` (in-row drift ``v_k = tan(phi_k - theta_k)``,
    so a change of ``theta_k`` alone tilts the tiles while the row keeps its direction - the
    mechanism of the records 37, 88, 102, 123).  Row ``k`` is placed against row ``k-1``: its
    centre is at ``(t_k, D + g_k)`` in row ``k-1``'s tile frame, where ``D`` is the exact contact
    distance, so any parameter change keeps the rows touching and non-overlapping.  For even
    ``R`` joint 0 joins the two central rows; for odd ``R`` row 0 is centred at the origin.
    The lower half is the half-turn image of the upper half (``sym="rot"``, e.g. n = 18, 70,
    88) or its mirror image across the central across-axis (``sym="mirror"``, e.g. n = 19:
    tiles are reflected in row 0's tile frame, so their tilts become ``2 theta_0 - theta_k``).
    """
    name = "chain"

    def __init__(self, p: int, R: int, sym: str = "rot"):
        self.p, self.R, self.H = int(p), int(R), (int(R) + 1) // 2
        self.sym = sym
        H = self.H
        self.ith, self.iph, self.it, self.ig = 4, 4 + H, 4 + 2 * H, 4 + 3 * H
        self.m = 4 + 4 * H
        k0 = 0 if R % 2 == 0 else 1
        kph = 1 if (sym == "mirror" and R % 2 == 1) else 0   # an odd central row has no drift in mirror mode
        self.cont = tuple([(2, 0.1), (3, 0.1)] + [(self.ith + k, math.radians(1.0)) for k in range(H)]
                          + [(self.iph + k, math.radians(1.0)) for k in range(kph, H)]
                          + [(self.it + k, 0.05) for k in range(k0, H)] + [(self.ig + k, 0.02) for k in range(k0, H)])

    @classmethod
    def from_block(cls, p: int, R: int, theta: float, dx: float, dy: float, sym: str = "rot") -> Tuple["RowChain", np.ndarray]:
        fam = cls(p, R, sym)
        params = np.zeros(fam.m)
        params[:4] = (p, R, dx, dy)
        params[fam.ith:fam.ith + fam.H] = theta
        params[fam.iph:fam.iph + fam.H] = theta
        return fam, params

    def row_geometry(self, params):
        """Row centres ``(P, H, 2)`` (upper half, relative to the chain centre), tilts, drifts."""
        params = np.asarray(params, float)
        P, H, p = len(params), self.H, self.p
        th = params[:, self.ith:self.ith + H]
        ph = params[:, self.iph:self.iph + H]
        t = params[:, self.it:self.it + H]
        g = np.maximum(params[:, self.ig:self.ig + H], 0.0)
        v = np.tan(np.clip(ph - th, -QUARTER_PI + 1e-6, QUARTER_PI - 1e-6))
        if self.sym == "mirror" and self.R % 2 == 1:
            v[:, 0] = 0.0                      # the central row must be its own mirror image
        cen = np.zeros((P, H, 2))
        if self.R % 2 == 0:
            if self.sym == "rot":
                # the half-turn image of row 0 has the same drift and sits (t0, D) away
                D = _row_contact(th[:, 0], v[:, 0], th[:, 0], v[:, 0], t[:, 0], p) + g[:, 0]
            else:
                # the mirror image of row 0 has drift -v0 and the same lateral position
                D = _row_contact(th[:, 0], v[:, 0], th[:, 0], -v[:, 0], np.zeros(P), p) + g[:, 0]
            c, s = np.cos(th[:, 0]), np.sin(th[:, 0])
            cen[:, 0, 0] = 0.5 * (c * t[:, 0] - s * D)
            cen[:, 0, 1] = 0.5 * (s * t[:, 0] + c * D)
        for k in range(1, H):
            D = _row_contact(th[:, k - 1], v[:, k - 1], th[:, k], v[:, k], t[:, k], p) + g[:, k]
            c, s = np.cos(th[:, k - 1]), np.sin(th[:, k - 1])
            cen[:, k, 0] = cen[:, k - 1, 0] + c * t[:, k] - s * D
            cen[:, k, 1] = cen[:, k - 1, 1] + s * t[:, k] + c * D
        return cen, th, v

    def specs(self, params):
        params = np.asarray(params, float)
        P, H, p, R = len(params), self.H, self.p, self.R
        cen, th, v = self.row_geometry(params)
        i = np.arange(p) - (p - 1) / 2.0
        c, s = np.cos(th), np.sin(th)                               # (P, H)
        ox = cen[:, :, 0:1] + c[:, :, None] * i - s[:, :, None] * v[:, :, None] * i      # (P, H, p)
        oy = cen[:, :, 1:2] + s[:, :, None] * i + c[:, :, None] * v[:, :, None] * i
        thk = np.broadcast_to(th[:, :, None], ox.shape)
        if self.sym == "rot":
            mx, my, mt = -ox, -oy, thk
        else:
            # reflect across row 0's across-axis (the x axis of its tile frame): in that frame
            # (x, y) -> (x, -y) and angles theta -> 2 theta_0 - theta
            c0, s0 = np.cos(th[:, 0:1, None]), np.sin(th[:, 0:1, None])
            xf = c0 * ox + s0 * oy
            yf = -s0 * ox + c0 * oy
            mx = c0 * xf + s0 * yf
            my = s0 * xf - c0 * yf
            mt = 2 * th[:, 0:1, None] - thk
        if R % 2 == 0:
            X = np.concatenate([ox, mx], axis=1)
            Y = np.concatenate([oy, my], axis=1)
            T = np.concatenate([thk, mt], axis=1)
        else:
            X = np.concatenate([ox, mx[:, 1:]], axis=1)
            Y = np.concatenate([oy, my[:, 1:]], axis=1)
            T = np.concatenate([thk, mt[:, 1:]], axis=1)
        K = R * p
        spec = np.zeros((P, K, 6))
        spec[:, :, SPEC_W] = 1
        spec[:, :, SPEC_H] = 1
        spec[:, :, SPEC_OX] = X.reshape(P, K) + params[:, 2:3]
        spec[:, :, SPEC_OY] = Y.reshape(P, K) + params[:, 3:4]
        spec[:, :, SPEC_TH] = T.reshape(P, K)
        return spec

    def label(self, params, transpose):
        params = np.asarray(params, float)
        H = self.H
        th = np.degrees(params[self.ith:self.ith + H])
        drift = np.degrees(params[self.iph:self.iph + H] - params[self.ith:self.ith + H])
        t = params[self.it:self.it + H]
        tilts = "/".join(f"{a:.3g}" for a in th)
        return (f"row chain {self.p}x{self.R} ({self.sym}): tilts {tilts}deg, row-drift {'/'.join(f'{a:+.2g}' for a in drift)}deg, "
                f"slides {'/'.join(f'{a:+.3g}' for a in t)}, offset ({params[2]:+.4g},{params[3]:+.4g})"
                + (" transposed" if transpose else ""))

    def meta(self, params, transpose):
        return {"family": self.name, "p": self.p, "R": self.R, "sym": self.sym, "params": [float(v) for v in params],
                "transpose": bool(transpose)}


FAMILIES = {"single": SingleBlock(), "shear": ShearBlock(), "two": TwoBlocks(), "strip": BentStrip(0),
            "strip_w": BentStrip(1)}


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
DEFAULT_OFFSETS = tuple(k / 4.0 for k in range(-5, 6))
DEFAULT_ANGLES = tuple(math.radians(a) for a in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0))
COARSE_OFFSETS = tuple(k / 2.0 for k in range(-3, 4))
DEFAULT_BENDS = tuple(math.radians(a) for a in (1.0, 2.0, 4.0))


def _cand_key(c):
    return (round(c.s, 9), c.family.rank)


class _Candidate:
    __slots__ = ("s", "family", "params", "transpose")

    def __init__(self, s, family, params, transpose):
        self.s = float(s)
        self.family = family
        self.params = np.asarray(params, float)
        self.transpose = bool(transpose)

    def spec(self):
        return self.family.specs(self.params[None])[0]

    def build(self, n):
        pk = build_spec_packing(self.s, self.spec(), self.transpose, "tilted_block",
                                self.family.label(self.params, self.transpose),
                                self.family.meta(self.params, self.transpose))
        if pk is None or pk.n < n:
            return None
        pk = pk.take(n)
        # the bisection stops within ~EPS of the exact threshold; scaling by the
        # minimal repair factor makes the packing exactly valid (strict check)
        s_fix, sq_fix = repair(pk.s, pk.squares)
        pk = Packing(pk.n, s_fix, sq_fix, pk.method, pk.exact, pk.meta)
        if not verify(pk.s, pk.squares, 1e-12).ok:
            return None
        return pk


def _grid_stage(n, family, params, transposes, s_lo, s_hi, margin, keep):
    """Evaluate a batch of grid candidates; return the best ``keep`` as ``_Candidate`` (sorted by s)."""
    params = np.asarray(params, float)
    if len(params) == 0:
        return []
    transposes = np.broadcast_to(np.asarray(transposes, bool), (len(params),))
    spec = family.specs(params)
    sides = min_side_batch(n, spec, transposes, s_lo, s_hi, margin=margin)
    order = np.argsort(sides, kind="stable")[:keep]
    return [_Candidate(sides[k], family, params[k], transposes[k]) for k in order if np.isfinite(sides[k])]


def _bisect_side(n, fam, params, transposes, s_lo, s_hi, iters):
    """Smallest side (bisection, ``iters`` steps) of each row of ``params`` starting from
    the achievable upper bounds ``s_hi`` (array)."""
    spec = fam.specs(params)
    lo = np.full(len(params), s_lo)
    hi = np.asarray(s_hi, float).copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = spec_capacity(mid, spec, transposes) >= n
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


def _refine(n, cands: List[_Candidate], s_lo: float, rounds: int = 40, shrink: float = 0.5,
            min_step: float = 1e-7, iters: int = 24, time_limit: Optional[float] = None) -> List[_Candidate]:
    """Pattern search on the continuous parameters of each candidate (batched over candidates).

    Every round tries ``+-step`` on each continuous parameter; a trial that admits ``n``
    squares below the candidate's current side replaces it (side re-bisected with
    ``iters`` steps), otherwise the step shrinks.  The final sides are re-bisected to
    full precision."""
    import time
    t0 = time.time()
    cands = [_Candidate(c.s, c.family, c.params.copy(), c.transpose) for c in cands]
    if not cands:
        return cands
    steps = [np.array([st for _, st in c.family.cont]) for c in cands]
    touched = np.zeros(len(cands), bool)

    def out_of_time():
        return time_limit is not None and time.time() - t0 > time_limit

    for _ in range(rounds):
        if out_of_time():
            break
        trial_params, owners, trial_tr, fams = [], [], [], []
        for ci, c in enumerate(cands):
            if steps[ci].max() < min_step:
                continue
            for (pi, _), st in zip(c.family.cont, steps[ci]):
                if st < min_step:
                    continue
                for sgn in (1.0, -1.0):
                    q = c.params.copy()
                    q[pi] += sgn * st
                    trial_params.append(q)
                    owners.append(ci)
                    trial_tr.append(c.transpose)
                    fams.append(c.family)
        if not trial_params:
            break
        improved = np.zeros(len(cands), bool)
        # group by family instance (specs have family-specific shapes)
        for fam in {id(f): f for f in fams}.values():
            if out_of_time():
                break
            sel = [i for i, f in enumerate(fams) if f is fam]
            P_ = np.array([trial_params[i] for i in sel])
            T_ = np.array([trial_tr[i] for i in sel])
            S_ = np.array([cands[owners[i]].s for i in sel]) - 1e-9
            spec = fam.specs(P_)
            cap = spec_capacity(S_, spec, T_)
            good = np.nonzero(cap >= n)[0]
            if not len(good):
                continue
            # bisect the improving trials individually (each against its owner's current s)
            hi = _bisect_side(n, fam, P_[good], T_[good], s_lo, S_[good], iters)
            for g, s_new in zip(good, hi):
                ci = owners[sel[g]]
                if s_new < cands[ci].s - 1e-12:
                    cands[ci] = _Candidate(s_new, fam, P_[g], bool(T_[g]))
                    improved[ci] = True
                    touched[ci] = True
        for ci in range(len(cands)):
            if not improved[ci]:
                steps[ci] = steps[ci] * shrink
    # full-precision sides for the candidates that moved
    for fam in {id(c.family): c.family for c in cands}.values():
        sel = [i for i, c in enumerate(cands) if c.family is fam and touched[i]]
        if not sel:
            continue
        hi = _bisect_side(n, fam, np.array([cands[i].params for i in sel]), np.array([cands[i].transpose for i in sel]),
                          s_lo, np.array([cands[i].s for i in sel]), 44)
        for i, s_new in zip(sel, hi):
            cands[i] = _Candidate(min(s_new, cands[i].s), fam, cands[i].params, cands[i].transpose)
    cands.sort(key=_cand_key)
    return cands


def _chain_seeds(pool: List[_Candidate], limit: int, s_max: float, offsets: Sequence[float],
                 max_tiles: int = 48, small_tiles: int = 0) -> List[_Candidate]:
    """Row-chain relaxations of the best distinct single-block candidates (both row assignments
    and both symmetries).  Chains of at most ``small_tiles`` squares are additionally seeded at
    every grid offset, starting from ``s_max`` (off by default: the record chains, e.g. n = 19,
    need several coordinated moves from the straight chain that a coordinate search does not find)."""
    seeds, seen, shapes_done = [], set(), set()
    single = FAMILIES["single"]
    for c in sorted(pool, key=_cand_key):
        if c.family is not single:
            continue
        p, q = int(round(c.params[0])), int(round(c.params[1]))
        if p * q > max_tiles:
            continue
        th, dx, dy = float(c.params[4]), float(c.params[2]), float(c.params[3])
        key = (p, q, round(th, 6), round(dx, 6), round(dy, 6), c.transpose)
        if key in seen:
            continue
        seen.add(key)
        for (pp, R, ang) in ((p, q, th), (q, p, th + 0.5 * math.pi)):
            for sym in ("rot", "mirror"):
                if R < 2 and sym == "mirror":
                    continue
                fam, params = RowChain.from_block(pp, R, ang, dx, dy, sym)
                seeds.append(_Candidate(c.s, fam, params, c.transpose))
                if p * q <= small_tiles and (p, q, round(th, 6)) not in shapes_done:
                    for ox in offsets:
                        for oy in offsets:
                            if abs(ox - dx) < 1e-9 and abs(oy - dy) < 1e-9:
                                continue
                            fam2, params2 = RowChain.from_block(pp, R, ang, ox, oy, sym)
                            seeds.append(_Candidate(s_max, fam2, params2, False))
        shapes_done.add((p, q, round(th, 6)))
        if len(seen) >= limit:
            break
    return seeds


def _single_grid(shapes, angles, offsets):
    """Parameter rows ``[p, q, dx, dy, theta, 0]`` and transposes for all combinations."""
    DX, DY = np.meshgrid(np.asarray(offsets, float), np.asarray(offsets, float), indexing="ij")
    DX, DY = DX.ravel(), DY.ravel()
    rows, trs = [], []
    for p, q in shapes:
        for th in angles:
            for tr in (False, True):
                if tr and p == q and abs(th - QUARTER_PI) < 1e-12:
                    continue
                blk = np.column_stack([np.full(len(DX), p), np.full(len(DX), q), DX, DY,
                                       np.full(len(DX), th), np.zeros(len(DX))])
                rows.append(blk)
                trs.append(np.full(len(DX), tr))
    if not rows:
        return np.zeros((0, 6)), np.zeros(0, bool)
    return np.vstack(rows), np.concatenate(trs)


def tilted_block_search(n: int, s_max: Optional[float] = None,
                        offsets: Sequence[float] = DEFAULT_OFFSETS,
                        shapes: Optional[Iterable[Tuple[int, int]]] = None,
                        angle: float = QUARTER_PI, verbose: bool = False,
                        families: Sequence[str] = ("single", "two", "strip", "shear"),
                        angles: Sequence[float] = DEFAULT_ANGLES,
                        bends: Sequence[float] = DEFAULT_BENDS,
                        refine: bool = True, keep: int = 12, relax: bool = True, relax_seeds: int = 2,
                        relax_rounds: int = 30, relax_time: Optional[float] = 1.5) -> Optional[Packing]:
    """Best tilted-block packing for ``n`` with side below ``s_max`` (default: grid side).

    Stage 1 scans single blocks at ``angle`` (45 degrees) over ``shapes`` (default:
    all ``p <= q`` that can fit) and centre offsets ``offsets``, in both the row
    and the column model.  Further families (``"single"`` at the extra
    ``angles``, ``"two"`` symmetric blocks, bent ``"strip"``s with the bend
    angles ``bends``, ``"shear"``ed blocks) are scanned on coarser grids, and the
    best ``keep`` candidates of every stage are polished by a pattern search
    over their continuous parameters.  Returns the verified packing with the
    smallest ``s`` or ``None`` if nothing beats ``s_max``.
    """
    import time
    t0 = time.time()
    if s_max is None:
        s_max = grid(n).s
    s_lo = math.sqrt(n)
    best_s = s_max - 1e-12
    if shapes is None:
        pmax = int(s_max * SQRT2) + 1
        shapes = [(p, q) for p in range(1, pmax + 1) for q in range(p, pmax + 1)
                  if p * q <= n and (p + q) / SQRT2 <= s_max + 1e-9]
    shapes = list(shapes)
    pool: List[_Candidate] = []

    def log(msg):
        if verbose:
            print(f"  n={n} [{time.time() - t0:6.2f}s] {msg}")

    def add(cs, what):
        nonlocal best_s
        pool.extend(cs)
        if cs:
            best_s = min(best_s, cs[0].s)
            log(f"{what}: best {cs[0].s:.10f} {cs[0].family.label(cs[0].params, cs[0].transpose)}")
        else:
            log(f"{what}: nothing below {best_s:.10f}")

    single = FAMILIES["single"]
    large = n > 120          # coarser secondary grids keep the search under ~10 s up to n = 300
    # stage 1: single blocks at the base angle on the fine offset grid
    params, trs = _single_grid(shapes, [angle], offsets)
    add(_grid_stage(n, single, params, trs, s_lo, best_s, 0.05, keep), "single 45deg")
    base = list(pool)  # remember the straight-block results for the strip family
    if "single" in families and len(angles):
        params, trs = _single_grid(shapes, list(angles), COARSE_OFFSETS if not large else (-1.0, -0.5, 0.0, 0.5, 1.0))
        add(_grid_stage(n, single, params, trs, s_lo, best_s + 0.05, 0.05, keep), "single angles")
    if "two" in families:
        e = np.arange(0.0, 3.6, 0.5) if not large else np.arange(0.0, 3.1, 0.5)
        EX, EY = np.meshgrid(e, e, indexing="ij")
        EX, EY = EX.ravel(), EY.ravel()
        rows, trs = [], []
        two_angles = [angle] + list(angles)
        if large:
            two_angles = [angle] + [a for a in angles if abs(a - math.radians(30)) < 1e-9 or abs(a - math.radians(60)) < 1e-9]
        for p, q in shapes:
            if 2 * p * q > n or 2 * p * q < (n // 4 if not large else n // 2):
                continue
            for th in two_angles:
                for tr in (False, True):
                    rows.append(np.column_stack([np.full(len(EX), p), np.full(len(EX), q), EX, EY, np.full(len(EX), th)]))
                    trs.append(np.full(len(EX), tr))
        if rows:
            add(_grid_stage(n, FAMILIES["two"], np.vstack(rows), np.concatenate(trs), s_lo, best_s + 0.05, 0.05, keep),
                "two blocks")
    if "strip" in families and len(bends):
        # bend the straight strips whose shape is competitive
        good_shapes = []
        for c in base[:keep]:
            p, q = int(round(c.params[0])), int(round(c.params[1]))
            if (p, q) not in good_shapes and p <= 4 and q >= 3:
                good_shapes.append((p, q))
        DX, DY = np.meshgrid(np.asarray(offsets, float), np.asarray(offsets, float), indexing="ij")
        DX, DY = DX.ravel(), DY.ravel()
        for kind in (0, 1):
            rows, trs = [], []
            for p, q in good_shapes:
                splits = []
                if kind == 0:
                    splits = [(q1, q - 2 * q1) for q1 in range(1, q // 2 + 1) if q - 2 * q1 >= 1]
                else:
                    splits = [(q1, q - 2 * q1 - 2) for q1 in range(1, q // 2 + 1) if q - 2 * q1 - 2 >= 1]
                for q1, q2 in splits:
                    for a in bends:
                        for tr in (False, True):
                            rows.append(np.column_stack([np.full(len(DX), p), np.full(len(DX), q1), np.full(len(DX), q2),
                                                         np.full(len(DX), a), DX, DY, np.full(len(DX), kind)]))
                            trs.append(np.full(len(DX), tr))
            if rows:
                fam = FAMILIES["strip" if kind == 0 else "strip_w"]
                add(_grid_stage(n, fam, np.vstack(rows), np.concatenate(trs), s_lo, best_s + 0.05, 0.05, keep),
                    f"bent strips kind {kind}")
    if "shear" in families:
        seen, groups = set(), {}
        DX, DY = np.meshgrid(np.asarray(offsets, float), np.asarray(offsets, float), indexing="ij")
        DX, DY = DX.ravel(), DY.ravel()
        for c in sorted(pool, key=_cand_key)[:keep]:
            if c.family is not single:
                continue
            key = (int(round(c.params[0])), int(round(c.params[1])), round(float(c.params[4]), 9))
            if key in seen:
                continue
            seen.add(key)
            p, q, th = key
            for (w, h) in ((p, q), (q, p)):      # shear along either direction of the block
                if h < 2 or h > 12:               # every row is a piece: keep the batches small
                    continue
                rows, trs = groups.setdefault(h, ([], []))
                for u in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75):
                    for tr in (False, True):
                        rows.append(np.column_stack([np.full(len(DX), w), np.full(len(DX), h), DX, DY,
                                                     np.full(len(DX), th), np.full(len(DX), u)]))
                        trs.append(np.full(len(DX), tr))
        for h, (rows, trs) in sorted(groups.items()):
            add(_grid_stage(n, FAMILIES["shear"], np.vstack(rows), np.concatenate(trs), s_lo, best_s + 0.05, 0.05, keep),
                f"sheared blocks ({h} rows)")
    pool.sort(key=_cand_key)
    if refine and pool:
        # polish the best candidates of every family (a few per family so that the 45deg block
        # candidates do not crowd out the others)
        chosen, per_family = [], {}
        for c in pool:
            k = per_family.get(c.family.name, 0)
            if k < max(3, keep // 2):
                chosen.append(c)
                per_family[c.family.name] = k + 1
        refined = _refine(n, chosen, s_lo)
        pool = sorted(refined + pool, key=_cand_key)
        if refined:
            log(f"refined: best {refined[0].s:.10f} {refined[0].family.label(refined[0].params, refined[0].transpose)}")
    if relax and pool:
        # relax the best rigid blocks into row chains (per-row tilt / direction / slide)
        seeds = _chain_seeds(pool, relax_seeds, s_max, offsets)
        if seeds:
            relaxed = _refine(n, seeds, s_lo, rounds=relax_rounds,
                              time_limit=relax_time if relax_time is None or n > 40 else 2 * relax_time)
            pool = sorted(relaxed + pool, key=_cand_key)
            log(f"relaxed: best {relaxed[0].s:.10f} {relaxed[0].family.label(relaxed[0].params, relaxed[0].transpose)}")
    best: Optional[Packing] = None
    for c in pool:
        if c.s >= s_max - 1e-12:
            break
        pk = c.build(n)
        if pk is None:
            log(f"candidate failed verification: {c.family.label(c.params, c.transpose)} s={c.s:.10f}")
            continue
        if best is None or pk.s < best.s - 1e-12:
            best = pk
            log(f"verified s={pk.s:.10f} {pk.exact}")
            break
    return best
