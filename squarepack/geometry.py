"""Geometry kernels for packing unit squares in a square.

Conventions
-----------
* A square is the triple ``(x, y, theta)``: centre coordinates and rotation angle
  in **radians** (counter-clockwise).  Every small square has side length 1.
* The container is the axis-aligned square ``[0, s] x [0, s]``; ``s`` is the
  quantity we minimise.
* A packing is *valid* when every small square lies inside the container and
  no two small squares overlap (touching is allowed).

The overlap test is the separating-axis theorem specialised to squares: two
unit squares with relative angle ``phi = theta_j - theta_i`` are separated iff
the projection of the centre difference ``d`` on one of the four edge normals
exceeds the sum of the two half-extents on that normal.  Because a square's
edge normals are its own edge directions, the half-extent sum is the same on
all four axes, namely ``1/2 + (|cos phi| + |sin phi|)/2``.  Hence the
*penetration depth* used throughout the package is

    p(i, j) = 1/2 + (|cos phi| + |sin phi|)/2 - max_a |a . d|,

with ``a`` ranging over the edge directions of both squares.  ``p > 0`` means
the squares overlap, ``p <= 0`` means they are separated (``p == 0`` touching).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

SQRT2 = float(np.sqrt(2.0))
PI = float(np.pi)
HALF_PI = PI / 2.0
QUARTER_PI = PI / 4.0

Array = np.ndarray


# --------------------------------------------------------------------------- #
# basic helpers
# --------------------------------------------------------------------------- #
def support_half(phi):
    """Half-width of a unit square projected on an axis at angle ``phi`` to its edges.

    ``(|cos phi| + |sin phi|) / 2``.  For ``phi = theta`` this is the half-width of
    the axis-aligned bounding box of a square rotated by ``theta``.
    """
    return 0.5 * (np.abs(np.cos(phi)) + np.abs(np.sin(phi)))


def d_support_half(phi):
    """Derivative of :func:`support_half` (a sub-gradient at the kinks)."""
    c = np.cos(phi)
    s = np.sin(phi)
    return 0.5 * (-np.sign(c) * s + np.sign(s) * c)


def bbox_half_width(theta):
    """Half side of the axis-aligned bounding box of a unit square at angle ``theta``."""
    return support_half(theta)


def canonical_angle(theta):
    """Reduce an angle to ``(-pi/4, pi/4]`` using the 4-fold symmetry of a square
    (so that a 45-degree square is reported as ``+pi/4``)."""
    theta = np.asarray(theta, dtype=float)
    out = theta - HALF_PI * np.floor((theta + QUARTER_PI) / HALF_PI)
    out = np.where(out <= -QUARTER_PI + 1e-12, out + HALF_PI, out)
    return out


def square_vertices(x, y, theta) -> Array:
    """Vertices of unit squares, shape ``(..., 4, 2)``, counter-clockwise."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    theta = np.asarray(theta, float)
    c = np.cos(theta)
    s = np.sin(theta)
    e1 = 0.5 * np.stack([c, s], axis=-1)
    e2 = 0.5 * np.stack([-s, c], axis=-1)
    ctr = np.stack([x, y], axis=-1)
    return np.stack([ctr + e1 + e2, ctr - e1 + e2, ctr - e1 - e2, ctr + e1 - e2], axis=-2)


# --------------------------------------------------------------------------- #
# pair enumeration
# --------------------------------------------------------------------------- #
def all_pairs(n: int) -> Tuple[Array, Array]:
    I, J = np.triu_indices(n, k=1)
    return I.astype(np.intp), J.astype(np.intp)


def candidate_pairs(x: Array, y: Array, cutoff: float = SQRT2 + 1e-9,
                    dense_limit: int = 512) -> Tuple[Array, Array]:
    """Index pairs ``(i < j)`` whose centres are within ``cutoff`` in both x and y.

    Two unit squares can only overlap if ``|dx| < sqrt(2)`` and ``|dy| < sqrt(2)``
    (the diagonal is the largest extent).  For small ``n`` the dense
    enumeration is used; otherwise a uniform cell grid gives expected ``O(n)``
    candidate pairs for any non-degenerate arrangement.
    """
    n = len(x)
    if n < 2:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    if n <= dense_limit:
        I, J = all_pairs(n)
        m = (np.abs(x[I] - x[J]) < cutoff) & (np.abs(y[I] - y[J]) < cutoff)
        return I[m], J[m]
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        I, J = all_pairs(n)
        m = (np.abs(x[I] - x[J]) < cutoff) & (np.abs(y[I] - y[J]) < cutoff)
        return I[m], J[m]
    cx = np.floor(x / cutoff).astype(np.int64)
    cy = np.floor(y / cutoff).astype(np.int64)
    cx -= cx.min()
    cy -= cy.min()
    # group indices by cell with a lexicographic sort on the (cx, cy) pair; the lookup is keyed by
    # the Python-int tuple, so no scalar key arithmetic can overflow for huge coordinate spreads
    order = np.lexsort((cy, cx))
    cs = cx[order]
    ds = cy[order]
    change = np.nonzero((cs[1:] != cs[:-1]) | (ds[1:] != ds[:-1]))[0] + 1
    start = np.concatenate([[0], change])
    end = np.append(start[1:], n)
    lookup = {(int(cs[a]), int(ds[a])): (int(a), int(b)) for a, b in zip(start, end)}
    Is = []
    Js = []
    for (cxk, cyk), (a, b) in lookup.items():
        idx = order[a:b]
        if b - a > 1:
            ii, jj = np.triu_indices(b - a, 1)
            Is.append(idx[ii])
            Js.append(idx[jj])
        for dxc, dyc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            nb = lookup.get((cxk + dxc, cyk + dyc))
            if nb is None:
                continue
            idx2 = order[nb[0]:nb[1]]
            gi, gj = np.meshgrid(idx, idx2, indexing="ij")
            Is.append(gi.ravel())
            Js.append(gj.ravel())
    if not Is:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    I = np.concatenate(Is).astype(np.intp)
    J = np.concatenate(Js).astype(np.intp)
    swap = I > J
    I[swap], J[swap] = J[swap], I[swap]
    m = (np.abs(x[I] - x[J]) < cutoff) & (np.abs(y[I] - y[J]) < cutoff)
    return I[m], J[m]


# --------------------------------------------------------------------------- #
# separating-axis penetration depth
# --------------------------------------------------------------------------- #
def pair_projection_max(x, y, theta, I, J):
    """``max_a |a . d|`` over the four edge directions, plus data for gradients.

    Returns ``(M, k, sgn, A, ci, si, cj, sj)`` where ``A`` has shape ``(4, m)``
    with the signed projections and ``k`` is the arg-max axis index.
    """
    dx = x[I] - x[J]
    dy = y[I] - y[J]
    ci = np.cos(theta[I])
    si = np.sin(theta[I])
    cj = np.cos(theta[J])
    sj = np.sin(theta[J])
    A = np.stack([ci * dx + si * dy,
                  -si * dx + ci * dy,
                  cj * dx + sj * dy,
                  -sj * dx + cj * dy])
    absA = np.abs(A)
    k = np.argmax(absA, axis=0)
    cols = np.arange(A.shape[1])
    M = absA[k, cols]
    sgn = np.sign(A[k, cols])
    return M, k, sgn, A, ci, si, cj, sj


def pair_penetration(x, y, theta, I, J) -> Array:
    """Penetration depth for each pair; ``> 0`` overlap, ``<= 0`` separated."""
    if len(I) == 0:
        return np.zeros(0)
    M = pair_projection_max(x, y, theta, I, J)[0]
    return 0.5 + support_half(theta[J] - theta[I]) - M


def containment_violation(x, y, theta, s) -> Array:
    """Per-square distance by which the bounding box leaves ``[0, s]^2`` (``<= 0`` inside)."""
    w = support_half(theta)
    return np.max(np.stack([w - x, x + w - s, w - y, y + w - s]), axis=0)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
@dataclass
class VerifyReport:
    ok: bool
    n: int
    s: float
    max_penetration: float
    max_outside: float
    tol: float
    worst_pair: Optional[Tuple[int, int]] = None
    worst_square: Optional[int] = None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        state = "VALID" if self.ok else "INVALID"
        return (f"{state}: n={self.n} s={self.s:.12g} "
                f"max_penetration={self.max_penetration:.3e} (pair {self.worst_pair}) "
                f"max_outside={self.max_outside:.3e} (square {self.worst_square}) tol={self.tol:g}")


def as_arrays(squares) -> Tuple[Array, Array, Array]:
    arr = np.asarray(squares, dtype=float).reshape(-1, 3)
    return arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()


def verify(s: float, squares, tol: float = 1e-9) -> VerifyReport:
    """Check that ``squares`` (``[[x, y, theta], ...]``) is a valid packing in ``[0, s]^2``.

    ``tol`` is the largest penetration / protrusion accepted (touching squares are
    exact zeros in exact arithmetic, but constructions with irrational
    coordinates need a few ulps of slack).
    """
    x, y, t = as_arrays(squares)
    n = len(x)
    s = float(s)
    if n == 0:
        return VerifyReport(True, 0, s, -np.inf, -np.inf, tol)
    cont = containment_violation(x, y, t, s)
    worst_sq = int(np.argmax(cont))
    max_out = float(cont[worst_sq])
    I, J = candidate_pairs(x, y)
    if len(I):
        pen = pair_penetration(x, y, t, I, J)
        w = int(np.argmax(pen))
        max_pen = float(pen[w])
        worst_pair = (int(I[w]), int(J[w]))
    else:
        max_pen, worst_pair = -np.inf, None
    ok = bool(max_out <= tol and max_pen <= tol)
    return VerifyReport(ok, n, s, max_pen, max_out, tol, worst_pair, worst_sq)


def repair_scale(s: float, x, y, theta, I=None, J=None) -> float:
    """Smallest factor ``lam >= 1`` such that scaling ``s``, ``x`` and ``y`` by ``lam``
    (angles unchanged) removes every overlap and protrusion.

    Scaling the centres by ``lam`` multiplies every centre projection by ``lam``
    while the half-extent sums stay fixed, so pair ``(i, j)`` becomes separated
    exactly when ``lam >= (1/2 + R(phi)) / M``.  Returns ``inf`` when two
    centres coincide (no scaling can separate them).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    theta = np.asarray(theta, float)
    lam = 1.0
    if I is None:
        I, J = candidate_pairs(x, y, cutoff=SQRT2 * 1.5)
    if len(I):
        M = pair_projection_max(x, y, theta, I, J)[0]
        need = 0.5 + support_half(theta[J] - theta[I])
        over = need > M
        if np.any(over):
            Mo = M[over]
            if np.any(Mo <= 0):
                return float("inf")
            lam = max(lam, float(np.max(need[over] / Mo)))
    w = support_half(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        for pos in (x, y):
            for val in (pos, s - pos):
                bad = val < w
                if np.any(bad):
                    vals = val[bad]
                    if np.any(vals <= 0):
                        return float("inf")
                    lam = max(lam, float(np.max(w[bad] / vals)))
    return lam


def repair(s: float, squares, margin: float = 1e-12):
    """Return ``(s', squares')``: the packing scaled by the minimal factor that
    makes it exactly valid (plus ``margin`` relative slack)."""
    x, y, t = as_arrays(squares)
    lam = repair_scale(s, x, y, t)
    if not np.isfinite(lam):
        raise ValueError("packing cannot be repaired by scaling (coincident centres)")
    lam *= (1.0 + margin)
    return s * lam, np.column_stack([x * lam, y * lam, t])
