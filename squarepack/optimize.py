"""Numerical search for tight packings.

Mathematical formulation
------------------------
Variables: centres ``(x_i, y_i)`` and angles ``theta_i`` of the ``n`` squares.
For a *fixed* container side ``s`` the constraint violation is measured by

    E(z; s) = sum_{i<j} max(0, p_ij)^2 + sum_i sum_{4 sides} max(0, v_i)^2

where ``p_ij`` is the separating-axis penetration depth of squares ``i`` and
``j`` (see :mod:`squarepack.geometry`) and ``v_i`` the protrusion of square
``i`` beyond a side of the container.  ``E`` is zero exactly on valid packings,
continuous, and piecewise smooth; its gradient is derived analytically below.

Two outer strategies are provided and benchmarked against each other:

``bisect`` (default)
    Fix ``s``; minimise ``E`` with L-BFGS-B from the previous configuration
    scaled into the smaller container; if ``E`` vanishes the container is
    shrunk further, otherwise a few basin-hopping perturbations are tried,
    and the step is halved on failure.  This is the "compaction" scheme used
    for dense circle packings and it converges geometrically in ``s``.
``penalty``
    Treat ``s`` as an additional variable and minimise ``s + mu * E`` with a
    continuation schedule ``mu -> infinity``.  Faster per run, but the
    quadratic penalty is only exact in the limit, so a final bisection polish
    is applied.
``anneal`` / ``anneal-hop``
    Simulated annealing of ``E`` at fixed ``s`` (Metropolis moves on single
    squares, implemented in the C core; see :func:`anneal`), used either as a
    cold-start compaction with a slowly shrinking container (the approach
    behind the 2025/26 records of T. Schadt and D. Ellsworth) followed by the
    L-BFGS bisection polish, or as the basin-hopping move of the bisection.

Basin hopping can use the ``"classic"`` random perturbations or the
``"extended"`` move set (re-insert stressed squares into the largest holes,
swap-and-rotate pairs, coordinated row/column shifts, 45-degree cluster
rotations); seeds may also be tilted-block members of ``n .. n+3`` with random
squares removed and re-inserted; and improvements can be polished by snapping
angles to multiples of pi/4 or to a common value (:func:`snap_polish`).

Every result is passed through :func:`squarepack.geometry.repair`, which
scales the packing by the *minimal* factor that removes every overlap, so
all returned packings are exactly valid (verified with tolerance 1e-9).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from . import fastcore
from .constructions import Packing, analytic_candidates, best_analytic, grid
from .geometry import (QUARTER_PI, SQRT2, candidate_pairs, canonical_angle,
                       containment_violation, d_support_half, pair_penetration,
                       repair, support_half, verify)


# --------------------------------------------------------------------------- #
# energy and gradient
# --------------------------------------------------------------------------- #
def energy_grad(z: np.ndarray, n: int, s: float, I: np.ndarray, J: np.ndarray,
                with_s: bool = False) -> Tuple[float, np.ndarray]:
    """Penalty energy and its gradient.

    ``z = [x (n), y (n), theta (n)]`` (plus ``s`` as the last entry when
    ``with_s``).  The pair list ``(I, J)`` must contain every pair that can
    overlap during the optimisation.
    """
    x = z[:n]
    y = z[n:2 * n]
    t = z[2 * n:3 * n]
    if with_s:
        s = float(z[3 * n])
    gx = np.zeros(n)
    gy = np.zeros(n)
    gt = np.zeros(n)
    E = 0.0

    if len(I):
        dx = x[I] - x[J]
        dy = y[I] - y[J]
        ci = np.cos(t[I])
        si = np.sin(t[I])
        cj = np.cos(t[J])
        sj = np.sin(t[J])
        A = np.stack([ci * dx + si * dy, -si * dx + ci * dy, cj * dx + sj * dy, -sj * dx + cj * dy])
        absA = np.abs(A)
        k = np.argmax(absA, axis=0)
        cols = np.arange(len(I))
        M = absA[k, cols]
        phi = t[J] - t[I]
        p = 0.5 + support_half(phi) - M
        act = p > 0
        if np.any(act):
            Ia = I[act]
            Ja = J[act]
            pa = p[act]
            ka = k[act]
            ca = cols[act]
            sgn = np.sign(A[ka, ca])
            E += float(np.dot(pa, pa))
            g = 2.0 * pa
            ax = np.choose(ka, [ci[act], -si[act], cj[act], -sj[act]])
            ay = np.choose(ka, [si[act], ci[act], sj[act], cj[act]])
            # dp/dx_i = -sgn*ax, dp/dx_j = +sgn*ax
            gxi = -g * sgn * ax
            gyi = -g * sgn * ay
            gx += np.bincount(Ia, gxi, minlength=n)
            gx -= np.bincount(Ja, gxi, minlength=n)
            gy += np.bincount(Ia, gyi, minlength=n)
            gy -= np.bincount(Ja, gyi, minlength=n)
            # angle derivatives of the max projection
            A1 = A[1, ca]
            A0 = A[0, ca]
            A3 = A[3, ca]
            A2 = A[2, ca]
            dM_dti = sgn * np.where(ka == 0, A1, np.where(ka == 1, -A0, 0.0))
            dM_dtj = sgn * np.where(ka == 2, A3, np.where(ka == 3, -A2, 0.0))
            dR = d_support_half(phi[act])
            gt += np.bincount(Ia, g * (-dR - dM_dti), minlength=n)
            gt += np.bincount(Ja, g * (dR - dM_dtj), minlength=n)

    w = support_half(t)
    v = np.stack([w - x, x + w - s, w - y, y + w - s])
    vp = np.maximum(v, 0.0)
    E += float(np.sum(vp * vp))
    gx += 2.0 * (vp[1] - vp[0])
    gy += 2.0 * (vp[3] - vp[2])
    gt += 2.0 * np.sum(vp, axis=0) * d_support_half(t)
    if with_s:
        gs = -2.0 * float(np.sum(vp[1] + vp[3]))
        return E, np.concatenate([gx, gy, gt, [gs]])
    return E, np.concatenate([gx, gy, gt])


def energy_only(x, y, t, s) -> float:
    I, J = candidate_pairs(x, y)
    n = len(x)
    return energy_grad(np.concatenate([x, y, t]), n, s, I, J)[0]


# --------------------------------------------------------------------------- #
# configuration helpers
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    s: float
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray

    @property
    def n(self) -> int:
        return len(self.x)

    def copy(self) -> "Config":
        return Config(self.s, self.x.copy(), self.y.copy(), self.t.copy())

    def scaled(self, s_new: float) -> "Config":
        f = s_new / self.s
        return Config(s_new, self.x * f, self.y * f, self.t.copy())

    def squares(self) -> np.ndarray:
        return np.column_stack([self.x, self.y, canonical_angle(self.t)])

    @staticmethod
    def from_packing(p: Packing) -> "Config":
        a = np.asarray(p.squares, float)
        return Config(float(p.s), a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy())

    def to_packing(self, method: str, exact: str = "") -> Packing:
        return Packing(self.n, float(self.s), self.squares(), method, exact)

    def max_violation(self) -> float:
        if fastcore.load() is not None:
            return fastcore.max_violation(np.concatenate([self.x, self.y, self.t]), self.n, self.s)
        cont = float(np.max(containment_violation(self.x, self.y, self.t, self.s)))
        I, J = candidate_pairs(self.x, self.y)
        pen = float(np.max(pair_penetration(self.x, self.y, self.t, I, J))) if len(I) else -1.0
        return max(cont, pen)


def repaired(cfg: Config, margin: float = 1e-12) -> Config:
    s2, sq = repair(cfg.s, cfg.squares(), margin)
    return Config(s2, sq[:, 0].copy(), sq[:, 1].copy(), sq[:, 2].copy())


# --------------------------------------------------------------------------- #
# local optimisation (fixed s)
# --------------------------------------------------------------------------- #
PAIR_CUTOFF = 2.6  # generous: squares may move ~0.6 during one L-BFGS run
LOCAL_MAXITER = 250  # L-BFGS iterations per local optimisation (tunable)


def local_opt(cfg: Config, maxiter: Optional[int] = None, gtol: float = 1e-11,
              ftol: float = 1e-18, rounds: int = 3) -> Tuple[Config, float]:
    """Minimise ``E`` at fixed ``s`` starting from ``cfg``; returns ``(config, energy)``.

    The pair list is rebuilt between rounds so that squares which travelled
    far and now overlap something not in the initial list are accounted for.
    """
    n = cfg.n
    s = cfg.s
    if maxiter is None:
        maxiter = LOCAL_MAXITER
    z = np.concatenate([cfg.x, cfg.y, cfg.t])
    if fastcore.load() is not None:
        z, _E, _it = fastcore.lbfgs(z, n, s, maxiter, gtol, ftol, PAIR_CUTOFF)
        I2, J2 = candidate_pairs(z[:n], z[n:2 * n])
        E = fastcore.energy_grad(z, n, s, I2, J2)[0]
        return Config(s, z[:n].copy(), z[n:2 * n].copy(), z[2 * n:].copy()), E
    bounds = [(-0.5, s + 0.5)] * (2 * n) + [(None, None)] * n
    E = float("inf")
    for _ in range(rounds):
        I, J = candidate_pairs(z[:n], z[n:2 * n], cutoff=PAIR_CUTOFF)
        res = minimize(energy_grad, z, args=(n, s, I, J), jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": maxiter, "gtol": gtol, "ftol": ftol, "maxcor": 30})
        z = res.x
        I2, J2 = candidate_pairs(z[:n], z[n:2 * n])
        E = energy_grad(z, n, s, I2, J2)[0]
        if E <= 1e-30 or abs(E - res.fun) < 1e-14:
            break
    out = Config(s, z[:n].copy(), z[n:2 * n].copy(), z[2 * n:].copy())
    return out, E


def is_feasible(cfg: Config, ptol: float) -> bool:
    return cfg.max_violation() <= ptol


# --------------------------------------------------------------------------- #
# perturbations for basin hopping
# --------------------------------------------------------------------------- #
def _random_angle(rng) -> float:
    r = rng.random()
    if r < 0.4:
        return 0.0
    if r < 0.7:
        return QUARTER_PI
    return float(rng.uniform(-QUARTER_PI, QUARTER_PI))


def _pair_energy_per_square(cfg: Config) -> np.ndarray:
    I, J = candidate_pairs(cfg.x, cfg.y)
    e = np.zeros(cfg.n)
    if len(I):
        p = np.maximum(pair_penetration(cfg.x, cfg.y, cfg.t, I, J), 0.0) ** 2
        e += np.bincount(I, p, minlength=cfg.n) + np.bincount(J, p, minlength=cfg.n)
    e += np.maximum(containment_violation(cfg.x, cfg.y, cfg.t, cfg.s), 0.0) ** 2
    return e


def _emptiest_point(cfg: Config, rng, samples: int = 400) -> Tuple[float, float]:
    """A point far (in L-inf) from all other centres, found by random sampling."""
    w = 0.5
    px = rng.uniform(w, cfg.s - w, samples)
    py = rng.uniform(w, cfg.s - w, samples)
    d = np.maximum(np.abs(px[:, None] - cfg.x[None, :]), np.abs(py[:, None] - cfg.y[None, :]))
    clearance = np.min(d, axis=1)
    k = int(np.argmax(clearance))
    return float(px[k]), float(py[k])


PERTURB_MIXES = {
    # the original basin-hopping move set
    "classic": {"jitter": 0.25, "kick": 0.25, "rotate": 0.15, "swap": 0.1, "hole": 0.15, "shake": 0.1},
    # classic moves plus structured ones: re-insert the worst squares into the largest holes,
    # swap-and-rotate a pair, shift a whole row/column band, rotate a cluster by 45 degrees
    "extended": {"jitter": 0.18, "kick": 0.15, "rotate": 0.1, "swap": 0.05, "hole": 0.1, "shake": 0.05,
                 "reinsert": 0.12, "swaprot": 0.08, "bandshift": 0.09, "cluster45": 0.08},
}


def _hole_point(cfg: Config, rng, exclude: Sequence[int] = (), samples: int = 600) -> Tuple[float, float, float]:
    """Centre of the largest empty axis-aligned square (approximately) not covered by the
    squares outside ``exclude``; returns ``(x, y, clearance)``."""
    keep = np.ones(cfg.n, bool)
    keep[list(exclude)] = False
    w = support_half(cfg.t[keep])
    px = rng.uniform(0.5, cfg.s - 0.5, samples)
    py = rng.uniform(0.5, cfg.s - 0.5, samples)
    if keep.any():
        d = np.maximum(np.abs(px[:, None] - cfg.x[keep][None, :]), np.abs(py[:, None] - cfg.y[keep][None, :]))
        clearance = np.min(d - w[None, :], axis=1)
    else:
        clearance = np.full(samples, np.inf)
    clearance = np.minimum(clearance, np.minimum(np.minimum(px, cfg.s - px), np.minimum(py, cfg.s - py)))
    k = int(np.argmax(clearance))
    return float(px[k]), float(py[k]), float(clearance[k])


def perturb(cfg: Config, rng, kind: Optional[str] = None, mix="classic") -> Config:
    """Return a perturbed copy of ``cfg`` (used when the local optimiser is stuck).

    ``mix`` names an entry of :data:`PERTURB_MIXES` or is a ``{kind: weight}`` dict.
    """
    c = cfg.copy()
    n = c.n
    if kind is None:
        weights = PERTURB_MIXES[mix] if isinstance(mix, str) else mix
        kinds = list(weights)
        p = np.array([weights[k] for k in kinds], float)
        kind = kinds[int(rng.choice(len(kinds), p=p / p.sum()))]
    if kind == "jitter":
        sig = rng.choice([0.03, 0.1, 0.25])
        c.x += rng.normal(0, sig, n)
        c.y += rng.normal(0, sig, n)
        c.t += rng.normal(0, sig, n)
    elif kind == "shake":
        c.x += rng.normal(0, 0.5, n)
        c.y += rng.normal(0, 0.5, n)
        c.t += rng.normal(0, 0.3, n)
    elif kind == "kick":
        e = _pair_energy_per_square(c)
        m = int(rng.integers(1, min(4, n) + 1))
        prob = e + 1e-9
        prob /= prob.sum()
        idx = rng.choice(n, size=m, replace=False, p=prob)
        for i in idx:
            c.x[i] = rng.uniform(0.5, c.s - 0.5)
            c.y[i] = rng.uniform(0.5, c.s - 0.5)
            c.t[i] = _random_angle(rng)
    elif kind == "rotate":
        m = int(rng.integers(1, min(4, n) + 1))
        idx = rng.choice(n, size=m, replace=False)
        for i in idx:
            c.t[i] = _random_angle(rng) if rng.random() < 0.7 else c.t[i] + rng.normal(0, 0.4)
    elif kind == "swap":
        if n >= 2:
            i, j = rng.choice(n, size=2, replace=False)
            c.x[i], c.x[j] = c.x[j], c.x[i]
            c.y[i], c.y[j] = c.y[j], c.y[i]
    elif kind == "hole":
        e = _pair_energy_per_square(c)
        i = int(np.argmax(e)) if e.max() > 0 else int(rng.integers(n))
        c.x[i], c.y[i] = _emptiest_point(c, rng)
        c.t[i] = _random_angle(rng)
    elif kind == "reinsert":
        # remove the 1-3 most stressed squares and re-insert each in the largest remaining hole
        e = _pair_energy_per_square(c)
        m = int(rng.integers(1, min(3, n) + 1))
        prob = e + 1e-6 * (1 + e.max())
        prob /= prob.sum()
        idx = list(rng.choice(n, size=m, replace=False, p=prob))
        removed = list(idx)
        for i in idx:
            hx, hy, _clear = _hole_point(c, rng, exclude=removed)
            c.x[i], c.y[i] = hx, hy
            c.t[i] = _random_angle(rng)
            removed.remove(i)
    elif kind == "swaprot":
        if n >= 2:
            i, j = rng.choice(n, size=2, replace=False)
            c.x[i], c.x[j] = c.x[j], c.x[i]
            c.y[i], c.y[j] = c.y[j], c.y[i]
            c.t[i] = _random_angle(rng)
            c.t[j] = _random_angle(rng)
    elif kind == "bandshift":
        # coordinated shift of a horizontal (or vertical) band of squares, or of a half-plane
        axis = rng.random() < 0.5
        pos = c.y if axis else c.x
        y0 = rng.uniform(0.0, c.s)
        if rng.random() < 0.5:
            sel = np.abs(pos - y0) < 0.75
        else:
            sel = pos > y0
        if not sel.any():
            sel = np.zeros(n, bool)
            sel[rng.integers(n)] = True
        d = rng.choice([-1, 1]) * rng.uniform(0.1, 0.6)
        if axis:
            c.x[sel] += d
        else:
            c.y[sel] += d
        if rng.random() < 0.3:
            c.t[sel] = np.where(rng.random(sel.sum()) < 0.5, 0.0, QUARTER_PI)
    elif kind == "cluster45":
        # rotate a cluster of nearby squares by 45 degrees about its centre
        k = int(rng.integers(n))
        r = rng.uniform(1.0, 2.0)
        sel = np.maximum(np.abs(c.x - c.x[k]), np.abs(c.y - c.y[k])) <= r
        cx, cy = c.x[sel].mean(), c.y[sel].mean()
        ang = QUARTER_PI * rng.choice([-1, 1])
        ca, sa = math.cos(ang), math.sin(ang)
        dx, dy = c.x[sel] - cx, c.y[sel] - cy
        c.x[sel] = cx + ca * dx - sa * dy
        c.y[sel] = cy + sa * dx + ca * dy
        c.t[sel] += ang
    c.x = np.clip(c.x, 0.3, c.s - 0.3)
    c.y = np.clip(c.y, 0.3, c.s - 0.3)
    return c


# --------------------------------------------------------------------------- #
# seeds
# --------------------------------------------------------------------------- #
def seed_random(n: int, s: float, rng) -> Config:
    x = rng.uniform(0.5, s - 0.5, n)
    y = rng.uniform(0.5, s - 0.5, n)
    t = np.array([_random_angle(rng) for _ in range(n)])
    return Config(s, x, y, t)


def seed_grid_jitter(n: int, s: float, rng) -> Config:
    k = math.ceil(s - 1e-9)
    cells = rng.choice(k * k, size=min(n, k * k), replace=False)
    x = (cells % k + 0.5) * (s / k)
    y = (cells // k + 0.5) * (s / k)
    if n > k * k:
        x = np.concatenate([x, rng.uniform(0.5, s - 0.5, n - k * k)])
        y = np.concatenate([y, rng.uniform(0.5, s - 0.5, n - k * k)])
    x = x + rng.normal(0, 0.05, n)
    y = y + rng.normal(0, 0.05, n)
    t = np.where(rng.random(n) < 0.15, [_random_angle(rng) for _ in range(n)], 0.0)
    return Config(s, x, y, t)


def seed_analytic(n: int, rng, which: Optional[int] = None) -> Config:
    """An analytic packing for n (surplus squares dropped at random), possibly of a
    *larger* family member so the optimiser starts from a structured but loose state."""
    cands = analytic_candidates(n)
    if which is None:
        which = int(rng.integers(len(cands)))
    p = cands[which % len(cands)]
    return Config.from_packing(p)


def seed_analytic_random_drop(n: int, rng) -> Config:
    """Take a family member with capacity > n and drop random squares (not the last ones)."""
    from .constructions import gobel_square_members, gobel_strip_capacity, gobel_strip_member, add_L
    pool: List[Packing] = []
    a = 1
    while gobel_strip_capacity(a) < n:
        a += 1
    for aa in (a, a + 1):
        pool.append(gobel_strip_member(aa))
    for p in gobel_square_members(n)[:2]:
        pool.append(p)
    g = grid(n + int(rng.integers(0, 3)))
    pool.append(g)
    p = pool[int(rng.integers(len(pool)))]
    if p.n > n:
        keep = np.sort(rng.choice(p.n, size=n, replace=False))
        sq = p.squares[keep]
    else:
        sq = p.squares
        if p.n < n:  # add the missing squares at random spots
            extra = np.column_stack([rng.uniform(0.5, p.s - 0.5, n - p.n),
                                     rng.uniform(0.5, p.s - 0.5, n - p.n),
                                     [_random_angle(rng) for _ in range(n - p.n)]])
            sq = np.vstack([sq, extra])
    return Config(float(p.s), sq[:, 0].copy(), sq[:, 1].copy(), sq[:, 2].copy())


_BLOCK_POOLS: dict = {}


def _block_shape(pk: Packing):
    """(p, q) of a tilted-block packing, for both the legacy ``meta["p"]`` and the
    ``meta["params"] = [p, q, dx, dy, angle, shear]`` layouts of :mod:`squarepack.blocks`."""
    m = pk.meta or {}
    if "p" in m and "q" in m:
        return int(m["p"]), int(m["q"])
    params = m.get("params")
    if not params and m.get("spec"):
        params = m["spec"][0]
    try:
        return int(round(params[0])), int(round(params[1]))
    except (TypeError, IndexError, ValueError):
        return None


def block_seed_pool(n: int, extra: int = 2, per_n: int = 2) -> List[Packing]:
    """Structured seeds: the best tilted-block members (several block shapes) for ``n`` and for
    the next few larger ``n`` (their surplus squares are dropped at random by :func:`seed_block`).
    The pool is cached per ``(n, extra, per_n)``."""
    from .blocks import tilted_block_search
    key = (n, extra, per_n)
    if key in _BLOCK_POOLS:
        return _BLOCK_POOLS[key]
    pool: List[Packing] = []
    for m in range(n, n + extra + 1):
        s_max = grid(m).s + 0.3
        seen = set()
        for _ in range(per_n):
            shapes = None
            if seen:
                pmax = int(s_max * SQRT2) + 1
                shapes = [(p_, q_) for p_ in range(1, pmax + 1) for q_ in range(p_, pmax + 1)
                          if p_ * q_ <= m and (p_ + q_) / SQRT2 <= s_max + 1e-9 and (p_, q_) not in seen]
                if not shapes:
                    break
            pk = tilted_block_search(m, s_max=s_max, shapes=shapes, offsets=(-0.5, 0.0, 0.5))
            if pk is None:
                break
            shape = _block_shape(pk)
            if shape is None:
                pool.append(pk)
                break
            seen.add(shape)
            pool.append(pk)
        for pk in analytic_candidates(m)[:2]:
            if pk.method != "grid":
                pool.append(pk)
    pool.sort(key=lambda q: q.s)
    _BLOCK_POOLS[key] = pool
    return pool


def seed_block(n: int, rng, pool: Sequence[Packing], kick: int = 3) -> Config:
    """A tilted-block / analytic member (surplus squares dropped at random), with a few random
    squares removed and re-inserted into the largest holes, at a slightly enlarged side."""
    p = pool[int(rng.integers(len(pool)))]
    if p.n > n:
        keep = np.sort(rng.choice(p.n, size=n, replace=False))
        sq = p.squares[keep]
    else:
        sq = p.squares
    c = Config(float(p.s) * (1.0 + 0.01 * rng.random()), sq[:, 0].copy(), sq[:, 1].copy(), sq[:, 2].copy())
    m = int(rng.integers(0, min(kick, n) + 1))
    if m:
        idx = list(rng.choice(n, size=m, replace=False))
        removed = list(idx)
        for i in idx:
            hx, hy, _ = _hole_point(c, rng, exclude=removed)
            c.x[i], c.y[i], c.t[i] = hx, hy, _random_angle(rng)
            removed.remove(i)
    return c


# --------------------------------------------------------------------------- #
# compaction by bisection on s
# --------------------------------------------------------------------------- #
@dataclass
class SearchStats:
    local_opts: int = 0
    improvements: int = 0
    seeds: int = 0
    log: List[Tuple[float, float, str]] = field(default_factory=list)


def make_feasible(cfg: Config, rng, ptol: float, hops: int = 3, grow: float = 1.03,
                  max_grow: int = 12, stats: Optional[SearchStats] = None, mix="classic") -> Optional[Config]:
    """Locally optimise; on failure perturb, then enlarge ``s`` until a valid packing appears."""
    cur = cfg
    for _ in range(max_grow):
        res, E = local_opt(cur)
        if stats:
            stats.local_opts += 1
        if is_feasible(res, ptol):
            return repaired(res)
        for _h in range(hops):
            res2, E2 = local_opt(perturb(res, rng, mix=mix))
            if stats:
                stats.local_opts += 1
            if is_feasible(res2, ptol):
                return repaired(res2)
            if E2 < E:  # keep the better of the two as the base
                res, E = res2, E2
        cur = res.scaled(res.s * grow)
    return None


def compact(cfg: Config, rng, deadline: float, ptol: float = 1e-8, step0: float = 0.01,
            min_step: float = 1e-7, hops: int = 4, hop_min_step: float = 5e-4,
            stats: Optional[SearchStats] = None,
            on_improve: Optional[Callable[[Config], None]] = None, mix="classic",
            anneal_opts: Optional[dict] = None, max_fails: Optional[int] = None) -> Config:
    """Shrink a valid packing as far as the local optimiser (plus basin hopping) allows.

    ``cfg`` must be valid.  Returns the best valid configuration found.  ``mix`` selects
    the perturbation move set (see :data:`PERTURB_MIXES`).  With ``anneal_opts`` each
    basin-hopping move is a short simulated-annealing run at the trial side (see
    :func:`anneal`) instead of a random perturbation; ``max_fails`` stops the compaction
    after that many consecutive failed shrink attempts (default: run until ``min_step``).
    """
    best = repaired(cfg)
    step = step0 * best.s
    fails = 0
    while step > min_step * best.s and time.time() < deadline:
        s_try = best.s - step
        trial = best.scaled(s_try)
        res, E = local_opt(trial)
        if stats:
            stats.local_opts += 1
        ok = is_feasible(res, ptol)
        h = 0
        base = res
        # basin hopping only while the step is coarse; the fine bisection is pure local descent
        hops_now = hops if step > hop_min_step * best.s else 0
        while not ok and h < hops_now and time.time() < deadline:
            if anneal_opts is not None and fastcore.load() is not None:
                feas, fin, _Ef = anneal(base, rng, **anneal_opts)
                cand, E2 = local_opt(fin)
                if feas is not None and (E2 > 0 or not is_feasible(cand, ptol)):
                    cand, E2 = local_opt(feas)
            else:
                cand, E2 = local_opt(perturb(base, rng, mix=mix))
            if stats:
                stats.local_opts += 1
            h += 1
            if is_feasible(cand, ptol):
                res, ok = cand, True
            elif E2 < E:
                base, E = cand, E2
        if ok:
            best = repaired(res)
            if stats:
                stats.improvements += 1
            if on_improve:
                on_improve(best)
            fails = 0
            step = min(step * 1.5, 0.02 * best.s)
        else:
            fails += 1
            if max_fails is not None and fails >= max_fails:
                break
            step *= 0.5
    return best


# --------------------------------------------------------------------------- #
# simulated annealing (C core)
# --------------------------------------------------------------------------- #
ANNEAL_DEFAULTS = dict(sweeps=3000, T0=3e-3, T1=1e-7, step_xy=0.15, step_t=0.3, etol=1e-10,
                       cold_sweeps=20000, cold_T0=3e-2, cold_shrink=3e-3)
_ANNEAL_KEYS = ("sweeps", "T0", "T1", "step_xy", "step_t", "etol")


def _anneal_kw(opts: Optional[dict]) -> dict:
    d = dict(ANNEAL_DEFAULTS)
    if opts:
        d.update(opts)
    return d


def anneal(cfg: Config, rng, sweeps: int = 3000, T0: float = 3e-3, T1: float = 1e-7, step_xy: float = 0.15,
           step_t: float = 0.3, shrink: float = 0.0, etol: float = 1e-16,
           s: Optional[float] = None) -> Tuple[Optional[Config], Config, float]:
    """Simulated annealing of the penalty energy at fixed side ``s`` (default ``cfg.s``).

    Metropolis moves on single squares (displace, rotate, snap to 0 / 45 degrees, swap,
    teleport) with the temperature decaying geometrically from ``T0`` to ``T1`` over
    ``sweeps`` sweeps of ``n`` moves.  With ``shrink > 0`` the container is shrunk by that
    relative amount every time the energy vanishes, so the run compacts on its own.
    Returns ``(best_feasible_or_None, final_state, final_energy)``; the feasible state
    (``E < etol``) is the tightest one met.  Requires the C core (returns
    ``(None, cfg, inf)`` without it).
    """
    if fastcore.load() is None:
        return None, cfg, float("inf")
    s0 = cfg.s if s is None else float(s)
    sc = cfg.scaled(s0)
    z = np.concatenate([sc.x, sc.y, sc.t])
    seed = int(rng.integers(1, 2 ** 62))
    _hits, zf, sf, Ef, zb, sb, _Eb = fastcore.anneal(z, cfg.n, s0, sweeps, T0, T1, step_xy, step_t, shrink, etol, seed)
    n = cfg.n
    final = Config(sf, zf[:n].copy(), zf[n:2 * n].copy(), zf[2 * n:].copy())
    feas = None
    if sb > 0:
        feas = Config(sb, zb[:n].copy(), zb[n:2 * n].copy(), zb[2 * n:].copy())
    return feas, final, Ef


def anneal_feasible(cfg: Config, rng, ptol: float, stats: Optional[SearchStats] = None,
                    anneal_opts: Optional[dict] = None, mix="classic") -> Optional[Config]:
    """Cold-start annealing: cool a (typically random) seed at its side while shrinking the
    container every time the energy vanishes, then polish the tightest feasible state with
    L-BFGS.  Falls back to :func:`make_feasible` when nothing feasible is met."""
    d = _anneal_kw(anneal_opts)
    feas, fin, _Ef = anneal(cfg, rng, sweeps=d["cold_sweeps"], T0=d["cold_T0"], T1=d["T1"], step_xy=d["step_xy"],
                            step_t=d["step_t"], shrink=d["cold_shrink"], etol=d["etol"])
    for c in (feas, fin):
        if c is None:
            continue
        res, _E = local_opt(c)
        if stats:
            stats.local_opts += 1
        if is_feasible(res, ptol):
            return repaired(res)
        rep = c.scaled(c.s * (1 + 2 * math.sqrt(d["etol"])))
        res, _E = local_opt(rep)
        if stats:
            stats.local_opts += 1
        if is_feasible(res, ptol):
            return repaired(res)
    return make_feasible(fin, rng, ptol, stats=stats, mix=mix)


def anneal_compact(cfg: Config, rng, deadline: float, ptol: float = 1e-8, shrink0: float = 4e-3,
                   min_shrink: float = 1e-5, reheats: int = 3, stats: Optional[SearchStats] = None,
                   on_improve: Optional[Callable[[Config], None]] = None,
                   anneal_opts: Optional[dict] = None, mix="classic") -> Config:
    """Compaction driven by simulated annealing: shrink the side, anneal the (now overlapping)
    packing at the smaller side, polish with L-BFGS; on failure retry with a smaller shrink."""
    d = _anneal_kw(anneal_opts)
    opts = {k: d[k] for k in _ANNEAL_KEYS}
    best = repaired(cfg)
    shrink = shrink0
    while shrink >= min_shrink and time.time() < deadline:
        s_try = best.s * (1.0 - shrink)
        ok = False
        base = best.scaled(s_try)
        cand = base
        for r in range(reheats):
            if time.time() >= deadline:
                break
            feas, fin, _Ef = anneal(base, rng, **opts)
            cand, E = local_opt(fin)
            if stats:
                stats.local_opts += 1
            if is_feasible(cand, ptol):
                ok = True
                break
            if feas is not None:
                cand2, E2 = local_opt(feas)
                if stats:
                    stats.local_opts += 1
                if is_feasible(cand2, ptol):
                    cand, ok = cand2, True
                    break
            base = cand if E < 1e-3 else perturb(cand, rng, mix=mix)
        if ok:
            best = repaired(cand)
            if stats:
                stats.improvements += 1
            if on_improve:
                on_improve(best)
            shrink = min(shrink * 1.5, shrink0)
        else:
            shrink *= 0.5
    return best


# --------------------------------------------------------------------------- #
# angle snapping polish
# --------------------------------------------------------------------------- #
def snap_angles(t: np.ndarray, tol: float = 1e-3, cluster_tol: float = 0.02) -> Tuple[np.ndarray, bool]:
    """Snap angles within ``tol`` of a multiple of pi/4 to it exactly, and equalise groups of
    angles that agree within ``cluster_tol`` (record packings typically have a few squares
    sharing one exact angle).  Returns ``(angles, changed)``."""
    t = canonical_angle(t)
    out = t.copy()
    fixed = np.zeros(len(t), bool)
    for target in (0.0, QUARTER_PI, -QUARTER_PI):
        m = np.abs(t - target) < tol
        out[m] = target
        fixed |= m
    if cluster_tol > 0:
        free = np.nonzero(~fixed)[0]
        order = free[np.argsort(t[free])]
        group = []
        for i in list(order) + [None]:
            if i is not None and (not group or abs(t[i] - t[group[-1]]) < cluster_tol):
                group.append(i)
                continue
            if len(group) > 1:
                out[group] = float(np.mean(t[group]))
            group = [i] if i is not None else []
    return out, bool(np.any(out != t))


def snap_polish(cfg: Config, rng, deadline: float, ptol: float = 1e-8, tol: float = 1e-3,
                cluster_tol: float = 0.02, rounds: int = 3,
                stats: Optional[SearchStats] = None) -> Optional[Config]:
    """Angle-snapping polish: snap angles (see :func:`snap_angles`), re-optimise at the same
    (or a marginally larger) side, then re-compact by pure local descent; repeated while it
    keeps improving.  Returns the result when it is tighter than ``cfg``, else ``None``."""
    best = None
    cur = cfg
    for _ in range(rounds):
        if time.time() >= deadline:
            break
        snapped, changed = snap_angles(cur.t, tol, cluster_tol)
        if not changed:
            break
        c = Config(cur.s, cur.x.copy(), cur.y.copy(), snapped)
        res, _E = local_opt(c, maxiter=max(LOCAL_MAXITER, 500))
        if stats:
            stats.local_opts += 1
        if not is_feasible(res, ptol):
            res, _E = local_opt(c.scaled(cur.s * (1 + 2e-4)), maxiter=max(LOCAL_MAXITER, 500))
            if stats:
                stats.local_opts += 1
            if not is_feasible(res, ptol):
                break
        out = compact(repaired(res), rng, deadline, ptol=ptol, step0=1e-3, hops=0, stats=stats)
        if out.s >= cur.s - 1e-12:
            break
        best = cur = out
    return best


# --------------------------------------------------------------------------- #
# penalty continuation (s as a variable)
# --------------------------------------------------------------------------- #
def penalty_descent(cfg: Config, mus: Sequence[float] = (1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6),
                    maxiter: int = 300) -> Config:
    """Minimise ``s + mu * E(z; s)`` for increasing ``mu`` (quadratic penalty method)."""
    n = cfg.n
    z = np.concatenate([cfg.x, cfg.y, cfg.t, [cfg.s]])
    bounds = [(-0.5, None)] * (2 * n) + [(None, None)] * n + [(math.sqrt(n), None)]

    def f(zz, I, J, mu):
        E, g = energy_grad(zz, n, 0.0, I, J, with_s=True)
        g = mu * g
        g[-1] += 1.0
        return float(zz[-1]) + mu * E, g

    for mu in mus:
        I, J = candidate_pairs(z[:n], z[n:2 * n], cutoff=PAIR_CUTOFF)
        res = minimize(f, z, args=(I, J, mu), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "gtol": 1e-10, "ftol": 1e-16, "maxcor": 30})
        z = res.x
    out = Config(float(z[-1]), z[:n].copy(), z[n:2 * n].copy(), z[2 * n:3 * n].copy())
    return out


# --------------------------------------------------------------------------- #
# global driver
# --------------------------------------------------------------------------- #
# defaults chosen by the controlled benchmark in results/search_tuning.md
DEFAULT_SEED_MIX: Tuple[str, ...] = ("analytic", "drop", "grid", "random", "block")
DEFAULT_PERTURB_MIX = "classic"


@dataclass
class SearchResult:
    packing: Packing
    stats: SearchStats
    history: List[Tuple[float, float]]  # (time, s)


def search(n: int, time_budget: float = 30.0, seed: Optional[int] = None,
           start: Optional[Packing] = None, strategy: str = "bisect",
           seed_mix: Sequence[str] = DEFAULT_SEED_MIX,
           ptol: float = 1e-8, verbose: bool = False,
           on_improve: Optional[Callable[[Packing], None]] = None,
           perturb_mix=DEFAULT_PERTURB_MIX, snap: bool = False, anneal_opts: Optional[dict] = None,
           local_maxiter: Optional[int] = None, hops: int = 4,
           incumbent_fraction: float = 0.25) -> SearchResult:
    """Numerical search for a packing of ``n`` unit squares tighter than the analytic one.

    The analytic portfolio (or ``start``) provides the incumbent; random,
    grid and structured seeds are then compacted in turn until the time budget
    is exhausted.  The best *exactly valid* packing is returned.

    Options (defaults reproduce the original behaviour):

    ``strategy``
        ``"bisect"`` (L-BFGS compaction with basin hopping), ``"penalty"`` (penalty
        continuation first), ``"anneal"`` (simulated-annealing compaction, see
        :func:`anneal_compact`, followed by the bisection polish) or ``"anneal-hop"``
        (bisection whose basin-hopping moves are short annealing runs).
    ``seed_mix``
        Seed kinds cycled through: ``"analytic"``, ``"drop"``, ``"grid"``, ``"random"``,
        ``"block"`` (tilted-block members of ``n`` .. ``n+3`` with random squares removed and
        re-inserted into holes).
    ``perturb_mix``
        Name of a move set in :data:`PERTURB_MIXES` (``"classic"`` or ``"extended"``) or a
        ``{kind: weight}`` dict.
    ``snap``
        Apply the angle-snapping polish (:func:`snap_polish`) to every improvement.
    ``anneal_opts``
        Overrides for :data:`ANNEAL_DEFAULTS` (``sweeps``, ``T0``, ``T1``, ``step_xy``, ``step_t``).
    ``local_maxiter``
        L-BFGS iterations per local optimisation for this search (default :data:`LOCAL_MAXITER`).
    ``hops``
        Basin-hopping moves per failed shrink step in the bisection.
    ``incumbent_fraction``
        Share of the budget spent compacting the incumbent before fresh seeds are tried.
    """
    global LOCAL_MAXITER
    rng = np.random.default_rng(seed)
    t0 = time.time()
    deadline = t0 + time_budget
    stats = SearchStats()
    incumbent = start if start is not None else best_analytic(n)
    best = Config.from_packing(incumbent)
    best_method = incumbent.method
    history = [(0.0, best.s)]
    saved_maxiter = LOCAL_MAXITER
    if local_maxiter is not None:
        LOCAL_MAXITER = int(local_maxiter)
    block_pool: Optional[List[Packing]] = None

    def improve(c: Config, label: str):
        nonlocal best, best_method
        if c.s < best.s - 1e-12 and verify(c.s, c.squares(), 1e-9).ok:
            best = c.copy()
            best_method = label
            history.append((time.time() - t0, c.s))
            if verbose:
                print(f"  [{time.time() - t0:7.1f}s] n={n} s={c.s:.10f} via {label}")
            if on_improve:
                on_improve(best.to_packing(label))
            if snap and time.time() < deadline:
                pol = snap_polish(best, rng, deadline, ptol=ptol, stats=stats)
                if pol is not None:
                    improve(pol, label + "+snap")

    if n <= 1:
        LOCAL_MAXITER = saved_maxiter
        return SearchResult(best.to_packing(best_method), stats, history)

    hop_anneal = None
    if strategy == "anneal-hop":
        d = _anneal_kw(anneal_opts)
        hop_anneal = {k: d[k] for k in _ANNEAL_KEYS}
        hop_anneal["sweeps"] = max(200, int(hop_anneal["sweeps"] // 5))

    def compact_from(feas: Config, label: str, until: float) -> Config:
        if strategy == "anneal":
            res = anneal_compact(feas, rng, until, ptol=ptol, stats=stats, anneal_opts=anneal_opts,
                                 mix=perturb_mix, on_improve=lambda c: improve(c, label))
            improve(res, label)
            feas = res
        return compact(feas, rng, until, ptol=ptol, stats=stats, hops=hops, mix=perturb_mix,
                       anneal_opts=hop_anneal, on_improve=lambda c: improve(c, label))

    try:
        # first: compact the incumbent itself (it may not be locally optimal); cap at a
        # fraction of the budget so that fresh seeds always get their turn
        cfg = compact_from(best, (incumbent.method if incumbent.method.startswith("numeric") else "numeric:" + incumbent.method), min(deadline, t0 + incumbent_fraction * time_budget))
        improve(cfg, (incumbent.method if incumbent.method.startswith("numeric") else "numeric:" + incumbent.method))

        k = 0
        while time.time() < deadline:
            kind = seed_mix[k % len(seed_mix)]
            k += 1
            stats.seeds += 1
            s0 = best.s * 1.02
            if kind == "analytic":
                cfg = seed_analytic(n, rng)
                if cfg.s > best.s * 1.15:
                    cfg = seed_random(n, s0, rng)
            elif kind == "drop":
                cfg = seed_analytic_random_drop(n, rng)
            elif kind == "grid":
                cfg = seed_grid_jitter(n, s0, rng)
            elif kind == "block":
                if block_pool is None:
                    # building the pool costs a few seconds: skip it for tiny budgets
                    block_pool = block_seed_pool(n) if deadline - time.time() > 12.0 else []
                cfg = seed_block(n, rng, block_pool) if block_pool else seed_random(n, s0, rng)
            else:
                cfg = seed_random(n, s0, rng)
            if strategy == "penalty":
                cfg = penalty_descent(cfg)
            if strategy == "anneal" and fastcore.load() is not None:
                feas = anneal_feasible(cfg, rng, ptol, stats=stats, anneal_opts=anneal_opts, mix=perturb_mix)
            else:
                feas = make_feasible(cfg, rng, ptol, stats=stats, mix=perturb_mix)
            if feas is None:
                continue
            label = f"numeric:{strategy}:{kind}"
            improve(feas, label)
            res = compact_from(feas, label, deadline)
            improve(res, label)
    finally:
        LOCAL_MAXITER = saved_maxiter
    return SearchResult(best.to_packing(best_method), stats, history)
