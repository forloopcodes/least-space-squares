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


def perturb(cfg: Config, rng, kind: Optional[str] = None) -> Config:
    """Return a perturbed copy of ``cfg`` (used when the local optimiser is stuck)."""
    c = cfg.copy()
    n = c.n
    if kind is None:
        kind = rng.choice(["jitter", "kick", "rotate", "swap", "hole", "shake"],
                          p=[0.25, 0.25, 0.15, 0.1, 0.15, 0.1])
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
                  max_grow: int = 12, stats: Optional[SearchStats] = None) -> Optional[Config]:
    """Locally optimise; on failure perturb, then enlarge ``s`` until a valid packing appears."""
    cur = cfg
    for _ in range(max_grow):
        res, E = local_opt(cur)
        if stats:
            stats.local_opts += 1
        if is_feasible(res, ptol):
            return repaired(res)
        for _h in range(hops):
            res2, E2 = local_opt(perturb(res, rng))
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
            on_improve: Optional[Callable[[Config], None]] = None) -> Config:
    """Shrink a valid packing as far as the local optimiser (plus basin hopping) allows.

    ``cfg`` must be valid.  Returns the best valid configuration found.
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
            cand, E2 = local_opt(perturb(base, rng))
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
            step *= 0.5
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
@dataclass
class SearchResult:
    packing: Packing
    stats: SearchStats
    history: List[Tuple[float, float]]  # (time, s)


def search(n: int, time_budget: float = 30.0, seed: Optional[int] = None,
           start: Optional[Packing] = None, strategy: str = "bisect",
           seed_mix: Sequence[str] = ("analytic", "drop", "grid", "random"),
           ptol: float = 1e-8, verbose: bool = False,
           on_improve: Optional[Callable[[Packing], None]] = None) -> SearchResult:
    """Numerical search for a packing of ``n`` unit squares tighter than the analytic one.

    The analytic portfolio (or ``start``) provides the incumbent; random,
    grid and structured seeds are then compacted in turn until the time budget
    is exhausted.  The best *exactly valid* packing is returned.
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    deadline = t0 + time_budget
    stats = SearchStats()
    incumbent = start if start is not None else best_analytic(n)
    best = Config.from_packing(incumbent)
    best_method = incumbent.method
    history = [(0.0, best.s)]

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

    if n <= 1:
        return SearchResult(best.to_packing(best_method), stats, history)

    # first: compact the incumbent itself (it may not be locally optimal); cap at a
    # quarter of the budget so that fresh seeds always get their turn
    cfg = compact(best, rng, min(deadline, t0 + 0.25 * time_budget), ptol=ptol, stats=stats,
                  on_improve=lambda c: improve(c, "numeric:" + incumbent.method))
    improve(cfg, "numeric:" + incumbent.method)

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
        else:
            cfg = seed_random(n, s0, rng)
        if strategy == "penalty":
            cfg = penalty_descent(cfg)
            feas = make_feasible(cfg, rng, ptol, stats=stats)
        else:
            feas = make_feasible(cfg, rng, ptol, stats=stats)
        if feas is None:
            continue
        label = f"numeric:{strategy}:{kind}"
        improve(feas, label)
        res = compact(feas, rng, deadline, ptol=ptol, stats=stats,
                      on_improve=lambda c, label=label: improve(c, label))
        improve(res, label)
    return SearchResult(best.to_packing(best_method), stats, history)
