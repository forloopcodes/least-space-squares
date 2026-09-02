"""Top-level solver: ``pack(n)`` returns ``(s, [[x, y, angle], ...])``.

Pipeline (each stage only runs when the previous one has not settled the answer)
---------------------------------------------------------------------------------
1. **cache** - packings found by earlier searches (``data/best_packings.json``),
   re-verified on load.  O(n).
2. **closed-form families** (:mod:`squarepack.constructions`) - grid, Göbel
   strips, Göbel squares and their "L" extensions.  O(n).
3. **tilted-block search** (:mod:`squarepack.blocks`) - a p x q block rotated
   45 degrees with a greedy row fill; the exact ``s`` is found by bisection.
   O(s^2 * offsets * log(1/eps)); ~1-5 s for n <= 300 in numpy.
4. **numerical search** (:mod:`squarepack.optimize`) - penalty-energy
   compaction with basin hopping, only when a time budget is given.

Every candidate is validated with :func:`squarepack.geometry.verify`; the
smallest verified side wins.  A lower bound ``sqrt(n)`` (area) and, when
available, the best known literature value are reported alongside.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .blocks import tilted_block_search
from .constructions import Packing, analytic_candidates, best_analytic
from .geometry import verify
from .exact import exact_form
from .known import best_known, is_proved

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "best_packings.json"
_cache_lock = threading.Lock()
_cache: Optional[Dict[int, Packing]] = None


def _load_cache(path: Path = CACHE_FILE) -> Dict[int, Packing]:
    global _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        out: Dict[int, Packing] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            for key, ent in raw.items():
                n = int(key)
                sq = np.asarray(ent["squares"], float).reshape(-1, 3)
                if len(sq) == n and verify(ent["s"], sq, 1e-9).ok:
                    out[n] = Packing(n, float(ent["s"]), sq, ent.get("method", "cache"), ent.get("exact", ""))
        _cache = out
        return out


def cached(n: int) -> Optional[Packing]:
    return _load_cache().get(n)


def update_cache(p: Packing, path: Path = CACHE_FILE, only_if_better: bool = True) -> bool:
    """Store ``p`` in the on-disk cache (verified first).  Returns True if written."""
    rep = verify(p.s, p.squares, 1e-9)
    if not rep.ok:
        raise ValueError(f"refusing to cache an invalid packing: {rep}")
    with _cache_lock:
        raw: Dict[str, dict] = {}
        if path.exists():
            raw = json.loads(path.read_text())
        old = raw.get(str(p.n))
        if only_if_better and old is not None and old["s"] <= p.s + 1e-12:
            return False
        raw[str(p.n)] = {"s": float(p.s), "method": p.method, "exact": p.exact,
                         "squares": np.asarray(p.squares, float).round(15).tolist()}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(sorted(raw.items(), key=lambda kv: int(kv[0]))), indent=0))
        os.replace(tmp, path)
        global _cache
        if _cache is not None:
            _cache[p.n] = p
    return True


@dataclass
class Solution:
    n: int
    s: float
    squares: np.ndarray
    method: str
    exact: str
    lower_bound: float
    best_known: Optional[float]
    proved_optimal: bool

    @property
    def gap_to_best_known(self) -> Optional[float]:
        return None if self.best_known is None else self.s - self.best_known

    def as_list(self, degrees: bool = False) -> List[List[float]]:
        out = np.array(self.squares, float)
        if degrees:
            out[:, 2] = np.degrees(out[:, 2])
        return out.tolist()

    def to_dict(self, degrees: bool = False) -> dict:
        return {"n": self.n, "s": self.s, "method": self.method, "exact": self.exact,
                "lower_bound": self.lower_bound, "best_known": self.best_known,
                "proved_optimal": self.proved_optimal,
                "angle_unit": "degrees" if degrees else "radians",
                "squares": self.as_list(degrees)}


def solve(n: int, time_budget: float = 0.0, use_cache: bool = True, use_blocks: bool = True,
          seed: Optional[int] = None, verbose: bool = False, save: bool = False) -> Solution:
    """Find the tightest packing of ``n`` unit squares that the portfolio can produce."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    cands: List[Packing] = []
    best = best_analytic(n)
    cands.append(best)
    if use_cache:
        c = cached(n)
        if c is not None:
            cands.append(c)
    best = min(cands, key=lambda p: p.s)
    lower = math.sqrt(n)
    k = math.isqrt(n)
    proved = is_proved(n)
    if not proved and best.s > lower + 1e-12:
        if use_blocks and not (best.method == "grid" and (k + 1) ** 2 - n in (1, 2)):
            blk = tilted_block_search(n, s_max=best.s)
            if blk is not None and blk.s < best.s - 1e-9:
                best = blk
        if time_budget > 0:
            from .optimize import search
            res = search(n, time_budget=time_budget, seed=seed, start=best, verbose=verbose)
            if res.packing.s < best.s - 1e-12:
                best = res.packing
    rep = verify(best.s, best.squares, 1e-9)
    if not rep.ok:  # pragma: no cover - every producer verifies already
        raise RuntimeError(f"internal error: produced an invalid packing: {rep}")
    if save and best.method not in ("grid",):
        update_cache(best)
    exact = best.exact
    form = exact_form(best.s)
    if form and form not in exact:
        exact = f"{exact} = {form}" if exact else form
    return Solution(n, float(best.s), np.asarray(best.squares, float), best.method, exact,
                    lower, best_known(n), proved or abs(best.s - lower) < 1e-12)


def pack(n: int, time_budget: float = 0.0, degrees: bool = False, **kw) -> Tuple[float, List[List[float]]]:
    """``pack(n) -> (s, [[x_center, y_center, angle], ...])``.

    ``s`` is the side of the enclosing square; coordinates are in ``[0, s]`` with
    the origin at the bottom-left corner; ``angle`` is in radians (or degrees
    when ``degrees=True``), counter-clockwise, reduced to ``[-pi/4, pi/4)``.
    """
    sol = solve(n, time_budget=time_budget, **kw)
    return sol.s, sol.as_list(degrees)
