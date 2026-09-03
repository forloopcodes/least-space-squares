"""Recognise closed forms of a side length: s ≈ (a + b√2 + c√7 + d√3 + e√5) / q for small integers.

Used only for reporting: when a numerically found ``s`` agrees with a short closed form to
``tol``, the form is printed next to the number so that a structure can be recognised
(e.g. 4.822875655532... = 7/2 + √7/2).
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

_ROOTS = ((2, math.sqrt(2)), (7, math.sqrt(7)), (3, math.sqrt(3)), (5, math.sqrt(5)))


def exact_form(s: float, tol: float = 1e-7, max_int: int = 40, max_coef: int = 12) -> Optional[str]:
    """Shortest closed form ``(a + b*sqrt(r)) / q`` (one surd) matching ``s`` within ``tol``, else None."""
    best: Optional[str] = None
    best_cost = 10 ** 9
    for q in (1, 2, 3, 4, 6, 7, 8, 41):
        target = s * q
        for r, root in _ROOTS:
            for b in range(-max_coef, max_coef + 1):
                a = round(target - b * root)
                if abs(a) > max_int * q:
                    continue
                val = (a + b * root) / q
                if abs(val - s) <= tol:
                    cost = abs(a) + 3 * abs(b) + 2 * (q - 1) + (0 if b == 0 else 1)
                    if cost < best_cost:
                        best_cost = cost
                        best = _fmt(a, b, r, q)
    return best


def _fmt(a: int, b: int, r: int, q: int) -> str:
    terms = []
    if a:
        terms.append(str(a))
    if b:
        sign = "-" if b < 0 else "+"
        mag = abs(b)
        core = f"sqrt({r})" if mag == 1 else f"{mag}*sqrt({r})"
        terms.append(f"{sign} {core}" if terms else (f"-{core}" if b < 0 else core))
    body = " ".join(terms) if terms else "0"
    if q == 1:
        return body
    return f"({body})/{q}" if len(terms) > 1 else f"{body}/{q}"
