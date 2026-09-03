"""SVG rendering of packings (for visual inspection)."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def to_svg(s: float, squares: Sequence[Sequence[float]], size: int = 480, title: str = "") -> str:
    arr = np.asarray(squares, float).reshape(-1, 3)
    scale = size / s
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {s:.12g} {s:.12g}">']
    if title:
        out.append(f"<title>{title}</title>")
    out.append(f'<rect x="0" y="0" width="{s:.12g}" height="{s:.12g}" fill="#fff" stroke="#000" stroke-width="{2 / scale:.6g}"/>')
    # flip y so the origin is the bottom-left corner
    out.append(f'<g transform="translate(0 {s:.12g}) scale(1 -1)">')
    for x, y, t in arr:
        deg = math.degrees(t)
        fill = "#b3cde3" if abs(math.sin(2 * t)) < 1e-9 else "#fbb4ae"
        out.append(f'<rect x="-0.5" y="-0.5" width="1" height="1" fill="{fill}" stroke="#000" '
                   f'stroke-width="{1.2 / scale:.6g}" transform="translate({x:.12g} {y:.12g}) rotate({deg:.9g})"/>')
    out.append("</g></svg>")
    return "\n".join(out)


def save_svg(path: str, s: float, squares, size: int = 480, title: str = "") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_svg(s, squares, size, title))
