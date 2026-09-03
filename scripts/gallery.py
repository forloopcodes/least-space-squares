#!/usr/bin/env python3
"""Render SVG drawings of the solver's packings for a set of n into results/svg/ and an index page."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from squarepack.render import save_svg  # noqa: E402
from squarepack.solver import solve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ns", nargs="*", type=int, default=[5, 10, 11, 17, 18, 19, 26, 27, 28, 29, 37, 38, 40, 41, 50, 52, 65, 66, 67, 85, 89])
    ap.add_argument("--out", default=str(ROOT / "results" / "svg"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in a.ns:
        sol = solve(n)
        f = out / f"square-{n}.svg"
        save_svg(str(f), sol.s, sol.squares, size=300, title=f"n={n} s={sol.s:.8f}")
        bk = "" if sol.best_known is None else f"{sol.best_known:.6f}"
        rows.append(f'<figure><img src="{f.name}" width="300"><figcaption>n={n}, s={sol.s:.6f} ({sol.method}); best known {bk}</figcaption></figure>')
        print(n, sol.s, sol.method)
    (out / "index.html").write_text("<html><body style='font-family:sans-serif'><h1>squarepack gallery</h1>"
                                    "<div style='display:flex;flex-wrap:wrap;gap:12px'>" + "".join(rows) + "</div></body></html>")


if __name__ == "__main__":
    main()
