"""Command line interface: ``python -m squarepack N [options]``."""
from __future__ import annotations

import argparse
import json
import signal
import sys

from .render import save_svg
from .solver import solve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="squarepack", description="Pack n unit squares into the smallest square.")
    ap.add_argument("n", type=int, help="number of unit squares")
    ap.add_argument("--budget", type=float, default=0.0, help="seconds of numerical search (0 = analytic only)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for the numerical search")
    ap.add_argument("--degrees", action="store_true", help="output angles in degrees instead of radians")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--svg", metavar="FILE", help="also write an SVG drawing of the packing")
    ap.add_argument("--no-cache", action="store_true", help="ignore the cache of previously found packings")
    ap.add_argument("--no-blocks", action="store_true", help="skip the tilted-block search")
    ap.add_argument("--save", action="store_true", help="store an improved packing in the cache")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    sol = solve(a.n, time_budget=a.budget, use_cache=not a.no_cache, use_blocks=not a.no_blocks,
                seed=a.seed, verbose=a.verbose, save=a.save)
    if a.svg:
        save_svg(a.svg, sol.s, sol.squares, title=f"n={sol.n} s={sol.s:.10f}")
    if a.json:
        json.dump(sol.to_dict(a.degrees), sys.stdout, indent=1)
        print()
    else:
        bk = "" if sol.best_known is None else f"  best known: {sol.best_known:.10f}"
        print(f"n = {sol.n}\ns = {sol.s:.12f}   ({sol.exact or sol.method}){bk}"
              f"  lower bound sqrt(n) = {sol.lower_bound:.6f}" + ("  [proved optimal]" if sol.proved_optimal else ""))
        unit = "deg" if a.degrees else "rad"
        for i, (x, y, t) in enumerate(sol.as_list(a.degrees)):
            print(f"{i:4d}  x={x:.12f}  y={y:.12f}  angle({unit})={t:.12f}")
    return 0


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # let `| head` end the output quietly
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
