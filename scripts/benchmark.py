#!/usr/bin/env python3
"""Benchmark every method of the portfolio over a range of n and record the best packings.

For each n the script runs, in order, and times:
  grid            s = ceil(sqrt n)                                   O(n)
  closed-form     Göbel strips / squares + L extensions              O(n)
  tilted-block    p x q block at 45 deg + row fill, bisection on s   ~seconds
  numeric         penalty-energy compaction + basin hopping          --budget seconds
and compares with the best known value from the literature (squarepack.known).

Outputs: <out>/benchmark.json, <out>/benchmark.md and (with --save) improved packings
in data/best_packings.json (verified before writing).

Usage: python scripts/benchmark.py --n-min 1 --n-max 100 --budget 30 --workers 4 --save
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

from squarepack.blocks import tilted_block_search  # noqa: E402
from squarepack.constructions import best_analytic, grid  # noqa: E402
from squarepack.exact import exact_form  # noqa: E402
from squarepack.geometry import verify  # noqa: E402
from squarepack.known import best_known, is_proved  # noqa: E402
from squarepack.solver import cached, update_cache  # noqa: E402


def run_one(n: int, budget: float, seed: int, use_cache: bool) -> dict:
    from squarepack.optimize import search
    row = {"n": n, "best_known": best_known(n), "proved": is_proved(n), "lower_bound": math.sqrt(n)}
    t = time.perf_counter()
    g = grid(n)
    row["grid"] = {"s": g.s, "time": time.perf_counter() - t}
    t = time.perf_counter()
    a = best_analytic(n)
    row["closed_form"] = {"s": a.s, "time": time.perf_counter() - t, "method": a.method, "exact": a.exact}
    best = a
    t = time.perf_counter()
    skip_block = a.method == "grid" and ((math.isqrt(n) + 1) ** 2 - n in (0, 1, 2) or math.isqrt(n) ** 2 == n)
    b = None if skip_block else tilted_block_search(n, s_max=best.s)
    row["tilted_block"] = {"s": b.s if b else best.s, "time": time.perf_counter() - t,
                           "exact": b.exact if b else "(none better)"}
    if b is not None and b.s < best.s - 1e-9:
        best = b
    c = cached(n) if use_cache else None
    if c is not None and c.s < best.s - 1e-12:
        best = c
    row["numeric"] = {"s": best.s, "time": 0.0, "method": best.method}
    if budget > 0 and not row["proved"] and best.s > math.sqrt(n) + 1e-9:
        t = time.perf_counter()
        r = search(n, time_budget=budget, seed=seed, start=best)
        row["numeric"] = {"s": r.packing.s, "time": time.perf_counter() - t, "method": r.packing.method,
                          "local_opts": r.stats.local_opts, "seeds": r.stats.seeds}
        if r.packing.s < best.s - 1e-12:
            best = r.packing
    rep = verify(best.s, best.squares, 1e-9)
    row["best"] = {"s": best.s, "method": best.method, "exact": best.exact, "valid": rep.ok,
                   "max_penetration": rep.max_penetration, "max_outside": rep.max_outside}
    row["packing"] = {"s": best.s, "method": best.method, "exact": best.exact,
                      "squares": np.asarray(best.squares, float).tolist()}
    return row


def fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def write_markdown(rows, path: Path, budget: float):
    lines = ["# Benchmark: n unit squares in the smallest square", "",
             f"Numeric budget per n: {budget:g} s.  `gap` = best found - best known (literature).", "",
             "| n | grid | closed-form | tilted-block | numeric | best | best known | gap | proved | method | t_block (s) | t_num (s) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--|---:|---:|"]
    for r in rows:
        bk = r["best_known"]
        gap = "" if bk is None else f"{r['best']['s'] - bk:+.6f}"
        form = exact_form(r["best"]["s"]) or ""
        lines.append(f"| {r['n']} | {r['grid']['s']:.4f} | {r['closed_form']['s']:.6f} | {r['tilted_block']['s']:.6f} | "
                     f"{r['numeric']['s']:.6f} | **{r['best']['s']:.6f}** | {'' if bk is None else f'{bk:.6f}'} | {gap} | "
                     f"{'yes' if r['proved'] else ''} | {r['best']['method']} {form} | {r['tilted_block']['time']:.1f} | {r['numeric']['time']:.1f} |")
    nn = len(rows)
    def cnt(*keys):  # running minimum over the pipeline stages listed
        return sum(1 for r in rows if r["best_known"] is not None
                   and min(r[k]["s"] for k in keys) <= r["best_known"] + 1e-9)
    lines += ["", "## Summary", "",
              f"* n range: {rows[0]['n']}..{rows[-1]['n']} ({nn} values)",
              f"* matches best known: grid {cnt('grid')}, +closed-form {cnt('grid', 'closed_form')}, "
              f"+tilted-block {cnt('grid', 'closed_form', 'tilted_block')}, +numeric {cnt('best')} of {nn}",
              f"* mean gap to best known: {np.mean([r['best']['s'] - r['best_known'] for r in rows if r['best_known'] is not None]):.5f}",
              f"* all packings verified valid: {all(r['best']['valid'] for r in rows)}",
              f"* total time: closed-form {sum(r['closed_form']['time'] for r in rows):.1f}s, "
              f"tilted-block {sum(r['tilted_block']['time'] for r in rows):.1f}s, numeric {sum(r['numeric']['time'] for r in rows):.1f}s"]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=100)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--save", action="store_true", help="store improved packings in data/best_packings.json")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--only", type=str, default="", help="comma separated list of n (overrides range)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ns = [int(v) for v in a.only.split(",") if v] if a.only else list(range(a.n_min, a.n_max + 1))
    rows = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, n, a.budget, a.seed + n, not a.no_cache): n for n in ns}
        for f in as_completed(futs):
            r = f.result()
            n = r["n"]
            rows[n] = r
            bk = r["best_known"]
            gap = "" if bk is None else f" gap={r['best']['s'] - bk:+.6f}"
            print(f"[{time.time() - t0:7.0f}s] n={n:4d} best={r['best']['s']:.8f} ({r['best']['method']}){gap}", flush=True)
            if a.save and r["best"]["method"] != "grid" and r["best"]["valid"]:
                from squarepack.constructions import Packing
                p = Packing(n, r["packing"]["s"], np.asarray(r["packing"]["squares"]), r["packing"]["method"], r["packing"]["exact"])
                if update_cache(p):
                    print(f"          cached n={n} s={p.s:.10f}", flush=True)
    # monotonicity pass: s(n) <= s(m) for m > n, so a better packing of m squares (minus m - n
    # of them) also serves n; propagate downwards and cache the result
    from squarepack.constructions import Packing
    ordered = [rows[n] for n in sorted(rows)]
    for i in range(len(ordered) - 2, -1, -1):
        r, nxt = ordered[i], ordered[i + 1]
        if nxt["best"]["s"] < r["best"]["s"] - 1e-9 and nxt["best"]["valid"]:
            p = Packing(nxt["n"], nxt["packing"]["s"], np.asarray(nxt["packing"]["squares"]),
                        f"monotone from n={nxt['n']}", nxt["packing"]["exact"]).take(r["n"])
            rep = verify(p.s, p.squares, 1e-9)
            if rep.ok:
                r["best"] = {"s": p.s, "method": p.method, "exact": p.exact, "valid": True,
                             "max_penetration": rep.max_penetration, "max_outside": rep.max_outside}
                r["packing"] = {"s": p.s, "method": p.method, "exact": p.exact, "squares": p.squares.tolist()}
                print(f"monotone: n={r['n']} improved to {p.s:.8f} from n={nxt['n']}", flush=True)
                if a.save:
                    update_cache(p)
    slim = [{k: v for k, v in r.items() if k != "packing"} for r in ordered]
    (out / "benchmark.json").write_text(json.dumps(slim, indent=1))
    write_markdown(ordered, out / "benchmark.md", a.budget)
    print("wrote", out / "benchmark.md")


if __name__ == "__main__":
    main()
