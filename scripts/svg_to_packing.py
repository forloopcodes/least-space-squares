#!/usr/bin/env python
"""Interpret the reference SVG drawings of record square packings
(https://kingbird.myphotos.cc/packing/square-N.svg) and extract exact coordinates.

The drawings encode the packing through nested ``translate/scale/rotate``
transforms, ``<use>`` references into ``<defs>``, unit-cell ``<rect>``
elements (``width="1" height="2"`` is a domino, etc.), filled rectilinear
``<path>`` polyominoes such as ``M2,0 V1 H1 V2 H0 V0`` and numeric
``<!ENTITY>`` constants in the DOCTYPE.  Elements with ``fill:none`` are
outlines and are ignored; the container is the ``<rect id="outer">``.

Usage::

    python scripts/svg_to_packing.py square-18.svg [--json] [--no-flip]

prints ``s`` and the rows ``[x_center, y_center, angle_radians]`` (in the
conventions of :mod:`squarepack.geometry`: y up, angle counter-clockwise) and
runs :func:`squarepack.geometry.verify` on the result.

The module can also be imported: :func:`load_svg` returns ``(s, squares)``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squarepack.geometry import verify  # noqa: E402

Matrix = np.ndarray  # 3x3 homogeneous transform


# --------------------------------------------------------------------------- #
# entities and parsing
# --------------------------------------------------------------------------- #
_ENTITY_RE = re.compile(r'<!ENTITY\s+(\w+)\s+"([^"]*)"\s*>')
_DOCTYPE_RE = re.compile(r"<!DOCTYPE[^\[>]*(\[.*?\])?\s*>", re.S)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def parse_entities(text: str) -> Dict[str, str]:
    m = _DOCTYPE_RE.search(text)
    if not m or not m.group(1):
        return {}
    body = _COMMENT_RE.sub("", m.group(1))
    ents = {k: v.strip() for k, v in _ENTITY_RE.findall(body)}
    # entity values may reference earlier entities (e.g. tr "translate(-&u2; 1)")
    for _ in range(16):
        new = {k: expand_entities(v, ents) for k, v in ents.items()}
        if new == ents:
            break
        ents = new
    return ents


def expand_entities(text: str, ents: Dict[str, str]) -> str:
    def sub(m):
        name = m.group(1)
        if name in ents:
            return ents[name]
        return m.group(0)
    return re.sub(r"&(\w+);", sub, text)


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def parse_svg(text: str) -> Tuple[ET.Element, Dict[str, str]]:
    ents = parse_entities(text)
    text = _DOCTYPE_RE.sub("", text)
    text = _COMMENT_RE.sub("", text)
    text = expand_entities(text, ents)
    # drop the <script> tag (external reference) and the xml declaration
    text = re.sub(r"<script[^>]*/>", "", text)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.S)
    root = ET.fromstring(text)
    for el in root.iter():
        el.tag = strip_ns(el.tag)
        for k in list(el.attrib):
            if "}" in k:
                el.attrib[strip_ns(k)] = el.attrib.pop(k)
    return root, ents


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
_TRANSFORM_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _nums(s: str) -> List[float]:
    return [float(v) for v in _NUM_RE.findall(s)]


def parse_transform(spec: Optional[str]) -> Matrix:
    M = np.eye(3)
    if not spec:
        return M
    for kind, args in _TRANSFORM_RE.findall(spec):
        v = _nums(args)
        T = np.eye(3)
        if kind == "translate":
            T[0, 2] = v[0]
            T[1, 2] = v[1] if len(v) > 1 else 0.0
        elif kind == "scale":
            T[0, 0] = v[0]
            T[1, 1] = v[1] if len(v) > 1 else v[0]
        elif kind == "rotate":
            a = math.radians(v[0])
            c, s = math.cos(a), math.sin(a)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            if len(v) == 3:
                cx, cy = v[1], v[2]
                P = np.eye(3); P[0, 2], P[1, 2] = cx, cy
                Q = np.eye(3); Q[0, 2], Q[1, 2] = -cx, -cy
                T = P @ R @ Q
            else:
                T = R
        elif kind == "matrix":
            a, b, c, d, e, f = v
            T = np.array([[a, c, e], [b, d, f], [0, 0, 1.0]])
        elif kind == "skewX":
            T[0, 1] = math.tan(math.radians(v[0]))
        elif kind == "skewY":
            T[1, 0] = math.tan(math.radians(v[0]))
        M = M @ T
    return M


# --------------------------------------------------------------------------- #
# paths -> unit cells
# --------------------------------------------------------------------------- #
_PATH_TOKEN_RE = re.compile(r"([MmLlHhVvZz])|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def path_polygons(d: str) -> List[List[Tuple[float, float]]]:
    """Sub-polygons of an SVG path made of M/L/H/V/Z commands (absolute or relative)."""
    tokens = _PATH_TOKEN_RE.findall(d)
    polys: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    x = y = 0.0
    cmd = None
    i = 0
    items: List[Tuple[str, Optional[float]]] = [(c, None) if c else (None, float(n)) for c, n in tokens]
    while i < len(items):
        c, n = items[i]
        if c is not None:
            cmd = c
            i += 1
            if cmd in "Zz":
                if cur:
                    polys.append(cur)
                cur = []
                if polys:
                    x, y = polys[-1][0]
            continue
        # numbers following the current command
        if cmd in ("M", "m", "L", "l"):
            nx, ny = items[i][1], items[i + 1][1]
            i += 2
            if cmd in "ml":
                nx += x; ny += y
            if cmd in "Mm":
                if cur:
                    polys.append(cur)
                cur = []
                cmd = "L" if cmd == "M" else "l"
            x, y = nx, ny
            cur.append((x, y))
        elif cmd in ("H", "h"):
            nx = items[i][1]; i += 1
            x = nx + (x if cmd == "h" else 0.0)
            cur.append((x, y))
        elif cmd in ("V", "v"):
            ny = items[i][1]; i += 1
            y = ny + (y if cmd == "v" else 0.0)
            cur.append((x, y))
        else:
            raise ValueError(f"unsupported path command {cmd!r} in {d!r}")
    if cur:
        polys.append(cur)
    return polys


def point_in_polygon(px: float, py: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > py) != (y2 > py):
            xi = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < xi:
                inside = not inside
    return inside


def polygon_cells(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Unit-cell centres of a rectilinear polygon with integer vertices (even-odd fill)."""
    if len(poly) < 3:
        return []
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    x0, x1 = math.floor(min(xs)), math.ceil(max(xs))
    y0, y1 = math.floor(min(ys)), math.ceil(max(ys))
    cells = []
    for i in range(x0, x1):
        for j in range(y0, y1):
            if point_in_polygon(i + 0.5, j + 0.5, poly):
                cells.append((i + 0.5, j + 0.5))
    return cells


# --------------------------------------------------------------------------- #
# the interpreter
# --------------------------------------------------------------------------- #
def _style(el: ET.Element) -> Dict[str, str]:
    st = {}
    for part in (el.get("style") or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            st[k.strip()] = v.strip()
    if el.get("fill") is not None:
        st["fill"] = el.get("fill")
    return st


class SvgPacking:
    def __init__(self, root: ET.Element):
        self.root = root
        self.ids: Dict[str, ET.Element] = {}
        for el in root.iter():
            # duplicate ids occur in some drawings; like browsers, the first one wins
            if el.get("id") and el.get("id") not in self.ids:
                self.ids[el.get("id")] = el
        vb = _nums(root.get("viewBox", ""))
        if len(vb) == 4:
            self.s = vb[2]
            if abs(vb[3] - vb[2]) > 1e-9 or vb[0] != 0 or vb[1] != 0:
                raise ValueError("viewBox is not a square at the origin")
        else:
            outer = self.ids.get("outer")
            self.s = float(outer.get("width"))
        self.cells: List[Tuple[float, float, float]] = []  # (cx, cy, angle) in SVG user space
        self.depth = 0

    # -- emit one unit square: local unit cell centred at (cx, cy) under transform M
    def _emit(self, M: Matrix, cx: float, cy: float):
        L = M[:2, :2]
        # the linear part must be orthogonal (possibly a reflection) up to rounding
        if abs(abs(np.linalg.det(L)) - 1.0) > 1e-9 or abs(L[:, 0] @ L[:, 1]) > 1e-9:
            raise ValueError(f"non-rigid transform for a unit square: {L}")
        p = M @ np.array([cx, cy, 1.0])
        ex = L @ np.array([1.0, 0.0])
        ang = math.atan2(ex[1], ex[0])
        self.cells.append((float(p[0]), float(p[1]), ang))

    def _rect(self, el: ET.Element, M: Matrix):
        w = float(el.get("width")); h = float(el.get("height"))
        x = float(el.get("x", 0)); y = float(el.get("y", 0))
        if el.get("id") == "outer" or abs(w - self.s) < 1e-9:
            return
        nw, nh = round(w), round(h)
        if abs(nw - w) > 1e-9 or abs(nh - h) > 1e-9 or nw < 1 or nh < 1:
            raise ValueError(f"rect {w}x{h} is not a polyomino")
        for i in range(nw):
            for j in range(nh):
                self._emit(M, x + i + 0.5, y + j + 0.5)

    def _path(self, el: ET.Element, M: Matrix):
        d = el.get("d", "")
        for poly in path_polygons(d):
            for cx, cy in polygon_cells(poly):
                self._emit(M, cx, cy)

    def walk(self, el: ET.Element, M: Matrix, inherited_style: Dict[str, str]):
        tag = el.tag
        st = dict(inherited_style)
        st.update(_style(el))
        if tag == "defs":
            return
        M = M @ parse_transform(el.get("transform"))
        fill_none = st.get("fill") == "none"
        if tag == "rect":
            if not fill_none:
                self._rect(el, M)
        elif tag == "path":
            if not fill_none:
                self._path(el, M)
        elif tag == "use":
            href = el.get("href") or el.get("{http://www.w3.org/1999/xlink}href") or ""
            ref = href.lstrip("#")
            if ref == "outer":
                return
            target = self.ids[ref]
            U = np.eye(3)
            U[0, 2] = float(el.get("x", 0)); U[1, 2] = float(el.get("y", 0))
            self.depth += 1
            if self.depth > 64:
                raise RecursionError("use recursion too deep")
            self.walk(target, M @ U, st)
            self.depth -= 1
        elif tag in ("g", "svg", "a", "symbol"):
            for child in el:
                self.walk(child, M, st)
        # everything else (script, title, ...) is ignored

    def run(self):
        self.walk(self.root, np.eye(3), {})
        return self.s, self.cells


def load_svg(path: str, flip_y: bool = True):
    """Return ``(s, squares)`` with ``squares`` rows ``[x, y, angle]`` (y up when ``flip_y``)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    root, _ = parse_svg(text)
    s, cells = SvgPacking(root).run()
    arr = np.array(cells, dtype=float).reshape(-1, 3)
    if flip_y:
        arr[:, 1] = s - arr[:, 1]
        arr[:, 2] = -arr[:, 2]
    # canonical angle in (-pi/4, pi/4]
    arr[:, 2] = (arr[:, 2] + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return s, arr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("svg", nargs="+")
    ap.add_argument("--json", action="store_true", help="print JSON {s, squares}")
    ap.add_argument("--no-flip", action="store_true", help="keep the SVG y-down coordinates")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args(argv)
    rc = 0
    for path in a.svg:
        s, arr = load_svg(path, flip_y=not a.no_flip)
        rep = verify(s, arr, a.tol)
        if a.json:
            print(json.dumps({"file": path, "s": s, "n": len(arr), "valid": rep.ok, "squares": arr.tolist()}))
        else:
            print(f"{path}: n={len(arr)} s={s:.15g} {rep}")
            for i, (x, y, t) in enumerate(arr):
                print(f"{i:4d}  x={x:.12f}  y={y:.12f}  angle={t:.12f}  ({math.degrees(t):.6f} deg)")
        if not rep.ok:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
