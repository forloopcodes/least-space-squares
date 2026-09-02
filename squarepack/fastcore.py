"""ctypes loader for the optional C core (:file:`_fastcore.c`).

The shared library is compiled on first use with the system C compiler into a
build directory next to the package (or a temp directory).  If no compiler is
available, :func:`load` returns ``None`` and the numpy implementation is used.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

_lib = None
_tried = False
SRC = Path(__file__).with_name("_fastcore.c")


def load() -> Optional[ctypes.CDLL]:
    global _lib, _tried
    if _tried:
        return _lib or None
    _tried = True
    if os.environ.get("SQUAREPACK_NO_C"):
        return None
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if cc is None or not SRC.exists():
        return None
    h = hashlib.sha1(SRC.read_bytes()).hexdigest()[:12]
    name = f"fastcore_{h}_{sys.platform}.so"
    for d in (Path(__file__).parent / "_build", Path(tempfile.gettempdir()) / "squarepack_build"):
        try:
            d.mkdir(parents=True, exist_ok=True)
            so = d / name
            if not so.exists():
                tmp = d / (name + f".{os.getpid()}.tmp")
                subprocess.run([cc, "-O3", "-fPIC", "-shared", "-o", str(tmp), str(SRC), "-lm"],
                               check=True, capture_output=True)
                os.replace(tmp, so)
            lib = ctypes.CDLL(str(so))
            dp = ctypes.POINTER(ctypes.c_double)
            ip = ctypes.POINTER(ctypes.c_int)
            lib.energy_grad_c.restype = ctypes.c_double
            lib.energy_grad_c.argtypes = [ctypes.c_int, ctypes.c_double, dp, ctypes.c_int, ip, ip, dp]
            lib.max_violation_c.restype = ctypes.c_double
            lib.max_violation_c.argtypes = [ctypes.c_int, ctypes.c_double, dp]
            lib.lbfgs_c.restype = ctypes.c_int
            lib.lbfgs_c.argtypes = [ctypes.c_int, ctypes.c_double, dp, ctypes.c_int, ctypes.c_double,
                                    ctypes.c_double, ctypes.c_double, dp]
            _lib = lib
            return lib
        except Exception:
            continue
    return None


def _dptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def energy_grad(z: np.ndarray, n: int, s: float, I: np.ndarray, J: np.ndarray):
    lib = load()
    z = np.ascontiguousarray(z, dtype=np.float64)
    I = np.ascontiguousarray(I, dtype=np.int32)
    J = np.ascontiguousarray(J, dtype=np.int32)
    g = np.zeros(3 * n)
    E = lib.energy_grad_c(n, float(s), _dptr(z), len(I), I.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                          J.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), _dptr(g))
    return float(E), g


def max_violation(z: np.ndarray, n: int, s: float) -> float:
    lib = load()
    z = np.ascontiguousarray(z, dtype=np.float64)
    return float(lib.max_violation_c(n, float(s), _dptr(z)))


def lbfgs(z: np.ndarray, n: int, s: float, maxiter: int, gtol: float, ftol: float, cutoff: float):
    lib = load()
    z = np.array(z, dtype=np.float64, copy=True)
    E = ctypes.c_double(0.0)
    it = lib.lbfgs_c(n, float(s), _dptr(z), int(maxiter), float(gtol), float(ftol), float(cutoff), ctypes.byref(E))
    return z, float(E.value), int(it)
