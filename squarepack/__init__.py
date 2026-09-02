"""squarepack - tight packings of n unit squares in the smallest square.

>>> from squarepack import pack
>>> s, squares = pack(5)
>>> round(s, 6), len(squares)
(2.707107, 5)
"""
from .geometry import verify, repair
from .constructions import Packing, best_analytic, analytic_candidates
from .blocks import tilted_block_search
from .solver import pack, solve, Solution
from .known import best_known, RECORDS

__all__ = ["pack", "solve", "Solution", "verify", "repair", "Packing", "best_analytic",
           "analytic_candidates", "tilted_block_search", "best_known", "RECORDS"]
__version__ = "0.1.0"
