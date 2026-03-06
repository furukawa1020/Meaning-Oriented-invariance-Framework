"""
Meaning-Oriented Invariance Framework (MOIF)
A library for detecting and explaining where physiological invariance breaks relative to meaning.
"""

__version__ = "0.1.0"

# Expose core functionality at the top level so users can just:
# import moif
# moif.load_wesad(...)
# moif.apply_banding(...)
# moif.compute_jsd(...)
# moif.permutation_test(...)

from .loaders.wesad import load_wesad
from .invariance.banding import apply_banding
from .invariance.divergence import compute_jsd, get_distribution
from .invariance.stats import permutation_test, apply_fdr

__all__ = [
    "load_wesad",
    "apply_banding",
    "compute_jsd",
    "get_distribution",
    "permutation_test",
    "apply_fdr",
]
