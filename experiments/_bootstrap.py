"""Thread limits, imported first by every experiment.

The shared machine oversubscribes BLAS threads badly: a 512x512 complex SVD
costs 17 s with the default thread pool and 0.13 s pinned to one thread.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
