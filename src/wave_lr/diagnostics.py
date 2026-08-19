"""Representation diagnostics before fitting a large predictive model."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import svdvals


def singular_spectrum(
    array: ArrayLike, split_axis: int = 1, use_gpu: bool | None = None
) -> NDArray[np.float64]:
    """Return normalized singular values of one tensor unfolding.

    Axes before ``split_axis`` form rows and remaining axes form columns.
    """

    value = np.asarray(array)
    if not 0 < split_axis < value.ndim:
        raise ValueError("split_axis must lie strictly inside the array dimensions")
    rows = int(np.prod(value.shape[:split_axis]))
    matrix = value.reshape(rows, -1)
    spectrum = svdvals(matrix, use_gpu=use_gpu)
    norm = np.linalg.norm(spectrum)
    return spectrum / norm if norm > 0 else spectrum.astype(np.float64)


def best_rank_error(array: ArrayLike, rank: int, split_axis: int = 1) -> float:
    """Relative Frobenius error of the best rank-r unfolding approximation."""

    spectrum = singular_spectrum(array, split_axis=split_axis)
    if rank < 0:
        raise ValueError("rank must be non-negative")
    return float(np.linalg.norm(spectrum[rank:]))


def effective_rank(array: ArrayLike, energy: float = 0.99, split_axis: int = 1) -> int:
    """Smallest rank retaining the requested squared singular-value energy."""

    if not 0 < energy <= 1:
        raise ValueError("energy must be in (0, 1]")
    spectrum = singular_spectrum(array, split_axis=split_axis)
    if spectrum.size == 0 or np.all(spectrum == 0):
        return 0
    return int(np.searchsorted(np.cumsum(spectrum**2), energy) + 1)

