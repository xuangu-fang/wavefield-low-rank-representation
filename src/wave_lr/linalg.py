"""Singular values with an optional GPU backend for large unfoldings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_GPU_MIN_ELEMENTS = 250_000


def svdvals(matrix: ArrayLike, use_gpu: bool | None = None) -> NDArray[np.float64]:
    """Singular values of a 2-D array, on GPU when the matrix is large."""

    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError("svdvals expects a 2-D array")
    if use_gpu is None:
        use_gpu = array.size >= _GPU_MIN_ELEMENTS
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                tensor = torch.from_numpy(np.ascontiguousarray(array)).cuda()
                values = torch.linalg.svdvals(tensor)
                return values.double().cpu().numpy()
        except (ImportError, RuntimeError):  # pragma: no cover - falls back to CPU
            pass
    return np.linalg.svd(array, compute_uv=False)
