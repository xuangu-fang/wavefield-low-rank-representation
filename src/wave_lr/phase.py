"""Phase representations that do not introduce a branch-cut discontinuity."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def wrap_phase(phase: ArrayLike) -> NDArray[np.float64]:
    """Wrap angles to [-pi, pi)."""

    value = np.asarray(phase, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def phase_embedding(phase: ArrayLike) -> NDArray[np.float64]:
    """Represent phase continuously as [..., sin(phi), cos(phi)]."""

    value = np.asarray(phase, dtype=np.float64)
    return np.stack((np.sin(value), np.cos(value)), axis=-1)


def paired_phase_carriers(
    distance: ArrayLike,
    time: ArrayLike,
    wavenumbers: ArrayLike,
    speeds: ArrayLike,
) -> NDArray[np.float64]:
    """Build the four separable carriers for every (wavenumber, speed) pair.

    The final dimension is ordered as
    [cos(kd)cos(kct), sin(kd)sin(kct), cos(kd)sin(kct), sin(kd)cos(kct)].
    Hence terms 0+1 reconstruct cos(k(d-ct)), while 3-2 reconstruct
    sin(k(d-ct)). Inputs ``distance`` and ``time`` must broadcast.
    """

    d, t = np.broadcast_arrays(
        np.asarray(distance, dtype=np.float64), np.asarray(time, dtype=np.float64)
    )
    k = np.asarray(wavenumbers, dtype=np.float64).reshape(-1, 1)
    c = np.asarray(speeds, dtype=np.float64).reshape(1, -1)
    if k.size == 0 or c.size == 0:
        raise ValueError("wavenumbers and speeds must be non-empty")

    spatial = d[..., None, None] * k * np.ones_like(c)
    temporal = t[..., None, None] * k * c
    return np.stack(
        (
            np.cos(spatial) * np.cos(temporal),
            np.sin(spatial) * np.sin(temporal),
            np.cos(spatial) * np.sin(temporal),
            np.sin(spatial) * np.cos(temporal),
        ),
        axis=-1,
    )

