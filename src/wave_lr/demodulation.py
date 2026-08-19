"""Carrier removal and reconstruction for complex wave fields."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def demodulate(field: ArrayLike, phase: ArrayLike) -> NDArray[np.complex128]:
    """Remove a known or estimated phase carrier: z -> z exp(-i phi)."""

    z = np.asarray(field, dtype=np.complex128)
    phi = np.asarray(phase, dtype=np.float64)
    return z * np.exp(-1j * phi)


def remodulate(envelope: ArrayLike, phase: ArrayLike) -> NDArray[np.complex128]:
    """Reapply a phase carrier: a -> a exp(i phi)."""

    a = np.asarray(envelope, dtype=np.complex128)
    phi = np.asarray(phase, dtype=np.float64)
    return a * np.exp(1j * phi)

