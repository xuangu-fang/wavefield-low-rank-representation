"""Complex-field metrics that keep amplitude and phase errors separate."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _pair(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a)
    y = np.asarray(b)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    return x, y


def complex_nrmse(prediction: ArrayLike, target: ArrayLike) -> float:
    """Relative Frobenius error of the complex field."""

    p, t = _pair(prediction, target)
    denom = np.linalg.norm(t)
    if denom == 0:
        return float("nan")
    return float(np.linalg.norm(p - t) / denom)


def amplitude_nrmse(prediction: ArrayLike, target: ArrayLike) -> float:
    p, t = _pair(prediction, target)
    denom = np.linalg.norm(np.abs(t))
    if denom == 0:
        return float("nan")
    return float(np.linalg.norm(np.abs(p) - np.abs(t)) / denom)


def log_amplitude_error(prediction: ArrayLike, target: ArrayLike, floor: float = 1e-6) -> float:
    p, t = _pair(prediction, target)
    scale = float(np.abs(t).max()) or 1.0
    lp = np.log(np.abs(p) / scale + floor)
    lt = np.log(np.abs(t) / scale + floor)
    return float(np.sqrt(np.mean((lp - lt) ** 2)))


def amplitude_weighted_phase_coherence(prediction: ArrayLike, target: ArrayLike) -> float:
    """AWPC in [-1, 1]; ignores phase where the target amplitude is negligible.

    Equal to ``1`` for a perfectly aligned phase and ``0`` for random phase.
    """

    p, t = _pair(prediction, target)
    weight = np.abs(t)
    total = weight.sum()
    if total == 0:
        return float("nan")
    unit_p = p / np.maximum(np.abs(p), 1e-30)
    unit_t = t / np.maximum(np.abs(t), 1e-30)
    return float(np.real(np.sum(weight * unit_p * np.conj(unit_t))) / total)


def summarize(prediction: ArrayLike, target: ArrayLike) -> dict[str, float]:
    return {
        "complex_nrmse": complex_nrmse(prediction, target),
        "amplitude_nrmse": amplitude_nrmse(prediction, target),
        "log_amplitude_rmse": log_amplitude_error(prediction, target),
        "awpc": amplitude_weighted_phase_coherence(prediction, target),
    }
