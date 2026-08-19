"""Delay-spread rank law for complex wave fields.

A frequency-domain field is the Fourier transform of an impulse response,

    u(x, f) = int g(x, tau) exp(2 pi i f tau) d tau,

so the ``(x, f)`` unfolding is a band-limited nonuniform Fourier operator.
Slepian-type degree-of-freedom counting then predicts

    rank_eps(U) ~ B * Lambda_B + O(log),

where ``B`` is the bandwidth and ``Lambda_B`` is the Lebesgue measure of the
delay set that carries energy, thickened by the resolution limit ``1 / B``.
Demodulating by a carrier ``exp(2 pi i f tau_ref(x))`` shifts each location's
impulse response in time, replacing the *absolute* delay occupancy by the
*relative* one -- which is the entire source of the compression gain.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_OVERSAMPLE = 8


def delay_occupancy(
    delays: ArrayLike,
    energies: ArrayLike | None = None,
    bandwidth: float = 1.0,
    threshold: float = 1e-3,
) -> float:
    """Bandwidth-resolved Lebesgue measure of the occupied delay set.

    ``delays`` holds arrival times (any shape) and ``energies`` their squared
    amplitudes. Arrivals closer than the resolution ``1 / bandwidth`` cannot be
    told apart by a band-limited measurement, so the occupancy is computed
    after smoothing with a kernel of that width. ``threshold`` is the fraction
    of the peak energy density below which a delay counts as empty.
    """

    tau = np.asarray(delays, dtype=float).ravel()
    if tau.size == 0:
        return 0.0
    weight = (
        np.ones_like(tau)
        if energies is None
        else np.asarray(energies, dtype=float).ravel() ** 1
    )
    if weight.shape != tau.shape:
        raise ValueError("delays and energies must have matching shapes")
    resolution = 1.0 / float(bandwidth)
    step = resolution / _OVERSAMPLE
    lo, hi = tau.min() - 2.0 * resolution, tau.max() + 2.0 * resolution
    n_bins = max(int(np.ceil((hi - lo) / step)), 4)
    density, _ = np.histogram(tau, bins=n_bins, range=(lo, hi), weights=weight)
    kernel = np.ones(_OVERSAMPLE)
    smoothed = np.convolve(density, kernel, mode="same")
    peak = smoothed.max()
    if peak <= 0:
        return 0.0
    return float(np.count_nonzero(smoothed > threshold * peak) * step)


def occupancy_from_traces(
    traces: ArrayLike,
    dt: float,
    bandwidth: float,
    threshold: float = 1e-3,
) -> float:
    """Delay occupancy read directly off time-domain traces ``(n_x, n_t)``.

    This is the deployable estimator: no path model is required, only the
    pooled energy envelope over the time axis.
    """

    data = np.asarray(traces, dtype=float)
    if data.ndim != 2:
        raise ValueError("traces must be (n_x, n_t)")
    energy = (data**2).sum(axis=0)
    resolution = 1.0 / float(bandwidth)
    width = max(int(round(resolution / dt)), 1)
    kernel = np.ones(width) / width
    smoothed = np.convolve(energy, kernel, mode="same")
    peak = smoothed.max()
    if peak <= 0:
        return 0.0
    return float(np.count_nonzero(smoothed > threshold * peak) * dt)


def predicted_rank(bandwidth: float, occupancy: float, offset: float = 1.0) -> float:
    """Predicted numerical rank of a band-limited delayed-arrival field."""

    return float(bandwidth) * float(occupancy) + offset


def predicted_gain(
    bandwidth: float, absolute_occupancy: float, relative_occupancy: float
) -> float:
    """Predicted rank reduction achieved by first-arrival demodulation."""

    return predicted_rank(bandwidth, absolute_occupancy) / predicted_rank(
        bandwidth, relative_occupancy
    )


def pooled_support(delays: ArrayLike) -> float:
    """Naive support length ``max - min`` (ignores gaps and resolution)."""

    tau = np.asarray(delays, dtype=float)
    return float(tau.max() - tau.min())


def union_of_ranges(delays: ArrayLike, axis: int = 0) -> float:
    """Total length of the union of per-path spatial delay ranges."""

    tau = np.asarray(delays, dtype=float)
    lo = tau.min(axis=axis)
    hi = tau.max(axis=axis)
    order = np.argsort(lo)
    total, current_lo, current_hi = 0.0, None, None
    for index in order:
        a, b = float(lo[index]), float(hi[index])
        if current_hi is None:
            current_lo, current_hi = a, b
            continue
        if a > current_hi:
            total += current_hi - current_lo
            current_lo, current_hi = a, b
        else:
            current_hi = max(current_hi, b)
    if current_hi is not None:
        total += current_hi - current_lo
    return total


def fit_slope(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Least-squares fit ``y = a x + b`` reported with R^2."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "r2": 1.0 - float(residual @ residual) / ss_tot if ss_tot > 0 else float("nan"),
        "median_abs_rel_error": float(
            np.median(np.abs(residual) / np.maximum(np.abs(y), 1e-12))
        ),
    }


def relative_error(measured: ArrayLike, predicted: ArrayLike) -> NDArray[np.float64]:
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return np.abs(measured - predicted) / np.maximum(predicted, 1e-12)
