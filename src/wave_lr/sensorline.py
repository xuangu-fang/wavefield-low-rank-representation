"""Identifiability criterion and reconstruction on a regular sensor line.

The two-dimensional version in :mod:`wave_lr.spatial` assumes a full grid. Most
measured physics arrives on a *line* of sensors -- a receiver cable, a PIV
centreline -- so the criterion is restated here for one sensor axis, with the
same three disciplines: a regular array (never random sampling, which turns the
bound into a compressed-sensing question), a cropped rather than zero-filled
support, and one taper applied identically to prediction and measurement.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .spectra import tukey

DEFAULT_TAPER = 0.25


def tapered(spectrum, taper: float = DEFAULT_TAPER):
    """The single object both the bound and the estimator are defined on.

    Applying the window inside the bound but only as an error weight inside the
    estimator lets the estimator beat the bound: it is scored on the edges the
    bound still counts. Tapering once, up front, removes the loophole -- after
    this call, prediction and measurement see literally the same signal.
    """

    return spectrum * tukey(spectrum.shape[-1], taper)


def aliased_energy_line(
    spectrum: NDArray[np.complex128],
    weights: NDArray[np.float64],
    fraction: float,
    taper: float = DEFAULT_TAPER,
) -> float:
    """Energy fraction above the sensor-line Nyquist, weighted over frequency.

    Keeping a fraction ``p`` of a regular line means a stride of ``1 / p``
    sensors and a Nyquist wavenumber of ``p / 2`` cycles per sensor. Energy
    above it is not determined by the samples at any frequency.
    """

    power = np.abs(np.fft.fft(tapered(spectrum, taper), axis=-1)) ** 2
    wavenumber = np.abs(np.fft.fftfreq(spectrum.shape[-1]))
    outside = power[:, wavenumber > 0.5 * fraction].sum(1)
    total = power.sum(1)
    share = np.where(total > 0, outside / np.maximum(total, 1e-300), 0.0)
    return float((weights * share).sum())


def _relative(values, estimate, weights) -> float:
    """Energy-weighted relative error, on whatever footing the caller set up."""

    residual = (np.abs(values - estimate) ** 2).sum(1)
    reference = (np.abs(values) ** 2).sum(1)
    share = np.where(reference > 0, residual / np.maximum(reference, 1e-300), 0.0)
    return float(np.sqrt((weights * share).sum()))


def bound_line(spectrum, weights, fraction, taper=DEFAULT_TAPER) -> float:
    """Lower bound on relative error for any estimator at this sensor density."""

    return float(np.sqrt(aliased_energy_line(spectrum, weights, fraction, taper)))


def apply_warp(spectrum, freqs, delays) -> NDArray[np.complex128]:
    """Reparameterise the sensor axis by a per-sensor delay."""

    return spectrum * np.exp(2j * np.pi * freqs[:, None] * delays[None, :])


def transport_delays(coords, source, speed) -> NDArray[np.float64]:
    """The one-parameter transport warp: distance from the origin over a speed."""

    return np.abs(coords - source) / speed


def scan_speed(case, fraction, speeds, taper=DEFAULT_TAPER):
    """Best transport warp at this sensor density, and what it buys.

    Scanning one scalar is deliberately the weakest possible warp family: any
    gain it finds is a lower bound on what a learned reparameterisation can do,
    and the speed it lands on is checkable against the physics.
    """

    raw = bound_line(case.spectrum, case.weights, fraction, taper)
    best_speed, best = float("nan"), raw
    for speed in speeds:
        delays = transport_delays(case.coords, case.source, speed)
        value = bound_line(
            apply_warp(case.spectrum, case.freqs, delays), case.weights, fraction, taper
        )
        if value < best:
            best_speed, best = float(speed), value
    return raw, best_speed, best


def reconstruct_line(spectrum, weights, stride, delays=None, freqs=None, taper=DEFAULT_TAPER):
    """Relative error of linear interpolation from every ``stride``-th sensor.

    A zero-parameter estimator on purpose: the claim under test is about what
    the samples determine, so the reference has to be something that cannot be
    accused of under-fitting the training set.
    """

    values = spectrum if delays is None else apply_warp(spectrum, freqs, delays)
    values = tapered(values, taper)
    n_sensor = values.shape[-1]
    index = np.arange(0, n_sensor, stride)
    if index[-1] != n_sensor - 1:
        index = np.append(index, n_sensor - 1)
    grid = np.arange(n_sensor)
    estimate = np.empty_like(values)
    for part in (values.real, values.imag):
        interp = np.stack([np.interp(grid, index, row[index]) for row in part])
        if part is values.real:
            estimate.real = interp
        else:
            estimate.imag = interp
    return _relative(values, estimate, weights)


def reconstruct_bandlimited(
    spectrum, weights, fraction, delays=None, freqs=None, taper=DEFAULT_TAPER, ridge=1e-8
):
    """Least-squares fit of the resolvable band to a regular subset of sensors.

    Linear interpolation leaves a smoothing error on top of the aliasing error,
    so it sits far above the bound and cannot say whether the bound is tight.
    This estimator solves for exactly the Fourier coefficients the sampling can
    determine, which is the estimator the bound is written about. It still has
    no trained parameters.
    """

    values = spectrum if delays is None else apply_warp(spectrum, freqs, delays)
    values = tapered(values, taper)
    n_sensor = values.shape[-1]
    stride = max(round(1.0 / fraction), 1)
    index = np.arange(0, n_sensor, stride)
    n_mode = max(int(np.floor(fraction * n_sensor / 2.0)) * 2 + 1, 1)
    modes = np.fft.fftfreq(n_sensor)[np.argsort(np.abs(np.fft.fftfreq(n_sensor)))][:n_mode]
    design = np.exp(2j * np.pi * np.outer(index, modes))
    gram = design.conj().T @ design
    gram.flat[:: n_mode + 1] += ridge * np.trace(gram).real / max(n_mode, 1)
    coeffs = np.linalg.solve(gram, design.conj().T @ values[:, index].T)
    estimate = (np.exp(2j * np.pi * np.outer(np.arange(n_sensor), modes)) @ coeffs).T
    return _relative(values, estimate, weights)
