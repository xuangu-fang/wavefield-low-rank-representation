"""Time-to-frequency transforms and carrier-aligned time realignment.

Demodulating a frequency-domain field by ``exp(2 pi i f tau(x))`` is exactly a
per-location time shift of the impulse response, so alignment and demodulation
are two views of one operation. Traces are zero-padded before the transform so
the shift never wraps around.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Spectrum:
    values: NDArray[np.complex128]  # (n_x, n_f) band-limited field
    frequencies: NDArray[np.float64]  # (n_f,)
    dt: float
    n_padded: int

    @property
    def bandwidth(self) -> float:
        return float(self.frequencies[-1] - self.frequencies[0])


def tukey(n: int, alpha: float = 0.1) -> NDArray[np.float64]:
    """Tapered window that suppresses truncation ringing at the record ends."""

    if alpha <= 0:
        return np.ones(n)
    window = np.ones(n)
    edge = int(np.floor(alpha * (n - 1) / 2.0))
    if edge < 1:
        return window
    ramp = 0.5 * (1.0 + np.cos(np.pi * (np.arange(edge) / edge - 1.0)))
    window[:edge] = ramp
    window[n - edge :] = ramp[::-1]
    return window


def to_spectrum(
    traces: ArrayLike,
    dt: float,
    f_min: float,
    f_max: float,
    pad_factor: int = 2,
    taper: float = 0.1,
) -> Spectrum:
    """Band-limited complex spectrum of real traces ``(n_x, n_t)``."""

    data = np.asarray(traces, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("traces must be (n_x, n_t)")
    n_t = data.shape[1]
    windowed = data * tukey(n_t, taper)[None, :]
    n_padded = int(pad_factor * n_t)
    if n_padded <= n_t:
        raise ValueError("pad_factor must exceed 1 so that shifts cannot wrap")
    spectrum = np.fft.rfft(windowed, n=n_padded, axis=1)
    frequencies = np.fft.rfftfreq(n_padded, dt)
    keep = (frequencies >= f_min) & (frequencies <= f_max)
    if keep.sum() < 2:
        raise ValueError("selected band contains fewer than two frequencies")
    return Spectrum(spectrum[:, keep], frequencies[keep], float(dt), n_padded)


def band_limited_traces(spectrum: Spectrum) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Invert a band-limited spectrum back to traces on the padded time axis."""

    full = np.zeros(
        (spectrum.values.shape[0], spectrum.n_padded // 2 + 1), dtype=np.complex128
    )
    all_freq = np.fft.rfftfreq(spectrum.n_padded, spectrum.dt)
    index = np.searchsorted(all_freq, spectrum.frequencies[0])
    full[:, index : index + spectrum.values.shape[1]] = spectrum.values
    traces = np.fft.irfft(full, n=spectrum.n_padded, axis=1)
    times = np.arange(spectrum.n_padded) * spectrum.dt
    return traces, times


def carrier(frequencies: ArrayLike, delays: ArrayLike) -> NDArray[np.complex128]:
    """Phase carrier ``exp(-2 pi i f tau)`` in the NumPy transform convention.

    ``numpy.fft.rfft`` uses ``exp(-2 pi i f t)``, so a trace whose arrival sits
    at ``tau`` carries the factor ``exp(-2 pi i f tau)``. Alignment therefore
    multiplies by the conjugate.
    """

    f = np.asarray(frequencies, dtype=np.float64)
    tau = np.asarray(delays, dtype=np.float64)
    return np.exp(-2j * np.pi * f[None, :] * tau[:, None])


def shift_spectrum(spectrum: Spectrum, delays: ArrayLike) -> Spectrum:
    """Align every location to zero delay: ``u -> u exp(+2 pi i f tau)``."""

    tau = np.asarray(delays, dtype=np.float64)
    if tau.shape[0] != spectrum.values.shape[0]:
        raise ValueError("one delay per location is required")
    ramp = np.conj(carrier(spectrum.frequencies, tau))
    return Spectrum(spectrum.values * ramp, spectrum.frequencies, spectrum.dt, spectrum.n_padded)
