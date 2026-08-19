"""Controlled multipath frequency-domain fields with known delay statistics.

Fields follow the impulse-response model used throughout this repository,

    u(x, f) = sum_m A_m(x) exp(2 pi i f tau_m(x)),

so that every quantity entering the delay-spread rank law is known exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MultipathField:
    """A synthetic field together with the delays that generated it."""

    field: NDArray[np.complex128]  # (n_x, n_f)
    delays: NDArray[np.float64]  # (n_x, n_paths)
    amplitudes: NDArray[np.float64]  # (n_x, n_paths)
    frequencies: NDArray[np.float64]  # (n_f,)

    @property
    def bandwidth(self) -> float:
        return float(self.frequencies[-1] - self.frequencies[0])

    @property
    def first_arrival(self) -> NDArray[np.float64]:
        return self.delays.min(axis=1)

    @property
    def absolute_spread(self) -> float:
        """Support length of all arrival times pooled over space."""

        return float(self.delays.max() - self.delays.min())

    @property
    def delay_spread(self) -> float:
        """Largest per-location coda length (relative delay support)."""

        if self.delays.shape[1] == 1:
            return 0.0
        return float((self.delays.max(axis=1) - self.delays.min(axis=1)).max())


def make_multipath_field(
    n_x: int = 256,
    n_f: int = 256,
    f_min: float = 20.0,
    f_max: float = 60.0,
    n_paths: int = 1,
    absolute_spread: float = 0.4,
    delay_spread: float = 0.0,
    amplitude_decay: float = 0.6,
    amplitude_roughness: float = 0.15,
    seed: int = 0,
) -> MultipathField:
    """Build a field whose absolute and relative delay supports are prescribed.

    ``absolute_spread`` sets how far first arrivals vary across space and
    ``delay_spread`` sets the per-location coda length; both are exact by
    construction so the rank law can be checked without estimation error.
    """

    rng = np.random.default_rng(seed)
    frequencies = np.linspace(f_min, f_max, n_f)
    x = np.linspace(0.0, 1.0, n_x)

    # First arrivals sweep the prescribed absolute support monotonically.
    first = absolute_spread * (0.5 * (1.0 - np.cos(np.pi * x)))
    first = first - first.min()

    delays = np.empty((n_x, n_paths))
    delays[:, 0] = first
    if n_paths > 1:
        # Later arrivals fill [0, delay_spread] with location-dependent jitter.
        base = np.linspace(0.0, 1.0, n_paths)[1:]
        jitter = 0.15 * rng.standard_normal((n_x, n_paths - 1))
        extra = delay_spread * np.clip(base[None, :] + jitter * base[None, :], 0.0, 1.0)
        # Force the prescribed spread to be attained somewhere in the domain.
        extra[:, -1] = delay_spread * (0.85 + 0.15 * np.cos(3.0 * np.pi * x))
        delays[:, 1:] = first[:, None] + extra

    amplitudes = np.exp(-amplitude_decay * np.arange(n_paths))[None, :] * (
        1.0 + amplitude_roughness * np.cos(2.0 * np.pi * x)[:, None]
    )
    amplitudes = amplitudes / (1.0 + 0.5 * first)[:, None]

    # NumPy transform convention: an arrival at tau carries exp(-2 pi i f tau).
    phase = -2.0 * np.pi * frequencies[None, :, None] * delays[:, None, :]
    field = np.sum(amplitudes[:, None, :] * np.exp(1j * phase), axis=2)
    return MultipathField(field, delays, amplitudes, frequencies)
