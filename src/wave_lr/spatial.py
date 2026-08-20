"""The spatial dual of the delay-occupancy law.

The same degree-of-freedom argument runs along either axis of a wave field:

    along frequency :  rank        ~ bandwidth   x  occupied delay support
    along space     :  sensors     ~ domain area x  occupied wavenumber support

The second form is what a sensing problem is governed by. A field whose energy
occupies ``M`` wavenumber cells cannot be determined by fewer than about ``M``
samples, whatever model is fitted to them -- that is an identifiability
statement, not a statement about any particular estimator. A carrier shrinks
the occupied wavenumber set from the propagation wavenumber ``omega / c`` down
to whatever the envelope varies at, which is why it changes how many sensors
are needed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def spatial_mode_count(
    field: NDArray[np.complex128], energy_fraction: float = 0.99
) -> float:
    """Number of wavenumber cells holding ``energy_fraction`` of the energy.

    ``field`` is a 2-D complex array on a regular grid. The count is the
    smallest number of cells of the discrete wavenumber grid whose combined
    power reaches the requested fraction, which is the field's degree-of-freedom
    count at that accuracy.
    """

    values = np.asarray(field)
    if values.ndim != 2:
        raise ValueError("field must be 2-D")
    power = np.abs(np.fft.fft2(values)) ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    ordered = np.sort(power.ravel())[::-1]
    cumulative = np.cumsum(ordered) / total
    return float(np.searchsorted(cumulative, energy_fraction) + 1)


def occupied_bandwidth(
    field: NDArray[np.complex128], energy_fraction: float = 0.99
) -> float:
    """Radius in cycles-per-pixel enclosing the occupied wavenumber set."""

    count = spatial_mode_count(field, energy_fraction)
    values = np.asarray(field)
    area = values.size
    return float(np.sqrt(count / area) * 0.5)


def infer_spacing(coords: NDArray[np.float64]) -> float:
    """Smallest positive step present in the coordinates, per axis.

    Cases loaded with a spatial stride carry the *original* spacing in their
    metadata; using it would leave empty rows in the reconstructed grid and
    manufacture high-wavenumber energy that is not in the field.
    """

    steps = []
    for axis in range(coords.shape[1]):
        unique = np.unique(coords[:, axis])
        if unique.size > 1:
            steps.append(float(np.min(np.diff(unique))))
    return min(steps) if steps else 1.0


def to_grid(
    values: NDArray[np.complex128],
    coords: NDArray[np.float64],
    spacing: float | None = None,
) -> NDArray[np.complex128]:
    """Scatter values back onto the regular grid they came from, zero-filling."""

    if spacing is None:
        spacing = infer_spacing(coords)
    rows = np.rint(coords[:, 0] / spacing).astype(int)
    cols = np.rint(coords[:, 1] / spacing).astype(int)
    rows -= rows.min()
    cols -= cols.min()
    grid = np.zeros((rows.max() + 1, cols.max() + 1), dtype=np.complex128)
    grid[rows, cols] = values
    return grid


def radial_power(field: NDArray[np.complex128]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Power and wavenumber magnitude (cycles per pixel) of a 2-D field."""

    values = np.asarray(field)
    power = np.abs(np.fft.fft2(values)) ** 2
    rows = np.fft.fftfreq(values.shape[0])[:, None]
    cols = np.fft.fftfreq(values.shape[1])[None, :]
    return power, np.sqrt(rows**2 + cols**2)


def aliased_energy(field: NDArray[np.complex128], fraction: float) -> float:
    """Energy fraction above the Nyquist wavenumber of a given sampling density.

    Sampling a fraction ``p`` of a 2-D grid corresponds to a mean sample spacing
    of ``1 / sqrt(p)`` pixels and therefore a Nyquist wavenumber of
    ``sqrt(p) / 2`` cycles per pixel. Whatever energy sits above that is not
    determined by the samples: no estimator can recover it, so its square root
    lower-bounds the achievable relative error.
    """

    power, wavenumber = radial_power(field)
    total = power.sum()
    if total <= 0:
        return 0.0
    cutoff = 0.5 * np.sqrt(fraction)
    return float(power[wavenumber > cutoff].sum() / total)


def identifiability_bound(field: NDArray[np.complex128], fraction: float) -> float:
    """Lower bound on relative reconstruction error at a sampling fraction."""

    return float(np.sqrt(aliased_energy(field, fraction)))


def required_fraction(
    field: NDArray[np.complex128], target: float, grid: int = 400
) -> float:
    """Smallest sampling fraction whose identifiability bound meets ``target``."""

    fractions = np.logspace(-4, 0, grid)
    power, wavenumber = radial_power(field)
    total = power.sum()
    if total <= 0:
        return float("nan")
    order = np.argsort(wavenumber.ravel())
    sorted_power = power.ravel()[order]
    sorted_k = wavenumber.ravel()[order]
    # Energy above each cutoff, from the cumulative sum below it.
    below = np.cumsum(sorted_power) / total
    for fraction in fractions:
        cutoff = 0.5 * np.sqrt(fraction)
        index = int(np.searchsorted(sorted_k, cutoff))
        outside = 1.0 - (below[index - 1] if index > 0 else 0.0)
        if np.sqrt(max(outside, 0.0)) <= target:
            return float(fraction)
    return float("nan")
