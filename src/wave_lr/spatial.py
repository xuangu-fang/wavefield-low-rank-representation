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


def largest_full_rectangle(
    coords: NDArray[np.float64], spacing: float | None = None
) -> NDArray[np.bool_]:
    """Mask selecting the largest axis-aligned block containing no missing cells.

    A masked domain -- a wall, an obstacle -- must be cropped rather than
    zero-filled before the transform: zero-filling inserts a step edge that is
    not in the field, and the spurious high-wavenumber energy inflates the
    bound until the measured error appears to beat it.

    Only one axis usually admits a full slab (a corrugated wall spans every
    column but leaves whole rows clear), so both orientations are tried and the
    larger block wins. The result is verified to be exactly rectangular.
    """

    if spacing is None:
        spacing = infer_spacing(coords)
    rows = np.rint(coords[:, 0] / spacing).astype(int)
    cols = np.rint(coords[:, 1] / spacing).astype(int)
    rows -= rows.min()
    cols -= cols.min()
    present = np.zeros((rows.max() + 1, cols.max() + 1), dtype=bool)
    present[rows, cols] = True

    def longest_run(flags: NDArray[np.bool_]) -> NDArray[np.int_]:
        best, current = [], []
        for index, flag in enumerate(flags):
            if flag:
                current.append(index)
            else:
                if len(current) > len(best):
                    best = current
                current = []
        return np.array(best if len(best) >= len(current) else current, dtype=int)

    row_slab = longest_run(present.all(axis=1))  # rows clear across every column
    col_slab = longest_run(present.all(axis=0))  # columns clear across every row
    by_rows = (row_slab, np.arange(present.shape[1])) if row_slab.size else None
    by_cols = (np.arange(present.shape[0]), col_slab) if col_slab.size else None
    candidates = [c for c in (by_rows, by_cols) if c is not None]
    if not candidates:
        return np.ones(len(coords), dtype=bool)
    keep_rows, keep_cols = max(candidates, key=lambda c: c[0].size * c[1].size)

    mask = np.isin(rows, keep_rows) & np.isin(cols, keep_cols)
    assert mask.sum() == keep_rows.size * keep_cols.size, "crop is not rectangular"
    return mask


DEFAULT_TAPER = 0.25


def taper_window(shape: tuple[int, int], alpha: float = DEFAULT_TAPER) -> NDArray[np.float64]:
    """Separable Tukey window over a 2-D block.

    A field that does not happen to be periodic across the block leaks a 1/k
    tail into the transform, and for a smooth, heavily oversampled field that
    leakage *is* the entire apparent out-of-band energy -- enough to make the
    bound exceed the error it is supposed to lower-bound. Tapering the outer
    quarter removes it. The same window weights the measured error, so
    prediction and measurement are defined on identical footing.
    """

    from .spectra import tukey

    return tukey(shape[0], alpha)[:, None] * tukey(shape[1], alpha)[None, :]


def radial_power(
    field: NDArray[np.complex128], taper: float = DEFAULT_TAPER
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Power and wavenumber magnitude (cycles per pixel) of a 2-D field."""

    values = np.asarray(field)
    if taper > 0:
        values = values * taper_window(values.shape, taper)
    power = np.abs(np.fft.fft2(values)) ** 2
    rows = np.fft.fftfreq(values.shape[0])[:, None]
    cols = np.fft.fftfreq(values.shape[1])[None, :]
    return power, np.sqrt(rows**2 + cols**2)


def aliased_energy(
    field: NDArray[np.complex128], fraction: float, taper: float = DEFAULT_TAPER
) -> float:
    """Energy fraction above the Nyquist wavenumber of a given sampling density.

    Sampling a fraction ``p`` of a 2-D grid corresponds to a mean sample spacing
    of ``1 / sqrt(p)`` pixels and therefore a Nyquist wavenumber of
    ``sqrt(p) / 2`` cycles per pixel. Whatever energy sits above that is not
    determined by the samples: no estimator can recover it, so its square root
    lower-bounds the achievable relative error.
    """

    power, wavenumber = radial_power(field, taper)
    total = power.sum()
    if total <= 0:
        return 0.0
    cutoff = 0.5 * np.sqrt(fraction)
    return float(power[wavenumber > cutoff].sum() / total)


def identifiability_bound(
    field: NDArray[np.complex128], fraction: float, taper: float = DEFAULT_TAPER
) -> float:
    """Lower bound on relative reconstruction error at a sampling fraction."""

    return float(np.sqrt(aliased_energy(field, fraction, taper)))


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


def block_weights(
    coords: NDArray[np.float64],
    spacing: float | None = None,
    taper: float = DEFAULT_TAPER,
) -> NDArray[np.float64]:
    """Per-location weights matching the taper used inside the bound."""

    if spacing is None:
        spacing = infer_spacing(coords)
    rows = np.rint(coords[:, 0] / spacing).astype(int)
    cols = np.rint(coords[:, 1] / spacing).astype(int)
    rows -= rows.min()
    cols -= cols.min()
    window = taper_window((rows.max() + 1, cols.max() + 1), taper)
    return window[rows, cols]
