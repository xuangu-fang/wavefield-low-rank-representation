"""Per-case representation analysis shared by every real-data experiment."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .diagnostics import singular_spectrum
from .fields import WaveCase
from .spectra import Spectrum, band_limited_traces, shift_spectrum, to_spectrum
from .theory import occupancy_from_traces, predicted_rank

RANKS = (1, 2, 4, 8, 16, 32)
ENERGY_LEVELS = (0.90, 0.99)


def pick_first_break(traces: NDArray[np.float64], dt: float, threshold: float = 0.05) -> NDArray[np.float64]:
    """Deployable data-driven carrier: first crossing of a relative envelope level."""

    envelope = np.abs(traces)
    peak = envelope.max(axis=1, keepdims=True)
    peak[peak == 0] = 1.0
    return np.argmax(envelope > threshold * peak, axis=1) * dt


def carrier_report(spectrum: Spectrum, delays: NDArray[np.float64] | None) -> dict:
    """Rank diagnostics and delay occupancy for one carrier choice.

    Advancing a trace by its own delay would wrap the record start around the
    padded time axis, so the occupancy is read from traces shifted by
    ``delays - delays.max()`` instead: a pure delay, which never wraps. A
    delay common to every location multiplies each frequency column by one
    unit scalar, so the singular values -- and therefore the ranks below --
    are identical either way.
    """

    shifted = spectrum if delays is None else shift_spectrum(spectrum, delays)
    for_traces = (
        spectrum if delays is None else shift_spectrum(spectrum, delays - delays.max())
    )
    traces, _ = band_limited_traces(for_traces)
    # One SVD serves every rank statistic below.
    spectrum_values = singular_spectrum(shifted.values)
    cumulative = np.cumsum(spectrum_values**2)

    report = {}
    for level in ENERGY_LEVELS:
        tag = int(level * 100)
        occupancy = occupancy_from_traces(
            traces, shifted.dt, bandwidth=shifted.bandwidth, energy_fraction=level
        )
        report[f"occupancy_{tag}"] = occupancy
        report[f"predicted_rank_{tag}"] = predicted_rank(shifted.bandwidth, occupancy)
        report[f"rank_{tag}"] = int(np.searchsorted(cumulative, level) + 1)
    report["rank_999"] = int(np.searchsorted(cumulative, 0.999) + 1)
    for rank in RANKS:
        report[f"residual_rank{rank}"] = float(np.linalg.norm(spectrum_values[rank:]))
    return report


def analyze_case(
    case: WaveCase,
    f_min: float,
    f_max: float,
    taper: float = 0.05,
    pad_factor: int = 2,
) -> dict:
    """Compare raw, eikonal, straight-ray and data-picked carriers on one case."""

    spectrum = to_spectrum(
        case.traces, case.dt, f_min, f_max, pad_factor=pad_factor, taper=taper
    )
    band_traces, _ = band_limited_traces(spectrum)
    carriers = {
        "raw": None,
        "eikonal": case.travel_time,
        "straight": case.straight_time,
        "data_pick": pick_first_break(band_traces, spectrum.dt),
    }
    result = {
        "case": case.name,
        "dataset": case.dataset,
        "f_min": float(f_min),
        "f_max": float(f_max),
        "bandwidth": spectrum.bandwidth,
        "n_x": case.n_x,
        "n_f": int(spectrum.values.shape[1]),
        "metadata": case.metadata,
    }
    for name, delays in carriers.items():
        for key, value in carrier_report(spectrum, delays).items():
            result[f"{name}_{key}"] = value
    for name in ("eikonal", "straight", "data_pick"):
        for level in ENERGY_LEVELS:
            tag = int(level * 100)
            result[f"{name}_measured_gain_{tag}"] = result[f"raw_rank_{tag}"] / max(
                result[f"{name}_rank_{tag}"], 1
            )
            result[f"{name}_predicted_gain_{tag}"] = (
                result[f"raw_predicted_rank_{tag}"] / result[f"{name}_predicted_rank_{tag}"]
            )
    return result
