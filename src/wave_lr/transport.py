"""Applying the same measurement to fields that are not waves.

Nothing in the degree-of-freedom argument mentions waves: the rank of an
``(x, f)`` unfolding is set by bandwidth times the measure of the time axis
that carries energy, whatever produced the field. What *is* wave-specific is
where the alignment delays come from. Here they are estimated from the data by
cross-correlation, so advection-dominated flows can be measured on exactly the
same footing as wave fields.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .fields import WaveCase, _grid_coords

DYNAMICS = Path("/mnt/data/xuangu-fang/ai-physical-dynamics/datasets")
PDEBENCH = Path("/mnt/data/xuangu-fang/operator-spectral-funbat/pdebench")


def estimate_delays_by_correlation(
    traces: NDArray[np.float64],
    reference: int | None = None,
    max_lag_fraction: float = 0.25,
) -> NDArray[np.float64]:
    """Per-location time shift aligning each trace to a reference, in samples.

    This is the transport-agnostic counterpart of a first-arrival pick: it asks
    only "how far is this location's history delayed relative to that one",
    which is meaningful for an advected structure as much as for an arrival.

    The lag of a periodic signal is only defined modulo its period, so the
    search is restricted to ``max_lag_fraction`` of the record; without that
    bound neighbouring locations can pick lags a whole period apart and the
    delay field comes out discontinuous.
    """

    data = np.asarray(traces, dtype=np.float64)
    data = data - data.mean(axis=1, keepdims=True)
    energy = (data**2).sum(axis=1)
    if reference is None:
        reference = int(np.argmax(energy))
    n_t = data.shape[1]
    padded = 2 * n_t
    spectrum = np.fft.rfft(data, n=padded, axis=1)
    correlation = np.fft.irfft(
        spectrum * np.conj(spectrum[reference]), n=padded, axis=1
    )
    limit = max(int(max_lag_fraction * n_t), 1)
    allowed = np.zeros(padded, dtype=bool)
    allowed[: limit + 1] = True
    allowed[-limit:] = True
    # Mask only the peak search; the refinement below must read real values,
    # or a masked neighbour turns the parabolic offset into a NaN.
    masked = np.where(allowed[None, :], correlation, -np.inf)
    peak = np.argmax(masked, axis=1)
    index = np.arange(len(peak))
    left = correlation[index, (peak - 1) % padded]
    centre = correlation[index, peak]
    right = correlation[index, (peak + 1) % padded]
    denominator = left - 2.0 * centre + right
    offset = 0.5 * (left - right) / np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    offset = np.clip(np.nan_to_num(offset), -0.5, 0.5)
    lag = np.where(peak > padded // 2, peak - padded, peak) + offset
    return -np.nan_to_num(lag).astype(np.float64)


def fit_uniform_advection(
    coords: NDArray[np.float64], delays: NDArray[np.float64], weights=None
) -> NDArray[np.float64]:
    """Least-squares plane through the delays: a constant-velocity prior.

    The counterpart of the eikonal carrier for a flow -- a single advection
    velocity explains the whole delay field if transport is uniform.
    """

    design = np.column_stack([coords, np.ones(len(coords))])
    if weights is not None:
        scale = np.sqrt(np.asarray(weights))[:, None]
        coefficients, *_ = np.linalg.lstsq(design * scale, delays * scale[:, 0], rcond=None)
    else:
        coefficients, *_ = np.linalg.lstsq(design, delays, rcond=None)
    return design @ coefficients


def estimate_delays_by_peak(traces: NDArray[np.float64]) -> NDArray[np.float64]:
    """Envelope-peak time per location, in samples.

    For a structure that passes through once, the peak is unambiguous and can
    span the whole record; bounded cross-correlation would clip exactly those
    delays. Which estimator is right is itself decidable without labels, by
    taking whichever leaves the smaller relative occupancy.
    """

    data = np.asarray(traces, dtype=np.float64)
    data = data - data.mean(axis=1, keepdims=True)
    envelope = np.abs(data)
    return np.argmax(envelope, axis=1).astype(np.float64)


def choose_delays(
    traces: NDArray[np.float64], dt: float, band: tuple[float, float]
) -> tuple[NDArray[np.float64], str]:
    """Pick the delay estimator that minimises the relative delay occupancy."""

    from .spectra import band_limited_traces, shift_spectrum, to_spectrum
    from .theory import occupancy_from_traces

    spectrum = to_spectrum(traces, dt, *band)
    candidates = {
        "correlation": estimate_delays_by_correlation(traces),
        "peak": estimate_delays_by_peak(traces) * dt,
    }
    best = None
    for label, delays in candidates.items():
        shifted = shift_spectrum(spectrum, delays - delays.max())
        aligned, _ = band_limited_traces(shifted)
        occupancy = occupancy_from_traces(aligned, spectrum.dt, spectrum.bandwidth)
        if best is None or occupancy < best[2]:
            best = (label, delays, occupancy)
    return best[1], best[0]


def _wrap(name: str, block: NDArray[np.float64], index: int) -> WaveCase:
    """Wrap ``(n_t, H, W)`` or ``(n_t, W)`` as a case with estimated delays."""

    block = np.asarray(block, dtype=np.float64)
    if block.ndim == 2:
        block = block[:, None, :]
    n_t, height, width = block.shape
    traces = block.reshape(n_t, -1).T
    # Time is measured in samples: the law involves the dimensionless product
    # of bandwidth and occupancy, so no physical dt calibration is needed.
    coords = _grid_coords(height, width, 1.0)
    delays, estimator = choose_delays(traces, 1.0, (0.02, 0.10))
    energy = (traces**2).sum(axis=1)
    return WaveCase(
        name=f"{name}_{index:03d}",
        dataset=name,
        traces=traces,
        dt=1.0,
        coords=coords,
        travel_time=delays - delays.min(),
        straight_time=fit_uniform_advection(coords, delays, energy)
        - delays.min(),
        metadata={
            "spacing": 1.0,
            "grid": [height, width],
            "n_t": int(n_t),
            "delay_estimator": estimator,
        },
    )


def load_transport_cases(name: str, limit: int = 4) -> list[WaveCase]:
    """Adapters for the locally available non-wave spatiotemporal datasets."""

    if name == "kolmogorov_flow":
        data = np.load(DYNAMICS / "kolmogorov_mno/re40_ladder_v2_stride1/sealed_test.npz")
        fields = data["fields"][:limit]
    elif name == "kuramoto_sivashinsky":
        data = np.load(DYNAMICS / "ks_forecast_object/v4_seed20260817/sealed_test.npz")
        fields = data["fields"][:limit]
    elif name == "cylinder_wake":
        data = np.load(
            DYNAMICS / "realpde_cylinder_subset/prepared/benchmark_r64.npz", allow_pickle=True
        )
        fields = data["test_fields"][:limit, :, 0]
    elif name == "active_matter":
        data = np.load(DYNAMICS / "active_matter_multi/benchmark_strict_r48.npz")
        fields = data["test_fields"][:limit]
    elif name == "diffusion_reaction":
        import h5py

        with h5py.File(PDEBENCH / "2D_diff-react_NA_NA.h5", "r") as handle:
            keys = sorted(handle.keys())[:limit]
            fields = np.stack([handle[key]["data"][:, ::2, ::2, 0] for key in keys])
    else:
        raise ValueError(f"unknown transport dataset {name}")
    return [_wrap(name, fields[index], index) for index in range(len(fields))]


def advection_diffusion(
    grid: int = 64,
    n_t: int = 128,
    velocity: tuple[float, float] = (0.55, 0.35),
    diffusivity: float = 0.0,
    width: float = 0.05,
    seed: int = 0,
) -> NDArray[np.float64]:
    """A transient, non-wave transported field: a blob advected and spread.

    The five recorded flows are statistically stationary, so their energy fills
    the record and the occupancy bound is loose by construction. This family is
    the missing control: transport without wave physics, and *transient*, with
    ``diffusivity`` tuning how coherent the transport stays.
    """

    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, grid)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    start = np.array([0.15, 0.2]) + 0.05 * rng.standard_normal(2)
    times = np.linspace(0.0, 1.0, n_t)
    frames = []
    for time in times:
        centre = start + np.array(velocity) * time
        spread = width**2 + 2.0 * diffusivity * time
        amplitude = width**2 / spread
        frames.append(
            amplitude
            * np.exp(-((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2.0 * spread))
        )
    return np.stack(frames)


def load_synthetic_transport(
    name: str = "advection_diffusion", limit: int = 4, diffusivity: float = 0.0
) -> list[WaveCase]:
    """Wrap the synthetic transported blobs with the same estimated delays."""

    return [
        _wrap(name, advection_diffusion(diffusivity=diffusivity, seed=index), index)
        for index in range(limit)
    ]
