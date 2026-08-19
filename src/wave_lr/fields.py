"""Dataset adapters producing a uniform ``WaveCase`` view.

Every case exposes real time traces at a set of spatial locations, a physical
coordinate for each location, and a *deployable* first-arrival carrier derived
from the medium rather than from the field being modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .eikonal import batched_travel_time

WELL_ACOUSTIC = Path(
    "/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/"
    "Geo-Aware-Tensor/data/the_well_acoustic_64x64"
)
OPENFWI = Path(
    "/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/openfwi_curvefault_a/raw"
)


@dataclass
class WaveCase:
    """One wave field with everything needed to build and test a carrier."""

    name: str
    dataset: str
    traces: NDArray[np.float64]  # (n_x, n_t)
    dt: float
    coords: NDArray[np.float64]  # (n_x, 2) physical positions
    travel_time: NDArray[np.float64]  # (n_x,) deployable eikonal first arrival
    straight_time: NDArray[np.float64]  # (n_x,) constant-speed control carrier
    metadata: dict = field(default_factory=dict)

    @property
    def n_x(self) -> int:
        return self.traces.shape[0]

    @property
    def duration(self) -> float:
        return self.traces.shape[1] * self.dt


def _grid_coords(n_r: int, n_c: int, spacing: float) -> NDArray[np.float64]:
    rows, cols = np.meshgrid(np.arange(n_r), np.arange(n_c), indexing="ij")
    return np.stack([rows.ravel() * spacing, cols.ravel() * spacing], axis=1)


def load_well_acoustic_case(path: Path, wall_fraction: float = 0.05) -> WaveCase:
    """The Well acoustic scattering maze: strong reverberation, known ``c(x)``.

    Sources are initial pressure pulses, so the source mask is read from the
    first frame. Wall cells carry no propagating energy and are dropped.
    """

    data = np.load(path)
    pressure = np.asarray(data["pressure"], dtype=np.float64)  # (n_t, n_r, n_c)
    speed = np.asarray(data["speed_of_sound"], dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64)
    spacing = float(data["x"][1] - data["x"][0])
    dt = float(times[1] - times[0])

    first = np.abs(pressure[0])
    source_mask = first > 0.05 * first.max()
    fluid = speed > wall_fraction * speed.max()

    tau = batched_travel_time(
        speed[None], source_mask[None], spacing=spacing, iterations=6 * speed.shape[0]
    )[0]
    coords = _grid_coords(*speed.shape, spacing)
    source_xy = coords.reshape(*speed.shape, 2)[source_mask]
    flat_coords = coords.reshape(-1, 2)
    distance = np.min(
        np.linalg.norm(flat_coords[:, None, :] - source_xy[None, :, :], axis=2), axis=1
    )
    keep = fluid.ravel()
    reference_speed = float(np.median(speed[fluid]))

    return WaveCase(
        name=path.stem,
        dataset="well_acoustic_maze",
        traces=pressure.reshape(len(times), -1).T[keep],
        dt=dt,
        coords=flat_coords[keep],
        travel_time=tau.ravel()[keep],
        straight_time=(distance / reference_speed)[keep],
        metadata={
            "spacing": spacing,
            "n_sources": int(source_mask.sum()),
            "reference_speed": reference_speed,
            "wall_fraction": float(1.0 - fluid.mean()),
            "grid": list(speed.shape),
        },
    )


def load_well_acoustic(split: str = "test", limit: int | None = None) -> list[WaveCase]:
    paths = sorted((WELL_ACOUSTIC / split).glob("trajectory_*.npz"))
    if limit is not None:
        paths = paths[:limit]
    return [load_well_acoustic_case(p) for p in paths]


OPENFWI_SOURCE_COLUMNS = (0, 17, 35, 52, 69)
OPENFWI_SPACING = 10.0
OPENFWI_DT = 1.0e-3


def load_openfwi(
    n_models: int = 32, shots: tuple[int, ...] = (2,), file_index: int = 0
) -> list[WaveCase]:
    """OpenFWI shot gathers.

    WARNING: the local copy pairs ``seis2_*`` with ``vel4_*`` files, which come
    from different OpenFWI families. The velocity models therefore do not
    describe the recorded gathers (see ``docs/DATA_INTEGRITY.md``), so no
    eikonal carrier is trustworthy here and ``travel_time`` is picked from the
    data itself. This dataset is used only as an estimated-carrier stress test.
    """

    seismic = np.load(OPENFWI / f"seis2_1_{file_index}.npy", mmap_mode="r")
    cases = []
    for model in range(n_models):
        for shot in shots:
            gather = np.asarray(seismic[model, shot], dtype=np.float64).T  # (n_rec, n_t)
            envelope = np.abs(gather)
            peak = envelope.max(axis=1, keepdims=True)
            picks = np.argmax(envelope > 0.05 * peak, axis=1) * OPENFWI_DT
            source_x = OPENFWI_SOURCE_COLUMNS[shot] * OPENFWI_SPACING
            coords = np.stack(
                [np.zeros(gather.shape[0]), np.arange(gather.shape[0]) * OPENFWI_SPACING],
                axis=1,
            )
            offset = np.abs(coords[:, 1] - source_x)
            apparent = np.polyfit(offset, picks, 1)[0]
            cases.append(
                WaveCase(
                    name=f"model{model:03d}_shot{shot}",
                    dataset="openfwi_gathers",
                    traces=gather,
                    dt=OPENFWI_DT,
                    coords=coords,
                    travel_time=picks,
                    straight_time=offset * apparent,
                    metadata={
                        "carrier_source": "data_picked_first_break",
                        "velocity_pairing": "broken",
                        "shot": shot,
                    },
                )
            )
    return cases


def fdtd_case(
    spec,
    peak_frequency: float = 12.0,
    duration: float = 6.0,
    record_every: int = 4,
    source_fraction: tuple[float, float] = (0.5, 0.28),
    interior_margin: float = 0.24,
) -> WaveCase:
    """Simulate one controlled regime and wrap it as a :class:`WaveCase`.

    Sponge cells are excluded from the recorded locations so that absorbed
    amplitudes never enter the rank diagnostics.
    """

    from .fdtd import build_medium, simulate

    speed, damping = build_medium(spec)
    n = spec.grid
    row = round(source_fraction[0] * (n - 1))
    col = round(source_fraction[1] * (n - 1))
    frames, record_dt, spacing = simulate(
        speed[None], damping[None], (row, col), peak_frequency, duration, record_every
    )
    margin = round(interior_margin * n)
    interior = np.zeros((n, n), dtype=bool)
    interior[margin : n - margin, margin : n - margin] = True

    source_mask = np.zeros((n, n), dtype=bool)
    source_mask[row, col] = True
    tau = batched_travel_time(
        speed[None], source_mask[None], spacing=spacing, iterations=6 * n
    )[0]
    coords = _grid_coords(n, n, spacing)
    distance = np.linalg.norm(coords - coords[row * n + col], axis=1)
    keep = interior.ravel()

    return WaveCase(
        name=spec.name,
        dataset="fdtd_regimes",
        traces=frames[0].reshape(frames.shape[1], -1).T[keep].astype(np.float64),
        dt=float(record_dt),
        coords=coords[keep],
        travel_time=tau.ravel()[keep],
        straight_time=distance[keep] / 1.0,
        metadata={
            "spacing": float(spacing),
            "peak_frequency": peak_frequency,
            "absorption": spec.absorption,
            "scatterer_fraction": spec.scatterer_fraction,
            "scatterer_contrast": spec.scatterer_contrast,
            "seed": spec.seed,
            "grid": n,
            "duration": duration,
            "source_xy": [float(row * spacing), float(col * spacing)],
            "box": [float((n - 1) * spacing), float((n - 1) * spacing)],
        },
    )


WELL_SCATTERING = Path(
    "/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/raw"
)


def load_well_scattering(
    variant: str = "acoustic_inclusions",
    chunk: str = "chunk_36.hdf5",
    limit: int = 12,
    stride: int = 2,
    wall_fraction: float = 0.05,
) -> list[WaveCase]:
    """Full-resolution Well acoustic scattering fields straight from HDF5.

    Unlike the locally cached 64x64 maze, these are 256x256 with two absorbing
    and two reflecting sides, which places them between the open and closed
    ends of the regime sweep. ``stride`` subsamples the spatial grid for the
    diagnostics; the time axis is always kept in full.
    """

    import h5py

    path = WELL_SCATTERING / variant / chunk
    cases = []
    with h5py.File(path, "r") as handle:
        times = handle["dimensions/time"][:]
        axis = handle["dimensions/x"][:]
        spacing = float(axis[1] - axis[0])
        dt = float(times[1] - times[0])
        for index in range(min(limit, handle["t0_fields/pressure"].shape[0])):
            pressure = np.asarray(
                handle["t0_fields/pressure"][index], dtype=np.float64
            )  # (n_t, n_r, n_c)
            speed = np.asarray(
                handle["t0_fields/speed_of_sound"][index], dtype=np.float64
            )
            first = np.abs(pressure[0])
            source_mask = first > 0.05 * first.max()
            fluid = speed > wall_fraction * speed.max()

            tau = batched_travel_time(
                speed[None], source_mask[None], spacing=spacing,
                iterations=6 * speed.shape[0],
            )[0]
            coords = _grid_coords(*speed.shape, spacing)
            source_xy = coords.reshape(*speed.shape, 2)[source_mask]
            distance = np.min(
                np.linalg.norm(coords[:, None, :] - source_xy[None, :, :], axis=2), axis=1
            )
            keep = fluid.ravel()
            if stride > 1:
                grid = np.zeros(speed.shape, dtype=bool)
                grid[::stride, ::stride] = True
                keep = keep & grid.ravel()
            reference_speed = float(np.median(speed[fluid]))

            cases.append(
                WaveCase(
                    name=f"{variant}_{index:03d}",
                    dataset=variant,
                    traces=pressure.reshape(len(times), -1).T[keep],
                    dt=dt,
                    coords=coords[keep],
                    travel_time=tau.ravel()[keep],
                    straight_time=(distance / reference_speed)[keep],
                    metadata={
                        "spacing": spacing,
                        "reference_speed": reference_speed,
                        "source_cells": int(source_mask.sum()),
                        "slow_fraction": float(1.0 - fluid.mean()),
                        "grid": list(speed.shape),
                        "stride": stride,
                    },
                )
            )
    return cases
