"""Frequency-domain (harmonic) datasets: one complex field per frequency.

The Well's ``helmholtz_staircase`` solves ``-(laplacian + omega^2) u = delta_x0``
with a sound-hard staircase boundary, for 16 frequencies and a set of point
source positions. Because the same geometry is solved at every frequency, it
is the natural place to test what a carrier does to the *frequency* axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .eikonal import batched_travel_time

STAIRCASE_ROOT = Path(
    "/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/raw"
)
SOURCE_X = np.round(np.arange(-0.4, 0.41, 0.1), 3)
SOURCE_Y = np.round(np.arange(-0.2, 0.41, 0.1), 3)


@dataclass
class HarmonicCase:
    """One source configuration observed at several frequencies."""

    name: str
    dataset: str
    fields: NDArray[np.complex128]  # (n_x, n_omega)
    omegas: NDArray[np.float64]
    coords: NDArray[np.float64]  # (n_x, 2)
    travel_time: NDArray[np.float64]
    straight_time: NDArray[np.float64]
    metadata: dict = field(default_factory=dict)

    @property
    def frequencies(self) -> NDArray[np.float64]:
        """Ordinary frequencies matching the ``exp(-2 pi i f tau)`` convention."""

        return self.omegas / (2.0 * np.pi)


def _snap(value: float, allowed: NDArray[np.float64]) -> float:
    return float(allowed[np.argmin(np.abs(allowed - value))])


def load_staircase(
    split: str = "test", limit: int | None = None, subsample: int = 1
) -> list[HarmonicCase]:
    """Load every frequency of the staircase dataset for each source position.

    The stored field uses the ``exp(-i omega t)`` time convention, so an
    outgoing arrival carries ``exp(+i omega tau)``. It is conjugated on load so
    that every downstream routine can assume the NumPy transform convention.
    """

    import h5py

    root = STAIRCASE_ROOT / (
        "helmholtz_staircase" if split == "test" else "helmholtz_staircase_train"
    )
    paths = sorted(root.glob("helmholtz_staircase_omega_*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no staircase files under {root}")

    omegas, stacks, wall, spacing = [], [], None, None
    for path in paths:
        with h5py.File(path, "r") as handle:
            omegas.append(float(handle["scalars/omega"][()]))
            real = handle["t0_fields/pressure_re"][:, 0]
            imaginary = handle["t0_fields/pressure_im"][:, 0]
            if wall is None:
                wall = handle["boundary_conditions/xy_wall/mask"][:]
                x_axis = handle["dimensions/x"][:]
                y_axis = handle["dimensions/y"][:]
                spacing = float(x_axis[1] - x_axis[0])
            # Conjugate into the exp(-2 pi i f tau) convention used repository-wide.
            stacks.append(np.conj(real + 1j * imaginary))
    order = np.argsort(omegas)
    omegas = np.asarray(omegas)[order]
    fields = np.stack([stacks[i] for i in order], axis=1)  # (n_traj, n_omega, nx, ny)

    n_traj = fields.shape[0]
    if limit is not None:
        n_traj = min(n_traj, limit)
    n_x, n_y = wall.shape
    fluid = ~wall.astype(bool)

    # The stored y axis is reversed with respect to ``dimensions/y``.
    grid_x = x_axis[:, None] * np.ones((1, n_y))
    grid_y = y_axis[-1] - np.arange(n_y)[None, :] * spacing * np.ones((n_x, 1))
    coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    keep = fluid.ravel()
    if subsample > 1:
        stride = np.zeros(n_x * n_y, dtype=bool)
        stride.reshape(n_x, n_y)[::subsample, ::subsample] = True
        keep = keep & stride

    speed = np.where(fluid, 1.0, 1e-3)
    cases = []
    for index in range(n_traj):
        # The source singularity dominates at the lowest frequency; at high
        # frequency the global maximum sits on a trapped surface mode instead.
        amplitude = np.abs(fields[index, 0])
        peak = np.unravel_index(int(np.argmax(np.where(fluid, amplitude, 0.0))), (n_x, n_y))
        source_xy = (
            _snap(coords.reshape(n_x, n_y, 2)[peak][0], SOURCE_X),
            _snap(coords.reshape(n_x, n_y, 2)[peak][1], SOURCE_Y),
        )
        mask = np.zeros((n_x, n_y), dtype=bool)
        mask[peak] = True
        tau = batched_travel_time(
            speed[None], mask[None], spacing=spacing, iterations=3 * (n_x + n_y)
        )[0]
        distance = np.linalg.norm(coords - coords[peak[0] * n_y + peak[1]], axis=1)
        cases.append(
            HarmonicCase(
                name=f"{split}_source{index}",
                dataset="helmholtz_staircase",
                fields=fields[index].reshape(len(omegas), -1).T[keep],
                omegas=omegas,
                coords=coords[keep],
                travel_time=tau.ravel()[keep],
                straight_time=distance[keep],
                metadata={
                    "spacing": spacing,
                    "source_index": peak,
                    "source_xy": source_xy,
                    "n_omega": len(omegas),
                    "wall_fraction": float(wall.mean()),
                    "subsample": subsample,
                },
            )
        )
    return cases
