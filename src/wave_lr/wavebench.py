"""WaveBench time-harmonic adapter.

The archive ships heterogeneous wavespeed maps paired with the complex
Helmholtz solution at a fixed point source, one file per frequency. The
absolute frequency and grid spacing are not recorded in the container, but only
their product enters the carrier: the per-pixel phase advance is
``kappa / c(x)`` with ``kappa = omega * spacing``. That single scalar is
calibrated from the field's own phase gradient rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .beton import BetonReader
from .eikonal import batched_travel_time

ROOT = Path(
    "/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/raw/wavebench"
)
SOURCE_INDEX = (1, 62)


@dataclass
class WaveBenchSample:
    index: int
    omega_label: int
    speed: NDArray[np.float64]
    field: NDArray[np.complex128]
    travel_time: NDArray[np.float64]  # grid units, spacing = 1
    kappa: float

    @property
    def wavelength_pixels(self) -> float:
        """Mean wavelength in pixels: ``2 pi c / kappa``."""

        return float(2.0 * np.pi * self.speed.mean() / self.kappa)

    @property
    def phase(self) -> NDArray[np.float64]:
        return self.kappa * self.travel_time


def calibrate_kappa(field: NDArray[np.complex128], speed: NDArray[np.float64]) -> float:
    """Estimate ``omega * spacing`` from the measured per-pixel phase advance."""

    unwrapped = np.unwrap(np.angle(field), axis=1)
    gradient = np.abs(np.gradient(unwrapped, axis=1))
    strong = np.abs(field) > 0.2 * np.abs(field).mean()
    return float(np.median((gradient * speed)[strong]))


def load_samples(omega_label: int, count: int = 24) -> list[WaveBenchSample]:
    """Read the first ``count`` readable samples of one frequency file."""

    reader = BetonReader(ROOT / f"isotropic_{omega_label}.beton")
    available = reader.readable_samples()[:count]
    grid = reader.fields[0]["shape"][-1]
    source = np.zeros((1, grid, grid), dtype=bool)
    source[0, SOURCE_INDEX[0], SOURCE_INDEX[1]] = True

    samples = []
    for index in available:
        record = reader.read(int(index))
        speed = record["input"][0].astype(np.float64)
        field = (record["target"][0] + 1j * record["target"][1]).astype(np.complex128)
        travel = batched_travel_time(
            speed[None], source, spacing=1.0, iterations=6 * grid
        )[0]
        samples.append(
            WaveBenchSample(
                index=int(index),
                omega_label=omega_label,
                speed=speed,
                field=field,
                travel_time=travel,
                kappa=calibrate_kappa(field, speed),
            )
        )
    return samples


def coordinates(grid: int) -> NDArray[np.float64]:
    rows, cols = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    return np.stack([rows.ravel(), cols.ravel()], axis=1).astype(np.float64)
