"""Batched 2-D acoustic FDTD used to sweep the delay-spread regime.

The public benchmarks in this repository sit at fixed points of the regime
diagram: the acoustic maze is fully reverberant, the Helmholtz staircase is
weakly scattering. To test *where* phase alignment starts and stops paying,
the regime itself has to be a controlled variable, so fields are generated
with a solver whose boundary absorption and scatterer density are dials.

Solved system (constant density, damped):

    p_tt + sigma(x) p_t = c(x)^2 laplacian(p) + s(t) delta(x - x_s)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MediumSpec:
    """One controlled regime: scattering strength and boundary absorption."""

    name: str
    grid: int = 128
    scatterer_fraction: float = 0.0
    scatterer_contrast: float = 0.5
    scatterer_radius: float = 0.045
    absorption: float = 40.0  # sponge strength; 0.0 gives a reverberant box
    sponge_width: float = 0.22
    seed: int = 0


def build_medium(spec: MediumSpec) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the speed map and the damping profile for one regime."""

    rng = np.random.default_rng(spec.seed)
    n = spec.grid
    axis = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    speed = np.ones((n, n))

    if spec.scatterer_fraction > 0:
        area = np.pi * spec.scatterer_radius**2
        count = max(round(spec.scatterer_fraction / area), 1)
        for _ in range(count):
            cx, cy = rng.uniform(0.1, 0.9, size=2)
            radius = spec.scatterer_radius * rng.uniform(0.6, 1.4)
            inside = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
            speed[inside] = 1.0 - spec.scatterer_contrast

    damping = np.zeros((n, n))
    if spec.absorption > 0 and spec.sponge_width > 0:
        edge = np.minimum(np.minimum(xx, 1.0 - xx), np.minimum(yy, 1.0 - yy))
        ramp = np.clip(1.0 - edge / spec.sponge_width, 0.0, 1.0)
        damping = spec.absorption * ramp**2
    return speed, damping


def ricker(times: NDArray[np.float64], peak_frequency: float) -> NDArray[np.float64]:
    delay = 1.2 / peak_frequency
    arg = (np.pi * peak_frequency * (times - delay)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def simulate(
    speeds: NDArray[np.float64],
    dampings: NDArray[np.float64],
    source_rc: tuple[int, int],
    peak_frequency: float,
    duration: float,
    record_every: int = 4,
    cfl: float = 0.4,
    device: str | None = None,
) -> tuple[NDArray[np.float32], float, float]:
    """Run a batch of simulations sharing one source and wavelet.

    ``speeds`` and ``dampings`` are ``(B, n, n)``. Returns recorded frames
    ``(B, n_rec, n, n)``, the recording interval and the grid spacing.
    """

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    c = torch.as_tensor(np.asarray(speeds, dtype=np.float32), device=device)
    sigma = torch.as_tensor(np.asarray(dampings, dtype=np.float32), device=device)
    if c.ndim == 2:
        c, sigma = c[None], sigma[None]
    _, n, _ = c.shape
    spacing = 1.0 / (n - 1)
    dt = cfl * spacing / float(np.max(speeds))
    n_steps = round(duration / dt)
    times = np.arange(n_steps) * dt
    wavelet = torch.as_tensor(ricker(times, peak_frequency).astype(np.float32), device=device)

    previous = torch.zeros_like(c)
    current = torch.zeros_like(c)
    coefficient = (c * dt / spacing) ** 2
    half = 0.5 * dt * sigma
    frames = []
    row, col = source_rc
    for step in range(n_steps):
        laplacian = torch.zeros_like(current)
        laplacian[:, 1:-1, 1:-1] = (
            current[:, 2:, 1:-1]
            + current[:, :-2, 1:-1]
            + current[:, 1:-1, 2:]
            + current[:, 1:-1, :-2]
            - 4.0 * current[:, 1:-1, 1:-1]
        )
        update = (
            2.0 * current
            - (1.0 - half) * previous
            + coefficient * laplacian
        )
        update[:, row, col] = update[:, row, col] + (dt**2) * wavelet[step]
        update = update / (1.0 + half)
        previous, current = current, update
        if step % record_every == 0:
            frames.append(current.clone())
    stacked = torch.stack(frames, dim=1).cpu().numpy()
    return stacked.astype(np.float32), dt * record_every, spacing
