"""Physics-informed, self-supervised learning of the alignment coordinates.

The eikonal carrier is one way to obtain the warp; it is exact only for
non-dispersive first arrivals and needs the medium. This module learns the
phase field instead, from a single objective:

    min_theta  ||U . exp(+i phi_theta)||_*  +  lambda * || |grad tau_theta| - 1/c ||^2
               ^ nuclear norm: how low-rank the aligned field is
                                              ^ eikonal residual: the physics prior

Pointwise multiplication by a unimodular factor leaves the Frobenius norm of
``U`` unchanged, so minimising the nuclear norm at fixed Frobenius norm is
exactly a push towards low rank -- the objective is well posed and needs no
labels. ``lambda -> infinity`` recovers the eikonal carrier; ``lambda = 0`` is
purely self-supervised; in between the physics acts as a prior. Allowing the
phase to depart from ``2 pi f tau(x)`` additionally covers dispersion, which a
travel-time carrier cannot express.

The nuclear norm is only a surrogate for low rank, and minimising it can trade
against the quantity actually reported -- the error of the best rank-``R``
approximation. ``objective="tail"`` therefore optimises that error directly,
which is equally differentiable through the singular values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CarrierConfig:
    features: int = 64
    feature_scale: float = 3.0
    width: int = 128
    depth: int = 3
    dispersive: bool = False
    dispersion_rank: int = 2
    steps: int = 600
    learning_rate: float = 3e-4
    objective: str = "tail"
    budget: int = 16
    physics_weight: float = 0.0
    warmup_steps: int = 300
    seed: int = 0


def _mlp(config: CarrierConfig, n_out: int, generator, device):
    import torch
    from torch import nn

    class Features(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            matrix = torch.randn(2, config.features, generator=generator) * config.feature_scale
            self.register_buffer("matrix", matrix)

        def forward(self, coords):
            projected = 2.0 * np.pi * coords @ self.matrix
            return torch.cat([coords, torch.sin(projected), torch.cos(projected)], dim=-1)

    layers: list[nn.Module] = [Features()]
    in_dim = 2 + 2 * config.features
    for index in range(config.depth):
        layers.append(nn.Linear(in_dim if index == 0 else config.width, config.width))
        layers.append(nn.GELU())
    layers.append(nn.Linear(config.width, n_out))
    return nn.Sequential(*layers).to(device)


class LearnedCarrier:
    """A coordinate network producing a travel-time field and, optionally, dispersion."""

    def __init__(self, config: CarrierConfig, device: str) -> None:
        import torch

        self.config = config
        self.device = device
        generator = torch.Generator().manual_seed(config.seed)
        n_out = 1 + (2 * config.dispersion_rank if config.dispersive else 0)
        self.net = _mlp(config, n_out, generator, device)

    def parameters(self):
        return self.net.parameters()

    def phase(self, coords, frequencies, scale: float):
        """Phase field ``phi(x, f)``; linear in ``f`` unless dispersion is enabled."""

        import torch

        output = self.net(coords)
        delays = output[:, :1] * scale
        phase = 2.0 * np.pi * frequencies[None, :] * delays
        if self.config.dispersive:
            rank = self.config.dispersion_rank
            spatial = output[:, 1 : 1 + rank]
            # A low-rank correction in (x, f) buys a dispersive phase without
            # letting the model represent an arbitrary per-entry rotation.
            basis = torch.stack(
                [torch.cos(np.pi * (index + 1) * frequencies / frequencies[-1])
                 for index in range(rank)],
                dim=0,
            )
            phase = phase + spatial @ basis
        return phase, delays[:, 0]


def _gradient_magnitude(delays, coords, grid_shape, spacing):
    """Central-difference |grad tau| on the regular grid the coordinates lie on."""

    import torch

    field = delays.reshape(grid_shape)
    d_row = torch.zeros_like(field)
    d_col = torch.zeros_like(field)
    d_row[1:-1] = (field[2:] - field[:-2]) / (2 * spacing)
    d_col[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * spacing)
    return torch.sqrt(d_row**2 + d_col**2 + 1e-12)


def fit_learned_carrier(
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    coords: NDArray[np.float64],
    config: CarrierConfig | None = None,
    initial_delays: NDArray[np.float64] | None = None,
    slowness: NDArray[np.float64] | None = None,
    grid_shape: tuple[int, int] | None = None,
    spacing: float | None = None,
    device: str | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """Learn the alignment phase by minimising the aligned field's nuclear norm.

    ``initial_delays`` (typically the eikonal solution) is used as a warm start
    only; ``slowness`` and ``grid_shape`` enable the eikonal residual penalty.
    Returns the learned delays and a history including the nuclear norm, which
    is the quantity the whole method is organised around.
    """

    import torch

    config = config or CarrierConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    values = torch.from_numpy(np.ascontiguousarray(field)).to(device).to(torch.complex64)
    freq = torch.from_numpy(np.ascontiguousarray(frequencies)).float().to(device)
    centre = coords.mean(axis=0)
    extent = float(np.abs(coords - centre).max()) or 1.0
    positions = torch.from_numpy(((coords - centre) / extent).astype(np.float32)).to(device)

    delay_scale = float(np.abs(initial_delays).max()) if initial_delays is not None else 1.0
    delay_scale = delay_scale or 1.0

    carrier = LearnedCarrier(config, device)
    optimiser = torch.optim.Adam(carrier.parameters(), lr=config.learning_rate)
    frobenius = torch.linalg.norm(values)

    target = None
    if initial_delays is not None:
        target = torch.from_numpy(
            (np.asarray(initial_delays) / delay_scale).astype(np.float32)
        ).to(device)

    slowness_t = None
    if slowness is not None and grid_shape is not None and spacing is not None:
        slowness_t = torch.from_numpy(
            np.asarray(slowness, dtype=np.float32).reshape(grid_shape)
        ).to(device)

    history = {"nuclear": [], "warmup": [], "physics": []}
    for step in range(config.warmup_steps + config.steps):
        optimiser.zero_grad(set_to_none=True)
        phase, delays = carrier.phase(positions, freq, delay_scale)

        if step < config.warmup_steps and target is not None:
            # Warm start: reproduce the physics solution before optimising rank.
            loss = torch.nn.functional.mse_loss(delays / delay_scale, target)
            history["warmup"].append(float(loss.detach()))
        else:
            # exp(+i phi) removes the carrier, matching spectra.shift_spectrum.
            aligned = values * torch.exp(1j * phase)
            spectrum = torch.linalg.svdvals(aligned)
            if config.objective == "nuclear":
                loss = spectrum.sum() / frobenius
            else:
                # Energy outside the leading R components: exactly the error the
                # experiments report, so the surrogate gap disappears.
                loss = torch.linalg.norm(spectrum[config.budget :]) / frobenius
            history["nuclear"].append(float((spectrum.sum() / frobenius).detach()))
            history.setdefault("objective", []).append(float(loss.detach()))
            if slowness_t is not None and config.physics_weight > 0:
                magnitude = _gradient_magnitude(delays, positions, grid_shape, spacing)
                residual = torch.nn.functional.mse_loss(magnitude, slowness_t)
                loss = loss + config.physics_weight * residual
                history["physics"].append(float(residual.detach()))
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        phase, delays = carrier.phase(positions, freq, delay_scale)
        aligned = values * torch.exp(1j * phase)
        spectrum = torch.linalg.svdvals(aligned)
        final = float(spectrum.sum() / frobenius)
    return delays.cpu().numpy().astype(np.float64), {
        "history": history,
        "final_nuclear": final,
        "delay_scale": delay_scale,
        # The full phase is what a dispersive model must be scored with; the
        # delays alone do not describe it.
        "phase": phase.cpu().numpy().astype(np.float64),
    }


def nuclear_ratio(field: NDArray[np.complex128], phase: NDArray[np.float64] | None = None) -> float:
    """Nuclear norm divided by Frobenius norm: a scale-free soft rank."""

    values = np.asarray(field)
    if phase is not None:
        values = values * np.exp(1j * np.asarray(phase))
    spectrum = np.linalg.svd(values, compute_uv=False)
    return float(spectrum.sum() / np.linalg.norm(spectrum))
