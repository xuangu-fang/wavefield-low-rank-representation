"""Coordinate networks for sparse-sensor reconstruction.

A reviewer's first objection to a phase carrier is that Fourier features or a
SIREN already handle oscillation. These models make that comparison concrete:
the same architecture and budget either predicts the field directly or
predicts the carrier-aligned envelope, with the carrier reapplied at the
output. Everything else is held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrainConfig:
    width: int = 256
    depth: int = 3
    features: int = 128
    feature_scale: float = 8.0
    steps: int = 3000
    learning_rate: float = 2e-3
    seed: int = 0


def _build(config: TrainConfig, n_out: int, encoding: str, device: str):
    import torch
    from torch import nn

    generator = torch.Generator().manual_seed(config.seed)

    class FourierFeatures(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            matrix = torch.randn(2, config.features, generator=generator) * config.feature_scale
            self.register_buffer("matrix", matrix)

        def forward(self, coords):
            projected = 2.0 * np.pi * coords @ self.matrix
            return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

    class Sine(nn.Module):
        def __init__(self, omega: float = 30.0) -> None:
            super().__init__()
            self.omega = omega

        def forward(self, values):
            return torch.sin(self.omega * values)

    layers: list[nn.Module] = []
    if encoding == "fourier":
        layers.append(FourierFeatures())
        in_dim = 2 * config.features
        activation: type[nn.Module] | None = nn.GELU
    elif encoding == "siren":
        in_dim = 2
        activation = None
    else:
        in_dim = 2
        activation = nn.GELU

    for index in range(config.depth):
        layers.append(nn.Linear(in_dim if index == 0 else config.width, config.width))
        layers.append(Sine() if encoding == "siren" else activation())
    layers.append(nn.Linear(config.width, n_out))
    model = nn.Sequential(*layers)

    if encoding == "siren":
        with torch.no_grad():
            first = True
            for module in model:
                if isinstance(module, nn.Linear):
                    fan_in = module.weight.shape[1]
                    bound = 1.0 / fan_in if first else np.sqrt(6.0 / fan_in) / 30.0
                    module.weight.uniform_(-bound, bound, generator=generator)
                    first = False
    return model.to(device)


def fit_field_network(
    coords: NDArray[np.float64],
    field: NDArray[np.complex128],
    observed: NDArray[np.bool_],
    encoding: str = "fourier",
    delays: NDArray[np.float64] | None = None,
    frequencies: NDArray[np.float64] | None = None,
    config: TrainConfig | None = None,
    device: str | None = None,
) -> NDArray[np.complex128]:
    """Fit ``coords -> complex spectrum`` on observed locations only.

    With ``delays`` supplied the network learns the aligned envelope and the
    carrier is reapplied afterwards, so the target it must represent is smooth
    rather than oscillatory. The prediction returned is always the physical
    field, so errors are comparable across encodings.
    """

    import torch

    from .spectra import carrier

    config = config or TrainConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.seed)

    ramp = None
    target = field
    if delays is not None:
        if frequencies is None:
            raise ValueError("frequencies are required when delays are given")
        ramp = np.conj(carrier(frequencies, delays))
        target = field * ramp

    # Coordinates are centred and scaled so the feature bandwidth means the same
    # thing on every dataset.
    centre = coords.mean(axis=0)
    scale = float(np.abs(coords - centre).max()) or 1.0
    inputs = torch.from_numpy(((coords - centre) / scale).astype(np.float32)).to(device)

    amplitude = float(np.abs(target).std()) or 1.0
    stacked = np.concatenate([target.real, target.imag], axis=1) / amplitude
    values = torch.from_numpy(stacked.astype(np.float32)).to(device)
    mask = torch.from_numpy(observed).to(device)

    model = _build(config, stacked.shape[1], encoding, device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.steps)
    train_inputs, train_values = inputs[mask], values[mask]

    for _ in range(config.steps):
        optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_inputs), train_values)
        loss.backward()
        optimiser.step()
        schedule.step()

    with torch.no_grad():
        prediction = model(inputs).cpu().numpy() * amplitude
    half = prediction.shape[1] // 2
    complex_prediction = prediction[:, :half] + 1j * prediction[:, half:]
    if ramp is not None:
        complex_prediction = complex_prediction * np.conj(ramp)
    return complex_prediction
