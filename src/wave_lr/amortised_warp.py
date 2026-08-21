"""Amortise the sensor-line warp across instances.

Fitting a warp to one sparsely-observed instance is itself an identifiability
problem: a handful of observed sensors do not pin down a delay at the sensors in
between, so a flexible warp fits the observations and is wrong everywhere else.
The way out is not a stronger regulariser but more instances -- the warps of a
family of gathers share structure, and a network trained across them learns that
structure instead of re-deriving it from fourteen numbers.

Training uses dense fields, because at training time the dense fields exist; the
objective on them is still the label-free identifiability criterion, never a
ground-truth delay. Deployment sees only the sparse sensors. That split is the
whole point: it is what makes the warp something you can actually apply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from .spectra import tukey


@dataclass
class AmortisedConfig:
    channels: int = 64
    depth: int = 4
    kernel: int = 5
    steps: int = 400
    batch: int = 8
    learning_rate: float = 1e-3
    fraction: float = 0.2  # the sensor density the warp is being optimised for
    taper: float = 0.25
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class WarpNet(torch.nn.Module):
    """Sparse observations in, a delay per sensor out.

    Convolutional along the sensor axis so the map is shift-equivariant: the
    same local pattern of arrivals implies the same local delay wherever on the
    array it appears, which is what lets it transfer between source positions.
    """

    def __init__(self, n_freq: int, config: AmortisedConfig):
        super().__init__()
        pad = config.kernel // 2
        layers: list[torch.nn.Module] = []
        n_in = 2 * n_freq + 1  # real, imaginary, and an observed/interpolated flag
        for _ in range(config.depth):
            layers += [
                torch.nn.Conv1d(n_in, config.channels, config.kernel, padding=pad),
                torch.nn.GELU(),
            ]
            n_in = config.channels
        layers.append(torch.nn.Conv1d(n_in, 1, config.kernel, padding=pad))
        self.body = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)[:, 0]


def observed_input(spectrum: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero the unobserved sensors and say which ones they were.

    Interpolating first would hand the network a smoothed field and hide the
    aliasing it is supposed to react to, so the gaps stay as gaps.
    """

    kept = spectrum * mask[None, None, :]
    scale = torch.linalg.matrix_norm(kept, dim=(-2, -1), keepdim=True) + 1e-12
    kept = kept / scale
    flag = mask[None, None, :].expand(kept.shape[0], 1, -1)
    return torch.cat([kept.real, kept.imag, flag], dim=1)


def aliased_energy_torch(
    aligned: torch.Tensor, weights: torch.Tensor, fraction: float, taper: float
) -> torch.Tensor:
    """The identifiability criterion, differentiable in the delay."""

    n_sensor = aligned.shape[-1]
    window = torch.as_tensor(
        tukey(n_sensor, taper), dtype=aligned.real.dtype, device=aligned.device
    )
    power = torch.abs(torch.fft.fft(aligned * window, dim=-1)) ** 2
    wavenumber = torch.abs(torch.fft.fftfreq(n_sensor, device=aligned.device))
    outside = power[..., wavenumber > 0.5 * fraction].sum(-1)
    total = power.sum(-1) + 1e-30
    return (weights * (outside / total)).sum(-1)


def train_warp_net(
    spectra: NDArray[np.complex128],
    freqs: NDArray[np.float64],
    coords: NDArray[np.float64],
    weights: NDArray[np.float64],
    config: AmortisedConfig | None = None,
):
    """Train on dense instances, deploy on sparse ones.

    ``spectra`` is ``(n_instance, n_freq, n_sensor)``; every instance shares the
    frequency axis and the sensor line so the network sees one problem, not a
    ragged collection.
    """

    config = config or AmortisedConfig()
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    n_instance, n_freq, n_sensor = spectra.shape
    stride = max(round(1.0 / config.fraction), 1)
    mask = torch.zeros(n_sensor, device=device)
    mask[::stride] = 1.0

    values = torch.as_tensor(spectra, dtype=torch.complex64, device=device)
    frequency = torch.as_tensor(freqs, dtype=torch.float32, device=device)
    weight = torch.as_tensor(weights, dtype=torch.float32, device=device)
    span = float(np.ptp(coords)) or 1.0
    # One period of the top frequency across the array: beyond that a delay
    # stops being a relabelling of the axis and starts wrapping.
    scale = float(n_sensor / (2.0 * max(freqs.max(), 1e-12)))

    net = WarpNet(n_freq, config).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    history = []
    for _ in range(config.steps):
        index = torch.randperm(n_instance, generator=generator)[: config.batch].to(device)
        batch = values[index]
        optimiser.zero_grad(set_to_none=True)
        delays = torch.tanh(net(observed_input(batch, mask))) * scale
        phase = 2.0 * np.pi * frequency[None, :, None] * delays[:, None, :]
        aligned = batch * torch.polar(torch.ones_like(phase), phase)
        loss = aliased_energy_torch(aligned, weight, config.fraction, config.taper).mean()
        loss.backward()
        optimiser.step()
        history.append(float(loss.detach()))

    def predict(sparse: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Delay for one instance, from its observed sensors only."""

        with torch.no_grad():
            block = torch.as_tensor(sparse, dtype=torch.complex64, device=device)[None]
            out = torch.tanh(net(observed_input(block, mask))) * scale
        delays = out[0].cpu().numpy().astype(np.float64)
        return delays - delays.min()

    return predict, {"loss": history, "delay_scale": scale, "span": span}
