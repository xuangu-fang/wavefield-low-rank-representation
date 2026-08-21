"""Learn the sensor-line reparameterisation from the sparse data you actually have.

The speed scan in :mod:`wave_lr.sensorline` picks a warp by looking at the dense
field. That is legitimate when the dense field exists -- deciding where to put
sensors from a pilot survey -- but it cannot be part of a reconstruction method,
because a reconstruction only ever sees the sparse samples. The warp has to be
estimated from those samples alone, or the gain it reports is circular.

That rules out optimising the identifiability criterion itself: the criterion is
about energy above the array's Nyquist, and the sparse array cannot see it. What
the sparse samples *can* see is that the correct warp makes the observed
``(frequency, sensor)`` matrix low rank -- alignment is what collapses a
transported transient onto a few components. So the objective here is a rank
surrogate on the observed columns only, and the warp is a smooth function of the
sensor coordinate, which is what lets it be evaluated at sensors that were never
measured.

Nothing in the objective knows about waves, travel times, or any PDE. The delay
is a coordinate, and the coordinate is fitted to the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class LineCarrierConfig:
    """Everything that shapes the fitted warp, kept small on purpose."""

    features: int = 24  # Fourier features on the sensor coordinate
    feature_scale: float = 6.0
    width: int = 64
    depth: int = 2
    rank: int = 3  # how many components "aligned" is allowed to need
    objective: str = "tail"  # "tail" or "nuclear"
    steps: int = 500
    learning_rate: float = 3e-3
    delay_scale: float | None = None  # seconds; defaults to the array's own scale
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class _DelayNet(torch.nn.Module):
    """A smooth delay as a function of the sensor coordinate.

    Smoothness is the whole point: a free delay per observed sensor would fit the
    observations and say nothing about the sensors in between, which are exactly
    the ones the reconstruction has to fill in.
    """

    def __init__(self, config: LineCarrierConfig, generator: torch.Generator, device):
        super().__init__()
        self.frequencies = torch.nn.Parameter(
            torch.randn(config.features, generator=generator, device=device)
            * config.feature_scale,
            requires_grad=False,
        )
        layers: list[torch.nn.Module] = []
        n_in = 2 * config.features + 1
        for _ in range(config.depth):
            layers += [torch.nn.Linear(n_in, config.width), torch.nn.Tanh()]
            n_in = config.width
        layers.append(torch.nn.Linear(n_in, 1))
        self.body = torch.nn.Sequential(*layers).to(device)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        arg = coords[:, None] * self.frequencies[None, :]
        features = torch.cat([coords[:, None], torch.sin(arg), torch.cos(arg)], dim=1)
        return self.body(features)[:, 0]


def _rank_loss(aligned: torch.Tensor, config: LineCarrierConfig) -> torch.Tensor:
    """A rank surrogate that is invariant to the warp's overall scale.

    Both branches divide by the Frobenius norm, which a pointwise unimodular
    factor leaves untouched -- so the objective cannot be driven down by
    inflating the field, only by making it genuinely simpler.
    """

    frob = torch.linalg.matrix_norm(aligned) + 1e-12
    values = torch.linalg.svdvals(aligned)
    if config.objective == "nuclear":
        return values.sum() / frob
    tail = values[config.rank :]
    return torch.sqrt((tail**2).sum()) / frob


def fit_line_carrier(
    spectrum: NDArray[np.complex128],
    freqs: NDArray[np.float64],
    coords: NDArray[np.float64],
    config: LineCarrierConfig | None = None,
):
    """Fit a delay to observed columns, return it evaluated wherever asked.

    ``spectrum`` and ``coords`` must contain only the sensors that were actually
    measured. The returned callable maps any sensor coordinate to a delay.
    """

    config = config or LineCarrierConfig()
    device = torch.device(config.device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    torch.manual_seed(config.seed)

    span = float(np.ptp(coords)) or 1.0
    centre, half = float(coords.mean()), 0.5 * span
    scale = config.delay_scale
    if scale is None:
        # One period of the highest frequency across the array is the natural
        # unit: a warp larger than that is not a relabelling, it is a wrap.
        scale = float(coords.size / (2.0 * max(freqs.max(), 1e-12)))

    values = torch.as_tensor(spectrum, dtype=torch.complex64, device=device)
    frequency = torch.as_tensor(freqs, dtype=torch.float32, device=device)
    position = torch.as_tensor((coords - centre) / half, dtype=torch.float32, device=device)

    net = _DelayNet(config, generator, device)
    optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    history = []
    for _ in range(config.steps):
        optimiser.zero_grad(set_to_none=True)
        delays = torch.tanh(net(position)) * scale
        phase = 2.0 * np.pi * frequency[:, None] * delays[None, :]
        aligned = values * torch.polar(torch.ones_like(phase), phase)
        loss = _rank_loss(aligned, config)
        loss.backward()
        optimiser.step()
        history.append(float(loss.detach()))

    def evaluate(query: NDArray[np.float64]) -> NDArray[np.float64]:
        with torch.no_grad():
            grid = torch.as_tensor(
                (np.asarray(query, dtype=np.float64) - centre) / half,
                dtype=torch.float32,
                device=device,
            )
            out = torch.tanh(net(grid)) * scale
        return out.cpu().numpy().astype(np.float64)

    return evaluate, {"loss": history, "delay_scale": scale}
