"""A compact Fourier neural operator for medium-to-field prediction.

Used to ask the operator-learning version of the representation question: at
equal capacity and on identical data, is it easier to predict the field or the
carrier-aligned envelope, given that the carrier is recomputable from the
medium at test time?
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorConfig:
    modes: int = 16
    width: int = 32
    depth: int = 4
    in_channels: int = 3
    out_channels: int = 16


def build_operator(config: OperatorConfig):
    import torch
    from torch import nn

    class SpectralConv2d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
            super().__init__()
            self.modes = modes
            scale = 1.0 / (in_channels * out_channels)
            self.weight_low = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
            )
            self.weight_high = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
            )

        def forward(self, values):
            batch, _, height, width = values.shape
            spectrum = torch.fft.rfft2(values)
            output = torch.zeros(
                batch, self.weight_low.shape[1], height, width // 2 + 1,
                dtype=torch.cfloat, device=values.device,
            )
            modes = self.modes
            output[:, :, :modes, :modes] = torch.einsum(
                "bixy,ioxy->boxy", spectrum[:, :, :modes, :modes], self.weight_low
            )
            output[:, :, -modes:, :modes] = torch.einsum(
                "bixy,ioxy->boxy", spectrum[:, :, -modes:, :modes], self.weight_high
            )
            return torch.fft.irfft2(output, s=(height, width))

    class FourierOperator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lift = nn.Conv2d(config.in_channels, config.width, 1)
            self.spectral = nn.ModuleList(
                SpectralConv2d(config.width, config.width, config.modes)
                for _ in range(config.depth)
            )
            self.pointwise = nn.ModuleList(
                nn.Conv2d(config.width, config.width, 1) for _ in range(config.depth)
            )
            self.head = nn.Sequential(
                nn.Conv2d(config.width, 2 * config.width, 1),
                nn.GELU(),
                nn.Conv2d(2 * config.width, config.out_channels, 1),
            )

        def forward(self, values):
            values = self.lift(values)
            for spectral, pointwise in zip(self.spectral, self.pointwise):
                values = torch.nn.functional.gelu(spectral(values) + pointwise(values))
            return self.head(values)

    return FourierOperator()
