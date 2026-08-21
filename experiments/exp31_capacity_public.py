"""Experiment 31: on public data, does capacity substitute for the coordinate?

The capacity control in this project was run on fields our own solver produced,
which is exactly the objection a reviewer should raise. This repeats it on
OpenFWI shot gathers: fit a coordinate network to the fourteen receivers that
were measured, ask it for the fifty-six that were not, and sweep its size over
two orders of magnitude.

The comparison that matters is against an estimator with no trained parameters
at all, evaluated in a reparameterised coordinate. If a warp with zero
parameters beats every network in the sweep, the missing ingredient was never
capacity.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
import time
from pathlib import Path

import numpy as np
import torch

from wave_lr.amortised_warp import AmortisedConfig, train_warp_net
from wave_lr.families import openfwi_stack
from wave_lr.sensorline import (
    apply_warp,
    bound_line,
    reconstruct_bandlimited,
    reconstruct_line,
    tapered,
    transport_delays,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRACTION = 0.20
WIDTHS = (16, 32, 64, 128, 256, 512)
DEPTH = 4
STEPS = 3000
SEEDS = (0, 1, 2)
N_TEST = 12
SPEEDS = np.linspace(1000.0, 6000.0, 40)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CoordinateNet(torch.nn.Module):
    """A Fourier-feature coordinate network over (sensor, frequency).

    The standard tool for this job, given the standard treatment: enough
    features to represent the field, trained to convergence on what was
    measured.
    """

    def __init__(self, width: int, depth: int, features: int = 32, scale: float = 8.0):
        super().__init__()
        self.register_buffer("bank", torch.randn(2, features) * scale)
        layers: list[torch.nn.Module] = []
        n_in = 2 + 2 * features
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.GELU()]
            n_in = width
        layers.append(torch.nn.Linear(n_in, 2))
        self.body = torch.nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        arg = 2.0 * np.pi * coords @ self.bank
        features = torch.cat([coords, torch.sin(arg), torch.cos(arg)], dim=-1)
        out = self.body(features)
        return torch.complex(out[..., 0], out[..., 1])


def fit_network(field, freqs, coords, observed, width, depth, seed):
    """Fit the measured columns, report error on the measured and the withheld."""

    torch.manual_seed(seed)
    device = torch.device(DEVICE)
    target = torch.as_tensor(field, dtype=torch.complex64, device=device)
    scale = torch.abs(target).max() + 1e-12
    target = target / scale
    x = torch.as_tensor(
        (coords - coords.mean()) / (0.5 * np.ptp(coords)), dtype=torch.float32, device=device
    )
    f = torch.as_tensor(
        (freqs - freqs.mean()) / (0.5 * np.ptp(freqs)), dtype=torch.float32, device=device
    )
    grid = torch.stack(torch.meshgrid(f, x, indexing="ij"), dim=-1)
    mask = torch.zeros(coords.size, dtype=torch.bool, device=device)
    mask[observed] = True

    net = CoordinateNet(width, depth).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=2e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, STEPS)
    for _ in range(STEPS):
        optimiser.zero_grad(set_to_none=True)
        predicted = net(grid[:, mask])
        loss = ((predicted - target[:, mask]).abs() ** 2).mean()
        loss.backward()
        optimiser.step()
        schedule.step()
    with torch.no_grad():
        full = net(grid)
    n_param = sum(p.numel() for p in net.parameters())
    return full.cpu().numpy(), float(loss.detach()), n_param


def relative(field, estimate, weights, taper=0.25):
    """Same footing as the bound: taper once, then score."""

    truth = tapered(field, taper)
    guess = tapered(estimate, taper)
    residual = (np.abs(truth - guess) ** 2).sum(1)
    reference = (np.abs(truth) ** 2).sum(1)
    share = np.where(reference > 0, residual / np.maximum(reference, 1e-300), 0.0)
    return float(np.sqrt((weights * share).sum()))


def main() -> None:
    start = time.time()
    train = openfwi_stack(path_index=0, n_sample=96)
    test = openfwi_stack(path_index=1, n_sample=8)
    freqs, coords, weights = train["freqs"], train["coords"], train["weights"]
    observed = np.arange(0, coords.size, max(round(1.0 / FRACTION), 1))
    print(f"observed {observed.size}/{coords.size} sensors", flush=True)

    predict, info = train_warp_net(
        train["spectra"],
        freqs,
        coords,
        weights,
        AmortisedConfig(fraction=FRACTION, steps=1200, batch=16, seed=0),
    )
    print(
        f"amortised warp objective {info['loss'][0]:.4f} -> {np.mean(info['loss'][-50:]):.4f}"
        f"  ({time.time()-start:.0f}s)",
        flush=True,
    )

    rows = []
    for index in range(min(N_TEST, test["spectra"].shape[0])):
        field = test["spectra"][index]
        raw_bound = bound_line(field, weights, FRACTION)

        best = (float("nan"), raw_bound)
        for speed in SPEEDS:
            delays = transport_delays(coords, test["sources"][index], speed)
            value = bound_line(apply_warp(field, freqs, delays), weights, FRACTION)
            if value < best[1]:
                best = (float(speed), value)
        oracle_delays = (
            transport_delays(coords, test["sources"][index], best[0])
            if np.isfinite(best[0])
            else np.zeros(coords.size)
        )
        learned_delays = predict(field)

        zero_parameter = {
            "linear": reconstruct_line(field, weights, max(round(1.0 / FRACTION), 1)),
            "bandlimited": reconstruct_bandlimited(field, weights, FRACTION),
            "bandlimited_oracle_warp": reconstruct_bandlimited(
                field, weights, FRACTION, oracle_delays, freqs
            ),
            "bandlimited_learned_warp": reconstruct_bandlimited(
                field, weights, FRACTION, learned_delays, freqs
            ),
        }

        sweep = []
        for width in WIDTHS:
            errors, trains, n_param = [], [], 0
            for seed in SEEDS:
                estimate, train_loss, n_param = fit_network(
                    field, freqs, coords, observed, width, DEPTH, seed
                )
                norm = np.abs(field).max()
                errors.append(relative(field, estimate * norm, weights))
                trains.append(train_loss)
            sweep.append(
                {
                    "width": width,
                    "n_param": n_param,
                    "test_error": float(np.mean(errors)),
                    "test_error_std": float(np.std(errors)),
                    "train_loss": float(np.mean(trains)),
                }
            )
        rows.append(
            {
                "name": test["names"][index],
                "bound_raw": raw_bound,
                "bound_oracle_warp": best[1],
                "bound_learned_warp": bound_line(
                    apply_warp(field, freqs, learned_delays), weights, FRACTION
                ),
                "zero_parameter": zero_parameter,
                "sweep": sweep,
            }
        )
        best_net = min(s["test_error"] for s in sweep)
        print(
            f"  {test['names'][index]:22s} bound {raw_bound:.3f} | best net {best_net:.3f} "
            f"({sweep[int(np.argmin([s['test_error'] for s in sweep]))]['n_param']/1e3:.0f}k par) "
            f"| bandlimited {zero_parameter['bandlimited']:.3f} "
            f"| +learned warp {zero_parameter['bandlimited_learned_warp']:.3f}  "
            f"({time.time()-start:.0f}s)",
            flush=True,
        )

    params = np.array([s["n_param"] for s in rows[0]["sweep"]], dtype=float)
    curves = np.array([[s["test_error"] for s in r["sweep"]] for r in rows])
    summary = {
        "n_test": len(rows),
        "fraction": FRACTION,
        "n_observed_sensors": int(observed.size),
        "widths": list(WIDTHS),
        "parameter_range": [float(params.min()), float(params.max())],
        "parameter_ratio": float(params.max() / params.min()),
        "network_test_error_by_width": [float(v) for v in curves.mean(0)],
        "network_best_test_error_median": float(np.median(curves.min(1))),
        "network_train_loss_median": float(
            np.median([[s["train_loss"] for s in r["sweep"]] for r in rows])
        ),
        "zero_parameter_median": {
            key: float(np.median([r["zero_parameter"][key] for r in rows]))
            for key in rows[0]["zero_parameter"]
        },
        "bound_median": {
            key: float(np.median([r[key] for r in rows]))
            for key in ("bound_raw", "bound_oracle_warp", "bound_learned_warp")
        },
        "runtime_s": time.time() - start,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "exp31_capacity_public.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
