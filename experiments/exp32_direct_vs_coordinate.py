"""Experiment 32: the baseline a reviewer will actually ask for.

Experiment 31 fits a coordinate network to one gather's fourteen sensors, which
is the setting where capacity has nothing to work with. The fair comparison is
the method a practitioner would really reach for: an amortised reconstruction
network, trained *supervised* on hundreds of dense gathers, that maps the sparse
sensors straight to the full array.

That baseline is given every advantage this project's own method is denied. It
sees ground-truth dense fields during training; the warp network sees only the
identifiability criterion and no targets at all. It spends its parameters on the
reconstruction; the warp network spends them on a single delay per sensor, after
which the reconstruction is done by an estimator with no parameters.

If the label-free coordinate route still competes, the claim that the missing
ingredient was the coordinate rather than the capacity survives its hardest
test. If it does not, that has to be reported as it comes out.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
import time
from pathlib import Path

import numpy as np
import torch

from wave_lr.amortised_warp import (
    AmortisedConfig,
    observed_input,
    train_warp_net,
)
from wave_lr.families import openfwi_stack
from wave_lr.sensorline import (
    apply_warp,
    bound_line,
    reconstruct_bandlimited,
    tapered,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRACTION = 0.20
SEEDS = (0, 1, 2)
WIDTHS = (32, 64, 128, 256)
DEPTH = 6
STEPS = 3000
BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ReconstructionNet(torch.nn.Module):
    """Masked sparse spectrum in, the whole array out.

    Same convolutional stem as the warp network so the comparison is about what
    the parameters are spent on, not about architecture family.
    """

    def __init__(self, n_freq: int, width: int, depth: int, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        layers: list[torch.nn.Module] = []
        n_in = 2 * n_freq + 1
        for _ in range(depth):
            layers += [torch.nn.Conv1d(n_in, width, kernel, padding=pad), torch.nn.GELU()]
            n_in = width
        layers.append(torch.nn.Conv1d(n_in, 2 * n_freq, kernel, padding=pad))
        self.body = torch.nn.Sequential(*layers)
        self.n_freq = n_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        return torch.complex(out[:, : self.n_freq], out[:, self.n_freq :])


def train_reconstruction(spectra, mask, width, depth, seed, taper=0.25):
    """Supervised on dense targets: every advantage the warp route does not get."""

    torch.manual_seed(seed)
    device = torch.device(DEVICE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_instance, n_freq, n_sensor = spectra.shape
    values = torch.as_tensor(spectra, dtype=torch.complex64, device=device)
    window = torch.as_tensor(
        np.asarray(tapered(np.ones((1, n_sensor)), taper))[0], dtype=torch.float32, device=device
    )
    net = ReconstructionNet(n_freq, width, depth).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, STEPS)
    history = []
    for _ in range(STEPS):
        index = torch.randperm(n_instance, generator=generator)[:BATCH].to(device)
        batch = values[index]
        # Normalise by what the network can actually see. Using the dense norm
        # would hand it the answer's overall scale for free, which is exactly
        # the kind of leak that makes an amortised baseline look stronger than
        # it is.
        scale = (
            torch.linalg.matrix_norm(batch * mask[None, None, :], dim=(-2, -1), keepdim=True)
            + 1e-12
        )
        optimiser.zero_grad(set_to_none=True)
        predicted = net(observed_input(batch, mask))
        residual = ((predicted - batch / scale).abs() ** 2) * window
        loss = residual.sum((-2, -1)).mean()
        loss.backward()
        optimiser.step()
        schedule.step()
        history.append(float(loss.detach()))

    def predict(field):
        with torch.no_grad():
            block = torch.as_tensor(field, dtype=torch.complex64, device=device)[None]
            scale = (
                torch.linalg.matrix_norm(
                    block * mask[None, None, :], dim=(-2, -1), keepdim=True
                )
                + 1e-12
            )
            out = net(observed_input(block, mask)) * scale
        return out[0].cpu().numpy()

    n_param = sum(p.numel() for p in net.parameters())
    return predict, n_param, history


def relative(field, estimate, weights, taper=0.25):
    truth, guess = tapered(field, taper), tapered(estimate, taper)
    residual = (np.abs(truth - guess) ** 2).sum(1)
    reference = (np.abs(truth) ** 2).sum(1)
    share = np.where(reference > 0, residual / np.maximum(reference, 1e-300), 0.0)
    return float(np.sqrt((weights * share).sum()))


def main() -> None:
    start = time.time()
    train = openfwi_stack(path_index=0, n_sample=96)
    test = openfwi_stack(path_index=1, n_sample=8)
    freqs, coords, weights = train["freqs"], train["coords"], train["weights"]
    stride = max(round(1.0 / FRACTION), 1)
    mask = torch.zeros(coords.size, device=DEVICE)
    mask[::stride] = 1.0
    print(
        f"train {train['spectra'].shape}  test {test['spectra'].shape}  "
        f"{int(mask.sum())}/{coords.size} sensors",
        flush=True,
    )

    warp_predictors, warp_params = [], 0
    for seed in SEEDS:
        predict, info = train_warp_net(
            train["spectra"],
            freqs,
            coords,
            weights,
            AmortisedConfig(fraction=FRACTION, steps=1200, batch=BATCH, seed=seed),
        )
        warp_predictors.append(predict)
        print(
            f"  warp net seed {seed}: {info['loss'][0]:.4f} -> "
            f"{np.mean(info['loss'][-50:]):.4f}  ({time.time()-start:.0f}s)",
            flush=True,
        )
    from wave_lr.amortised_warp import WarpNet

    warp_params = sum(
        p.numel() for p in WarpNet(freqs.size, AmortisedConfig()).parameters()
    )

    direct = {}
    for width in WIDTHS:
        predictors, n_param, losses = [], 0, []
        for seed in SEEDS:
            predict, n_param, history = train_reconstruction(
                train["spectra"], mask, width, DEPTH, seed
            )
            predictors.append(predict)
            losses.append(float(np.mean(history[-50:])))
        direct[width] = {"predictors": predictors, "n_param": n_param, "loss": losses}
        print(
            f"  direct net width {width}: {n_param/1e3:.0f}k parameters, "
            f"train loss {np.mean(losses):.4e}  ({time.time()-start:.0f}s)",
            flush=True,
        )

    rows = []
    for index in range(test["spectra"].shape[0]):
        field = test["spectra"][index]
        entry = {
            "name": test["names"][index],
            "bound_raw": bound_line(field, weights, FRACTION),
            "bandlimited": reconstruct_bandlimited(field, weights, FRACTION),
            "warp_bandlimited": [],
            "bound_warp": [],
            "direct": {},
        }
        for predict in warp_predictors:
            delays = predict(field)
            entry["warp_bandlimited"].append(
                reconstruct_bandlimited(field, weights, FRACTION, delays, freqs)
            )
            entry["bound_warp"].append(
                bound_line(apply_warp(field, freqs, delays), weights, FRACTION)
            )
        for width, block in direct.items():
            entry["direct"][str(width)] = [
                relative(field, predict(field), weights) for predict in block["predictors"]
            ]
        rows.append(entry)

    def median_of(key):
        return float(np.median([np.mean(np.atleast_1d(r[key])) for r in rows]))

    summary = {
        "n_test": len(rows),
        "fraction": FRACTION,
        "n_observed_sensors": int(mask.sum().item()),
        "n_train_instances": int(train["spectra"].shape[0]),
        "bound_raw_median": median_of("bound_raw"),
        "bound_warp_median": median_of("bound_warp"),
        "bandlimited_median": median_of("bandlimited"),
        "warp_bandlimited_median": median_of("warp_bandlimited"),
        "warp_net_parameters": int(warp_params),
        "direct": {
            str(width): {
                "n_param": int(block["n_param"]),
                "train_loss": float(np.mean(block["loss"])),
                "test_error_median": float(
                    np.median([np.mean(r["direct"][str(width)]) for r in rows])
                ),
            }
            for width, block in direct.items()
        },
        "runtime_s": time.time() - start,
    }
    best = min(v["test_error_median"] for v in summary["direct"].values())
    summary["direct_best_test_error_median"] = best
    summary["coordinate_route_relative_to_direct"] = best / max(
        summary["warp_bandlimited_median"], 1e-12
    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "exp32_direct_vs_coordinate.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
