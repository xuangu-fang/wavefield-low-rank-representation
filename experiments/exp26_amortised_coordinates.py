"""Experiment 26: do the learned coordinates transfer to unseen media?

Everything learned so far was fitted per field. That is a representation of one
case, not a representation *rule*. Here a network is trained across many media
to map the medium to its alignment coordinates, with the identifiability bound
as the only loss -- no labels, no eikonal targets -- and is then applied to
unseen media in a single forward pass.

Two ablations separate the contributions: whether a physics warm start helps,
and how far the amortised network falls short of fitting each test field
directly.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from wave_lr.learned_carrier import CarrierConfig, fit_learned_carrier
from wave_lr.operator import OperatorConfig, build_operator
from wave_lr.spatial import identifiability_bound, taper_window

RESULTS = Path(__file__).resolve().parents[1] / "results"
CACHE = Path(
    "/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/prepared"
)
TARGET_STRIDE = 6


def aliasing_loss(field, delays, frequencies, window, outside):
    """Energy above the array's Nyquist wavenumber, averaged over the band."""

    import torch

    phase = 2.0 * np.pi * frequencies[None, :, None, None] * delays[:, None]
    aligned = field * torch.exp(1j * phase)
    power = torch.abs(torch.fft.fft2(aligned * window[None, None])) ** 2
    return (power * outside[None, None]).sum() / power.sum().clamp_min(1e-30)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="operator_open_128_240.npz")
    parser.add_argument(
        "--test-dataset", default=None,
        help="train on --dataset, evaluate on a different medium family",
    )
    parser.add_argument("--train", type=int, default=180)
    parser.add_argument("--test", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(CACHE / args.dataset)
    field = data["field"]
    speed = data["speed"]
    travel = data["travel_time"]
    frequencies = data["frequencies"]
    grid = speed.shape[-1]

    cross_family = args.test_dataset is not None
    if cross_family:
        # Train on one medium family, evaluate on a different one entirely.
        other = np.load(CACHE / args.test_dataset)
        assert np.allclose(other["frequencies"], frequencies), "frequency grids differ"
        offset = len(speed)
        field = np.concatenate([field, other["field"][: args.test]])
        speed = np.concatenate([speed, other["speed"][: args.test]])
        travel = np.concatenate([travel, other["travel_time"][: args.test]])

    axis = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    mesh = np.stack(np.meshgrid(axis, axis, indexing="ij"))
    inputs = np.concatenate(
        [speed[:, None], np.broadcast_to(mesh, (len(speed), 2, grid, grid))], axis=1
    ).astype(np.float32)

    scale = float(np.abs(travel[: args.train]).max())
    inputs_t = torch.from_numpy(inputs).to(device)
    field_t = torch.from_numpy(field).to(device).to(torch.complex64)
    travel_t = torch.from_numpy(travel / scale).to(device)
    freq_t = torch.from_numpy(frequencies.astype(np.float32)).to(device)
    window = torch.from_numpy(taper_window((grid, grid))).to(device).float()
    rows = np.fft.fftfreq(grid)[:, None]
    cols = np.fft.fftfreq(grid)[None, :]
    outside = torch.from_numpy(
        (np.sqrt(rows**2 + cols**2) > 0.5 / TARGET_STRIDE).astype(np.float32)
    ).to(device)

    train_index = np.arange(args.train)
    if cross_family:
        test_index = np.arange(offset, offset + args.test)
    else:
        test_index = np.arange(args.train, min(args.train + args.test, len(speed)))

    def train_model(use_warmup: bool, seed: int = 0):
        torch.manual_seed(seed)
        model = build_operator(
            OperatorConfig(in_channels=3, out_channels=1, width=32, modes=20, depth=4)
        ).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.steps)
        generator = torch.Generator().manual_seed(seed)
        batch = 8
        history = []
        total = (args.warmup if use_warmup else 0) + args.steps
        for step in range(total):
            pick = torch.randperm(len(train_index), generator=generator)[:batch]
            index = torch.from_numpy(train_index[pick.numpy()]).to(device)
            optimiser.zero_grad(set_to_none=True)
            predicted = model(inputs_t[index])[:, 0]
            if use_warmup and step < args.warmup:
                # Physics only as a starting point, never as a target later.
                loss = torch.nn.functional.mse_loss(predicted, travel_t[index])
            else:
                loss = aliasing_loss(
                    field_t[index], predicted * scale, freq_t, window, outside
                )
                history.append(float(loss.detach()))
                schedule.step()
            loss.backward()
            optimiser.step()
        return model, history

    def evaluate(delays_by_case, label):
        values = []
        for position, case in enumerate(test_index):
            aligned = field[case] * np.exp(
                2j * np.pi * frequencies[:, None, None] * delays_by_case[position]
            )
            middle = aligned.shape[0] // 2
            values.append(
                identifiability_bound(aligned[middle], 1.0 / TARGET_STRIDE**2)
            )
        return {f"{label}_bound": float(np.mean(values)),
                f"{label}_bound_std": float(np.std(values))}

    summary = {"n_train": len(train_index), "n_test": len(test_index),
               "dataset": args.dataset, "test_dataset": args.test_dataset or args.dataset,
               "cross_family": cross_family, "target_stride": TARGET_STRIDE}
    summary.update(evaluate([np.zeros((grid, grid))] * len(test_index), "raw"))
    summary.update(evaluate([travel[c] for c in test_index], "eikonal"))

    start = time.time()
    for use_warmup, label in ((False, "amortised_scratch"), (True, "amortised_warmstart")):
        per_seed = []
        for seed in range(args.seeds):
            model, history = train_model(use_warmup, seed=seed)
            with torch.no_grad():
                predicted = (
                    model(inputs_t[torch.from_numpy(test_index).to(device)])[:, 0] * scale
                ).cpu().numpy()
            scored = evaluate([predicted[i] for i in range(len(test_index))], label)
            per_seed.append(scored[f"{label}_bound"])
        summary[f"{label}_bound"] = float(np.mean(per_seed))
        summary[f"{label}_bound_seed_std"] = float(np.std(per_seed))
        summary[f"{label}_per_seed"] = per_seed
        print(
            f"{label}: test bound {np.mean(per_seed):.3f} +- {np.std(per_seed):.3f} "
            f"over {args.seeds} seeds ({time.time() - start:.0f}s)",
            flush=True,
        )

    # Ceiling: fit each test field directly, as in experiment 24.
    per_case = []
    for case in test_index[:8]:
        coords = np.stack(
            np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij"), axis=-1
        ).reshape(-1, 2).astype(float)
        delays, _ = fit_learned_carrier(
            field[case].reshape(len(frequencies), -1).T, frequencies, coords,
            CarrierConfig(steps=500, warmup_steps=0, learning_rate=1e-3,
                          objective="aliasing", target_stride=TARGET_STRIDE),
        )
        aligned = field[case] * np.exp(
            2j * np.pi * frequencies[:, None, None] * delays.reshape(grid, grid)
        )
        per_case.append(
            identifiability_bound(aligned[aligned.shape[0] // 2], 1.0 / TARGET_STRIDE**2)
        )
    summary["per_case_fitted_bound"] = float(np.mean(per_case))
    summary["per_case_n"] = len(per_case)

    tag = "cross" if cross_family else Path(args.dataset).stem
    (RESULTS / f"exp26_amortised_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
