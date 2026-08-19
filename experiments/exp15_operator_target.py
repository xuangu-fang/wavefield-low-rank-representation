"""Experiment 15: which target should a neural operator predict?

The same FNO, the same media, the same budget: predict the complex field, or
predict the carrier-aligned envelope and reapply a carrier that is recomputed
from the medium at test time. Both are legitimate deployable pipelines, so any
difference is a property of the target representation.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wave_lr.eikonal import batched_travel_time
from wave_lr.fdtd import MediumSpec, build_medium, simulate
from wave_lr.operator import OperatorConfig, build_operator

RESULTS = Path(__file__).resolve().parents[1] / "results"
CACHE = Path(
    "/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/prepared"
)
FREQUENCIES = 8
BAND = (6.0, 24.0)
SOURCE = (0.5, 0.28)


def single_inclusion(grid: int, rng) -> np.ndarray:
    """One circular inclusion: a three-parameter family an operator can learn."""

    axis = np.linspace(0.0, 1.0, grid)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    speed = np.ones((grid, grid))
    centre = rng.uniform(0.32, 0.68, size=2)
    radius = float(rng.uniform(0.06, 0.16))
    contrast = float(rng.uniform(0.25, 0.55))
    inside = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 < radius**2
    speed[inside] = 1.0 - contrast
    return speed


def generate(
    n_media: int, absorption: float, grid: int, batch: int, seed0: int, family: str = "random"
) -> dict:
    """Simulate a family of random media and keep a few frequency slices."""

    speeds, dampings = [], []
    rng = np.random.default_rng(seed0)
    reference = build_medium(MediumSpec(name="damping", grid=grid, absorption=absorption))[1]
    for index in range(n_media):
        if family == "single":
            speeds.append(single_inclusion(grid, rng))
            dampings.append(reference)
            continue
        spec = MediumSpec(
            name=f"m{index}",
            grid=grid,
            scatterer_fraction=float(rng.uniform(0.04, 0.20)),
            scatterer_contrast=float(rng.uniform(0.3, 0.6)),
            absorption=absorption,
            seed=seed0 + index,
        )
        speed, damping = build_medium(spec)
        speeds.append(speed)
        dampings.append(damping)

    row = round(SOURCE[0] * (grid - 1))
    col = round(SOURCE[1] * (grid - 1))
    fields, spacing, record_dt = [], None, None
    for start in range(0, n_media, batch):
        block_speed = np.stack(speeds[start : start + batch])
        block_damping = np.stack(dampings[start : start + batch])
        frames, record_dt, spacing = simulate(
            block_speed, block_damping, (row, col), 12.0, 6.0, record_every=4
        )
        n_t = frames.shape[1]
        padded = 2 * n_t
        spectrum = np.fft.rfft(frames.astype(np.float64), n=padded, axis=1)
        freqs = np.fft.rfftfreq(padded, record_dt)
        keep = np.linspace(
            int(np.searchsorted(freqs, BAND[0])),
            int(np.searchsorted(freqs, BAND[1])) - 1,
            FREQUENCIES,
        ).astype(int)
        fields.append(spectrum[:, keep])
        selected = freqs[keep]
    field = np.concatenate(fields)  # (n, n_f, grid, grid)

    source_mask = np.zeros((n_media, grid, grid), dtype=bool)
    source_mask[:, row, col] = True
    travel = batched_travel_time(
        np.stack(speeds), source_mask, spacing=spacing, iterations=6 * grid
    )
    return {
        "field": field.astype(np.complex64),
        "speed": np.stack(speeds).astype(np.float32),
        "travel_time": travel.astype(np.float32),
        "frequencies": selected,
        "spacing": spacing,
    }


def train(inputs, targets, config, steps, learning_rate, device, seed):
    import torch

    torch.manual_seed(seed)
    model = build_operator(config).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    n_train = inputs.shape[0]
    batch = min(16, n_train)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        index = torch.randperm(n_train, generator=generator)[:batch]
        optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs[index]), targets[index])
        loss.backward()
        optimiser.step()
        schedule.step()
    return model, float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=192)
    parser.add_argument("--test", type=int, default=48)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--absorption", type=float, default=40.0)
    parser.add_argument("--regime", default="open")
    parser.add_argument("--family", default="random", choices=("random", "single"))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / (
        f"operator_{args.family}_{args.regime}_{args.grid}_{args.train + args.test}.npz"
    )
    if cache.exists():
        data = dict(np.load(cache))
        print(f"loaded {cache.name}")
    else:
        start = time.time()
        data = generate(
            args.train + args.test, args.absorption, args.grid, 16,
            seed0=0, family=args.family,
        )
        np.savez_compressed(cache, **data)
        print(f"generated {args.train + args.test} media in {time.time() - start:.0f}s")

    field = data["field"]
    speed = data["speed"]
    travel = data["travel_time"]
    frequencies = data["frequencies"]

    # Carrier is rebuilt from the medium, so it is available at test time too.
    ramp = np.exp(
        2j * np.pi * frequencies[None, :, None, None] * travel[:, None, :, :]
    ).astype(np.complex64)
    aligned = field * ramp

    def to_channels(values):
        return np.concatenate([values.real, values.imag], axis=1).astype(np.float32)

    grid_axis = np.linspace(0.0, 1.0, args.grid, dtype=np.float32)
    mesh = np.stack(np.meshgrid(grid_axis, grid_axis, indexing="ij"))
    inputs = np.concatenate(
        [speed[:, None], np.broadcast_to(mesh, (len(speed), 2, args.grid, args.grid))], axis=1
    ).astype(np.float32)

    split = args.train
    inputs_t = torch.from_numpy(inputs).to(device)
    rows = []
    for label, values in (("raw", field), ("aligned", aligned)):
        scale = float(np.abs(values[:split]).std())
        targets = torch.from_numpy(to_channels(values) / scale).to(device)
        config = OperatorConfig(
            out_channels=2 * FREQUENCIES, width=args.width, modes=args.modes
        )
        model, final_loss = train(
            inputs_t[:split], targets[:split], config, args.steps, 2e-3, device, seed=0
        )
        with torch.no_grad():
            prediction = model(inputs_t[split:]).cpu().numpy() * scale
        half = prediction.shape[1] // 2
        predicted = prediction[:, :half] + 1j * prediction[:, half:]
        if label == "aligned":
            predicted = predicted * np.conj(ramp[split:])
        truth = field[split:]
        error = float(np.linalg.norm(predicted - truth) / np.linalg.norm(truth))
        per_case = [
            float(np.linalg.norm(predicted[i] - truth[i]) / np.linalg.norm(truth[i]))
            for i in range(len(truth))
        ]
        parameters = sum(p.numel() for p in model.parameters())
        rows.append(
            {
                "target": label,
                "regime": args.regime,
                "test_nrmse": error,
                "test_nrmse_median": float(np.median(per_case)),
                "train_loss": final_loss,
                "parameters": int(parameters),
                "n_train": split,
                "n_test": len(truth),
            }
        )
        print(
            f"{label:8s} test NRMSE {error:.4f} (median {np.median(per_case):.4f}) "
            f"train loss {final_loss:.5f}  params {parameters / 1e6:.2f}M",
            flush=True,
        )

    payload = {"rows": rows, "gain": rows[0]["test_nrmse"] / rows[1]["test_nrmse"]}
    payload["family"] = args.family
    (RESULTS / f"exp15_operator_{args.family}_{args.regime}.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(f"carrier gain: {payload['gain']:.2f}x")


if __name__ == "__main__":
    main()
