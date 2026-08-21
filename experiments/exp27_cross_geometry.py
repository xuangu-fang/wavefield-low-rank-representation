"""Experiment 27: do the learned coordinates survive a change of geometry?

Experiment 26 amortised the coordinates across *media* while the source sat in
one place and the boundary never changed. That is transfer across content, not
across geometry. Here the source position and the boundary absorption become
held-out variables, and the network is asked for coordinates on configurations
it never saw.

The network is given what a practitioner actually knows -- the wavespeed, the
damping profile and where the source is -- and nothing else. In particular it
is never shown a travel time, and its only loss is the identifiability bound.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np

from wave_lr.eikonal import batched_travel_time
from wave_lr.fdtd import MediumSpec, build_medium, simulate
from wave_lr.operator import OperatorConfig, build_operator
from wave_lr.spatial import identifiability_bound, taper_window

RESULTS = Path(__file__).resolve().parents[1] / "results"
CACHE = Path(
    "/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/prepared"
)
BAND = (6.0, 24.0)
FREQUENCIES = 8
TARGET_STRIDE = 6
GRID = 128
MARGIN = 0.24
SOURCES = (
    (0.50, 0.28), (0.30, 0.30), (0.70, 0.30), (0.40, 0.55), (0.60, 0.50),
    (0.25, 0.60), (0.75, 0.62), (0.50, 0.72), (0.35, 0.45),
)
ABSORPTIONS = {"open": 40.0, "partial": 3.0, "closed": 0.0}


def generate(media_per_config: int, seed0: int = 0) -> dict:
    """Simulate every (source, boundary) configuration on shared random media."""

    rng = np.random.default_rng(seed0)
    specs = [
        MediumSpec(
            name=f"m{index}", grid=GRID,
            scatterer_fraction=float(rng.uniform(0.04, 0.20)),
            scatterer_contrast=float(rng.uniform(0.3, 0.6)),
            seed=seed0 + index,
        )
        for index in range(media_per_config)
    ]
    fields, speeds, dampings, sources, boundaries, travels = [], [], [], [], [], []
    frequencies = None
    for (source_index, source), (boundary, absorption) in itertools.product(
        enumerate(SOURCES), ABSORPTIONS.items()
    ):
        block_speed, block_damping = [], []
        for spec in specs:
            speed, damping = build_medium(
                MediumSpec(**{**spec.__dict__, "absorption": absorption})
            )
            block_speed.append(speed)
            block_damping.append(damping)
        block_speed = np.stack(block_speed)
        block_damping = np.stack(block_damping)
        row = round(source[0] * (GRID - 1))
        col = round(source[1] * (GRID - 1))
        frames, record_dt, spacing = simulate(
            block_speed, block_damping, (row, col), 12.0, 6.0, record_every=4
        )
        padded = 2 * frames.shape[1]
        spectrum = np.fft.rfft(frames.astype(np.float64), n=padded, axis=1)
        freqs = np.fft.rfftfreq(padded, record_dt)
        keep = np.linspace(
            int(np.searchsorted(freqs, BAND[0])),
            int(np.searchsorted(freqs, BAND[1])) - 1,
            FREQUENCIES,
        ).astype(int)
        frequencies = freqs[keep]
        mask = np.zeros((len(specs), GRID, GRID), dtype=bool)
        mask[:, row, col] = True
        travel = batched_travel_time(
            block_speed, mask, spacing=spacing, iterations=6 * GRID
        )
        fields.append(spectrum[:, keep].astype(np.complex64))
        speeds.append(block_speed.astype(np.float32))
        dampings.append(block_damping.astype(np.float32))
        travels.append(travel.astype(np.float32))
        sources.append(np.repeat([[source_index, *source]], len(specs), axis=0))
        boundaries.append([boundary] * len(specs))
        print(f"  simulated source {source_index} / {boundary}", flush=True)
    return {
        "field": np.concatenate(fields),
        "speed": np.concatenate(speeds),
        "damping": np.concatenate(dampings),
        "travel_time": np.concatenate(travels),
        "source": np.concatenate(sources).astype(np.float32),
        "boundary": np.concatenate([np.array(b) for b in boundaries]),
        "frequencies": frequencies,
        "spacing": spacing,
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"geometry_pool_{GRID}_{args.media}.npz"
    if cache.exists():
        data = dict(np.load(cache, allow_pickle=True))
        print(f"loaded {cache.name}")
    else:
        start = time.time()
        data = generate(args.media)
        np.savez_compressed(cache, **data)
        print(f"generated pool in {time.time() - start:.0f}s")

    field = data["field"]
    speed = data["speed"]
    damping = data["damping"]
    travel = data["travel_time"]
    source = data["source"]
    boundary = np.array([str(b) for b in data["boundary"]])
    frequencies = data["frequencies"]

    axis = np.linspace(0.0, 1.0, GRID, dtype=np.float32)
    mesh = np.stack(np.meshgrid(axis, axis, indexing="ij"))
    distance = np.stack(
        [
            np.hypot(mesh[0] - s[1], mesh[1] - s[2]).astype(np.float32)
            for s in source
        ]
    )
    inputs = np.concatenate(
        [
            speed[:, None],
            damping[:, None] / max(float(damping.max()), 1e-6),
            distance[:, None],
            np.broadcast_to(mesh, (len(speed), 2, GRID, GRID)),
        ],
        axis=1,
    ).astype(np.float32)

    margin = round(MARGIN * GRID)
    interior = (slice(margin, GRID - margin), slice(margin, GRID - margin))
    window = torch.from_numpy(
        taper_window((GRID - 2 * margin, GRID - 2 * margin))
    ).to(device).float()
    rows = np.fft.fftfreq(GRID - 2 * margin)[:, None]
    cols = np.fft.fftfreq(GRID - 2 * margin)[None, :]
    outside = torch.from_numpy(
        (np.sqrt(rows**2 + cols**2) > 0.5 / TARGET_STRIDE).astype(np.float32)
    ).to(device)

    scale = float(np.abs(travel).max())
    inputs_t = torch.from_numpy(inputs).to(device)
    field_t = torch.from_numpy(field[:, :, interior[0], interior[1]]).to(device)
    freq_t = torch.from_numpy(frequencies.astype(np.float32)).to(device)

    def loss_on(index, delays):
        phase = 2.0 * np.pi * freq_t[None, :, None, None] * delays[:, None]
        aligned = field_t[index] * torch.exp(1j * phase)
        power = torch.abs(torch.fft.fft2(aligned * window[None, None])) ** 2
        return (power * outside[None, None]).sum() / power.sum().clamp_min(1e-30)

    source_index = source[:, 0].astype(int)
    splits = {
        "in_distribution": (
            np.flatnonzero(np.arange(len(speed)) % 5 != 0),
            np.flatnonzero(np.arange(len(speed)) % 5 == 0),
        ),
        "unseen_source": (
            np.flatnonzero(source_index < 6),
            np.flatnonzero(source_index >= 6),
        ),
        "unseen_boundary": (
            np.flatnonzero(boundary == "open"),
            np.flatnonzero(boundary != "open"),
        ),
        # Isolates held-out-ness from heterogeneity: one boundary regime only,
        # so the model does not have to serve three at once.
        "unseen_source_open_only": (
            np.flatnonzero((boundary == "open") & (source_index < 6)),
            np.flatnonzero((boundary == "open") & (source_index >= 6)),
        ),
        "in_distribution_open_only": (
            np.flatnonzero((boundary == "open") & (np.arange(len(speed)) % 5 != 0)),
            np.flatnonzero((boundary == "open") & (np.arange(len(speed)) % 5 == 0)),
        ),
    }

    def bound_for(index, delays):
        values = []
        for position, case in enumerate(index):
            phase = 2.0 * np.pi * frequencies[:, None, None] * delays[position]
            aligned = field[case][:, interior[0], interior[1]] * np.exp(1j * phase)
            values.append(
                identifiability_bound(
                    aligned[aligned.shape[0] // 2], 1.0 / TARGET_STRIDE**2
                )
            )
        return float(np.mean(values)), float(np.std(values))

    rows_out = []
    for name, (train_index, test_index) in splits.items():
        zeros = [np.zeros((GRID - 2 * margin,) * 2) for _ in test_index]
        raw_mean, _ = bound_for(test_index, zeros)
        eik_mean, eik_std = bound_for(
            test_index, [travel[c][interior[0], interior[1]] for c in test_index]
        )
        learned = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = build_operator(
                OperatorConfig(in_channels=5, out_channels=1, width=32, modes=20, depth=4)
            ).to(device)
            optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=args.steps
            )
            generator = torch.Generator().manual_seed(seed)
            for _ in range(args.steps):
                pick = torch.randperm(len(train_index), generator=generator)[:8]
                index = torch.from_numpy(train_index[pick.numpy()]).to(device)
                optimiser.zero_grad(set_to_none=True)
                delays = model(inputs_t[index])[:, 0] * scale
                loss = loss_on(index, delays[:, interior[0], interior[1]])
                loss.backward()
                optimiser.step()
                schedule.step()
            with torch.no_grad():
                predicted = (
                    model(inputs_t[torch.from_numpy(test_index).to(device)])[:, 0] * scale
                ).cpu().numpy()
            mean, _ = bound_for(
                test_index, [predicted[i][interior[0], interior[1]] for i in range(len(test_index))]
            )
            learned.append(mean)
        rows_out.append(
            {
                "split": name,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "raw_bound": raw_mean,
                "eikonal_bound": eik_mean,
                "eikonal_bound_std": eik_std,
                "learned_bound": float(np.mean(learned)),
                "learned_bound_seed_std": float(np.std(learned)),
            }
        )
        last = rows_out[-1]
        print(
            f"{name:18s} train {last['n_train']:4d} test {last['n_test']:4d} | "
            f"raw {raw_mean:.3f}  eikonal {eik_mean:.3f}  learned "
            f"{last['learned_bound']:.3f} ± {last['learned_bound_seed_std']:.3f}",
            flush=True,
        )

    (RESULTS / "exp27_cross_geometry.json").write_text(
        json.dumps({"target_stride": TARGET_STRIDE, "sources": SOURCES,
                    "absorptions": ABSORPTIONS, "rows": rows_out}, indent=2)
    )
    print(json.dumps(rows_out, indent=2))


if __name__ == "__main__":
    main()
