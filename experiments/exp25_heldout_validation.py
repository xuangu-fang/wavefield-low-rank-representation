"""Experiment 25: the frozen procedure on media never used for any choice.

Every measurement decision in this project -- regular arrays, cropping instead
of zero-filling, the Tukey taper and its width, the energy quantile -- was made
while looking at the fields in experiments 21 to 24. That is exactly the
situation in which a result can be an artefact of its own tuning.

Nothing is tuned here. The procedure is frozen and run on twelve media whose
seeds appear nowhere else, and the three headline relationships are re-measured:
the bound holds, capacity does not help, and learning recovers the coordinates.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
from wave_lr.inr import TrainConfig, fit_field_network
from wave_lr.learned_carrier import CarrierConfig, fit_learned_carrier
from wave_lr.spatial import (
    block_weights,
    identifiability_bound,
    infer_spacing,
    largest_full_rectangle,
    to_grid,
)
from wave_lr.spectra import carrier, to_spectrum
from wave_lr.tasks import interpolate_from_sensors
from wave_lr.theory import fit_slope

RESULTS = Path(__file__).resolve().parents[1] / "results"
BAND = (6.0, 24.0)
STRIDES = (2, 3, 4, 6, 8, 11, 16)
HELD_OUT_SEEDS = range(100, 112)  # used nowhere else in the repository
REGIMES = {
    "open_clear": {"absorption": 40.0, "scatterer_fraction": 0.0},
    "open_sparse": {"absorption": 40.0, "scatterer_fraction": 0.08},
    "partial_clear": {"absorption": 3.0, "scatterer_fraction": 0.0},
    "closed_dense": {"absorption": 0.0, "scatterer_fraction": 0.22},
}


def uniform_mask(coords, spacing, stride):
    rows = np.rint(coords[:, 0] / spacing).astype(int)
    cols = np.rint(coords[:, 1] / spacing).astype(int)
    return ((rows - rows.min()) % stride == 0) & ((cols - cols.min()) % stride == 0)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-regime", type=int, default=3)
    args = parser.parse_args()

    seeds = list(HELD_OUT_SEEDS)
    rows = []
    start = time.time()
    for index, (regime, settings) in enumerate(REGIMES.items()):
        for offset in range(args.per_regime):
            seed = seeds[index * args.per_regime + offset]
            case = fdtd_case(MediumSpec(name=f"heldout_{regime}_s{seed}", seed=seed, **settings))
            spectrum = to_spectrum(case.traces, case.dt, *BAND)
            middle = spectrum.values.shape[1] // 2
            frequencies = spectrum.frequencies[middle : middle + 1]
            truth = spectrum.values[:, middle : middle + 1]
            spacing = infer_spacing(case.coords)
            block = largest_full_rectangle(case.coords, spacing)
            weights = np.zeros(len(case.coords))
            weights[block] = block_weights(case.coords[block])

            learned, _ = fit_learned_carrier(
                spectrum.values, spectrum.frequencies, case.coords,
                CarrierConfig(
                    steps=650, warmup_steps=0, learning_rate=1e-3, seed=seed,
                    objective="aliasing", target_stride=6,
                ),
            )
            for coordinate, delays in (
                ("raw", None),
                ("eikonal", case.travel_time),
                ("learned_scratch", learned),
            ):
                working = (
                    truth if delays is None else truth * np.conj(carrier(frequencies, delays))
                )
                grid = to_grid(working[block, 0], case.coords[block])
                for stride in STRIDES:
                    observed = uniform_mask(case.coords, spacing, stride) & block
                    hidden = (~observed) & block
                    if observed.sum() < 12 or hidden.sum() < 12:
                        continue
                    weight = np.sqrt(weights[hidden])[:, None]
                    predicted = interpolate_from_sensors(
                        case.coords, truth, frequencies, delays, observed
                    )
                    row = {
                        "regime": regime,
                        "seed": seed,
                        "coordinate": coordinate,
                        "stride": stride,
                        "bound": float(identifiability_bound(grid, 1.0 / stride**2)),
                        "linear": float(
                            np.linalg.norm((predicted[hidden] - truth[hidden]) * weight)
                            / np.linalg.norm(truth[hidden] * weight)
                        ),
                    }
                    if stride in (3, 6, 11):
                        network = fit_field_network(
                            case.coords, truth, observed, "fourier",
                            delays=delays, frequencies=frequencies,
                            config=TrainConfig(width=512, depth=4, steps=8000, seed=seed),
                        )
                        row["network_large"] = float(
                            np.linalg.norm((network[hidden] - truth[hidden]) * weight)
                            / np.linalg.norm(truth[hidden] * weight)
                        )
                    rows.append(row)
            print(
                f"heldout {regime:14s} seed {seed} done ({time.time() - start:5.0f}s)",
                flush=True,
            )

    pairs = [(r["bound"], r["linear"]) for r in rows if r["bound"] > 1e-6]
    with_network = [r for r in rows if "network_large" in r]
    summary = {
        "n_media": args.per_regime * len(REGIMES),
        "n_measurements": len(rows),
        "bound_violation_rate_linear": float(
            np.mean([m < 0.9 * b for b, m in pairs])
        ),
        "bound_violation_rate_network": float(
            np.mean([r["network_large"] < 0.9 * r["bound"] for r in with_network])
        ),
        "error_vs_bound": fit_slope(
            np.log([b for b, _ in pairs]), np.log([m for _, m in pairs])
        ),
        "mean_linear": float(np.mean([m for _, m in pairs])),
        "mean_network_large": float(np.mean([r["network_large"] for r in with_network])),
    }
    for coordinate in ("raw", "eikonal", "learned_scratch"):
        subset = [r for r in rows if r["coordinate"] == coordinate and r["stride"] == 6]
        summary[f"bound_s6_{coordinate}"] = float(np.mean([r["bound"] for r in subset]))
        summary[f"linear_s6_{coordinate}"] = float(np.mean([r["linear"] for r in subset]))
    (RESULTS / "exp25_heldout_validation.json").write_text(
        json.dumps({"strides": list(STRIDES), "summary": summary, "rows": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
