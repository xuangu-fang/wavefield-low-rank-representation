"""Experiment 24: can the missing information be learned rather than supplied?

The bound is a property of the field *in a coordinate system*. Physics lowers
it by supplying the travel time. The question here is whether the same
reduction can be obtained without the medium -- by minimising the field's own
out-of-band energy, which is exactly the quantity the bound measures and which
needs no labels and no solver.

Five coordinate systems are compared on the identical field: none, the eikonal
carrier, a corrupted eikonal carrier past the 1/B tolerance, a carrier learned
from scratch, and the corrupted carrier repaired by learning.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from exp07_carrier_error_tolerance import smooth_error

from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
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

RESULTS = Path(__file__).resolve().parents[1] / "results"
BAND = (6.0, 24.0)
STRIDES = (3, 6, 11)
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
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=250)
    args = parser.parse_args()

    rows = []
    start = time.time()
    for regime, settings in REGIMES.items():
        for seed in range(args.seeds):
            case = fdtd_case(MediumSpec(name=f"{regime}_s{seed}", seed=seed, **settings))
            spectrum = to_spectrum(case.traces, case.dt, *BAND)
            middle = spectrum.values.shape[1] // 2
            frequencies = spectrum.frequencies[middle : middle + 1]
            truth = spectrum.values[:, middle : middle + 1]
            spacing = infer_spacing(case.coords)
            block = largest_full_rectangle(case.coords, spacing)
            weights = np.zeros(len(case.coords))
            weights[block] = block_weights(case.coords[block])
            resolution = 1.0 / spectrum.bandwidth
            corrupted = case.travel_time + resolution * smooth_error(case.coords, seed=seed)

            def learn(
                initial, learning_rate, steps, warmup, objective,
                seed=seed, spectrum=spectrum, case=case,
            ):
                delays, _ = fit_learned_carrier(
                    spectrum.values, spectrum.frequencies, case.coords,
                    CarrierConfig(
                        steps=steps, warmup_steps=warmup,
                        learning_rate=learning_rate, seed=seed, budget=8,
                        objective=objective, target_stride=6,
                    ),
                    initial_delays=initial,
                )
                return delays

            total = args.steps + args.warmup
            coordinates = {
                "none": None,
                "eikonal": case.travel_time,
                "eikonal_corrupted": corrupted,
                # The bound itself is the loss: diagnostic, objective and metric
                # become one quantity.
                "learned_scratch": learn(None, 1e-3, total, 0, "aliasing"),
                "learned_repair": learn(corrupted, 3e-4, args.steps, args.warmup, "aliasing"),
                "learned_from_eikonal": learn(
                    case.travel_time, 3e-4, args.steps, args.warmup, "aliasing"
                ),
                # Ablation: optimising the frequency-axis objective instead.
                "learned_tail_objective": learn(None, 1e-3, total, 0, "tail"),
            }

            for name, delays in coordinates.items():
                working = (
                    truth
                    if delays is None
                    else truth * np.conj(carrier(frequencies, delays))
                )
                grid = to_grid(working[block, 0], case.coords[block])
                row = {"regime": regime, "seed": seed, "coordinate": name}
                for stride in STRIDES:
                    observed = uniform_mask(case.coords, spacing, stride) & block
                    hidden = (~observed) & block
                    row[f"bound_s{stride}"] = float(
                        identifiability_bound(grid, 1.0 / stride**2)
                    )
                    if observed.sum() < 12 or hidden.sum() < 12:
                        row[f"error_s{stride}"] = float("nan")
                        continue
                    predicted = interpolate_from_sensors(
                        case.coords, truth, frequencies, delays, observed
                    )
                    weight = np.sqrt(weights[hidden])[:, None]
                    row[f"error_s{stride}"] = float(
                        np.linalg.norm((predicted[hidden] - truth[hidden]) * weight)
                        / np.linalg.norm(truth[hidden] * weight)
                    )
                rows.append(row)
            last = {r["coordinate"]: r for r in rows[-len(coordinates):]}
            print(
                f"{regime:14s} s{seed} bound@6  "
                + "  ".join(
                    f"{k}={last[k]['bound_s6']:.3f}"
                    for k in ("none", "eikonal", "learned_scratch", "learned_repair")
                )
                + f"  ({time.time() - start:5.0f}s)",
                flush=True,
            )

    summary = {}
    for name in (
        "none", "eikonal", "eikonal_corrupted", "learned_scratch",
        "learned_repair", "learned_from_eikonal", "learned_tail_objective",
    ):
        subset = [r for r in rows if r["coordinate"] == name]
        summary[name] = {
            f"mean_bound_s{stride}": float(
                np.nanmean([r[f"bound_s{stride}"] for r in subset])
            )
            for stride in STRIDES
        }
        summary[name].update(
            {
                f"mean_error_s{stride}": float(
                    np.nanmean([r[f"error_s{stride}"] for r in subset])
                )
                for stride in STRIDES
            }
        )
    (RESULTS / "exp24_learned_identifiability.json").write_text(
        json.dumps({"strides": list(STRIDES), "summary": summary, "rows": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
