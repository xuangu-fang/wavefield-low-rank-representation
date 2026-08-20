"""Experiment 22: identifiability, not capacity.

If the failure to reconstruct between sensors were a modelling failure, a
bigger or better-trained model would fix it. If it is an identifiability
failure, nothing will -- and the same problem becomes easy once a carrier moves
the field's energy below the array's Nyquist wavenumber, without adding a
single measurement.

Five estimators spanning three capacity decades see identical regular arrays in
both coordinate systems, and every one is compared against the one-FFT bound.
Training error is recorded alongside test error so undertraining can be told
apart from unidentifiability.
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
from wave_lr.metrics import complex_nrmse
from wave_lr.spatial import identifiability_bound, infer_spacing, to_grid
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
NETWORKS = (
    ("fourier_small", "fourier", {"width": 64, "depth": 2, "steps": 1500}),
    ("fourier_base", "fourier", {"width": 256, "depth": 3, "steps": 4000}),
    ("fourier_large", "fourier", {"width": 512, "depth": 4, "steps": 8000}),
    ("siren", "siren", {"width": 256, "depth": 3, "steps": 4000}),
)


def uniform_mask(coords, spacing, stride):
    rows = np.rint(coords[:, 0] / spacing).astype(int)
    cols = np.rint(coords[:, 1] / spacing).astype(int)
    return ((rows - rows.min()) % stride == 0) & ((cols - cols.min()) % stride == 0)


def nearest_fill(coords, values, observed):
    from scipy.interpolate import NearestNDInterpolator

    return NearestNDInterpolator(coords[observed], values[observed])(coords)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
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

            for stride in STRIDES:
                observed = uniform_mask(case.coords, spacing, stride)
                hidden = ~observed
                if observed.sum() < 12 or hidden.sum() < 12:
                    continue
                for coordinate, delays in (("raw", None), ("aligned", case.travel_time)):
                    ramp = (
                        None if delays is None else np.conj(carrier(frequencies, delays))
                    )
                    working = truth if ramp is None else truth * ramp
                    grid = to_grid(working[:, 0], case.coords)
                    row = {
                        "regime": regime,
                        "seed": seed,
                        "stride": stride,
                        "coordinate": coordinate,
                        "n_sensors": int(observed.sum()),
                        "bound": float(identifiability_bound(grid, 1.0 / stride**2)),
                    }
                    predicted = interpolate_from_sensors(
                        case.coords, truth, frequencies, delays, observed
                    )
                    row["linear_test"] = complex_nrmse(predicted[hidden], truth[hidden])
                    filled = nearest_fill(case.coords, working, observed)
                    if ramp is not None:
                        filled = filled * np.conj(ramp)  # back to physical units
                    row["nearest_test"] = complex_nrmse(filled[hidden], truth[hidden])
                    for label, encoding, options in NETWORKS:
                        config = TrainConfig(seed=seed, **options)
                        estimate = fit_field_network(
                            case.coords, truth, observed, encoding,
                            delays=delays, frequencies=frequencies, config=config,
                        )
                        row[f"{label}_test"] = complex_nrmse(
                            estimate[hidden], truth[hidden]
                        )
                        row[f"{label}_train"] = complex_nrmse(
                            estimate[observed], truth[observed]
                        )
                        row[f"{label}_parameters"] = (
                            options["width"] ** 2 * options["depth"]
                        )
                    keys = [k for k in row if k.endswith("_test")]
                    row["best_estimator"] = min(keys, key=lambda k: row[k])
                    row["best_test"] = min(row[k] for k in keys)
                    rows.append(row)
                    print(
                        f"{regime:14s} s{seed} stride={stride:2d} {coordinate:8s} "
                        f"bound {row['bound']:.3f}  best {row['best_test']:.3f} "
                        f"({row['best_estimator'].replace('_test', '')})  "
                        f"({time.time() - start:5.0f}s)",
                        flush=True,
                    )

    beaten = [r for r in rows if r["best_test"] < 0.9 * r["bound"]]
    capacity = {}
    for label, _, options in NETWORKS:
        capacity[label] = {
            "mean_test": float(np.mean([r[f"{label}_test"] for r in rows])),
            "mean_train": float(np.mean([r[f"{label}_train"] for r in rows])),
            "parameters": options["width"] ** 2 * options["depth"],
        }
    summary = {
        "n_settings": len(rows),
        "bound_beaten": len(beaten),
        "bound_beaten_rate": len(beaten) / max(len(rows), 1),
        "capacity_sweep": capacity,
        "mean_linear_over_bound": float(
            np.mean([r["linear_test"] / max(r["bound"], 1e-9) for r in rows])
        ),
    }
    (RESULTS / "exp22_no_estimator_beats_it.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
