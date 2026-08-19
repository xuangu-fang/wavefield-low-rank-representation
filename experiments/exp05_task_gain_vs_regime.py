"""Experiment 5: does the predicted rank gain predict the task gain?

For every regime in the FDTD phase diagram the same two tasks are solved in
raw and in carrier coordinates, with identical observations and identical
solvers. The question is not whether alignment helps -- experiment 3 showed it
helps in open media -- but whether the *size* of the help is predicted by the
delay-occupancy law, including where it predicts no help at all.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from wave_lr.analysis import analyze_case
from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
from wave_lr.spectra import to_spectrum
from wave_lr.tasks import completion_curve, random_entry_mask, sensor_interpolation_report
from wave_lr.theory import fit_slope

RESULTS = Path(__file__).resolve().parents[1] / "results"
ABSORPTIONS = {"open": 40.0, "partial": 3.0, "closed": 0.0}
SCATTERERS = {"clear": 0.0, "sparse": 0.08, "dense": 0.22, "cluttered": 0.4}
SENSOR_FRACTIONS = (0.01, 0.02, 0.05, 0.10)
COMPLETION_FRACTION = 0.05
CARRIERS = ("raw", "eikonal", "straight")
BAND = (6.0, 24.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--completion", action="store_true")
    args = parser.parse_args()

    rows = []
    start = time.time()
    for (boundary, absorption), (clutter, fraction), seed in itertools.product(
        ABSORPTIONS.items(), SCATTERERS.items(), range(args.seeds)
    ):
        spec = MediumSpec(
            name=f"{boundary}_{clutter}_s{seed}",
            scatterer_fraction=fraction,
            absorption=absorption,
            seed=seed,
        )
        case = fdtd_case(spec)
        row = analyze_case(case, *BAND)
        row.update({"boundary": boundary, "clutter": clutter, "seed": seed})
        spectrum = to_spectrum(case.traces, case.dt, *BAND)
        delays = {"raw": None, "eikonal": case.travel_time, "straight": case.straight_time}

        for name, sensor_fraction in itertools.product(CARRIERS, SENSOR_FRACTIONS):
            scores = sensor_interpolation_report(
                case.coords, spectrum.values, spectrum.frequencies,
                delays[name], sensor_fraction, seed=seed,
            )
            for key, value in scores.items():
                row[f"interp_{name}_p{int(sensor_fraction * 100)}_{key}"] = value

        if args.completion:
            observed = random_entry_mask(spectrum.values.shape, COMPLETION_FRACTION, seed=seed)
            for name in CARRIERS:
                scores = completion_curve(
                    spectrum.values, spectrum.frequencies, delays[name], observed,
                    ranks=(2, 4, 8, 16, 32, 64), iterations=60,
                )
                for key, value in scores.items():
                    if key.startswith("best"):
                        row[f"complete_{name}_{key}"] = value
        rows.append(row)
        print(
            f"{spec.name:24s} rank {row['raw_rank_90']:3d}->{row['eikonal_rank_90']:3d} "
            f"interp2% {row['interp_raw_p2_complex_nrmse']:.3f}->"
            f"{row['interp_eikonal_p2_complex_nrmse']:.3f} "
            f"({time.time() - start:5.1f}s)",
            flush=True,
        )

    summary = {}
    predicted_gain = np.array([r["eikonal_predicted_gain_90"] for r in rows])
    measured_gain = np.array([r["eikonal_measured_gain_90"] for r in rows])
    for sensor_fraction in SENSOR_FRACTIONS:
        tag = f"p{int(sensor_fraction * 100)}"
        task_gain = np.array(
            [
                r[f"interp_raw_{tag}_complex_nrmse"]
                / max(r[f"interp_eikonal_{tag}_complex_nrmse"], 1e-9)
                for r in rows
            ]
        )
        summary[f"task_gain_vs_predicted_rank_gain_{tag}"] = fit_slope(
            np.log(predicted_gain), np.log(task_gain)
        )
        summary[f"task_gain_vs_measured_rank_gain_{tag}"] = fit_slope(
            np.log(measured_gain), np.log(task_gain)
        )
    payload = {"bands": BAND, "sensor_fractions": SENSOR_FRACTIONS, "fits": summary, "rows": rows}
    (RESULTS / "exp05_task_gain_vs_regime.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
