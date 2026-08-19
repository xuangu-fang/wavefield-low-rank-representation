"""Experiment 6: the same two tasks on public data, in both regimes.

The Well's acoustic maze is the reverberant extreme and its Helmholtz
staircase is a waveguide with a moderate delay spread. Running identical task
code on both tests whether the regime -- not the dataset -- decides how much a
carrier is worth.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from wave_lr.analysis import analyze_case
from wave_lr.diagnostics import singular_spectrum
from wave_lr.fields import load_well_acoustic, load_well_scattering
from wave_lr.harmonic import load_staircase
from wave_lr.spectra import to_spectrum
from wave_lr.tasks import completion_curve, random_entry_mask, sensor_interpolation_report

RESULTS = Path(__file__).resolve().parents[1] / "results"
SENSOR_FRACTIONS = (0.01, 0.02, 0.05, 0.10)
CARRIERS = ("raw", "eikonal", "straight")
MAZE_BAND = (3.0, 13.0)
INCLUSION_BAND = (3.0, 13.0)


def ranks(values: np.ndarray) -> dict:
    spectrum = singular_spectrum(values)
    cumulative = np.cumsum(spectrum**2)
    return {
        f"rank_{int(level * 100)}": int(np.searchsorted(cumulative, level) + 1)
        for level in (0.90, 0.99)
    }


def run_tasks(coords, values, frequencies, delays_by_name, seed, completion) -> dict:
    row = {}
    for name, sensor_fraction in itertools.product(CARRIERS, SENSOR_FRACTIONS):
        scores = sensor_interpolation_report(
            coords, values, frequencies, delays_by_name[name], sensor_fraction, seed=seed
        )
        for key, value in scores.items():
            row[f"interp_{name}_p{int(sensor_fraction * 100)}_{key}"] = value
    if completion:
        observed = random_entry_mask(values.shape, 0.05, seed=seed)
        for name in CARRIERS:
            scores = completion_curve(
                values, frequencies, delays_by_name[name], observed,
                ranks=(1, 2, 4, 8, 16, 32), iterations=60,
            )
            for key, value in scores.items():
                if key.startswith("best"):
                    row[f"complete_{name}_{key}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze-limit", type=int, default=12)
    parser.add_argument("--subsample", type=int, default=2)
    parser.add_argument("--inclusion-limit", type=int, default=10)
    parser.add_argument("--staircase-split", default="test")
    parser.add_argument("--staircase-limit", type=int, default=None)
    parser.add_argument("--skip", default="", help="comma-separated: maze,inclusions,staircase")
    parser.add_argument("--no-completion", action="store_true")
    parser.add_argument("--out", default="exp06_public_data_tasks.json")
    args = parser.parse_args()
    completion = not args.no_completion
    skip = {name for name in args.skip.split(",") if name}

    rows = []
    for index, case in enumerate(
        [] if "maze" in skip else load_well_acoustic("test", limit=args.maze_limit)
    ):
        row = analyze_case(case, *MAZE_BAND)
        spectrum = to_spectrum(case.traces, case.dt, *MAZE_BAND)
        delays = {"raw": None, "eikonal": case.travel_time, "straight": case.straight_time}
        row.update(
            run_tasks(case.coords, spectrum.values, spectrum.frequencies, delays, index, completion)
        )
        rows.append(row)
        print(
            f"maze {case.name} rank {row['raw_rank_90']:3d}->{row['eikonal_rank_90']:3d} "
            f"interp2% {row['interp_raw_p2_complex_nrmse']:.3f}->"
            f"{row['interp_eikonal_p2_complex_nrmse']:.3f}",
            flush=True,
        )

    for index, case in enumerate(
        [] if "inclusions" in skip else load_well_scattering(limit=args.inclusion_limit)
    ):
        row = analyze_case(case, *INCLUSION_BAND)
        spectrum = to_spectrum(case.traces, case.dt, *INCLUSION_BAND)
        delays = {"raw": None, "eikonal": case.travel_time, "straight": case.straight_time}
        row.update(
            run_tasks(case.coords, spectrum.values, spectrum.frequencies, delays, index, completion)
        )
        rows.append(row)
        print(
            f"inclusions {case.name} rank {row['raw_rank_90']:3d}->{row['eikonal_rank_90']:3d} "
            f"interp2% {row['interp_raw_p2_complex_nrmse']:.3f}->"
            f"{row['interp_eikonal_p2_complex_nrmse']:.3f}",
            flush=True,
        )

    for index, case in enumerate(
        []
        if "staircase" in skip
        else load_staircase(
            args.staircase_split, limit=args.staircase_limit, subsample=args.subsample
        )
    ):
        delays = {"raw": None, "eikonal": case.travel_time, "straight": case.straight_time}
        row = {"case": case.name, "dataset": case.dataset, "n_x": case.fields.shape[0]}
        for name, tau in delays.items():
            from wave_lr.spectra import carrier

            values = (
                case.fields
                if tau is None
                else case.fields * np.conj(carrier(case.frequencies, tau))
            )
            for key, value in ranks(values).items():
                row[f"{name}_{key}"] = value
        row.update(
            run_tasks(case.coords, case.fields, case.frequencies, delays, index, completion)
        )
        rows.append(row)
        print(
            f"staircase {case.name} rank {row['raw_rank_90']:3d}->{row['eikonal_rank_90']:3d} "
            f"interp2% {row['interp_raw_p2_complex_nrmse']:.3f}->"
            f"{row['interp_eikonal_p2_complex_nrmse']:.3f}",
            flush=True,
        )

    name = args.out
    (RESULTS / name).write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {name}")


if __name__ == "__main__":
    main()
