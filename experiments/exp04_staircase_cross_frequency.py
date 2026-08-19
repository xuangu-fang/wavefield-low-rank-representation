"""Experiment 4: cross-frequency extrapolation on The Well's Helmholtz staircase.

The carrier ``exp(-i omega tau(x))`` carries the entire fast dependence of the
field on frequency, so the residual should be far easier to continue to unseen
higher frequencies than the field itself. This is the APEX setting -- predict
higher frequencies from a data-rich low band -- on public frequency-domain data.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
from pathlib import Path

import numpy as np

from wave_lr.diagnostics import singular_spectrum
from wave_lr.harmonic import load_staircase
from wave_lr.spectra import carrier
from wave_lr.tasks import (
    copy_last_baseline,
    extrapolation_report,
    low_rank_extrapolation_report,
)
from wave_lr.theory import delay_occupancy, predicted_rank

RESULTS = Path(__file__).resolve().parents[1] / "results"


def rank_table(case) -> dict:
    """Numerical rank of the (x, omega) matrix under each carrier."""

    frequencies = case.frequencies
    bandwidth = float(frequencies[-1] - frequencies[0])
    out = {"bandwidth_hz": bandwidth, "n_omega": len(frequencies)}
    for name, delays in (
        ("raw", None),
        ("eikonal", case.travel_time),
        ("straight", case.straight_time),
    ):
        values = (
            case.fields
            if delays is None
            else case.fields * np.conj(carrier(frequencies, delays))
        )
        spectrum = singular_spectrum(values)
        cumulative = np.cumsum(spectrum**2)
        for level in (0.90, 0.99):
            out[f"{name}_rank_{int(level * 100)}"] = int(
                np.searchsorted(cumulative, level) + 1
            )
        if delays is not None:
            out[f"{name}_occupancy"] = delay_occupancy(
                delays - delays.min(), np.ones_like(delays), bandwidth=bandwidth
            )
            out[f"{name}_predicted_rank"] = predicted_rank(
                bandwidth, out[f"{name}_occupancy"]
            )
    out["raw_occupancy"] = delay_occupancy(
        case.travel_time, np.ones_like(case.travel_time), bandwidth=bandwidth
    )
    out["raw_predicted_rank"] = predicted_rank(bandwidth, out["raw_occupancy"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--subsample", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = load_staircase(args.split, limit=args.limit, subsample=args.subsample)
    rows = []
    for case in cases:
        frequencies = case.frequencies
        base = {"case": case.name, "source_xy": case.metadata["source_xy"]}
        base.update(rank_table(case))
        for n_train in (6, 8, 10, 12):
            row = dict(base)
            row["n_train"] = n_train
            row["train_f_max"] = float(frequencies[n_train - 1])
            row["target_f_max"] = float(frequencies[-1])
            row["frequency_reach"] = float(frequencies[-1] / frequencies[n_train - 1])
            for key, value in copy_last_baseline(case.fields, n_train).items():
                row[f"copy_{key}"] = value
            for name, delays in (
                ("raw", None),
                ("eikonal", case.travel_time),
                ("straight", case.straight_time),
            ):
                for mode in ("complex", "amplitude_phase"):
                    report = extrapolation_report(
                        case.fields, frequencies, n_train, delays, mode=mode
                    )
                    tag = f"{name}_{mode}"
                    for key, value in report.items():
                        if key.startswith("best"):
                            row[f"{tag}_{key}"] = value
                report = low_rank_extrapolation_report(
                    case.fields, frequencies, n_train, delays
                )
                for key, value in report.items():
                    if key.startswith("best"):
                        row[f"{name}_lowrank_{key}"] = value
            rows.append(row)
            print(
                f"{case.name} n_train={n_train:2d} reach={row['frequency_reach']:.2f} "
                f"copy={row['copy_complex_nrmse']:.3f} "
                f"raw={row['raw_complex_best_complex_nrmse']:.3f} "
                f"ampphase={row['raw_amplitude_phase_best_complex_nrmse']:.3f} "
                f"eikonal={row['eikonal_complex_best_complex_nrmse']:.3f} | "
                f"LR raw={row['raw_lowrank_best_complex_nrmse']:.3f} "
                f"LR eik={row['eikonal_lowrank_best_complex_nrmse']:.3f} "
                f"(r={row['eikonal_lowrank_best_rank']:.0f})",
                flush=True,
            )

    payload = {"split": args.split, "n_cases": len(cases), "rows": rows}
    (RESULTS / f"exp04_staircase_{args.split}.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote exp04_staircase_{args.split}.json")


if __name__ == "__main__":
    main()
