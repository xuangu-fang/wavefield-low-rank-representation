"""Experiment 23: the identifiability bound on public frequency-domain data.

Experiment 21 measured the bound on our own solver plus two time-domain public
sets. The two harmonic public datasets -- The Well's Helmholtz staircase and
WaveBench's time-harmonic media -- are the ones with a supplied wavespeed, so
the carrier is deployable and the bound can be checked exactly where a
practitioner would use it.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from wave_lr.harmonic import load_staircase
from wave_lr.spatial import (
    block_weights,
    identifiability_bound,
    infer_spacing,
    largest_full_rectangle,
    to_grid,
)
from wave_lr.spectra import carrier
from wave_lr.tasks import interpolate_from_sensors
from wave_lr.theory import fit_slope
from wave_lr.wavebench import coordinates, load_samples

RESULTS = Path(__file__).resolve().parents[1] / "results"
STRIDES = (2, 3, 4, 6, 8, 11, 16)


def uniform_mask(coords, spacing, stride):
    rows = np.rint(coords[:, 0] / spacing).astype(int)
    cols = np.rint(coords[:, 1] / spacing).astype(int)
    return ((rows - rows.min()) % stride == 0) & ((cols - cols.min()) % stride == 0)


def usable(coords, mask, spacing, minimum=3):
    """Reject sub-arrays too degenerate to triangulate (a line of sensors)."""

    if mask.sum() < 12:
        return False
    rows = np.rint(coords[mask, 0] / spacing).astype(int)
    cols = np.rint(coords[mask, 1] / spacing).astype(int)
    return len(np.unique(rows)) >= minimum and len(np.unique(cols)) >= minimum


def measure(label, name, coords, values, frequencies, delays_by_coordinate) -> list[dict]:
    spacing = infer_spacing(coords)
    # Crop away masked cells before transforming; see largest_full_rectangle.
    block = largest_full_rectangle(coords, spacing)
    weights = np.zeros(len(coords))
    weights[block] = block_weights(coords[block])
    rows = []
    for coordinate, delays in delays_by_coordinate.items():
        working = (
            values if delays is None else values * np.conj(carrier(frequencies, delays))
        )
        grid = to_grid(working[block, 0], coords[block])
        for stride in STRIDES:
            observed = uniform_mask(coords, spacing, stride) & block
            hidden = (~observed) & block
            if not usable(coords, observed, spacing) or hidden.sum() < 12:
                continue
            predicted = interpolate_from_sensors(
                coords, values, frequencies, delays, observed
            )
            weight = np.sqrt(weights[hidden])[:, None]
            rows.append(
                {
                    "dataset": label,
                    "case": name,
                    "coordinate": coordinate,
                    "stride": stride,
                    "n_sensors": int(observed.sum()),
                    "bound": float(identifiability_bound(grid, 1.0 / stride**2)),
                    "measured": float(
                        np.linalg.norm((predicted[hidden] - values[hidden]) * weight)
                        / np.linalg.norm(values[hidden] * weight)
                    ),
                }
            )
    return rows


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--staircase", type=int, default=4)
    parser.add_argument("--wavebench", type=int, default=6)
    parser.add_argument("--subsample", type=int, default=2)
    args = parser.parse_args()

    rows = []
    for case in load_staircase("train", limit=args.staircase, subsample=args.subsample):
        # The highest frequency is the hardest and the most informative.
        index = len(case.frequencies) - 1
        rows += measure(
            "helmholtz_staircase", case.name, case.coords,
            case.fields[:, index : index + 1],
            case.frequencies[index : index + 1],
            {"raw": None, "aligned": case.travel_time},
        )
        print(f"staircase {case.name} done", flush=True)

    for label in (10, 40):
        for sample in load_samples(label, count=args.wavebench):
            grid = sample.speed.shape[0]
            coords = coordinates(grid)
            values = sample.field.ravel()[:, None]
            keep = np.abs(values[:, 0]) > 0
            rows += measure(
                f"wavebench_omega{label}", f"sample{sample.index}", coords[keep],
                values[keep], np.array([sample.kappa / (2.0 * np.pi)]),
                {"raw": None, "aligned": sample.travel_time.ravel()[keep]},
            )
        print(f"wavebench omega {label} done", flush=True)

    usable = [r for r in rows if r["bound"] > 1e-6 and np.isfinite(r["measured"])]
    fits = {
        "error_vs_bound": fit_slope(
            np.log([r["bound"] for r in usable]), np.log([r["measured"] for r in usable])
        ),
        "bound_violation_rate": float(
            np.mean([r["measured"] < 0.9 * r["bound"] for r in usable])
        ),
        "n_pairs": len(usable),
    }
    for dataset in sorted({r["dataset"] for r in usable}):
        subset = [r for r in usable if r["dataset"] == dataset]
        fits[f"violation_{dataset}"] = float(
            np.mean([r["measured"] < 0.9 * r["bound"] for r in subset])
        )
    (RESULTS / "exp23_public_identifiability.json").write_text(
        json.dumps({"strides": list(STRIDES), "fits": fits, "rows": rows}, indent=2)
    )
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
