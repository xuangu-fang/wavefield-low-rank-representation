"""Experiment 21: how many sensors does a wave field actually need?

The claim under test is an identifiability statement, not a claim about any
estimator: on a *regular* sensor array of spacing ``m``, energy above the
wavenumber ``1 / (2m)`` is indistinguishable from energy below it, so its
square root lower-bounds the relative error of every method. The bound costs
one FFT and no training. What a carrier does is move energy below the
threshold, which is why it changes how dense the array has to be.

Regular arrays are used deliberately. Under random sampling the aliasing wall
is not strict -- sparse high-wavenumber content can be recovered, as
compressed sensing exploits -- and the same quantity becomes a heuristic
rather than a bound. Regular arrays are also what sensing hardware looks like.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case, load_well_acoustic, load_well_scattering
from wave_lr.spatial import (
    block_weights,
    identifiability_bound,
    infer_spacing,
    largest_full_rectangle,
    required_fraction,
    to_grid,
)
from wave_lr.spectra import carrier, to_spectrum
from wave_lr.tasks import interpolate_from_sensors
from wave_lr.theory import fit_slope

RESULTS = Path(__file__).resolve().parents[1] / "results"
# Regular arrays: keep every ``m``-th grid point along each axis.
STRIDES = (2, 3, 4, 6, 8, 11, 16)
TARGETS = (0.5, 0.3)
ABSORPTIONS = {"open": 40.0, "partial": 3.0, "closed": 0.0}
SCATTERERS = {"clear": 0.0, "sparse": 0.08, "dense": 0.22, "cluttered": 0.4}


def crossing(fractions, errors, target):
    """Smallest sampling fraction whose measured error reaches ``target``."""

    errors = np.asarray(errors)
    below = np.flatnonzero(np.nan_to_num(errors, nan=np.inf) <= target)
    if below.size == 0:
        return float("nan")
    index = int(below[0])
    if index == 0:
        return float(fractions[0])
    x0, x1 = np.log(fractions[index - 1]), np.log(fractions[index])
    y0, y1 = np.log(errors[index - 1]), np.log(errors[index])
    if y0 == y1:
        return float(fractions[index])
    return float(np.exp(x0 + (np.log(target) - y0) * (x1 - x0) / (y1 - y0)))


def uniform_mask(coords, spacing, stride):
    """Boolean mask selecting a regular sub-array of the original grid."""

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


def evaluate(case, band, label, seeds=2):
    spectrum = to_spectrum(case.traces, case.dt, *band)
    middle = spectrum.values.shape[1] // 2
    frequencies = spectrum.frequencies[middle : middle + 1]
    # Inferred, not taken from metadata: strided cases would leave empty rows.
    spacing = infer_spacing(case.coords)
    # Masked domains are cropped, never zero-filled, before the transform.
    block = largest_full_rectangle(case.coords, spacing)
    weights = np.zeros(len(case.coords))
    weights[block] = block_weights(case.coords[block])
    rows_in = np.rint(case.coords[block, 0] / spacing).astype(int)
    cols_in = np.rint(case.coords[block, 1] / spacing).astype(int)
    if len(np.unique(rows_in)) < 32 or len(np.unique(cols_in)) < 32:
        # A crop this thin cannot host the sparser arrays; skip rather than
        # report a number the geometry does not support.
        return []
    rows = []
    for coordinate, delays in (("raw", None), ("aligned", case.travel_time)):
        values = (
            spectrum.values[:, middle : middle + 1]
            if delays is None
            else (spectrum.values * np.conj(carrier(spectrum.frequencies, delays)))[
                :, middle : middle + 1
            ]
        )
        grid = to_grid(values[block, 0], case.coords[block])
        measured = []
        for stride in STRIDES:
            observed = uniform_mask(case.coords, spacing, stride) & block
            hidden = (~observed) & block
            if not usable(case.coords, observed, spacing) or hidden.sum() < 12:
                measured.append(float("nan"))
                continue
            predicted = interpolate_from_sensors(
                case.coords,
                spectrum.values[:, middle : middle + 1],
                frequencies,
                delays,
                observed,
            )
            truth = spectrum.values[:, middle : middle + 1]
            # Same taper as the bound, so the two are defined identically.
            weight = np.sqrt(weights[hidden])[:, None]
            measured.append(
                float(
                    np.linalg.norm((predicted[hidden] - truth[hidden]) * weight)
                    / np.linalg.norm(truth[hidden] * weight)
                )
            )
        row = {
            "dataset": label,
            "case": case.name,
            "band": list(band),
            "coordinate": coordinate,
            "n_x": int(case.n_x),
            "strides": list(STRIDES),
            "measured_errors": measured,
            "bounds": [float(identifiability_bound(grid, 1.0 / s**2)) for s in STRIDES],
        }
        fractions = np.array([1.0 / s**2 for s in STRIDES])
        for target in TARGETS:
            tag = f"t{int(target * 100)}"
            row[f"required_fraction_{tag}"] = crossing(fractions[::-1], measured[::-1], target)
            row[f"predicted_fraction_{tag}"] = required_fraction(grid, target)
        rows.append(row)
    return rows


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--public-limit", type=int, default=4)
    args = parser.parse_args()

    jobs = []
    for boundary, absorption in ABSORPTIONS.items():
        for clutter, fraction in SCATTERERS.items():
            for seed in range(args.seeds):
                spec = MediumSpec(
                    name=f"{boundary}_{clutter}_s{seed}",
                    scatterer_fraction=fraction,
                    absorption=absorption,
                    seed=seed,
                )
                jobs.append((f"fdtd_{boundary}_{clutter}", fdtd_case(spec), (6.0, 24.0)))
    for case in load_well_acoustic("test", limit=args.public_limit):
        jobs.append(("well_maze", case, (3.0, 13.0)))
    for case in load_well_scattering(limit=args.public_limit, stride=2):
        jobs.append(("acoustic_inclusions", case, (3.0, 13.0)))

    rows = []
    for label, case, band in jobs:
        rows += evaluate(case, band, label, seeds=args.seeds)
        pair = rows[-2:]
        print(
            f"{label:26s} {case.name:24s} "
            f"required@0.3  raw {pair[0]['required_fraction_t30']:.4f} "
            f"aligned {pair[1]['required_fraction_t30']:.4f}",
            flush=True,
        )

    fits = {}
    for target in TARGETS:
        tag = f"t{int(target * 100)}"
        usable = [
            r for r in rows
            if np.isfinite(r[f"required_fraction_{tag}"])
            and np.isfinite(r[f"predicted_fraction_{tag}"])
        ]
        if len(usable) > 3:
            fits[f"measured_vs_predicted_{tag}"] = fit_slope(
                np.log([r[f"predicted_fraction_{tag}"] for r in usable]),
                np.log([r[f"required_fraction_{tag}"] for r in usable]),
            )
            fits[f"n_used_{tag}"] = len(usable)

    flat_bound, flat_error = [], []
    for row in rows:
        for bound, error in zip(row["bounds"], row["measured_errors"]):
            if bound > 1e-6 and np.isfinite(error):
                flat_bound.append(bound)
                flat_error.append(error)
    fits["error_vs_bound"] = fit_slope(np.log(flat_bound), np.log(flat_error))
    fits["bound_violation_rate"] = float(
        np.mean(np.array(flat_error) < 0.9 * np.array(flat_bound))
    )
    fits["n_bound_pairs"] = len(flat_bound)

    (RESULTS / "exp21_identifiability_law.json").write_text(
        json.dumps({"strides": list(STRIDES), "targets": TARGETS,
                    "fits": fits, "rows": rows}, indent=2)
    )
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
