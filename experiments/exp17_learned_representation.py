"""Experiment 17: learning the alignment coordinates instead of deriving them.

The carrier so far came from the medium through an eikonal solve. Here it is
learned by minimising the nuclear norm of the aligned field -- a self-supervised
objective with no labels, since multiplying by a unimodular factor leaves the
Frobenius norm fixed. Three settings separate what the physics contributes from
what the learning contributes:

* from scratch      -- no medium, no source, no warm start;
* from the eikonal  -- physics as an initialisation, learning as refinement;
* repair            -- physics that is *wrong* by a rough error of one
                       resolution unit, which experiment 7 showed erases the
                       entire benefit.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import time
from pathlib import Path

import numpy as np
from exp07_carrier_error_tolerance import smooth_error

from wave_lr.diagnostics import singular_spectrum
from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
from wave_lr.learned_carrier import CarrierConfig, fit_learned_carrier, nuclear_ratio
from wave_lr.spectra import carrier, to_spectrum
from wave_lr.tasks import sensor_interpolation_report

RESULTS = Path(__file__).resolve().parents[1] / "results"
BAND = (6.0, 24.0)
BUDGET = 16
REGIMES = {
    "open_clear": {"absorption": 40.0, "scatterer_fraction": 0.0},
    "open_sparse": {"absorption": 40.0, "scatterer_fraction": 0.08},
    "partial_clear": {"absorption": 3.0, "scatterer_fraction": 0.0},
    "closed_dense": {"absorption": 0.0, "scatterer_fraction": 0.22},
}


def rank_at(values, level=0.90):
    spectrum = singular_spectrum(values)
    return int(np.searchsorted(np.cumsum(spectrum**2), level) + 1)


def truncation_error(values, rank):
    spectrum = np.linalg.svd(values, compute_uv=False)
    return float(np.linalg.norm(spectrum[rank:]) / np.linalg.norm(spectrum))


def main() -> None:
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
            values, frequencies = spectrum.values, spectrum.frequencies
            resolution = 1.0 / spectrum.bandwidth
            corrupted = case.travel_time + resolution * smooth_error(case.coords, seed=seed)

            def evaluate(
                label, delays, row, values=values, frequencies=frequencies,
                case=case, seed=seed,
            ):
                aligned = (
                    values if delays is None
                    else values * np.conj(carrier(frequencies, delays))
                )
                row[f"{label}_nuclear"] = nuclear_ratio(aligned)
                row[f"{label}_rank90"] = rank_at(aligned)
                row[f"{label}_truncation"] = truncation_error(aligned, BUDGET)
                row[f"{label}_sensor_nrmse"] = sensor_interpolation_report(
                    case.coords, values, frequencies, delays, 0.02, seed=seed
                )["complex_nrmse"]

            row = {"regime": regime, "seed": seed, "resolution": resolution}
            evaluate("raw", None, row)
            evaluate("eikonal", case.travel_time, row)
            evaluate("corrupted", corrupted, row)

            config = CarrierConfig(
                steps=args.steps, warmup_steps=args.warmup, learning_rate=3e-4, seed=seed
            )
            for label, initial, cfg in (
                ("learned_scratch", None,
                 CarrierConfig(steps=args.steps + args.warmup, warmup_steps=0,
                               learning_rate=1e-3, seed=seed)),
                ("learned_from_eikonal", case.travel_time, config),
                ("learned_repair", corrupted, config),
            ):
                delays, _ = fit_learned_carrier(
                    values, frequencies, case.coords, cfg, initial_delays=initial
                )
                evaluate(label, delays, row)
                row[f"{label}_delay_shift_in_resolutions"] = float(
                    np.abs(delays - case.travel_time).mean() / resolution
                )
            rows.append(row)
            print(
                f"{regime:14s} s{seed} | raw {row['raw_sensor_nrmse']:.3f}  "
                f"eik {row['eikonal_sensor_nrmse']:.3f}  "
                f"corrupt {row['corrupted_sensor_nrmse']:.3f}  "
                f"scratch {row['learned_scratch_sensor_nrmse']:.3f}  "
                f"repair {row['learned_repair_sensor_nrmse']:.3f}  "
                f"({time.time() - start:5.0f}s)",
                flush=True,
            )

    (RESULTS / "exp17_learned_representation.json").write_text(
        json.dumps({"band": BAND, "budget": BUDGET, "rows": rows}, indent=2)
    )
    print("wrote exp17_learned_representation.json")


if __name__ == "__main__":
    main()
