"""Experiment 9: more carriers, not a different low-rank container.

Where a single carrier fails, the law blames the *coda*, not the choice of
factorisation. If that diagnosis is right, then adding one carrier per
resolvable arrival should recover accuracy at an equal parameter budget, and
should stop helping when the coda comes from volume scattering rather than
from boundary reflections -- because image sources model the latter only.

Baselines get the optimal rank-R approximation (Eckart-Young); the
multi-carrier model is fitted by monotone alternating least squares, so the
comparison is conservative in favour of the baselines.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
from pathlib import Path

import numpy as np

from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
from wave_lr.multicarrier import fit_multicarrier_als, image_source_delays
from wave_lr.spectra import carrier, to_spectrum
from wave_lr.tasks import complete_low_rank, random_entry_mask

RESULTS = Path(__file__).resolve().parents[1] / "results"
BAND = (6.0, 24.0)
BUDGETS = (4, 8, 16, 24)
CARRIER_COUNTS = (2, 4, 8)
REGIMES = {
    "closed_clear": {"absorption": 0.0, "scatterer_fraction": 0.0},
    "closed_sparse": {"absorption": 0.0, "scatterer_fraction": 0.08},
    "closed_dense": {"absorption": 0.0, "scatterer_fraction": 0.22},
    "partial_clear": {"absorption": 3.0, "scatterer_fraction": 0.0},
    "open_clear": {"absorption": 40.0, "scatterer_fraction": 0.0},
}
OBSERVED_FRACTION = 0.05
RIDGES = (1e-6, 1e-3, 1e-2, 1e-1, 3e-1, 1.0)
VALIDATION_FRACTION = 0.25


def truncated_svd_error(values: np.ndarray, rank: int) -> float:
    spectrum = np.linalg.svd(values, compute_uv=False)
    return float(np.linalg.norm(spectrum[rank:]) / np.linalg.norm(spectrum))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--no-completion", action="store_true")
    args = parser.parse_args()

    rows = []
    for regime, settings in REGIMES.items():
        for seed in range(args.seeds):
            case = fdtd_case(MediumSpec(name=f"{regime}_s{seed}", seed=seed, **settings))
            spectrum = to_spectrum(case.traces, case.dt, *BAND)
            values = spectrum.values
            frequencies = spectrum.frequencies
            aligned = values * np.conj(carrier(frequencies, case.travel_time))
            source = np.array(case.metadata["source_xy"])
            box = tuple(case.metadata["box"])
            all_delays, orders = image_source_delays(
                case.coords, source, box, order=3, max_delay=case.duration
            )
            observed = random_entry_mask(values.shape, OBSERVED_FRACTION, seed=seed)
            hidden = ~observed

            for budget in BUDGETS:
                row = {
                    "regime": regime,
                    "seed": seed,
                    "budget_rank": budget,
                    "svd_error": truncated_svd_error(values, budget),
                    "carrier1_svd_error": truncated_svd_error(aligned, budget),
                }
                if not args.no_completion:
                    for name, matrix, ramp in (
                        ("svd", values, None),
                        ("carrier1", aligned, carrier(frequencies, case.travel_time)),
                    ):
                        filled = complete_low_rank(matrix, observed, budget, iterations=120)
                        restored = filled if ramp is None else filled * ramp
                        row[f"{name}_completion_nrmse"] = float(
                            np.linalg.norm(restored[hidden] - values[hidden])
                            / np.linalg.norm(values[hidden])
                        )
                for count in CARRIER_COUNTS:
                    if budget % count or budget // count < 1 or count > len(all_delays):
                        continue
                    rank = budget // count
                    delays = all_delays[:count]
                    estimate, info = fit_multicarrier_als(
                        values, frequencies, delays, rank=rank, sweeps=args.steps, seed=seed
                    )
                    row[f"multi{count}_error"] = float(
                        np.linalg.norm(estimate - values) / np.linalg.norm(values)
                    )
                    row[f"multi{count}_parameters"] = info["parameters"]
                    if not args.no_completion:
                        # The multi-carrier model has a free regulariser that the
                        # hard-thresholding baseline does not, so it is chosen on a
                        # validation split carved out of the observed entries only.
                        rng = np.random.default_rng(1000 + seed)
                        validation = observed & (
                            rng.random(values.shape) < VALIDATION_FRACTION
                        )
                        fitting = observed & ~validation
                        best = None
                        for ridge in RIDGES:
                            trial, _ = fit_multicarrier_als(
                                values, frequencies, delays, rank=rank,
                                observed=fitting, sweeps=args.steps, ridge=ridge, seed=seed,
                            )
                            score = float(
                                np.linalg.norm(trial[validation] - values[validation])
                                / np.linalg.norm(values[validation])
                            )
                            if best is None or score < best[1]:
                                best = (ridge, score)
                        estimate, _ = fit_multicarrier_als(
                            values, frequencies, delays, rank=rank,
                            observed=observed, sweeps=args.steps, ridge=best[0], seed=seed,
                        )
                        row[f"multi{count}_completion_nrmse"] = float(
                            np.linalg.norm(estimate[hidden] - values[hidden])
                            / np.linalg.norm(values[hidden])
                        )
                        row[f"multi{count}_completion_ridge"] = best[0]
                rows.append(row)
                best_multi = min(
                    (row[k] for k in row if k.startswith("multi") and k.endswith("_error")),
                    default=float("nan"),
                )
                print(
                    f"{regime:14s} s{seed} R={budget:3d} | svd {row['svd_error']:.3f} "
                    f"carrier1 {row['carrier1_svd_error']:.3f} multi {best_multi:.3f}",
                    flush=True,
                )
            row_note = {"regime": regime, "seed": seed, "n_images": len(all_delays),
                        "image_orders": orders[:12].tolist()}
            rows.append(row_note)

    (RESULTS / "exp09_multicarrier.json").write_text(
        json.dumps({"band": BAND, "budgets": BUDGETS, "rows": rows}, indent=2)
    )
    print("wrote exp09_multicarrier.json")


if __name__ == "__main__":
    main()
