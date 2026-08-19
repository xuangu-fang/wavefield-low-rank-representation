"""Experiment 12: does carrier estimation from data alone recover the oracle gain?

Experiment 9 built the carrier bank from image sources, which requires knowing
the boundary. Here the bank is grown from the field itself: scan the model
residual for the wavefront it stacks along best, add that virtual source, keep
it only if the fit improves at a fixed total parameter budget. The oracle
image-source bank is the ceiling, plain low rank and a single carrier are the
floors, and all four models get the same budget.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
from pathlib import Path

import numpy as np

from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
from wave_lr.multicarrier import (
    estimate_carriers,
    fit_multicarrier_als,
    image_source_delays,
)
from wave_lr.spectra import carrier, to_spectrum

RESULTS = Path(__file__).resolve().parents[1] / "results"
BAND = (6.0, 24.0)
BUDGET = 24
REGIMES = {
    "open_clear": {"absorption": 40.0, "scatterer_fraction": 0.0},
    "partial_clear": {"absorption": 3.0, "scatterer_fraction": 0.0},
    "closed_clear": {"absorption": 0.0, "scatterer_fraction": 0.0},
    "closed_sparse": {"absorption": 0.0, "scatterer_fraction": 0.08},
    "closed_dense": {"absorption": 0.0, "scatterer_fraction": 0.22},
}


def relative(estimate: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - target) / np.linalg.norm(target))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-carriers", type=int, default=8)
    args = parser.parse_args()

    rows = []
    for regime, settings in REGIMES.items():
        for seed in range(args.seeds):
            case = fdtd_case(MediumSpec(name=f"{regime}_s{seed}", seed=seed, **settings))
            spectrum = to_spectrum(case.traces, case.dt, *BAND)
            values = spectrum.values
            resolution = 1.0 / spectrum.bandwidth

            spectrum_values = np.linalg.svd(values, compute_uv=False)
            plain = float(
                np.linalg.norm(spectrum_values[BUDGET:]) / np.linalg.norm(spectrum_values)
            )
            aligned = values * np.conj(carrier(spectrum.frequencies, case.travel_time))
            aligned_values = np.linalg.svd(aligned, compute_uv=False)
            single = float(
                np.linalg.norm(aligned_values[BUDGET:]) / np.linalg.norm(aligned_values)
            )

            estimated, diagnostics = estimate_carriers(
                spectrum,
                n_carriers=args.max_carriers,
                rank=2,
                coords=case.coords,
                seed_delays=case.travel_time,
                budget=BUDGET,
            )
            share = max(BUDGET // len(estimated), 1)
            fitted, _ = fit_multicarrier_als(
                values, spectrum.frequencies, estimated, rank=share, sweeps=40
            )

            truth, _ = image_source_delays(
                case.coords, np.array(case.metadata["source_xy"]),
                tuple(case.metadata["box"]), order=3, max_delay=case.duration,
            )
            oracle_bank = truth[:8]
            oracle, _ = fit_multicarrier_als(
                values, spectrum.frequencies, oracle_bank, rank=BUDGET // 8, sweeps=40
            )

            # How close is each estimated wavefront to a true image wavefront?
            shape_errors = []
            for tau in estimated[1:]:
                centred = tau - tau.mean()
                errors = [np.abs(centred - (t - t.mean())).mean() for t in truth]
                shape_errors.append(float(min(errors) / resolution))

            rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "budget": BUDGET,
                    "n_estimated_carriers": len(estimated),
                    "plain_lowrank_error": plain,
                    "single_carrier_error": single,
                    "estimated_multicarrier_error": relative(fitted, values),
                    "oracle_multicarrier_error": relative(oracle, values),
                    "shape_errors_in_resolutions": shape_errors,
                    "diagnostics": [
                        {k: v for k, v in d.items() if k != "virtual_source"}
                        | {"virtual_source": d.get("virtual_source")}
                        for d in diagnostics
                    ],
                }
            )
            last = rows[-1]
            print(
                f"{regime:14s} s{seed} M={last['n_estimated_carriers']} | "
                f"plain {plain:.3f}  single {single:.3f}  "
                f"estimated {last['estimated_multicarrier_error']:.3f}  "
                f"oracle {last['oracle_multicarrier_error']:.3f} | "
                f"wavefront err {np.round(shape_errors, 2).tolist()} /B",
                flush=True,
            )

    (RESULTS / "exp12_estimated_carriers.json").write_text(
        json.dumps({"band": BAND, "budget": BUDGET, "rows": rows}, indent=2)
    )
    print("wrote exp12_estimated_carriers.json")


if __name__ == "__main__":
    main()
