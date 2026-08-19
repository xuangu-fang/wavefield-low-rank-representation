"""Experiment 14: does a learned model make the carrier unnecessary?

Fourier features and SIREN exist precisely to let a coordinate network
represent oscillation, so the honest question is whether they already absorb
what the carrier does. Every model here sees the same sensors, has the same
architecture and budget, and is scored on the same hidden locations; the
Fourier bandwidth is swept and the *best* setting is reported for each
representation, which favours the baselines.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
from wave_lr.inr import TrainConfig, fit_field_network
from wave_lr.metrics import summarize
from wave_lr.spectra import to_spectrum
from wave_lr.tasks import interpolate_from_sensors

RESULTS = Path(__file__).resolve().parents[1] / "results"
BAND = (6.0, 24.0)
REGIMES = {
    "open_clear": {"absorption": 40.0, "scatterer_fraction": 0.0},
    "open_sparse": {"absorption": 40.0, "scatterer_fraction": 0.08},
    "partial_clear": {"absorption": 3.0, "scatterer_fraction": 0.0},
    "closed_dense": {"absorption": 0.0, "scatterer_fraction": 0.22},
}
FEATURE_SCALES = (4.0, 16.0, 64.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=0.02)
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()

    rows = []
    start = time.time()
    for regime, settings in REGIMES.items():
        for seed in range(args.seeds):
            case = fdtd_case(MediumSpec(name=f"{regime}_s{seed}", seed=seed, **settings))
            spectrum = to_spectrum(case.traces, case.dt, *BAND)
            values, frequencies = spectrum.values, spectrum.frequencies
            rng = np.random.default_rng(seed)
            observed = rng.random(case.n_x) < args.fraction
            hidden = ~observed
            row = {
                "regime": regime,
                "seed": seed,
                "fraction": args.fraction,
                "n_sensors": int(observed.sum()),
            }

            for label, delays in (("raw", None), ("aligned", case.travel_time)):
                predicted = interpolate_from_sensors(
                    case.coords, values, frequencies, delays, observed
                )
                row[f"interp_{label}"] = summarize(predicted[hidden], values[hidden])[
                    "complex_nrmse"
                ]

            for encoding, scale, label in itertools.chain(
                ((("fourier"), s, "raw") for s in FEATURE_SCALES),
                ((("fourier"), s, "aligned") for s in FEATURE_SCALES),
                ((("siren"), 0.0, "raw"), ("siren", 0.0, "aligned")),
            ):
                config = TrainConfig(steps=args.steps, feature_scale=scale, seed=seed)
                delays = None if label == "raw" else case.travel_time
                predicted = fit_field_network(
                    case.coords, values, observed, encoding,
                    delays=delays, frequencies=frequencies, config=config,
                )
                scores = summarize(predicted[hidden], values[hidden])
                key = f"{encoding}{'' if encoding == 'siren' else f'_s{scale:g}'}_{label}"
                row[f"{key}_nrmse"] = scores["complex_nrmse"]
                row[f"{key}_awpc"] = scores["awpc"]
                row[f"{key}_train_nrmse"] = summarize(
                    predicted[observed], values[observed]
                )["complex_nrmse"]

            for label in ("raw", "aligned"):
                best = min(
                    value
                    for key, value in row.items()
                    if key.endswith(f"_{label}_nrmse")
                )
                row[f"best_network_{label}"] = best
            row["network_gain"] = row["best_network_raw"] / max(
                row["best_network_aligned"], 1e-12
            )
            rows.append(row)
            print(
                f"{regime:14s} s{seed} sensors={row['n_sensors']:4d} | "
                f"interp {row['interp_raw']:.3f}->{row['interp_aligned']:.3f}  "
                f"network {row['best_network_raw']:.3f}->{row['best_network_aligned']:.3f}  "
                f"gain {row['network_gain']:.2f}x  ({time.time() - start:5.0f}s)",
                flush=True,
            )

    name = args.out or f"exp14_learned_baselines_p{int(args.fraction * 100)}.json"
    (RESULTS / name).write_text(
        json.dumps({"band": BAND, "feature_scales": FEATURE_SCALES, "rows": rows}, indent=2)
    )
    print(f"wrote {name}")


if __name__ == "__main__":
    main()
