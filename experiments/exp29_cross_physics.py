"""Experiment 29: does the identifiability criterion transfer across physics?

Everything in this project so far was measured on wave fields, and most of the
large gains came from a solver we wrote. That leaves two questions open, and
they are the ones that decide whether the criterion is a general statement
about sampled representations or a fact about acoustics:

1. On public data nobody in this project generated, does the criterion still
   lower-bound what any estimator can do?
2. Does it predict *how much* a coordinate change buys -- including predicting
   zero, on families where it should buy nothing?

Four families are measured on identical footing: seismic shot gathers
(OpenFWI), real PIV of a cylinder wake (RealPDEBench, measured not simulated),
Kuramoto-Sivashinsky, and forced 2-D turbulence at two Reynolds numbers. The
warp family is the weakest one available -- a single transport speed -- so any
gain it finds is a floor on what a learned reparameterisation could reach, and
the speed it lands on is checkable against the physics.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
import time
from pathlib import Path

import numpy as np

from wave_lr.families import (
    cylinder_cases,
    kolmogorov_cases,
    ks_cases,
    openfwi_cases,
)
from wave_lr.sensorline import (
    reconstruct_bandlimited,
    scan_speed,
    transport_delays,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRACTIONS = (0.05, 0.10, 0.20, 0.34)

# Each family gets a speed scan centred on its own physics only so the scan is
# not absurdly wide; the range spans well over an order of magnitude either way,
# so it is not doing the work of picking the answer.
LOG_SCAN = np.concatenate(
    [-np.logspace(-1.0, 1.5, 24)[::-1], np.logspace(-1.0, 1.5, 24)]
)
SCANS = {
    "seismic": np.linspace(1000.0, 6000.0, 40),
    "wake": np.concatenate(
        [
            -np.logspace(np.log10(0.03), np.log10(3.0), 24)[::-1],
            np.logspace(np.log10(0.03), np.log10(3.0), 24),
        ]
    ),
}


def build_cases():
    groups = [
        ("seismic", lambda: openfwi_cases(limit=40)),
        ("wake_v", lambda: cylinder_cases(limit=12, field="v")),
        ("wake_vo", lambda: cylinder_cases(limit=12, field="vo")),
        ("ks", lambda: ks_cases(limit=20)),
        ("turbulence_re40", lambda: kolmogorov_cases(limit=16, reynolds=40)),
        ("turbulence_re5000", lambda: kolmogorov_cases(limit=16, reynolds=5000)),
    ]
    cases = []
    for label, loader in groups:
        try:
            block = loader()
        except (OSError, ValueError) as error:  # a family missing locally is not fatal
            print(f"  {label}: unavailable ({error})", flush=True)
            continue
        print(f"  {label}: {len(block)} cases", flush=True)
        cases.extend(block)
    return cases


def main() -> None:
    start = time.time()
    print("loading families", flush=True)
    cases = build_cases()
    rows = []
    for case in cases:
        speeds = SCANS.get(case.family, LOG_SCAN)
        for fraction in FRACTIONS:
            raw_bound, speed, warped_bound = scan_speed(case, fraction, speeds)
            raw_error = reconstruct_bandlimited(case.spectrum, case.weights, fraction)
            if np.isfinite(speed):  # a warp was found that lowers the bound
                delays = transport_delays(case.coords, case.source, speed)
                warped_error = reconstruct_bandlimited(
                    case.spectrum, case.weights, fraction, delays, case.freqs
                )
            else:
                speed, warped_bound, warped_error = float("nan"), raw_bound, raw_error
            rows.append(
                {
                    "family": case.family,
                    "name": case.name,
                    "fraction": fraction,
                    "n_sensor": int(case.coords.size),
                    "n_freq": int(case.freqs.size),
                    "bound_raw": raw_bound,
                    "bound_warped": warped_bound,
                    "speed": speed,
                    "error_raw": raw_error,
                    "error_warped": warped_error,
                    "gain_predicted": raw_bound / max(warped_bound, 1e-12),
                    "gain_measured": raw_error / max(warped_error, 1e-12),
                }
            )
        print(f"  {case.family:18s} {case.name:22s} ({time.time()-start:5.0f}s)", flush=True)

    # The bound is a bound: the measured error of an untrained estimator must
    # never fall below it. Both coordinate systems count as separate tests.
    checks = [(r["error_raw"], r["bound_raw"]) for r in rows]
    checks += [(r["error_warped"], r["bound_warped"]) for r in rows]
    violations = [1.0 for error, bound in checks if error < bound - 1e-9]
    ratios = np.array([error / max(bound, 1e-12) for error, bound in checks])

    predicted = np.array([r["gain_predicted"] for r in rows])
    measured = np.array([r["gain_measured"] for r in rows])
    finite = np.isfinite(predicted) & np.isfinite(measured) & (predicted > 0) & (measured > 0)
    slope, intercept = np.polyfit(np.log(predicted[finite]), np.log(measured[finite]), 1)
    resid = np.log(measured[finite]) - (slope * np.log(predicted[finite]) + intercept)
    r2 = 1.0 - resid.var() / np.log(measured[finite]).var()

    by_family = {}
    for family in sorted({r["family"] for r in rows}):
        block = [r for r in rows if r["family"] == family]
        speeds = [r["speed"] for r in block if np.isfinite(r["speed"])]
        by_family[family] = {
            "n_cases": len({r["name"] for r in block}),
            "n_measurements": len(block),
            "bound_raw_median": float(np.median([r["bound_raw"] for r in block])),
            "gain_predicted_median": float(np.median([r["gain_predicted"] for r in block])),
            "gain_measured_median": float(np.median([r["gain_measured"] for r in block])),
            "gain_predicted_max": float(np.max([r["gain_predicted"] for r in block])),
            "speed_median": float(np.median(speeds)) if speeds else float("nan"),
            "speed_iqr": (
                float(np.percentile(speeds, 75) - np.percentile(speeds, 25))
                if speeds
                else float("nan")
            ),
        }

    summary = {
        "n_measurements": len(rows),
        "n_bound_tests": len(checks),
        "bound_violation_rate": len(violations) / len(checks),
        "error_over_bound": {
            "median": float(np.median(ratios)),
            "p05": float(np.percentile(ratios, 5)),
            "p95": float(np.percentile(ratios, 95)),
        },
        "gain_predicted_vs_measured": {
            "log_slope": float(slope),
            "log_intercept": float(intercept),
            "r2": float(r2),
            "n": int(finite.sum()),
        },
        "by_family": by_family,
        "fractions": list(FRACTIONS),
        "runtime_s": time.time() - start,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "exp29_cross_physics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
