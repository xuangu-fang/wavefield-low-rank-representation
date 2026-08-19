"""Experiment 1: falsification test of the delay-spread rank law.

Controlled synthetic multipath fields have exactly known arrival times, so the
predicted rank ``B * Lambda_B + 1`` can be compared against the measured
numerical rank without any estimation error. Three candidate predictors are
scored against each other:

* ``support``   -- naive ``max - min`` of the delays;
* ``union``     -- union of the per-path spatial delay ranges;
* ``occupancy`` -- bandwidth-resolved occupied measure (the proposed law).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import itertools
import json
from pathlib import Path

import numpy as np

from wave_lr.demodulation import demodulate
from wave_lr.diagnostics import effective_rank
from wave_lr.synthetic import make_multipath_field
from wave_lr.theory import (
    delay_occupancy,
    fit_slope,
    pooled_support,
    predicted_rank,
    union_of_ranges,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
ENERGY = 0.99
SATURATION_LIMIT = 0.4  # discard cases whose rank approaches min(n_x, n_f)


def predictors(delays: np.ndarray, energies: np.ndarray, bandwidth: float) -> dict[str, float]:
    return {
        "support": pooled_support(delays),
        "union": union_of_ranges(delays, axis=0),
        "occupancy": delay_occupancy(delays, energies, bandwidth=bandwidth),
    }


def run_case(bandwidth: float, n_paths: int, absolute: float, spread: float, seed: int) -> dict:
    n_x = n_f = 512
    mp = make_multipath_field(
        n_x=n_x,
        n_f=n_f,
        f_min=100.0,
        f_max=100.0 + bandwidth,
        n_paths=n_paths,
        absolute_spread=absolute,
        delay_spread=spread,
        seed=seed,
    )
    carrier = 2.0 * np.pi * mp.frequencies[None, :] * mp.first_arrival[:, None]
    residual = demodulate(mp.field, carrier)
    energies = mp.amplitudes**2
    relative = mp.delays - mp.first_arrival[:, None]

    row = {
        "bandwidth": bandwidth,
        "n_paths": n_paths,
        "absolute_spread": absolute,
        "delay_spread": spread,
        "seed": seed,
        "measured_raw_rank": effective_rank(mp.field, energy=ENERGY),
        "measured_demodulated_rank": effective_rank(residual, energy=ENERGY),
        "grid": min(n_x, n_f),
    }
    for name, value in predictors(mp.delays, energies, bandwidth).items():
        row[f"raw_{name}"] = value
        row[f"raw_{name}_pred"] = predicted_rank(bandwidth, value)
    for name, value in predictors(relative, energies, bandwidth).items():
        row[f"demod_{name}"] = value
        row[f"demod_{name}_pred"] = predicted_rank(bandwidth, value)
    row["measured_gain"] = row["measured_raw_rank"] / max(row["measured_demodulated_rank"], 1)
    row["predicted_gain_occupancy"] = row["raw_occupancy_pred"] / row["demod_occupancy_pred"]
    return row


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    bandwidths = [10.0, 20.0, 40.0, 80.0, 160.0]
    absolutes = [0.05, 0.2, 0.4, 0.8]
    spreads = [0.0, 0.01, 0.05, 0.2, 0.6]
    path_counts = [1, 2, 4, 10, 30]
    for bandwidth, absolute, spread, n_paths in itertools.product(
        bandwidths, absolutes, spreads, path_counts
    ):
        if n_paths == 1 and spread > 0.0:
            continue
        rows.append(run_case(bandwidth, n_paths, absolute, spread, seed=n_paths))

    unsaturated = [
        r for r in rows if r["measured_raw_rank"] < SATURATION_LIMIT * r["grid"]
    ]
    fits = {}
    for stage in ("raw", "demod"):
        measured = np.array(
            [r[f"measured_{'raw' if stage == 'raw' else 'demodulated'}_rank"] for r in unsaturated],
            dtype=float,
        )
        for name in ("support", "union", "occupancy"):
            fits[f"{stage}_{name}"] = fit_slope(
                np.array([r[f"{stage}_{name}"] * r["bandwidth"] for r in unsaturated]), measured
            )

    gain_fit = fit_slope(
        np.array([r["predicted_gain_occupancy"] for r in unsaturated]),
        np.array([r["measured_gain"] for r in unsaturated], dtype=float),
    )

    payload = {
        "purpose": "falsification test of the delay-occupancy rank law",
        "energy_level": ENERGY,
        "n_cases": len(rows),
        "n_unsaturated": len(unsaturated),
        "fits_rank_vs_bandwidth_times_measure": fits,
        "fit_gain": gain_fit,
        "rows": rows,
    }
    out = RESULTS / "exp01_rank_law.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"fits": fits, "gain": gain_fit, "n": len(unsaturated)}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
