"""Experiment 16: shifted POD, the closest prior art, at an equal budget.

Shifted POD undoes a *rigid spatial* shift shared by each snapshot; the carrier
here undoes a *per-location temporal* warp. They agree for a translating
pattern and part ways for an expanding or refracted wavefront. Both are given
the same band-limited field and the same total rank, and shifted POD estimates
its own transport from the data with no physics supplied.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case, load_well_acoustic, load_well_scattering
from wave_lr.shifted_pod import carrier_pod, plain_pod, shifted_pod
from wave_lr.spectra import band_limited_traces, to_spectrum

RESULTS = Path(__file__).resolve().parents[1] / "results"
BUDGETS = (4, 8, 16, 32)
FRAME_COUNTS = (1, 2, 3)
REGIMES = {
    "open_clear": {"absorption": 40.0, "scatterer_fraction": 0.0},
    "open_sparse": {"absorption": 40.0, "scatterer_fraction": 0.08},
    "partial_clear": {"absorption": 3.0, "scatterer_fraction": 0.0},
    "closed_dense": {"absorption": 0.0, "scatterer_fraction": 0.22},
}


def as_grid(case, traces):
    """Place trace rows back on a regular grid, zero-filling absent cells."""

    spacing = float(case.metadata["spacing"])
    rows = np.rint(case.coords[:, 0] / spacing).astype(int)
    cols = np.rint(case.coords[:, 1] / spacing).astype(int)
    rows -= rows.min()
    cols -= cols.min()
    block = np.zeros((traces.shape[1], rows.max() + 1, cols.max() + 1), dtype=np.float64)
    block[:, rows, cols] = traces.T
    return block, rows, cols


def evaluate(case, band, budgets, sweeps: int) -> list[dict]:
    spectrum = to_spectrum(case.traces, case.dt, *band)
    traces, _ = band_limited_traces(spectrum)
    traces = traces[:, : case.traces.shape[1]]
    block, rows, cols = as_grid(case, traces)
    norm = np.linalg.norm(traces)

    def score(block_estimate):
        return float(np.linalg.norm(block_estimate[:, rows, cols].T - traces) / norm)

    out = []
    for budget in budgets:
        row = {"budget": budget}
        row["plain_pod"] = score(plain_pod(block, budget))
        for frames in FRAME_COUNTS:
            if budget // frames < 1:
                continue
            estimate, info = shifted_pod(
                block, n_frames=frames, total_rank=budget, sweeps=sweeps
            )
            row[f"shifted_pod_k{frames}"] = score(estimate)
            row[f"shifted_pod_k{frames}_rank"] = info["equivalent_rank"]
        restored = carrier_pod(traces, case.travel_time, spectrum.dt, budget)
        row["carrier_pod"] = float(np.linalg.norm(restored - traces) / norm)
        row["best_shifted_pod"] = min(
            value for key, value in row.items() if key.startswith("shifted_pod_k")
            and not key.endswith("_rank")
        )
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--maze-limit", type=int, default=4)
    parser.add_argument("--inclusion-limit", type=int, default=4)
    args = parser.parse_args()

    rows = []
    start = time.time()
    cases = []
    for regime, settings in REGIMES.items():
        for seed in range(args.seeds):
            cases.append(
                (
                    regime,
                    seed,
                    fdtd_case(MediumSpec(name=f"{regime}_s{seed}", seed=seed, **settings)),
                    (6.0, 24.0),
                )
            )
    for index, case in enumerate(load_well_acoustic("test", limit=args.maze_limit)):
        cases.append(("well_maze", index, case, (3.0, 13.0)))
    for index, case in enumerate(load_well_scattering(limit=args.inclusion_limit, stride=1)):
        cases.append(("acoustic_inclusions", index, case, (3.0, 13.0)))

    for regime, seed, case, band in cases:
        for row in evaluate(case, band, BUDGETS, args.sweeps):
            row.update({"regime": regime, "seed": seed, "case": case.name})
            rows.append(row)
        printed = [r for r in rows if r["regime"] == regime and r["seed"] == seed]
        best = printed[-1]
        print(
            f"{regime:20s} s{seed} R={best['budget']:3d} | plain {best['plain_pod']:.3f}  "
            f"sPOD {best['best_shifted_pod']:.3f}  carrier {best['carrier_pod']:.3f}  "
            f"({time.time() - start:5.0f}s)",
            flush=True,
        )

    (RESULTS / "exp16_shifted_pod.json").write_text(
        json.dumps({"budgets": BUDGETS, "frames": FRAME_COUNTS, "rows": rows}, indent=2)
    )
    print("wrote exp16_shifted_pod.json")


if __name__ == "__main__":
    main()
