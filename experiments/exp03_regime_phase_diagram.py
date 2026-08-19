"""Experiment 3: the regime phase diagram of phase-alignment gain.

Sweeps boundary absorption and scatterer density in a controlled FDTD solver,
then asks whether the delay-occupancy law predicts, per regime, how much rank
first-arrival demodulation removes -- including the regimes where it removes
nothing.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import itertools
import json
import time
from pathlib import Path

from wave_lr.analysis import analyze_case
from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
from wave_lr.reporting import summarize_fits

RESULTS = Path(__file__).resolve().parents[1] / "results"
BANDS = [(4.0, 10.0), (8.0, 16.0), (12.0, 24.0), (6.0, 24.0)]

ABSORPTIONS = {"open": 40.0, "partial": 3.0, "closed": 0.0}
SCATTERERS = {"clear": 0.0, "sparse": 0.08, "dense": 0.22, "cluttered": 0.4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--out", default="exp03_regime_phase_diagram.json")
    args = parser.parse_args()

    rows = []
    start = time.time()
    combos = list(itertools.product(ABSORPTIONS.items(), SCATTERERS.items(), range(args.seeds)))
    for (boundary, absorption), (clutter, fraction), seed in combos:
        spec = MediumSpec(
            name=f"{boundary}_{clutter}_s{seed}",
            grid=args.grid,
            scatterer_fraction=fraction,
            absorption=absorption,
            seed=seed,
        )
        case = fdtd_case(spec, duration=args.duration)
        for f_min, f_max in BANDS:
            row = analyze_case(case, f_min, f_max)
            row.update({"boundary": boundary, "clutter": clutter, "seed": seed})
            rows.append(row)
        last = rows[-1]
        print(
            f"{spec.name:26s} occ_raw={last['raw_occupancy_90']:6.3f} "
            f"occ_eik={last['eikonal_occupancy_90']:6.3f} "
            f"rank {last['raw_rank_90']:3d}->{last['eikonal_rank_90']:3d} "
            f"gain={last['eikonal_measured_gain_90']:5.2f} "
            f"(pred {last['eikonal_predicted_gain_90']:5.2f}) "
            f"({time.time() - start:5.1f}s)",
            flush=True,
        )

    fits = summarize_fits(rows)

    payload = {
        "absorptions": ABSORPTIONS,
        "scatterers": SCATTERERS,
        "bands": BANDS,
        "n_rows": len(rows),
        "fits": fits,
        "rows": rows,
    }
    (RESULTS / args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
