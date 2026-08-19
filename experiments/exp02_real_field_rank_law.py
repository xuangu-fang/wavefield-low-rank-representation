"""Experiment 2: does the delay-occupancy law hold on solved wave fields?

Runs the raw / eikonal / straight-ray / data-picked carrier comparison over
frequency bands for a dataset of real solved fields and writes a JSON table.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
from pathlib import Path

import numpy as np

from wave_lr.analysis import analyze_case
from wave_lr.fields import load_openfwi, load_well_acoustic
from wave_lr.reporting import summarize_fits

RESULTS = Path(__file__).resolve().parents[1] / "results"

BANDS = {
    "well_acoustic_maze": [(1.0, 3.0), (2.0, 6.0), (4.0, 8.0), (6.0, 12.0), (3.0, 13.0)],
    "openfwi_gathers": [(4.0, 10.0), (8.0, 16.0), (12.0, 24.0), (5.0, 25.0)],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="well_acoustic_maze")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    if args.dataset == "well_acoustic_maze":
        cases = load_well_acoustic("test", limit=args.limit)
    elif args.dataset == "openfwi_gathers":
        cases = load_openfwi(n_models=args.limit)
    else:
        raise SystemExit(f"unknown dataset {args.dataset}")

    rows = []
    for case in cases:
        for f_min, f_max in BANDS[args.dataset]:
            rows.append(analyze_case(case, f_min, f_max))

    fits = summarize_fits(rows)

    payload = {
        "dataset": args.dataset,
        "n_cases": len(cases),
        "n_rows": len(rows),
        "fits": fits,
        "rows": rows,
    }
    out = RESULTS / f"exp02_{args.dataset}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(fits, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
