"""Experiment 18: which objective, and does a dispersive phase pay?

Two questions the learned representation raises, answered on public data.

* The nuclear norm is a convex surrogate for low rank, but the quantity every
  table reports is the best rank-``R`` error. Optimising the surrogate can move
  the reported metric the wrong way, so both are run.
* A travel-time carrier forces the phase to be linear in frequency. The Well's
  Helmholtz staircase carries trapped modes whose on-surface wavenumber is a
  nonlinear function of frequency -- the documented reason experiment 4 failed.
  A learned phase need not be linear, so this is where it should pay.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wave_lr.diagnostics import singular_spectrum
from wave_lr.harmonic import load_staircase
from wave_lr.learned_carrier import CarrierConfig, fit_learned_carrier, nuclear_ratio
from wave_lr.spectra import carrier

RESULTS = Path(__file__).resolve().parents[1] / "results"
BUDGET = 4


def rank_at(values, level=0.90):
    spectrum = singular_spectrum(values)
    return int(np.searchsorted(np.cumsum(spectrum**2), level) + 1)


def truncation_error(values, rank):
    spectrum = np.linalg.svd(values, compute_uv=False)
    return float(np.linalg.norm(spectrum[rank:]) / np.linalg.norm(spectrum))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--subsample", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args()

    rows = []
    start = time.time()
    for case in load_staircase("train", limit=args.limit, subsample=args.subsample):
        values, frequencies = case.fields, case.frequencies
        row = {"case": case.name, "n_x": int(values.shape[0]), "budget": BUDGET}

        def record(label, aligned, row=row):
            row[f"{label}_nuclear"] = nuclear_ratio(aligned)
            row[f"{label}_rank90"] = rank_at(aligned)
            row[f"{label}_r2"] = truncation_error(aligned, 2)
            row[f"{label}_r4"] = truncation_error(aligned, BUDGET)

        record("raw", values)
        record("eikonal", values * np.conj(carrier(frequencies, case.travel_time)))

        for objective in ("nuclear", "tail"):
            for dispersive in (False, True):
                config = CarrierConfig(
                    steps=args.steps, warmup_steps=args.warmup, learning_rate=3e-4,
                    dispersive=dispersive, dispersion_rank=2,
                    objective=objective, budget=BUDGET,
                )
                _, info = fit_learned_carrier(
                    values, frequencies, case.coords, config,
                    initial_delays=case.travel_time,
                )
                label = f"learned_{objective}_{'disp' if dispersive else 'tau'}"
                record(label, values * np.exp(1j * info["phase"]))
        rows.append(row)
        print(
            f"{case.name:16s} eik r4={row['eikonal_r4']:.4f} | "
            f"tail-tau {row['learned_tail_tau_r4']:.4f}  "
            f"tail-disp {row['learned_tail_disp_r4']:.4f}  "
            f"nuc-disp {row['learned_nuclear_disp_r4']:.4f}  "
            f"({time.time() - start:5.0f}s)",
            flush=True,
        )

    (RESULTS / "exp18_objective_and_dispersion.json").write_text(
        json.dumps({"budget": BUDGET, "rows": rows}, indent=2)
    )
    print("wrote exp18_objective_and_dispersion.json")


if __name__ == "__main__":
    main()
