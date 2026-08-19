"""Experiment 13: the estimated carrier bank on public data.

Experiments 9 and 12 built and tested carrier banks on fields from our own
solver. This runs the same data-driven procedure on The Well's Helmholtz
staircase and acoustic inclusions, where the geometry is a corrugated
waveguide and a random inclusion field respectively -- neither of which the
virtual-source model was designed for.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wave_lr.fields import load_well_scattering
from wave_lr.harmonic import load_staircase
from wave_lr.multicarrier import estimate_carriers, fit_multicarrier_als
from wave_lr.spectra import carrier, to_spectrum

RESULTS = Path(__file__).resolve().parents[1] / "results"


@dataclass
class HarmonicSpectrum:
    """Minimal spectrum view for data that has no uniform time axis."""

    values: np.ndarray
    frequencies: np.ndarray
    dt: float = float("nan")
    n_padded: int = 0

    @property
    def bandwidth(self) -> float:
        return float(self.frequencies[-1] - self.frequencies[0])


def truncated_error(values: np.ndarray, rank: int) -> float:
    spectrum = np.linalg.svd(values, compute_uv=False)
    return float(np.linalg.norm(spectrum[rank:]) / np.linalg.norm(spectrum))


def evaluate(spectrum, coords, seed_delays, budgets, speed, uniform_grid: bool) -> list[dict]:
    rows = []
    values = spectrum.values
    aligned = values * np.conj(carrier(spectrum.frequencies, seed_delays))
    for budget in budgets:
        if budget > min(values.shape):
            continue
        delays, diagnostics = estimate_carriers(
            spectrum, n_carriers=6, rank=2, speed=speed, coords=coords,
            seed_delays=seed_delays, budget=budget, compute_occupancy=uniform_grid,
        )
        share = max(budget // len(delays), 1)
        fitted, info = fit_multicarrier_als(
            values, spectrum.frequencies, delays, rank=share, sweeps=40
        )
        rows.append(
            {
                "budget": budget,
                "n_carriers": len(delays),
                "plain_lowrank_error": truncated_error(values, budget),
                "single_carrier_error": truncated_error(aligned, budget),
                "estimated_multicarrier_error": float(
                    np.linalg.norm(fitted - values) / np.linalg.norm(values)
                ),
                "equivalent_rank": info["equivalent_rank"],
                "virtual_sources": [d.get("virtual_source") for d in diagnostics],
                "accepted": [d.get("accepted") for d in diagnostics],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staircase-limit", type=int, default=8)
    parser.add_argument("--inclusion-limit", type=int, default=6)
    parser.add_argument("--subsample", type=int, default=4)
    args = parser.parse_args()

    rows = []
    for case in load_staircase("train", limit=args.staircase_limit, subsample=args.subsample):
        spectrum = HarmonicSpectrum(case.fields, case.frequencies)
        for row in evaluate(
            spectrum, case.coords, case.travel_time, (4, 8), speed=1.0, uniform_grid=False
        ):
            row.update({"dataset": "helmholtz_staircase", "case": case.name})
            rows.append(row)
            print(
                f"staircase {case.name:16s} R={row['budget']:2d} M={row['n_carriers']} | "
                f"plain {row['plain_lowrank_error']:.3f} single "
                f"{row['single_carrier_error']:.3f} estimated "
                f"{row['estimated_multicarrier_error']:.3f}",
                flush=True,
            )

    for case in load_well_scattering(limit=args.inclusion_limit, stride=2):
        spectrum = to_spectrum(case.traces, case.dt, 3.0, 13.0)
        speed = float(case.metadata["reference_speed"])
        for row in evaluate(
            spectrum, case.coords, case.travel_time, (8, 16), speed=speed, uniform_grid=True
        ):
            row.update({"dataset": "acoustic_inclusions", "case": case.name})
            rows.append(row)
            print(
                f"inclusions {case.name:24s} R={row['budget']:2d} M={row['n_carriers']} | "
                f"plain {row['plain_lowrank_error']:.3f} single "
                f"{row['single_carrier_error']:.3f} estimated "
                f"{row['estimated_multicarrier_error']:.3f}",
                flush=True,
            )

    (RESULTS / "exp13_public_multicarrier.json").write_text(json.dumps({"rows": rows}, indent=2))
    print("wrote exp13_public_multicarrier.json")


if __name__ == "__main__":
    main()
