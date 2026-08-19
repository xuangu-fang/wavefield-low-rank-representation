"""Experiment 8: rank is set by bandwidth, not by centre frequency.

The usual intuition is that high-frequency wave fields are hard because they
oscillate fast. The delay-occupancy law says otherwise: the numerical rank of
the ``(x, f)`` unfolding is ``B * Lambda``, so a narrow band centred anywhere is
cheap and a wide band is expensive, at equal centre frequency. This experiment
separates the two variables that are usually confounded.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
from pathlib import Path

import numpy as np

from wave_lr.diagnostics import singular_spectrum
from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
from wave_lr.spectra import band_limited_traces, carrier, to_spectrum
from wave_lr.theory import fit_slope, occupancy_from_traces

RESULTS = Path(__file__).resolve().parents[1] / "results"
CENTRES = (9.0, 12.0, 15.0, 18.0, 21.0, 24.0)
BANDWIDTHS = (2.0, 4.0, 6.0, 9.0, 12.0)
REGIMES = {
    "open_clear": {"absorption": 40.0, "scatterer_fraction": 0.0},
    "open_sparse": {"absorption": 40.0, "scatterer_fraction": 0.08},
    "closed_dense": {"absorption": 0.0, "scatterer_fraction": 0.22},
}


def rank_at(values: np.ndarray, level: float = 0.90) -> int:
    spectrum = singular_spectrum(values)
    return int(np.searchsorted(np.cumsum(spectrum**2), level) + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    rows = []
    for regime, settings in REGIMES.items():
        for seed in range(args.seeds):
            case = fdtd_case(
                MediumSpec(name=f"{regime}_s{seed}", seed=seed, **settings),
                peak_frequency=16.0,
            )
            for centre in CENTRES:
                for bandwidth in BANDWIDTHS:
                    f_min, f_max = centre - bandwidth / 2, centre + bandwidth / 2
                    if f_min <= 1.0:
                        continue
                    spectrum = to_spectrum(case.traces, case.dt, f_min, f_max)
                    aligned = spectrum.values * np.conj(
                        carrier(spectrum.frequencies, case.travel_time)
                    )
                    shifted_traces, _ = band_limited_traces(
                        type(spectrum)(
                            spectrum.values
                            * np.conj(
                                carrier(
                                    spectrum.frequencies,
                                    case.travel_time - case.travel_time.max(),
                                )
                            ),
                            spectrum.frequencies,
                            spectrum.dt,
                            spectrum.n_padded,
                        )
                    )
                    raw_traces, _ = band_limited_traces(spectrum)
                    rows.append(
                        {
                            "regime": regime,
                            "seed": seed,
                            "centre": centre,
                            "bandwidth": float(spectrum.bandwidth),
                            "nominal_bandwidth": bandwidth,
                            "raw_rank_90": rank_at(spectrum.values),
                            "aligned_rank_90": rank_at(aligned),
                            "raw_occupancy_90": occupancy_from_traces(
                                raw_traces, spectrum.dt, spectrum.bandwidth
                            ),
                            "aligned_occupancy_90": occupancy_from_traces(
                                shifted_traces, spectrum.dt, spectrum.bandwidth
                            ),
                        }
                    )
            print(f"{regime} seed{seed} done", flush=True)

    fits = {}
    for regime in REGIMES:
        subset = [r for r in rows if r["regime"] == regime]
        fits[f"{regime}_rank_vs_bandwidth"] = fit_slope(
            [r["bandwidth"] for r in subset], [r["raw_rank_90"] for r in subset]
        )
        fits[f"{regime}_rank_vs_centre"] = fit_slope(
            [r["centre"] for r in subset], [r["raw_rank_90"] for r in subset]
        )
        fits[f"{regime}_rank_vs_B_occupancy"] = fit_slope(
            [r["bandwidth"] * r["raw_occupancy_90"] for r in subset],
            [r["raw_rank_90"] for r in subset],
        )
    (RESULTS / "exp08_bandwidth_not_frequency.json").write_text(
        json.dumps({"fits": fits, "rows": rows}, indent=2)
    )
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
