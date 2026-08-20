"""Experiment 19: does the law reach beyond wave fields?

The degree-of-freedom argument never mentions waves, so the bound
``rank <= bandwidth x occupancy`` should hold for any spatiotemporal field.
Whether it is *tight*, and whether alignment buys anything, is a different
question -- and the answer turns out to delimit the method's scope rather than
extend it. Five recorded non-wave flows, one synthetic non-wave transported
blob with tunable diffusion, and the wave regimes are all measured identically.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from wave_lr.diagnostics import singular_spectrum
from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
from wave_lr.spectra import band_limited_traces, shift_spectrum, to_spectrum
from wave_lr.theory import fit_slope, occupancy_from_traces
from wave_lr.transport import load_synthetic_transport, load_transport_cases

RESULTS = Path(__file__).resolve().parents[1] / "results"
BANDS = ((0.0125, 0.05), (0.02, 0.10), (0.05, 0.15), (0.02, 0.20))
RECORDED = (
    "cylinder_wake",
    "kuramoto_sivashinsky",
    "kolmogorov_flow",
    "active_matter",
    "diffusion_reaction",
)
DIFFUSIVITIES = (0.0, 0.0005, 0.002, 0.01)


def rank_at(values, level=0.99):
    spectrum = singular_spectrum(np.nan_to_num(values), use_gpu=False)
    return int(np.searchsorted(np.cumsum(spectrum**2), level) + 1)


def measure(case, band) -> dict | None:
    spectrum = to_spectrum(case.traces, case.dt, *band)
    if spectrum.values.shape[1] < 4:
        return None
    bandwidth = spectrum.bandwidth

    def occupancy(delays):
        shifted = (
            spectrum if delays is None else shift_spectrum(spectrum, delays - delays.max())
        )
        traces, _ = band_limited_traces(shifted)
        return occupancy_from_traces(traces, spectrum.dt, bandwidth)

    def rank(delays):
        return rank_at(
            spectrum.values if delays is None else shift_spectrum(spectrum, delays).values
        )

    raw_occupancy = occupancy(None)
    aligned_occupancy = occupancy(case.travel_time)
    raw_rank, aligned_rank = rank(None), rank(case.travel_time)
    return {
        "dataset": case.dataset,
        "case": case.name,
        "band": list(band),
        "bandwidth": bandwidth,
        "n_f": int(spectrum.values.shape[1]),
        "n_x": int(case.n_x),
        "delay_estimator": case.metadata.get("delay_estimator", "physics"),
        "raw_occupancy": raw_occupancy,
        "aligned_occupancy": aligned_occupancy,
        "raw_rank": raw_rank,
        "aligned_rank": aligned_rank,
        "predicted_gain": (bandwidth * raw_occupancy + 1) / (bandwidth * aligned_occupancy + 1),
        "measured_gain": raw_rank / max(aligned_rank, 1),
        "occupancy_fraction": raw_occupancy / (case.traces.shape[1] * 2 * case.dt),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    cases = []
    for name in RECORDED:
        cases += load_transport_cases(name, limit=args.limit)
    for diffusivity in DIFFUSIVITIES:
        for case in load_synthetic_transport(limit=args.limit, diffusivity=diffusivity):
            case.metadata["diffusivity"] = diffusivity
            case.dataset = f"advection_diffusion_D{diffusivity}"
            cases.append(case)
    for regime, settings in (
        ("wave_open_clear", {"absorption": 40.0, "scatterer_fraction": 0.0}),
        ("wave_open_sparse", {"absorption": 40.0, "scatterer_fraction": 0.08}),
        ("wave_closed_dense", {"absorption": 0.0, "scatterer_fraction": 0.22}),
    ):
        for seed in range(min(args.limit, 2)):
            case = fdtd_case(MediumSpec(name=f"{regime}_s{seed}", seed=seed, **settings))
            case.dataset = regime
            cases.append(case)

    rows = []
    for case in cases:
        is_wave = case.dataset.startswith("wave_")
        bands = ((6.0, 24.0), (8.0, 16.0), (12.0, 24.0)) if is_wave else BANDS
        for band in bands:
            row = measure(case, band)
            if row is not None:
                row["family"] = "wave" if is_wave else "non_wave"
                rows.append(row)
        print(
            f"{case.dataset:28s} {case.name:22s} "
            f"occ frac {rows[-1]['occupancy_fraction']:.2f}  "
            f"gain pred {rows[-1]['predicted_gain']:.2f} meas {rows[-1]['measured_gain']:.2f}",
            flush=True,
        )

    fits = {}
    for family in ("wave", "non_wave", "all"):
        subset = [r for r in rows if family == "all" or r["family"] == family]
        x = np.array([r["bandwidth"] * r["raw_occupancy"] for r in subset])
        y = np.array([r["raw_rank"] for r in subset], dtype=float)
        fits[f"rank_vs_occupancy_{family}"] = fit_slope(x, y)
        fits[f"bound_violation_rate_{family}"] = float(np.mean(y > x + 1))
    (RESULTS / "exp19_transport_generality.json").write_text(
        json.dumps({"bands": BANDS, "fits": fits, "rows": rows}, indent=2)
    )
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
