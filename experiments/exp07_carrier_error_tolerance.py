"""Experiment 7: how accurate must the carrier be?

The law caps the usable gain at ``Lambda_abs / max(Lambda_rel, |delta tau|)``,
so a travel-time error acts exactly like an extra delay spread. That predicts a
knee when the error reaches the resolution ``1 / B`` of the observed band, and
predicts that the knee moves with bandwidth rather than with centre frequency.
Both are tested here by perturbing an otherwise exact eikonal carrier.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from wave_lr.diagnostics import singular_spectrum
from wave_lr.fdtd import MediumSpec
from wave_lr.fields import fdtd_case
from wave_lr.spectra import carrier, to_spectrum
from wave_lr.tasks import sensor_interpolation_report

RESULTS = Path(__file__).resolve().parents[1] / "results"
BANDS = [(6.0, 12.0), (6.0, 24.0), (12.0, 30.0)]
REGIMES = {
    "open_clear": {"absorption": 40.0, "scatterer_fraction": 0.0},
    "open_sparse": {"absorption": 40.0, "scatterer_fraction": 0.08},
    "partial_clear": {"absorption": 3.0, "scatterer_fraction": 0.0},
}
ERROR_SCALES = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5)


def smooth_error(coords: np.ndarray, seed: int, length_scale: float = 0.25) -> np.ndarray:
    """Unit-variance spatially smooth travel-time error."""

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(6, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    phases = rng.uniform(0, 2 * np.pi, size=6)
    field = np.zeros(coords.shape[0])
    for direction, phase in zip(directions, phases):
        field += np.cos(2 * np.pi * (coords @ direction) / length_scale + phase)
    return field / field.std()


def measure(values: np.ndarray) -> int:
    spectrum = singular_spectrum(values)
    return int(np.searchsorted(np.cumsum(spectrum**2), 0.90) + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    rows = []
    for regime, settings in REGIMES.items():
        for seed in range(args.seeds):
            case = fdtd_case(MediumSpec(name=f"{regime}_s{seed}", seed=seed, **settings))
            perturbation = smooth_error(case.coords, seed=seed)
            for f_min, f_max in BANDS:
                spectrum = to_spectrum(case.traces, case.dt, f_min, f_max)
                bandwidth = spectrum.bandwidth
                resolution = 1.0 / bandwidth
                raw_rank = measure(spectrum.values)
                raw_task = sensor_interpolation_report(
                    case.coords, spectrum.values, spectrum.frequencies, None, 0.02, seed=seed
                )
                for scale, kind in itertools.product(ERROR_SCALES, ("smooth", "scaling")):
                    if kind == "smooth":
                        delays = case.travel_time + scale * resolution * perturbation
                    else:
                        span = case.travel_time.max() - case.travel_time.min()
                        delays = case.travel_time * (
                            1.0 + scale * resolution / max(span, 1e-9)
                        )
                    aligned = spectrum.values * np.conj(
                        carrier(spectrum.frequencies, delays)
                    )
                    task = sensor_interpolation_report(
                        case.coords, spectrum.values, spectrum.frequencies, delays, 0.02, seed=seed
                    )
                    rows.append(
                        {
                            "regime": regime,
                            "seed": seed,
                            "band": [f_min, f_max],
                            "bandwidth": bandwidth,
                            "error_kind": kind,
                            "error_scale_in_resolutions": scale,
                            "error_rms": float(np.std(delays - case.travel_time)),
                            "raw_rank_90": raw_rank,
                            "aligned_rank_90": measure(aligned),
                            "raw_interp_nrmse": raw_task["complex_nrmse"],
                            "aligned_interp_nrmse": task["complex_nrmse"],
                            "aligned_interp_awpc": task["awpc"],
                        }
                    )
            print(f"{regime} seed{seed} done", flush=True)

    (RESULTS / "exp07_carrier_error_tolerance.json").write_text(
        json.dumps({"error_scales": ERROR_SCALES, "bands": BANDS, "rows": rows}, indent=2)
    )
    for kind in ("smooth", "scaling"):
        print(f"\n--- {kind} error, band 6-24 ---")
        for scale in ERROR_SCALES:
            subset = [
                r for r in rows
                if r["error_kind"] == kind and r["error_scale_in_resolutions"] == scale
                and r["band"] == [6.0, 24.0] and r["regime"] == "open_clear"
            ]
            if subset:
                print(
                    f"  delta_tau = {scale:4.2f}/B : rank "
                    f"{np.mean([r['raw_rank_90'] for r in subset]):5.1f} -> "
                    f"{np.mean([r['aligned_rank_90'] for r in subset]):5.1f}   "
                    f"nrmse {np.mean([r['raw_interp_nrmse'] for r in subset]):.3f} -> "
                    f"{np.mean([r['aligned_interp_nrmse'] for r in subset]):.3f}"
                )


if __name__ == "__main__":
    main()
