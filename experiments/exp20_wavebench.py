"""Experiment 20: WaveBench, a public benchmark with the medium supplied.

WaveBench's time-harmonic files hold the same heterogeneous media solved at
four frequencies with a fixed point source, which makes it the public dataset
this project was missing: a deployable eikonal carrier can be built from the
provided wavespeed, and the frequency can be swept on identical media.

The prediction being tested is sharp. A carrier removes spatial aliasing, so it
can only help once sensors are sparser than half a wavelength. On a 128 grid,
sensor spacing is ``1/sqrt(fraction)`` pixels, and the measured wavelengths are
65, 43, 31 and 15 pixels -- so nothing should happen except at the highest
frequency and the sparsest sampling.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from wave_lr.diagnostics import singular_spectrum
from wave_lr.tasks import sensor_interpolation_report
from wave_lr.wavebench import coordinates, load_samples

RESULTS = Path(__file__).resolve().parents[1] / "results"
OMEGA_LABELS = (10, 15, 20, 40)
FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)


def rank_at(values, level=0.99):
    spectrum = singular_spectrum(np.nan_to_num(values), use_gpu=False)
    return int(np.searchsorted(np.cumsum(spectrum**2), level) + 1)


def truncation_error(values, rank):
    spectrum = np.linalg.svd(values, compute_uv=False)
    return float(np.linalg.norm(spectrum[rank:]) / np.linalg.norm(spectrum))


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    families = {label: load_samples(label, count=args.count) for label in OMEGA_LABELS}
    grid = families[OMEGA_LABELS[0]][0].speed.shape[0]
    coords = coordinates(grid)

    rows = []
    for label in OMEGA_LABELS:
        for sample in families[label]:
            values = sample.field.ravel()[:, None]
            keep = np.abs(values[:, 0]) > 0
            frequencies = np.array([sample.kappa / (2.0 * np.pi)])
            delays = sample.travel_time.ravel()
            row = {
                "omega_label": label,
                "index": sample.index,
                "kappa": sample.kappa,
                "wavelength_px": sample.wavelength_pixels,
            }
            for fraction in FRACTIONS:
                spacing = 1.0 / np.sqrt(fraction)
                raw = sensor_interpolation_report(
                    coords[keep], values[keep], frequencies, None, fraction,
                    seed=sample.index,
                )["complex_nrmse"]
                aligned = sensor_interpolation_report(
                    coords[keep], values[keep], frequencies, delays[keep], fraction,
                    seed=sample.index,
                )["complex_nrmse"]
                tag = f"p{fraction * 1000:.0f}"
                row[f"{tag}_raw"] = raw
                row[f"{tag}_aligned"] = aligned
                row[f"{tag}_gain"] = raw / max(aligned, 1e-12)
                row[f"{tag}_spacing_px"] = spacing
                row[f"{tag}_samples_per_wavelength"] = sample.wavelength_pixels / spacing
            rows.append(row)
        print(f"omega_label {label}: {len(families[label])} samples done", flush=True)

    # The same media appear in every file, so the four frequencies of one index
    # form a genuine (x, omega) matrix.
    cross = []
    n_shared = min(len(families[label]) for label in OMEGA_LABELS)
    for position in range(n_shared):
        samples = [families[label][position] for label in OMEGA_LABELS]
        assert all(
            np.array_equal(samples[0].speed, other.speed) for other in samples[1:]
        ), "frequency files are not index-aligned"
        matrix = np.stack([s.field.ravel() for s in samples], axis=1)
        keep = np.abs(matrix).sum(axis=1) > 0
        matrix = matrix[keep]
        phases = np.stack([s.phase.ravel()[keep] for s in samples], axis=1)
        aligned = matrix * np.exp(1j * phases)
        cross.append(
            {
                "index": samples[0].index,
                "raw_rank": rank_at(matrix),
                "aligned_rank": rank_at(aligned),
                "raw_r1": truncation_error(matrix, 1),
                "aligned_r1": truncation_error(aligned, 1),
                "raw_r2": truncation_error(matrix, 2),
                "aligned_r2": truncation_error(aligned, 2),
            }
        )

    payload = {"fractions": FRACTIONS, "omega_labels": OMEGA_LABELS,
               "rows": rows, "cross_frequency": cross}
    (RESULTS / "exp20_wavebench.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{'omega':>6s}{'lambda/px':>11s}" + "".join(f"{f'p={f:.3f}':>12s}" for f in FRACTIONS))
    for label in OMEGA_LABELS:
        subset = [r for r in rows if r["omega_label"] == label]
        line = f"{label:6d}{np.mean([r['wavelength_px'] for r in subset]):11.1f}"
        for fraction in FRACTIONS:
            tag = f"p{fraction * 1000:.0f}"
            line += f"{np.mean([r[f'{tag}_gain'] for r in subset]):12.2f}"
        print(line)
    print("\n(表中为 raw/aligned 增益；>1 表示载波有帮助)")
    print(
        f"cross-frequency: rank {np.mean([c['raw_rank'] for c in cross]):.2f} -> "
        f"{np.mean([c['aligned_rank'] for c in cross]):.2f}, "
        f"r1 {np.mean([c['raw_r1'] for c in cross]):.3f} -> "
        f"{np.mean([c['aligned_r1'] for c in cross]):.3f}"
    )


if __name__ == "__main__":
    main()
