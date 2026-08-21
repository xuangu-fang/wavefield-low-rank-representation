"""Experiment 34: can the transport speed be measured from the sparse sensors alone?

Experiment 29 scanned each transport speed against the identifiability criterion
computed on the *dense* field. That is a legitimate thing to do -- a pilot survey
really is dense, and the question "where should sensors go" is answered before
any sparse deployment exists. But it does not support the sentence "from 20% of
sensors, an equation-free estimate of the convection speed", and that sentence
is the one the measurement-instrument framing needs.

The objection to sparse-only estimation is exact: the criterion is an integral
of energy *above* the array's Nyquist, and a sparse array cannot see above its
own Nyquist. What it sees is the folded spectrum.

The objection is not fatal, for one reason. The warp multiplies each sensor by a
unit-modulus complex number, so it leaves the energy at every sampled point
untouched -- the sparse array's total energy carries no information about the
speed at all. What does change is how that fixed energy is *distributed* across
the folded band. At one frequency this is useless: the true wavenumber
k0 + f/c folds to (k0 + f/c) mod 1/m, a spike that moves with c but never
changes its concentration. Across many frequencies it is not useless: the
correct speed folds every frequency onto k = 0 at once, and a wrong one
scatters them. The information lives in the joint structure over frequency,
which fourteen columns can still show.

So two sparse-only objectives are scanned over the same one-parameter family
that experiment 29 used, and their argmin is compared against the dense scan and
against the physics. The estimator that failed in experiment 30 was a
twenty-four-feature network fitted through fourteen points; a single scalar is a
different question and has never been asked.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
import time
from pathlib import Path

import numpy as np

from wave_lr.families import cylinder_cases, openfwi_stack
from wave_lr.sensorline import (
    apply_warp,
    bound_line,
    reconstruct_bandlimited,
    tapered,
    transport_delays,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRACTION = 0.20
CUTOFFS = (0.10, 0.20, 0.30)


def folded_spread(observed, weights, cutoff, taper=0.25):
    """Energy outside the lowest folded wavenumbers, seen by the sparse array only.

    This is what survives of the criterion when the array is all you have. The
    band above the *dense* Nyquist is invisible by construction; what is visible
    is whether the fourteen columns look alike, and the correct warp is the one
    that makes them look most alike at every frequency at once.
    """

    power = np.abs(np.fft.fft(tapered(observed, taper), axis=-1)) ** 2
    wavenumber = np.abs(np.fft.fftfreq(observed.shape[-1]))
    outside = power[:, wavenumber > cutoff].sum(1)
    total = power.sum(1)
    share = np.where(total > 0, outside / np.maximum(total, 1e-300), 0.0)
    return float((weights * share).sum())


def nuclear_ratio(observed, weights):
    """Rank surrogate on the observed columns: nuclear norm over Frobenius.

    Scale-free by construction, and a unit-modulus factor cannot move it, so it
    can only fall when the columns genuinely become more alike.
    """

    del weights
    values = tapered(observed)
    singular = np.linalg.svd(values, compute_uv=False)
    return float(singular.sum() / (np.linalg.norm(values) + 1e-30))


def scan(spectrum, freqs, coords, source, speeds, observed, weights, objective):
    """Argmin of a sparse-only objective over the one-parameter transport family."""

    best_value, best_speed = np.inf, float("nan")
    for speed in speeds:
        delays = transport_delays(coords, source, speed)
        warped = apply_warp(spectrum, freqs, delays)[:, observed]
        value = objective(warped, weights)
        if value < best_value:
            best_value, best_speed = value, float(speed)
    return best_speed, best_value


def dense_scan(spectrum, freqs, coords, source, speeds, weights):
    raw = bound_line(spectrum, weights, FRACTION)
    best_value, best_speed = raw, float("nan")
    for speed in speeds:
        delays = transport_delays(coords, source, speed)
        value = bound_line(apply_warp(spectrum, freqs, delays), weights, FRACTION)
        if value < best_value:
            best_value, best_speed = value, float(speed)
    return best_speed, raw, best_value


def evaluate(spectrum, freqs, coords, source, speed, weights):
    """Reconstruction gain actually delivered by a speed, whatever chose it."""

    raw = reconstruct_bandlimited(spectrum, weights, FRACTION)
    if not np.isfinite(speed):
        return raw, raw, 1.0
    delays = transport_delays(coords, source, speed)
    warped = reconstruct_bandlimited(spectrum, weights, FRACTION, delays, freqs)
    return raw, warped, raw / max(warped, 1e-12)


def run_family(name, cases, speeds, truth_speed=None):
    rows = []
    for case in cases:
        spectrum, freqs, coords, source, weights = case
        observed = np.arange(0, coords.size, max(round(1.0 / FRACTION), 1))
        d_speed, raw_bound, dense_bound = dense_scan(
            spectrum, freqs, coords, source, speeds, weights
        )
        entry = {
            "family": name,
            "bound_raw": raw_bound,
            "bound_dense_scan": dense_bound,
            "speed_dense": d_speed,
            "n_observed": int(observed.size),
            "n_sensor": int(coords.size),
        }
        raw_err, _, dense_gain = evaluate(
            spectrum, freqs, coords, source, d_speed, weights
        )
        entry["error_raw"] = raw_err
        entry["gain_dense"] = dense_gain

        for cutoff in CUTOFFS:
            key = f"fold{int(cutoff*100)}"
            speed, _ = scan(
                spectrum,
                freqs,
                coords,
                source,
                speeds,
                observed,
                weights,
                lambda block, w, cut=cutoff: folded_spread(block, w, cut),
            )
            _, _, gain = evaluate(spectrum, freqs, coords, source, speed, weights)
            entry[f"speed_{key}"] = speed
            entry[f"gain_{key}"] = gain

        speed, _ = scan(
            spectrum, freqs, coords, source, speeds, observed, weights, nuclear_ratio
        )
        _, _, gain = evaluate(spectrum, freqs, coords, source, speed, weights)
        entry["speed_nuclear"] = speed
        entry["gain_nuclear"] = gain
        if truth_speed is not None:
            entry["speed_reference"] = float(truth_speed)
        rows.append(entry)
    return rows


def main() -> None:
    start = time.time()
    rows = []

    stack = openfwi_stack(path_index=1, n_sample=12)
    seismic = [
        (
            stack["spectra"][i],
            stack["freqs"],
            stack["coords"],
            float(stack["sources"][i]),
            stack["weights"],
        )
        for i in range(stack["spectra"].shape[0])
    ]
    rows += run_family("seismic", seismic, np.linspace(1000.0, 6000.0, 40))
    print(f"seismic done, {len(rows)} cases  ({time.time()-start:.0f}s)", flush=True)

    wake_speeds = np.concatenate(
        [
            -np.logspace(np.log10(0.03), np.log10(3.0), 24)[::-1],
            np.logspace(np.log10(0.03), np.log10(3.0), 24),
        ]
    )
    wake = [
        (case.spectrum, case.freqs, case.coords, case.source, case.weights)
        for case in cylinder_cases(limit=12, field="v")
    ]
    rows += run_family("wake", wake, wake_speeds)
    print(f"wake done, {len(rows)} cases  ({time.time()-start:.0f}s)", flush=True)

    summary = {"fraction": FRACTION, "cutoffs": list(CUTOFFS), "by_family": {}}
    for family in sorted({r["family"] for r in rows}):
        block = [r for r in rows if r["family"] == family]
        dense = np.array([r["speed_dense"] for r in block])
        entry = {
            "n_cases": len(block),
            "speed_dense_median": float(np.nanmedian(dense)),
            "gain_dense_median": float(np.median([r["gain_dense"] for r in block])),
        }
        for key in [f"fold{int(c*100)}" for c in CUTOFFS] + ["nuclear"]:
            speeds = np.array([r[f"speed_{key}"] for r in block])
            gains = np.array([r[f"gain_{key}"] for r in block])
            valid = np.isfinite(speeds) & np.isfinite(dense) & (np.abs(dense) > 0)
            relative = (
                np.abs(speeds[valid] - dense[valid]) / np.abs(dense[valid])
                if valid.any()
                else np.array([np.nan])
            )
            entry[key] = {
                "speed_median": float(np.nanmedian(speeds)),
                "relative_error_vs_dense_median": float(np.nanmedian(relative)),
                "within_20pct_of_dense": float(np.mean(relative < 0.2)) if valid.any() else 0.0,
                "gain_median": float(np.median(gains)),
                "gain_fraction_of_dense": float(
                    np.median(gains) / max(entry["gain_dense_median"], 1e-12)
                ),
                "helped_fraction": float(np.mean(gains > 1.0)),
            }
        summary["by_family"][family] = entry

    summary["runtime_s"] = time.time() - start
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "exp34_sparse_only_speed.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
