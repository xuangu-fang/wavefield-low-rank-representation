"""Experiment 30: can the warp be obtained from the data a reconstruction has?

Experiment 29 picked each warp by scanning the *dense* field. That is a fair way
to answer "where should I put sensors", because a pilot survey really is dense.
It is not a reconstruction method: a reconstruction only ever sees the sparse
sensors, so a warp chosen with the dense field in hand would be circular.

Three ways of getting the warp are compared on identical footing, all scored by
the same untrained band-limited estimator:

  oracle-scan   one transport speed, chosen using the dense field   (upper reference)
  sparse-fit    a delay fitted to the observed columns of one gather (label-free)
  amortised     a network trained across gathers, given only the observed columns

The interesting question is not whether learning beats the oracle -- it is
whether the warp survives being estimated from fourteen sensors at all, and if
not, whether sharing structure across instances is what rescues it.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
import time
from pathlib import Path

import numpy as np

from wave_lr.amortised_warp import AmortisedConfig, train_warp_net
from wave_lr.families import openfwi_stack
from wave_lr.line_carrier import LineCarrierConfig, fit_line_carrier
from wave_lr.sensorline import (
    apply_warp,
    bound_line,
    reconstruct_bandlimited,
    transport_delays,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRACTION = 0.20
SPEEDS = np.linspace(1000.0, 6000.0, 40)
N_TEST = 40
SEEDS = (0, 1, 2)


def score(spectra, freqs, weights, delays, index):
    """Bound and measured error for one instance under one warp."""

    field = spectra[index]
    warped = field if delays is None else apply_warp(field, freqs, delays)
    return (
        bound_line(warped, weights, FRACTION),
        reconstruct_bandlimited(field, weights, FRACTION, delays, freqs),
    )


def main() -> None:
    start = time.time()
    train = openfwi_stack(path_index=0, n_sample=96)
    # A different file means different velocity models, not just different shots.
    test = openfwi_stack(path_index=1, n_sample=16)
    freqs, coords, weights = train["freqs"], train["coords"], train["weights"]
    stride = max(round(1.0 / FRACTION), 1)
    observed = np.arange(0, coords.size, stride)
    print(
        f"train {train['spectra'].shape}  test {test['spectra'].shape}  "
        f"observed {observed.size}/{coords.size} sensors",
        flush=True,
    )

    print("training amortised warp (label-free, criterion as the objective)", flush=True)
    predictors = {}
    for seed in SEEDS:
        predict, info = train_warp_net(
            train["spectra"],
            freqs,
            coords,
            weights,
            AmortisedConfig(fraction=FRACTION, steps=1200, batch=16, seed=seed),
        )
        predictors[seed] = predict
        print(
            f"  seed {seed}: objective {info['loss'][0]:.4f} -> "
            f"{np.mean(info['loss'][-50:]):.4f}  ({time.time()-start:.0f}s)",
            flush=True,
        )

    rows = []
    for index in range(min(N_TEST, test["spectra"].shape[0])):
        field = test["spectra"][index]
        raw_bound, raw_error = score(test["spectra"], freqs, weights, None, index)

        best = (float("nan"), raw_bound)
        for speed in SPEEDS:
            delays = transport_delays(coords, test["sources"][index], speed)
            value = bound_line(apply_warp(field, freqs, delays), weights, FRACTION)
            if value < best[1]:
                best = (float(speed), value)
        if np.isfinite(best[0]):
            oracle_delays = transport_delays(coords, test["sources"][index], best[0])
            oracle_bound, oracle_error = score(
                test["spectra"], freqs, weights, oracle_delays, index
            )
        else:
            oracle_bound, oracle_error = raw_bound, raw_error

        sparse = []
        for seed in SEEDS:
            evaluate, _ = fit_line_carrier(
                field[:, observed],
                freqs,
                coords[observed],
                LineCarrierConfig(steps=400, rank=3, seed=seed),
            )
            delays = evaluate(coords)
            sparse.append(score(test["spectra"], freqs, weights, delays - delays.min(), index))

        amortised = []
        for seed in SEEDS:
            delays = predictors[seed](field)
            amortised.append(score(test["spectra"], freqs, weights, delays, index))

        rows.append(
            {
                "name": test["names"][index],
                "source": float(test["sources"][index]),
                "bound_raw": raw_bound,
                "error_raw": raw_error,
                "speed": best[0],
                "bound_oracle": oracle_bound,
                "error_oracle": oracle_error,
                "bound_sparse": [b for b, _ in sparse],
                "error_sparse": [e for _, e in sparse],
                "bound_amortised": [b for b, _ in amortised],
                "error_amortised": [e for _, e in amortised],
            }
        )
        print(
            f"  {test['names'][index]:22s} err raw {raw_error:.3f} | oracle {oracle_error:.3f} "
            f"| sparse {np.mean([e for _, e in sparse]):.3f} "
            f"| amortised {np.mean([e for _, e in amortised]):.3f}  ({time.time()-start:.0f}s)",
            flush=True,
        )

    def gains(key):
        return np.array(
            [
                r["error_raw"] / max(np.mean(np.atleast_1d(r[key])), 1e-12)
                for r in rows
            ]
        )

    summary = {
        "n_test": len(rows),
        "fraction": FRACTION,
        "n_observed_sensors": int(observed.size),
        "n_train_instances": int(train["spectra"].shape[0]),
        "seeds": list(SEEDS),
        "error_raw_median": float(np.median([r["error_raw"] for r in rows])),
        "gain": {
            key: {
                "median": float(np.median(gains(f"error_{key}"))),
                "p25": float(np.percentile(gains(f"error_{key}"), 25)),
                "p75": float(np.percentile(gains(f"error_{key}"), 75)),
                "fraction_above_one": float((gains(f"error_{key}") > 1.0).mean()),
            }
            for key in ("oracle", "sparse", "amortised")
        },
        "seed_spread_amortised": float(
            np.median([np.std(r["error_amortised"]) for r in rows])
        ),
        "runtime_s": time.time() - start,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "exp30_learned_warp.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
