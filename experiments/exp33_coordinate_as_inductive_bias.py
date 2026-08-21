"""Experiment 33: is the coordinate a better route, or a better representation?

Experiment 32 put the two ways of spending a learned prior side by side and the
direct supervised model won: 0.114 against 0.156, with twenty-one times the
parameters and dense targets the coordinate route never sees. That settles the
comparison but leaves the more interesting question open.

The coordinate route is capped by the bound in its own coordinate system,
because its estimator has no parameters and therefore no prior. The direct model
has no such cap. So the question is not which route wins on its own -- it is
whether the coordinate helps the model that does win. If a reconstruction
network trained in the reparameterised coordinate beats the identical network
trained in the raw one at matched capacity, the warp is an inductive bias for
amortised models, not merely an alternative to them.

The warp is predicted from the sparse sensors by the same label-free network as
before, so nothing here sees a delay label or a dense field at deployment.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
import time
from pathlib import Path

import numpy as np
import torch
from exp32_direct_vs_coordinate import relative, train_reconstruction

from wave_lr.amortised_warp import AmortisedConfig, WarpNet, train_warp_net
from wave_lr.families import openfwi_stack
from wave_lr.sensorline import apply_warp, bound_line, tapered

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRACTION = 0.20
SEEDS = (0, 1, 2)
WIDTHS = (32, 64, 128, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def warp_all(spectra, freqs, predict):
    """Reparameterise every instance by the delay its own sparse sensors imply."""

    out = np.empty_like(spectra)
    delays = np.empty((spectra.shape[0], spectra.shape[2]))
    for index in range(spectra.shape[0]):
        delays[index] = predict(spectra[index])
        out[index] = apply_warp(spectra[index], freqs, delays[index])
    return out, delays


def main() -> None:
    start = time.time()
    train = openfwi_stack(path_index=0, n_sample=96)
    test = openfwi_stack(path_index=1, n_sample=8)
    freqs, coords, weights = train["freqs"], train["coords"], train["weights"]
    stride = max(round(1.0 / FRACTION), 1)
    mask = torch.zeros(coords.size, device=DEVICE)
    mask[::stride] = 1.0
    print(f"train {train['spectra'].shape}  test {test['spectra'].shape}", flush=True)

    predict_warp, info = train_warp_net(
        train["spectra"],
        freqs,
        coords,
        weights,
        AmortisedConfig(fraction=FRACTION, steps=1200, batch=16, seed=0),
    )
    warp_params = sum(p.numel() for p in WarpNet(freqs.size, AmortisedConfig()).parameters())
    print(
        f"warp net {warp_params/1e3:.0f}k parameters, objective {info['loss'][0]:.4f} -> "
        f"{np.mean(info['loss'][-50:]):.4f}  ({time.time()-start:.0f}s)",
        flush=True,
    )

    train_warped, _ = warp_all(train["spectra"], freqs, predict_warp)
    test_warped, test_delays = warp_all(test["spectra"], freqs, predict_warp)
    print(f"warped both splits  ({time.time()-start:.0f}s)", flush=True)

    # A unimodular factor per sensor leaves every norm in the error untouched,
    # so an error measured in the warped coordinate is the error in the raw one.
    check = np.abs(
        np.abs(tapered(test_warped[0])) - np.abs(tapered(test["spectra"][0]))
    ).max()
    assert check < 1e-6, f"warp changed the magnitudes by {check:.2e}"

    results = {}
    for label, data_train, data_test in (
        ("raw", train["spectra"], test["spectra"]),
        ("warped", train_warped, test_warped),
    ):
        results[label] = {}
        for width in WIDTHS:
            errors, losses, n_param = [], [], 0
            for seed in SEEDS:
                predict, n_param, history = train_reconstruction(
                    data_train, mask, width, 6, seed
                )
                block = [
                    relative(data_test[i], predict(data_test[i]), weights)
                    for i in range(data_test.shape[0])
                ]
                errors.append(float(np.median(block)))
                losses.append(float(np.mean(history[-50:])))
            results[label][str(width)] = {
                "n_param": int(n_param),
                "test_error_median": float(np.mean(errors)),
                "test_error_std": float(np.std(errors)),
                "train_loss": float(np.mean(losses)),
            }
            print(
                f"  {label:6s} width {width:4d}: {n_param/1e3:6.0f}k par  "
                f"test {np.mean(errors):.4f} +/- {np.std(errors):.4f}  "
                f"train {np.mean(losses):.3e}  ({time.time()-start:.0f}s)",
                flush=True,
            )

    bounds = {
        "raw": float(
            np.median([bound_line(f, weights, FRACTION) for f in test["spectra"]])
        ),
        "warped": float(np.median([bound_line(f, weights, FRACTION) for f in test_warped])),
    }
    ratios = {
        width: results["warped"][width]["test_error_median"]
        / max(results["raw"][width]["test_error_median"], 1e-12)
        for width in results["raw"]
    }
    matched = min(
        (
            v["n_param"]
            for v in results["raw"].values()
            if v["n_param"] >= warp_params
        ),
        default=None,
    )
    summary = {
        "n_test": int(test["spectra"].shape[0]),
        "n_train_instances": int(train["spectra"].shape[0]),
        "fraction": FRACTION,
        "warp_net_parameters": int(warp_params),
        "bounds_median": bounds,
        "results": results,
        "warped_over_raw_by_width": ratios,
        "best_raw": float(min(v["test_error_median"] for v in results["raw"].values())),
        "best_warped": float(min(v["test_error_median"] for v in results["warped"].values())),
        "smallest_raw_net_matching_warp_parameters": matched,
        "delay_spread_median_s": float(np.median(np.ptp(test_delays, axis=1))),
        "runtime_s": time.time() - start,
    }
    summary["best_warped_over_best_raw"] = summary["best_warped"] / max(
        summary["best_raw"], 1e-12
    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "exp33_coordinate_bias.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
