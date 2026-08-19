"""Shared fit summaries so every experiment scores the law the same way."""

from __future__ import annotations

import numpy as np

from .analysis import ENERGY_LEVELS
from .theory import fit_slope

CARRIERS = ("raw", "eikonal", "straight", "data_pick")


def summarize_fits(rows: list[dict]) -> dict:
    """Fit measured rank against ``bandwidth x occupancy`` at each energy level."""

    fits: dict[str, dict] = {}
    for level in ENERGY_LEVELS:
        tag = int(level * 100)
        pooled_x, pooled_y = [], []
        for carrier in CARRIERS:
            if f"{carrier}_rank_{tag}" not in rows[0]:
                continue
            x = np.array([r[f"{carrier}_occupancy_{tag}"] * r["bandwidth"] for r in rows])
            y = np.array([r[f"{carrier}_rank_{tag}"] for r in rows], dtype=float)
            fits[f"rank_vs_occupancy_{carrier}_{tag}"] = fit_slope(x, y)
            pooled_x.append(x)
            pooled_y.append(y)
        fits[f"rank_vs_occupancy_pooled_{tag}"] = fit_slope(
            np.concatenate(pooled_x), np.concatenate(pooled_y)
        )
        for carrier in CARRIERS[1:]:
            key = f"{carrier}_predicted_gain_{tag}"
            if key not in rows[0]:
                continue
            predicted = np.array([r[key] for r in rows])
            measured = np.array([r[f"{carrier}_measured_gain_{tag}"] for r in rows])
            fits[f"gain_{carrier}_{tag}"] = fit_slope(predicted, measured)
            fits[f"log_gain_{carrier}_{tag}"] = fit_slope(
                np.log(np.maximum(predicted, 1e-9)), np.log(np.maximum(measured, 1e-9))
            )
    return fits
