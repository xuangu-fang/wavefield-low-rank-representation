"""Aggregate every result file into one compact summary for the report."""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"


def load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def fit(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.stack([x, np.ones_like(x)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "r2": 1.0 - float(residual @ residual) / total if total > 0 else float("nan"),
        "n": int(x.size),
    }


def through_origin(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope = float(np.sum(x * y) / np.sum(x * x))
    residual = y - slope * x
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": slope,
        "r2": 1.0 - float(residual @ residual) / total if total > 0 else float("nan"),
        "n": int(x.size),
    }


def rank_law_summary() -> dict:
    out = {}
    synthetic = load("exp01_rank_law.json")
    if synthetic:
        rows = [r for r in synthetic["rows"] if r["measured_raw_rank_99"] < 0.4 * r["grid"]]
        for level in (90, 99):
            x = [r["bandwidth"] * r[f"raw_occupancy_{level}"] for r in rows] + [
                r["bandwidth"] * r[f"demod_occupancy_{level}"] for r in rows
            ]
            y = [r[f"measured_raw_rank_{level}"] for r in rows] + [
                r[f"measured_demodulated_rank_{level}"] for r in rows
            ]
            out[f"synthetic_{level}"] = {**fit(x, y), "origin": through_origin(x, y)}
            for alternative in ("support", "union"):
                xa = [r["bandwidth"] * r[f"raw_{alternative}_{level}"] for r in rows] + [
                    r["bandwidth"] * r[f"demod_{alternative}_{level}"] for r in rows
                ]
                out[f"synthetic_{alternative}_{level}"] = fit(xa, y)

    for label, name in (
        ("fdtd", "exp03_regime_phase_diagram.json"),
        ("maze", "exp02_well_acoustic_maze.json"),
        ("openfwi", "exp02_openfwi_gathers.json"),
    ):
        payload = load(name)
        if not payload:
            continue
        rows = payload["rows"]
        for level in (90, 99):
            x, y = [], []
            for carrier in ("raw", "eikonal", "straight", "data_pick"):
                x += [r["bandwidth"] * r[f"{carrier}_occupancy_{level}"] for r in rows]
                y += [r[f"{carrier}_rank_{level}"] for r in rows]
            out[f"{label}_{level}"] = {**fit(x, y), "origin": through_origin(x, y)}
    return out


def regime_table() -> list[dict]:
    payload = load("exp03_regime_phase_diagram.json")
    if not payload:
        return []
    groups = defaultdict(list)
    for row in payload["rows"]:
        if row["f_min"] != 6.0:
            continue
        groups[(row["boundary"], row["clutter"])].append(row)
    table = []
    for (boundary, clutter), rows in groups.items():
        table.append(
            {
                "boundary": boundary,
                "clutter": clutter,
                "occupancy_raw": float(np.mean([r["raw_occupancy_90"] for r in rows])),
                "occupancy_aligned": float(np.mean([r["eikonal_occupancy_90"] for r in rows])),
                "rank_raw": float(np.mean([r["raw_rank_90"] for r in rows])),
                "rank_aligned": float(np.mean([r["eikonal_rank_90"] for r in rows])),
                "measured_gain": float(np.mean([r["eikonal_measured_gain_90"] for r in rows])),
                "predicted_gain": float(np.mean([r["eikonal_predicted_gain_90"] for r in rows])),
                "n_seeds": len(rows),
            }
        )
    return sorted(table, key=lambda r: -r["measured_gain"])


def task_table() -> dict:
    payload = load("exp05_task_gain_vs_regime.json")
    if not payload:
        return {}
    rows = payload["rows"]
    fractions = payload["sensor_fractions"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["boundary"], row["clutter"])].append(row)

    table = []
    for (boundary, clutter), group in groups.items():
        entry = {"boundary": boundary, "clutter": clutter,
                 "rank_gain": float(np.mean([r["eikonal_measured_gain_90"] for r in group]))}
        for fraction in fractions:
            tag = f"p{int(fraction * 100)}"
            raw = np.array([r[f"interp_raw_{tag}_complex_nrmse"] for r in group])
            aligned = np.array([r[f"interp_eikonal_{tag}_complex_nrmse"] for r in group])
            entry[f"interp_raw_{tag}"] = float(raw.mean())
            entry[f"interp_aligned_{tag}"] = float(aligned.mean())
            entry[f"interp_gain_{tag}"] = float((raw / np.maximum(aligned, 1e-9)).mean())
        if f"complete_raw_best_complex_nrmse" in group[0]:
            raw = np.array([r["complete_raw_best_complex_nrmse"] for r in group])
            aligned = np.array([r["complete_eikonal_best_complex_nrmse"] for r in group])
            entry["complete_raw"] = float(raw.mean())
            entry["complete_aligned"] = float(aligned.mean())
            entry["complete_gain"] = float((raw / np.maximum(aligned, 1e-9)).mean())
        table.append(entry)

    correlation = {}
    for fraction in fractions:
        tag = f"p{int(fraction * 100)}"
        rank_gain = np.array([r["eikonal_measured_gain_90"] for r in rows])
        task_gain = np.array(
            [
                r[f"interp_raw_{tag}_complex_nrmse"]
                / max(r[f"interp_eikonal_{tag}_complex_nrmse"], 1e-9)
                for r in rows
            ]
        )
        correlation[tag] = fit(np.log(rank_gain), np.log(task_gain))
    return {"per_regime": sorted(table, key=lambda r: -r["rank_gain"]), "log_fits": correlation}


def public_table() -> list[dict]:
    payload = load("exp06_public_data_tasks.json")
    if not payload:
        return []
    groups = defaultdict(list)
    for row in payload["rows"]:
        groups[row.get("dataset", "unknown")].append(row)
    table = []
    for dataset, rows in groups.items():
        entry = {"dataset": dataset, "n_cases": len(rows),
                 "rank_raw": float(np.mean([r["raw_rank_90"] for r in rows])),
                 "rank_aligned": float(np.mean([r["eikonal_rank_90"] for r in rows]))}
        entry["rank_gain"] = entry["rank_raw"] / max(entry["rank_aligned"], 1e-9)
        for tag in ("p1", "p2", "p5", "p10"):
            key = f"interp_raw_{tag}_complex_nrmse"
            if key not in rows[0]:
                continue
            raw = np.array([r[key] for r in rows])
            aligned = np.array([r[f"interp_eikonal_{tag}_complex_nrmse"] for r in rows])
            entry[f"interp_raw_{tag}"] = float(raw.mean())
            entry[f"interp_aligned_{tag}"] = float(aligned.mean())
            entry[f"interp_gain_{tag}"] = float((raw / np.maximum(aligned, 1e-9)).mean())
        if "complete_raw_best_complex_nrmse" in rows[0]:
            raw = np.array([r["complete_raw_best_complex_nrmse"] for r in rows])
            aligned = np.array([r["complete_eikonal_best_complex_nrmse"] for r in rows])
            entry["complete_raw"] = float(raw.mean())
            entry["complete_aligned"] = float(aligned.mean())
        table.append(entry)
    return table


def extrapolation_table() -> list[dict]:
    payload = load("exp04_staircase_test.json")
    if not payload:
        return []
    groups = defaultdict(list)
    for row in payload["rows"]:
        groups[row["n_train"]].append(row)
    table = []
    for n_train, rows in sorted(groups.items()):
        entry = {"n_train": n_train,
                 "frequency_reach": float(np.mean([r["frequency_reach"] for r in rows]))}
        for key, label in (
            ("copy_complex_nrmse", "copy_last"),
            ("raw_complex_best_complex_nrmse", "raw_pointwise"),
            ("raw_amplitude_phase_best_complex_nrmse", "raw_amplitude_phase"),
            ("eikonal_complex_best_complex_nrmse", "aligned_pointwise"),
            ("raw_lowrank_best_complex_nrmse", "raw_lowrank"),
            ("eikonal_lowrank_best_complex_nrmse", "aligned_lowrank"),
        ):
            if key in rows[0]:
                entry[label] = float(np.mean([r[key] for r in rows]))
        table.append(entry)
    return table


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    summary = {
        "rank_law": rank_law_summary(),
        "regimes": regime_table(),
        "tasks": task_table(),
        "public": public_table(),
        "staircase_extrapolation": extrapolation_table(),
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["rank_law"], indent=2))
    print(json.dumps(summary["regimes"], indent=2))


if __name__ == "__main__":
    main()
