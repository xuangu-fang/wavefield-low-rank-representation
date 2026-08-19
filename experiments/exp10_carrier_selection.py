"""Experiment 10: can the criterion pick the carrier without running the task?

The law says the right carrier is the one that minimises the relative delay
occupancy. That quantity costs one FFT and one sort per candidate, while the
task it is meant to predict costs a full reconstruction. This experiment scores
the rule post hoc on every case already produced by experiments 3, 5 and 6.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
from collections import Counter
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
CANDIDATES = ("raw", "eikonal", "straight", "data_pick")


def load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def score(rows: list[dict], task_key: str, level: int = 90) -> dict:
    """Compare the occupancy-selected carrier against the task-optimal one."""

    correct = 0
    regret, oracle_gain, chosen_gain = [], [], []
    picks = Counter()
    usable = 0
    for row in rows:
        available = [
            name
            for name in CANDIDATES
            if f"{name}_occupancy_{level}" in row and task_key.format(name=name) in row
        ]
        if len(available) < 2:
            continue
        usable += 1
        occupancies = {name: row[f"{name}_occupancy_{level}"] for name in available}
        errors = {name: row[task_key.format(name=name)] for name in available}
        selected = min(occupancies, key=occupancies.get)
        best = min(errors, key=errors.get)
        picks[selected] += 1
        correct += int(selected == best)
        regret.append(errors[selected] / max(errors[best], 1e-12))
        oracle_gain.append(errors["raw"] / max(errors[best], 1e-12))
        chosen_gain.append(errors["raw"] / max(errors[selected], 1e-12))
    if not usable:
        return {}
    return {
        "n_cases": usable,
        "accuracy": correct / usable,
        "median_regret": float(np.median(regret)),
        "mean_regret": float(np.mean(regret)),
        "mean_oracle_gain": float(np.mean(oracle_gain)),
        "mean_selected_gain": float(np.mean(chosen_gain)),
        "fraction_of_oracle_gain_captured": float(
            np.mean(np.log(chosen_gain)) / np.mean(np.log(oracle_gain))
        )
        if np.mean(np.log(oracle_gain)) > 0
        else float("nan"),
        "picks": dict(picks),
    }


def main() -> None:
    report = {}

    regimes = load("exp03_regime_phase_diagram.json")
    if regimes:
        report["rank_selection_fdtd"] = score(regimes["rows"], "{name}_rank_90")

    for label, name in (
        ("task_selection_fdtd", "exp05_task_gain_vs_regime.json"),
        ("task_selection_public", "exp06_public_data_tasks.json"),
    ):
        payload = load(name)
        if not payload:
            continue
        for fraction in ("p1", "p2", "p5", "p10"):
            key = "interp_{name}_" + fraction + "_complex_nrmse"
            result = score(payload["rows"], key)
            if result:
                report[f"{label}_{fraction}"] = result

    (RESULTS / "exp10_carrier_selection.json").write_text(json.dumps(report, indent=2))
    for key, value in report.items():
        if not value:
            continue
        print(
            f"{key:32s} n={value['n_cases']:4d} accuracy={value['accuracy']:.2f} "
            f"median regret={value['median_regret']:.3f} "
            f"selected gain={value['mean_selected_gain']:.2f} "
            f"(oracle {value['mean_oracle_gain']:.2f}) picks={value['picks']}"
        )


if __name__ == "__main__":
    main()
