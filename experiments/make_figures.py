"""Render every figure in the report from the JSON result files."""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"

PALETTE = {
    "raw": "#B4413C",
    "eikonal": "#1F6FB2",
    "straight": "#E0A030",
    "data_pick": "#4C9A6A",
    "grid": "#D8DCE0",
}
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 9,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def figure_rank_law() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    panels = []

    synthetic = load("exp01_rank_law.json")
    if synthetic:
        rows = [r for r in synthetic["rows"] if r["measured_raw_rank_99"] < 0.4 * r["grid"]]
        x = np.concatenate(
            [
                [r["bandwidth"] * r["raw_occupancy_99"] for r in rows],
                [r["bandwidth"] * r["demod_occupancy_99"] for r in rows],
            ]
        )
        y = np.concatenate(
            [
                [r["measured_raw_rank_99"] for r in rows],
                [r["measured_demodulated_rank_99"] for r in rows],
            ]
        )
        panels.append(("Synthetic multipath\n(420 configurations)", x, y))

    regimes = load("exp03_regime_phase_diagram.json")
    if regimes:
        rows = regimes["rows"]
        x, y = [], []
        for carrier in ("raw", "eikonal", "straight", "data_pick"):
            x += [r["bandwidth"] * r[f"{carrier}_occupancy_99"] for r in rows]
            y += [r[f"{carrier}_rank_99"] for r in rows]
        panels.append(("FDTD regime sweep\n(12 regimes, 4 bands)", np.array(x), np.array(y)))

    maze = load("exp02_well_acoustic_maze.json")
    if maze:
        rows = maze["rows"]
        x, y = [], []
        for carrier in ("raw", "eikonal", "straight", "data_pick"):
            x += [r["bandwidth"] * r[f"{carrier}_occupancy_99"] for r in rows]
            y += [r[f"{carrier}_rank_99"] for r in rows]
        panels.append(("The Well acoustic maze\n(24 trajectories, 5 bands)", np.array(x), np.array(y)))

    for axis, (title, x, y) in zip(axes, panels):
        axis.loglog(x, y, "o", ms=3, alpha=0.45, color=PALETTE["eikonal"], mew=0)
        lim = [min(x.min(), y.min()) * 0.7, max(x.max(), y.max()) * 1.4]
        axis.plot(lim, lim, "-", color="#333333", lw=1.0, label="rank = $B\\Lambda$")
        slope = np.sum(x * y) / np.sum(x * x)
        axis.plot(lim, [slope * v for v in lim], "--", color=PALETTE["raw"], lw=1.2,
                  label=f"fit: {slope:.2f}$\\,B\\Lambda$")
        axis.set_xlim(lim)
        axis.set_ylim(lim)
        for locator in (axis.xaxis, axis.yaxis):
            locator.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.minorticks_off()
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("bandwidth $\\times$ delay occupancy  $B\\Lambda$")
        axis.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("measured numerical rank (99% energy)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_rank_law.png")
    plt.close(fig)


def figure_phase_diagram() -> None:
    regimes = load("exp03_regime_phase_diagram.json")
    if not regimes:
        return
    boundaries = list(regimes["absorptions"])
    clutters = list(regimes["scatterers"])
    grid = np.zeros((len(boundaries), len(clutters)))
    counts = np.zeros_like(grid)
    for row in regimes["rows"]:
        if row["f_min"] != 6.0:
            continue
        i = boundaries.index(row["boundary"])
        j = clutters.index(row["clutter"])
        grid[i, j] += row["eikonal_measured_gain_90"]
        counts[i, j] += 1
    grid /= np.maximum(counts, 1)

    fig, axis = plt.subplots(figsize=(4.6, 3.2))
    image = axis.imshow(grid, cmap="magma_r", vmin=1.0, vmax=max(grid.max(), 1.2))
    axis.set_xticks(range(len(clutters)), clutters)
    axis.set_yticks(range(len(boundaries)), boundaries)
    axis.set_xlabel("scatterer density")
    axis.set_ylabel("boundary")
    axis.grid(False)
    for i in range(len(boundaries)):
        for j in range(len(clutters)):
            axis.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                      color="white" if grid[i, j] > 3 else "#222222", fontsize=9)
    axis.set_title("rank gain from first-arrival demodulation", fontsize=9)
    fig.colorbar(image, ax=axis, shrink=0.85, label="rank$_{raw}$ / rank$_{aligned}$")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_phase_diagram.png")
    plt.close(fig)


def figure_task_curves() -> None:
    tasks = load("exp05_task_gain_vs_regime.json")
    if not tasks:
        return
    fractions = tasks["sensor_fractions"]
    groups = {}
    for row in tasks["rows"]:
        groups.setdefault((row["boundary"], row["clutter"]), []).append(row)
    chosen = [("open", "clear"), ("open", "sparse"), ("partial", "clear"), ("closed", "dense")]

    fig, axes = plt.subplots(1, len(chosen), figsize=(3.0 * len(chosen), 3.1), sharey=True)
    for axis, key in zip(axes, chosen):
        rows = groups.get(key, [])
        if not rows:
            continue
        for carrier in ("raw", "eikonal"):
            values = np.array(
                [
                    [r[f"interp_{carrier}_p{int(f * 100)}_complex_nrmse"] for f in fractions]
                    for r in rows
                ]
            )
            axis.plot(
                np.array(fractions) * 100, values.mean(0), "o-", ms=3.5, lw=1.4,
                color=PALETTE[carrier],
                label="raw field" if carrier == "raw" else "carrier-aligned",
            )
            axis.fill_between(
                np.array(fractions) * 100, values.min(0), values.max(0),
                color=PALETTE[carrier], alpha=0.15, lw=0,
            )
        axis.axhline(1.0, color="#666666", lw=0.9, ls=":")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.minorticks_off()
        axis.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
        axis.set_title(f"{key[0]} boundary, {key[1]}", fontsize=9)
        axis.set_xlabel("sensors (% of locations)")
    axes[0].set_ylabel("complex NRMSE on hidden locations")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_sensor_curves.png")
    plt.close(fig)


def figure_task_vs_rank_gain() -> None:
    tasks = load("exp05_task_gain_vs_regime.json")
    if not tasks:
        return
    rows = tasks["rows"]
    fig, axis = plt.subplots(figsize=(4.4, 3.6))
    markers = {"open": "o", "partial": "s", "closed": "^"}
    for boundary, marker in markers.items():
        subset = [r for r in rows if r["boundary"] == boundary]
        if not subset:
            continue
        x = [r["eikonal_measured_gain_90"] for r in subset]
        y = [
            r["interp_raw_p2_complex_nrmse"] / max(r["interp_eikonal_p2_complex_nrmse"], 1e-9)
            for r in subset
        ]
        axis.loglog(x, y, marker, ms=5, alpha=0.75, mew=0, label=f"{boundary} boundary")
    axis.set_xlabel("rank gain  rank$_{raw}$ / rank$_{aligned}$")
    axis.set_ylabel("task gain  NRMSE$_{raw}$ / NRMSE$_{aligned}$\n(2% sensors)")
    axis.axhline(1.0, color="#666666", lw=0.9, ls=":")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_task_vs_rank_gain.png")
    plt.close(fig)


def figure_carrier_tolerance() -> None:
    payload = load("exp07_carrier_error_tolerance.json")
    if not payload:
        return
    rows = payload["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)
    regimes = ["open_clear", "open_sparse", "partial_clear"]
    colors = {"open_clear": PALETTE["eikonal"], "open_sparse": PALETTE["data_pick"],
              "partial_clear": PALETTE["straight"]}
    for axis, kind in zip(axes, ("smooth", "scaling")):
        for regime in regimes:
            subset = [
                r for r in rows
                if r["error_kind"] == kind and r["regime"] == regime and r["band"] == [6.0, 24.0]
            ]
            if not subset:
                continue
            scales = sorted({r["error_scale_in_resolutions"] for r in subset})
            ratio, baseline = [], []
            for scale in scales:
                group = [r for r in subset if r["error_scale_in_resolutions"] == scale]
                ratio.append(np.mean([r["aligned_interp_nrmse"] for r in group]))
                baseline.append(np.mean([r["raw_interp_nrmse"] for r in group]))
            axis.plot(scales, np.array(ratio) / np.array(baseline), "o-", ms=3.5, lw=1.4,
                      color=colors[regime], label=regime.replace("_", ", "))
        axis.axhline(1.0, color="#666666", lw=0.9, ls=":")
        axis.axvline(1.0, color=PALETTE["raw"], lw=0.9, ls="--")
        axis.set_yscale("log")
        axis.set_xlabel("carrier error  $\\delta\\tau$  (in units of $1/B$)")
        axis.set_title(
            {"smooth": "rough spatial error", "scaling": "smooth scaling error"}[kind],
            fontsize=9,
        )
    axes[0].set_ylabel("NRMSE$_{aligned}$ / NRMSE$_{raw}$\n(2% sensors; lower is better)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_carrier_tolerance.png")
    plt.close(fig)


def figure_bandwidth_not_frequency() -> None:
    payload = load("exp08_bandwidth_not_frequency.json")
    if not payload:
        return
    rows = [r for r in payload["rows"] if r["regime"] == "closed_dense"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    centres = sorted({r["centre"] for r in rows})
    bandwidths = sorted({r["nominal_bandwidth"] for r in rows})
    cmap = plt.get_cmap("viridis")

    for index, centre in enumerate(centres):
        subset = sorted(
            [r for r in rows if r["centre"] == centre], key=lambda r: r["bandwidth"]
        )
        axes[0].plot(
            [r["bandwidth"] for r in subset], [r["raw_rank_90"] for r in subset],
            "o-", ms=3.5, lw=1.2, color=cmap(index / max(len(centres) - 1, 1)),
            label=f"$f_c$={centre:.0f}",
        )
    for index, bandwidth in enumerate(bandwidths):
        subset = sorted(
            [r for r in rows if r["nominal_bandwidth"] == bandwidth], key=lambda r: r["centre"]
        )
        axes[1].plot(
            [r["centre"] for r in subset], [r["raw_rank_90"] for r in subset],
            "s-", ms=3.5, lw=1.2, color=cmap(index / max(len(bandwidths) - 1, 1)),
            label=f"$B$={bandwidth:.0f}",
        )
    axes[0].set_xlabel("bandwidth $B$")
    axes[1].set_xlabel("centre frequency $f_c$")
    axes[0].set_ylabel("numerical rank (90% energy)")
    axes[0].set_title("rank grows linearly with bandwidth", fontsize=9)
    axes[1].set_title("rank is flat in centre frequency", fontsize=9)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_bandwidth_not_frequency.png")
    plt.close(fig)


def figure_multicarrier() -> None:
    payload = load("exp09_multicarrier.json")
    if not payload:
        return
    rows = [r for r in payload["rows"] if "budget_rank" in r]
    if not rows:
        return
    regimes = ["open_clear", "partial_clear", "closed_clear", "closed_dense"]
    budgets = payload["budgets"]
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.0 * len(regimes), 3.2), sharey=True)
    for axis, regime in zip(axes, regimes):
        series = {"svd": [], "carrier1": [], "multi": []}
        for budget in budgets:
            subset = [r for r in rows if r["regime"] == regime and r["budget_rank"] == budget]
            if not subset:
                continue
            series["svd"].append(np.mean([r["svd_error"] for r in subset]))
            series["carrier1"].append(np.mean([r["carrier1_svd_error"] for r in subset]))
            best = [
                min(
                    (r[k] for k in r if k.startswith("multi") and k.endswith("_error")),
                    default=np.nan,
                )
                for r in subset
            ]
            series["multi"].append(np.nanmean(best))
        labels = {
            "svd": ("plain rank-$R$ (SVD)", PALETTE["raw"]),
            "carrier1": ("one carrier + rank-$R$", PALETTE["straight"]),
            "multi": ("$M$ carriers, rank $R/M$", PALETTE["eikonal"]),
        }
        for key, values in series.items():
            label, color = labels[key]
            axis.plot(budgets[: len(values)], values, "o-", ms=4, lw=1.5, color=color, label=label)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.minorticks_off()
        axis.set_xticks(budgets, [str(b) for b in budgets])
        axis.set_title(regime.replace("_", ", "), fontsize=9)
        axis.set_xlabel("parameter budget $R$")
    axes[0].set_ylabel("relative approximation error")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_multicarrier.png")
    plt.close(fig)


def figure_estimated_carriers() -> None:
    payload = load("exp12_estimated_carriers.json")
    if not payload:
        return
    rows = payload["rows"]
    order = ["open_clear", "partial_clear", "closed_clear", "closed_sparse", "closed_dense"]
    regimes = [r for r in order if any(row["regime"] == r for row in rows)]
    methods = [
        ("plain_lowrank_error", "plain rank-$R$", PALETTE["raw"]),
        ("single_carrier_error", "one carrier", PALETTE["straight"]),
        ("estimated_multicarrier_error", "estimated carriers (no geometry)", PALETTE["eikonal"]),
        ("oracle_multicarrier_error", "oracle image sources", PALETTE["data_pick"]),
    ]
    fig, (bars, gaps) = plt.subplots(
        1, 2, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [2.1, 1]}
    )
    positions = np.arange(len(regimes))
    width = 0.2
    for index, (key, label, color) in enumerate(methods):
        values = [
            np.mean([r[key] for r in rows if r["regime"] == regime]) for regime in regimes
        ]
        bars.bar(
            positions + (index - 1.5) * width, values, width * 0.9,
            color=color, label=label, linewidth=0,
        )
    bars.set_xticks(positions, [r.replace("_", "\n") for r in regimes], fontsize=8.5)
    bars.set_ylabel("relative error at $R=24$")
    bars.legend(fontsize=8, ncol=2)
    bars.set_title("equal parameter budget", fontsize=9)

    captured = []
    for regime in regimes:
        subset = [r for r in rows if r["regime"] == regime]
        plain = np.mean([r["plain_lowrank_error"] for r in subset])
        estimated = np.mean([r["estimated_multicarrier_error"] for r in subset])
        oracle = np.mean([r["oracle_multicarrier_error"] for r in subset])
        denominator = np.log(plain / oracle)
        captured.append(
            100.0 * np.log(plain / estimated) / denominator if denominator > 1e-9 else np.nan
        )
    gaps.barh(positions, captured, 0.55, color=PALETTE["eikonal"], linewidth=0)
    gaps.axvline(100, color="#555555", lw=1.0, ls="--")
    gaps.set_yticks(positions, [r.replace("_", ", ") for r in regimes], fontsize=8.5)
    gaps.invert_yaxis()
    gaps.set_xlabel("% of the oracle gain captured")
    gaps.set_title("no geometry required", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_estimated_carriers.png")
    plt.close(fig)


def figure_learned_baselines() -> None:
    panels = []
    for fraction in (2, 10):
        payload = load(f"exp14_learned_baselines_p{fraction}.json")
        if payload:
            panels.append((fraction, payload["rows"]))
    if not panels:
        return
    order = ["open_clear", "open_sparse", "partial_clear", "closed_dense"]
    series = [
        ("best_network_raw", "network, raw field", PALETTE["raw"], "o-"),
        ("best_network_aligned", "network, aligned", PALETTE["straight"], "s-"),
        ("interp_raw", "interpolation, raw", "#9AA5B1", "o--"),
        ("interp_aligned", "interpolation, aligned", PALETTE["eikonal"], "s--"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (fraction, rows) in zip(axes, panels):
        regimes = [r for r in order if any(row["regime"] == r for row in rows)]
        positions = np.arange(len(regimes))
        for key, label, color, style in series:
            values = [
                np.mean([r[key] for r in rows if r["regime"] == regime]) for regime in regimes
            ]
            axis.plot(positions, values, style, color=color, ms=5, lw=1.5, label=label)
        axis.axhline(1.0, color="#666666", lw=0.9, ls=":")
        axis.set_yscale("log")
        axis.set_xticks(positions, [r.replace("_", ",\n") for r in regimes], fontsize=8.5)
        axis.set_title(f"{fraction}% of locations observed", fontsize=9)
    axes[0].set_ylabel("complex NRMSE on hidden locations")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_learned_baselines.png")
    plt.close(fig)


def figure_shifted_pod() -> None:
    payload = load("exp16_shifted_pod.json")
    if not payload:
        return
    rows = payload["rows"]
    order = [
        "open_clear", "open_sparse", "partial_clear",
        "closed_dense", "acoustic_inclusions", "well_maze",
    ]
    regimes = [r for r in order if any(row["regime"] == r for row in rows)]
    budgets = payload["budgets"]
    series = [
        ("plain_pod", "plain POD", PALETTE["raw"], "o-"),
        ("best_shifted_pod", "shifted POD (best of $K$=1,2,3)", PALETTE["straight"], "^-"),
        ("carrier_pod", "carrier alignment (ours)", PALETTE["eikonal"], "s-"),
    ]
    columns = 3
    fig, axes = plt.subplots(
        2, columns, figsize=(3.4 * columns, 6.2), sharex=True, sharey=True
    )
    for axis, regime in zip(axes.ravel(), regimes):
        for key, label, color, style in series:
            values = [
                np.mean([r[key] for r in rows if r["regime"] == regime and r["budget"] == b])
                for b in budgets
            ]
            axis.plot(budgets, values, style, color=color, ms=5, lw=1.5, label=label)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.minorticks_off()
        axis.set_xticks(budgets, [str(b) for b in budgets])
        axis.set_title(regime.replace("_", ", "), fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel("parameter budget $R$")
    for axis in axes[:, 0]:
        axis.set_ylabel("relative reconstruction error")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig11_shifted_pod.png")
    plt.close(fig)


def figure_learned_representation() -> None:
    learned = load("exp17_learned_representation.json")
    ablation = load("exp18_objective_and_dispersion.json")
    if not learned and not ablation:
        return
    fig, axes = plt.subplots(
        1, 2, figsize=(10.6, 3.7), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    if learned:
        rows = learned["rows"]
        order = ["open_clear", "open_sparse", "partial_clear", "closed_dense"]
        regimes = [r for r in order if any(row["regime"] == r for row in rows)]
        bars = [
            ("raw_sensor_nrmse", "no carrier", "#9AA5B1"),
            ("corrupted_sensor_nrmse", "physics, but wrong", PALETTE["raw"]),
            ("learned_repair_sensor_nrmse", "wrong physics + learning", PALETTE["data_pick"]),
            ("learned_scratch_sensor_nrmse", "learned, no physics", PALETTE["straight"]),
            ("eikonal_sensor_nrmse", "physics, correct", PALETTE["eikonal"]),
        ]
        positions = np.arange(len(regimes))
        width = 0.16
        for index, (key, label, color) in enumerate(bars):
            values = [
                np.mean([r[key] for r in rows if r["regime"] == regime]) for regime in regimes
            ]
            axes[0].bar(
                positions + (index - 2) * width, values, width * 0.9,
                color=color, label=label, linewidth=0,
            )
        axes[0].axhline(1.0, color="#666666", lw=0.9, ls=":")
        axes[0].set_xticks(positions, [r.replace("_", ",\n") for r in regimes], fontsize=8.5)
        axes[0].set_ylabel("complex NRMSE, 2% sensors")
        axes[0].set_title("learning recovers what wrong physics loses", fontsize=9)
        axes[0].legend(fontsize=7.5, ncol=2)

    if ablation:
        rows = ablation["rows"]
        entries = [
            ("eikonal_r4", "physics\n(eikonal)", PALETTE["eikonal"]),
            ("learned_nuclear_disp_r4", "learned\nnuclear obj.", PALETTE["raw"]),
            ("learned_tail_tau_r4", "learned\ntail obj.", PALETTE["straight"]),
            ("learned_tail_disp_r4", "learned\ntail + dispersive", PALETTE["data_pick"]),
        ]
        means = [np.mean([r[key] for r in rows]) for key, _, _ in entries]
        errors = [np.std([r[key] for r in rows]) for key, _, _ in entries]
        axes[1].bar(
            np.arange(len(entries)), means, 0.6, yerr=errors, capsize=3,
            color=[color for _, _, color in entries], linewidth=0,
        )
        axes[1].set_xticks(
            np.arange(len(entries)), [label for _, label, _ in entries], fontsize=8
        )
        axes[1].set_ylabel("rank-4 error, Helmholtz staircase")
        axes[1].set_title("where the physics model is wrong\n(dispersive trapped modes)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig12_learned_representation.png")
    plt.close(fig)


def figure_identifiability() -> None:
    payload = load("exp21_identifiability_law.json")
    if not payload:
        return
    rows = payload["rows"]
    strides = payload["strides"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))

    for coordinate, color, label in (
        ("raw", PALETTE["raw"], "raw field"),
        ("aligned", PALETTE["eikonal"], "carrier-aligned"),
    ):
        bounds, errors = [], []
        for row in rows:
            if row["coordinate"] != coordinate:
                continue
            for bound, error in zip(row["bounds"], row["measured_errors"]):
                if bound > 1e-6 and np.isfinite(error):
                    bounds.append(bound)
                    errors.append(error)
        axes[0].loglog(bounds, errors, "o", ms=3.2, alpha=0.4, mew=0, color=color, label=label)
    limits = [8e-3, 2.0]
    axes[0].plot(limits, limits, "-", color="#333333", lw=1.1, label="error = bound")
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_xlabel("identifiability bound  $\\sqrt{E_{>k_N}}$  (one FFT, no model)")
    axes[0].set_ylabel("measured reconstruction error")
    axes[0].set_title("the bound holds and is nearly tight", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper left")

    chosen = ["fdtd_open_clear", "fdtd_open_sparse", "fdtd_closed_dense"]
    styles = {"fdtd_open_clear": "-", "fdtd_open_sparse": "--", "fdtd_closed_dense": ":"}
    for dataset in chosen:
        for coordinate, color in (("raw", PALETTE["raw"]), ("aligned", PALETTE["eikonal"])):
            subset = [
                r for r in rows if r["dataset"] == dataset and r["coordinate"] == coordinate
            ]
            if not subset:
                continue
            values = np.nanmean([r["measured_errors"] for r in subset], axis=0)
            axes[1].plot(
                strides, values, styles[dataset], color=color, lw=1.6,
                marker="o", ms=3.5,
                label=f"{dataset.replace('fdtd_', '').replace('_', ', ')} · {coordinate}",
            )
    axes[1].axhline(1.0, color="#666666", lw=0.9, ls=":")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].minorticks_off()
    axes[1].set_xticks(strides, [str(s) for s in strides])
    axes[1].set_xlabel("sensor array spacing (pixels)")
    axes[1].set_ylabel("reconstruction error")
    axes[1].set_title("the carrier moves the whole curve", fontsize=9)
    axes[1].legend(fontsize=6.6, ncol=1, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_identifiability.png")
    plt.close(fig)


def figure_not_capacity() -> None:
    payload = load("exp22_no_estimator_beats_it.json")
    if not payload:
        return
    rows = payload["rows"]
    capacity = payload["summary"]["capacity_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7))

    order = sorted(capacity.items(), key=lambda kv: kv[1]["parameters"])
    fourier = [(v["parameters"], v["mean_test"], v["mean_train"])
               for k, v in order if k.startswith("fourier")]
    params = [p for p, _, _ in fourier]
    axes[0].semilogx(params, [t for _, t, _ in fourier], "o-", ms=6, lw=1.8,
                     color=PALETTE["raw"], label="test error")
    axes[0].semilogx(params, [max(tr, 1e-8) for _, _, tr in fourier], "s--", ms=5, lw=1.5,
                     color=PALETTE["eikonal"], label="training error")
    for key, label, color in (
        ("linear", "linear interp. (0 parameters)", PALETTE["data_pick"]),
        ("nearest", "nearest neighbour", "#9AA5B1"),
    ):
        value = float(np.mean([r[f"{key}_test"] for r in rows]))
        axes[0].axhline(value, color=color, lw=1.5, ls=":", label=f"{label}: {value:.3f}")
    axes[0].set_yscale("log")
    axes[0].set_ylim(3e-8, 6.0)
    axes[0].set_xlabel("network parameters")
    axes[0].set_ylabel("complex NRMSE")
    axes[0].set_title("128$\\times$ more capacity, 3% less error", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="center left", framealpha=0.9, frameon=True)
    axes[0].annotate(
        "training error is at machine precision throughout",
        (params[0], 2e-7), fontsize=7.5, color=PALETTE["eikonal"], va="top",
    )

    labels, bounds, bests = [], [], []
    for coordinate in ("raw", "aligned"):
        subset = [r for r in rows if r["coordinate"] == coordinate]
        labels.append(coordinate)
        bounds.append(float(np.mean([r["bound"] for r in subset])))
        bests.append(float(np.mean([r["best_test"] for r in subset])))
    positions = np.arange(len(labels))
    axes[1].bar(positions - 0.18, bounds, 0.34, color=PALETTE["straight"],
                label="identifiability bound", linewidth=0)
    axes[1].bar(positions + 0.18, bests, 0.34, color=PALETTE["eikonal"],
                label="best of six estimators", linewidth=0)
    axes[1].set_xticks(positions, ["raw field", "carrier-aligned"])
    axes[1].set_ylabel("complex NRMSE")
    axes[1].set_title("changing coordinates moves both", fontsize=9)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig14_not_capacity.png")
    plt.close(fig)


def figure_learned_bound() -> None:
    payload = load("exp24_learned_identifiability.json")
    if not payload:
        return
    rows = payload["rows"]
    regimes = ["open_clear", "open_sparse", "partial_clear", "closed_dense"]
    entries = [
        ("none", "no carrier", "#9AA5B1"),
        ("eikonal_corrupted", "physics, corrupted", PALETTE["raw"]),
        ("eikonal", "physics (eikonal)", PALETTE["straight"]),
        ("learned_scratch", "learned, no physics", PALETTE["eikonal"]),
        ("learned_repair", "corrupted + learning", PALETTE["data_pick"]),
    ]
    fig, axis = plt.subplots(figsize=(7.4, 3.7))
    positions = np.arange(len(regimes))
    width = 0.16
    for index, (key, label, color) in enumerate(entries):
        values = [
            np.mean([
                r["bound_s6"] for r in rows
                if r["regime"] == regime and r["coordinate"] == key
            ])
            for regime in regimes
        ]
        axis.bar(positions + (index - 2) * width, values, width * 0.92,
                 color=color, label=label, linewidth=0)
    axis.set_xticks(positions, [r.replace("_", ",\n") for r in regimes], fontsize=8.5)
    axis.set_ylabel("identifiability bound at array spacing 6")
    axis.set_title(
        "the missing information can be learned as well as supplied", fontsize=9
    )
    axis.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig15_learned_bound.png")
    plt.close(fig)


def figure_amortised() -> None:
    files = [
        ("exp26_amortised_operator_open_128_240.json", "random scatterers\n→ unseen, same family"),
        ("exp26_amortised_operator_single_open_128_512.json", "single inclusion\n→ unseen, same family"),
        ("exp26_amortised_cross.json", "random scatterers\n→ single inclusion\n(across families)"),
    ]
    payloads = [(load(name), label) for name, label in files]
    payloads = [(p, l) for p, l in payloads if p]
    if not payloads:
        return
    entries = [
        ("raw_bound", "no carrier", "#9AA5B1"),
        ("eikonal_bound", "physics (a solver per medium)", PALETTE["straight"]),
        ("amortised_scratch_bound", "amortised, no physics (one forward pass)", PALETTE["eikonal"]),
        ("per_case_fitted_bound", "per-field fitting (ceiling)", PALETTE["data_pick"]),
    ]
    fig, axis = plt.subplots(figsize=(7.8, 3.8))
    positions = np.arange(len(payloads))
    width = 0.2
    for index, (key, label, color) in enumerate(entries):
        values = [p[key] for p, _ in payloads]
        errors = [p.get(key.replace("_bound", "_bound_seed_std"), 0.0) for p, _ in payloads]
        axis.bar(positions + (index - 1.5) * width, values, width * 0.9, yerr=errors,
                 capsize=2.5, color=color, label=label, linewidth=0)
    axis.set_xticks(positions, [label for _, label in payloads], fontsize=8.5)
    axis.set_ylabel("identifiability bound on unseen media")
    axis.set_title("the coordinates amortise, and transfer across medium families", fontsize=9)
    axis.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig16_amortised.png")
    plt.close(fig)


def figure_fields() -> None:
    from wave_lr.fdtd import MediumSpec
    from wave_lr.fields import fdtd_case, load_well_acoustic
    from wave_lr.spectra import carrier, to_spectrum

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6))
    specs = [
        ("open, clear", fdtd_case(MediumSpec(name="open_clear", absorption=40.0))),
        (
            "closed, dense",
            fdtd_case(MediumSpec(name="closed_dense", absorption=0.0, scatterer_fraction=0.22)),
        ),
    ]
    maze = load_well_acoustic("test", limit=1)[0]
    specs.append(("The Well maze", maze))

    for column, (title, case) in enumerate(specs):
        band = (3.0, 13.0) if case.dataset == "well_acoustic_maze" else (6.0, 24.0)
        spectrum = to_spectrum(case.traces, case.dt, *band)
        index = spectrum.values.shape[1] // 2
        raw = spectrum.values[:, index]
        aligned = raw * np.conj(carrier(spectrum.frequencies, case.travel_time))[:, index]
        span = float(np.percentile(np.abs(raw), 99.0))

        def rank_of(values):
            values = values.reshape(-1, 1) if values.ndim == 1 else values
        for row, (values, label) in enumerate(((raw, "raw"), (aligned, "aligned"))):
            axis = axes[row, column]
            scatter = axis.scatter(
                case.coords[:, 1], -case.coords[:, 0], c=np.real(values), s=1.4,
                cmap="RdBu_r", vmin=-span, vmax=span, linewidths=0,
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            if row == 0:
                axis.set_title(f"{title}\nRe $u(x, f)$", fontsize=9)
            else:
                axis.set_title("Re $u\\,e^{+2\\pi i f\\tau(x)}$", fontsize=9)
            axis.set_xlabel(label, fontsize=8)
        fig.colorbar(scatter, ax=axes[:, column], shrink=0.7)
    fig.savefig(FIGURES / "fig6_fields.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_rank_law()
    figure_phase_diagram()
    figure_task_curves()
    figure_task_vs_rank_gain()
    figure_carrier_tolerance()
    figure_bandwidth_not_frequency()
    figure_multicarrier()
    figure_estimated_carriers()
    figure_learned_baselines()
    figure_shifted_pod()
    figure_learned_representation()
    figure_identifiability()
    figure_not_capacity()
    figure_learned_bound()
    figure_amortised()
    figure_fields()
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
