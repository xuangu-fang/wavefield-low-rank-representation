"""Render every figure in the report from the JSON result files."""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
        span = np.abs(raw).max()
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
        fig.colorbar(scatter, ax=axes[:, column], shrink=0.7)
    fig.savefig(FIGURES / "fig6_fields.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_rank_law()
    figure_phase_diagram()
    figure_task_curves()
    figure_task_vs_rank_gain()
    figure_fields()
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
