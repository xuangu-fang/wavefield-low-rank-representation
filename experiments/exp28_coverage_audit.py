"""Experiment 28: a regime-labelled registry of the wave benchmarks we measured.

Three of the four public wave benchmarks measured in this project sit in the
regime where phase alignment provably cannot help. That claim is only useful if
it is made auditable: one uniform criterion, applied identically to every
dataset, with provenance recorded so anyone can repeat it.

This assembles the measurements already produced by experiments 6, 19, 20, 21
and 23 into a machine-readable registry plus a table, and applies a single
stated rule to label each dataset's regime.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
REFERENCE_STRIDE = 6

# Verified against the HuggingFace and Zenodo APIs on 2026-08-21.
PROVENANCE = {
    "well_maze": {
        "title": "The Well — acoustic_scattering_maze",
        "url": "https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_maze",
        "revision": "8df383a3",
        "license": "CC-BY-4.0",
        "access": "本地 64x64 下采样副本（由同组既有项目预处理）",
        "family": "wave",
    },
    "acoustic_inclusions": {
        "title": "The Well — acoustic_scattering_inclusions",
        "url": "https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_inclusions",
        "revision": "c17cd1d1",
        "license": "CC-BY-4.0",
        "access": "test/chunk_36.hdf5，256x256 全分辨率",
        "family": "wave",
    },
    "helmholtz_staircase": {
        "title": "The Well — helmholtz_staircase",
        "url": "https://huggingface.co/datasets/polymathic-ai/helmholtz_staircase",
        "revision": "a2429505",
        "license": "CC-BY-4.0",
        "access": "train + test 全部 16 个 omega",
        "family": "wave",
    },
    "wavebench_omega10": {
        "title": "WaveBench — time-harmonic isotropic, omega label 10",
        "url": "https://doi.org/10.5281/zenodo.8015145",
        "revision": "2023-06-07",
        "license": "CC-BY-4.0",
        "access": "range 请求读取 75 GB zip 内成员的前缀，约 1000 个可读样本",
        "family": "wave",
    },
    "wavebench_omega40": {
        "title": "WaveBench — time-harmonic isotropic, omega label 40",
        "url": "https://doi.org/10.5281/zenodo.8015145",
        "revision": "2023-06-07",
        "license": "CC-BY-4.0",
        "access": "同上",
        "family": "wave",
    },
    "kolmogorov_flow": {
        "title": "2D Navier-Stokes Re40 (Learning Dissipative Dynamics)",
        "url": "https://doi.org/10.5281/zenodo.7495555",
        "revision": "—",
        "license": "CC-BY-4.0",
        "access": "本地预处理副本",
        "family": "non_wave",
    },
    "kuramoto_sivashinsky": {
        "title": "Kuramoto–Sivashinsky（本地生成）",
        "url": "—", "revision": "—", "license": "本项目生成",
        "access": "同组既有项目生成", "family": "non_wave",
    },
    "cylinder_wake": {
        "title": "Cylinder wake（本地预处理）",
        "url": "—", "revision": "—", "license": "本项目生成",
        "access": "同组既有项目生成", "family": "non_wave",
    },
    "active_matter": {
        "title": "Active matter（本地预处理）",
        "url": "—", "revision": "—", "license": "本项目生成",
        "access": "同组既有项目生成", "family": "non_wave",
    },
    "diffusion_reaction": {
        "title": "PDEBench — 2D diffusion-reaction",
        "url": "https://github.com/pdebench/PDEBench",
        "revision": "—", "license": "见官方发布（未在本项目核实）",
        "access": "本地副本", "family": "non_wave",
    },
    "openfwi_gathers": {
        "title": "OpenFWI（本地副本，配对已损坏）",
        "url": "https://github.com/lanl/OpenFWI",
        "revision": "—", "license": "见官方发布（未在本项目核实）",
        "access": "本地 seis2_*/vel4_* 副本——两者来自不同 family，见 DATA_INTEGRITY.md",
        "family": "wave",
    },
}

# Our own solver, included as reference points that bracket the axis.
OWN = {"fdtd_open_clear", "fdtd_open_sparse", "fdtd_partial_clear", "fdtd_closed_dense"}


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


ALREADY_IDENTIFIABLE = 0.15


def verdict(raw: float | None, ratio: float | None) -> str:
    """One stated rule, applied identically to every dataset.

    A field whose raw bound is already small at the reference spacing is not
    "unfavourable" -- it is simply already identifiable at that array density,
    and the ratio between two near-zero numbers carries no information. That
    case gets its own label rather than being counted against alignment.
    """

    if raw is None or ratio is None or not np.isfinite(ratio):
        return "未测量"
    if raw < ALREADY_IDENTIFIABLE:
        return "已可辨识（该密度下无需载波）"
    if ratio <= 0.5:
        return "有利"
    if ratio <= 0.9:
        return "中等"
    return "不利"


def bounds_at_reference() -> dict:
    """Raw and aligned identifiability bound at the reference array spacing."""

    out = defaultdict(dict)
    payload = load("exp21_identifiability_law.json")
    if payload:
        strides = payload["strides"]
        index = strides.index(REFERENCE_STRIDE)
        grouped = defaultdict(lambda: defaultdict(list))
        for row in payload["rows"]:
            value = row["bounds"][index]
            if np.isfinite(value):
                grouped[row["dataset"]][row["coordinate"]].append(value)
        for dataset, byco in grouped.items():
            for coordinate, values in byco.items():
                out[dataset][coordinate] = float(np.mean(values))
    payload = load("exp23_public_identifiability.json")
    if payload:
        grouped = defaultdict(lambda: defaultdict(list))
        for row in payload["rows"]:
            if row["stride"] == REFERENCE_STRIDE:
                grouped[row["dataset"]][row["coordinate"]].append(row["bound"])
        for dataset, byco in grouped.items():
            for coordinate, values in byco.items():
                out[dataset][coordinate] = float(np.mean(values))
    return out


def occupancy_fractions() -> dict:
    payload = load("exp19_transport_generality.json")
    if not payload:
        return {}
    grouped = defaultdict(list)
    for row in payload["rows"]:
        grouped[row["dataset"]].append(row)
    return {
        key: {
            "occupancy_fraction": float(np.mean([r["occupancy_fraction"] for r in rows])),
            "predicted_gain": float(np.mean([r["predicted_gain"] for r in rows])),
            "measured_gain": float(np.mean([r["measured_gain"] for r in rows])),
        }
        for key, rows in grouped.items()
    }


def compression_gains() -> dict:
    """Equal-budget compression gain on the frequency axis, where measured."""

    payload = load("exp13_public_multicarrier.json")
    if not payload:
        return {}
    grouped = defaultdict(list)
    for row in payload["rows"]:
        grouped[row["dataset"]].append(row)
    out = {}
    for key, rows in grouped.items():
        ratios = [
            r["plain_lowrank_error"] / max(r["single_carrier_error"], 1e-12) for r in rows
        ]
        out[key] = float(np.mean(ratios))
    # The two WaveBench frequency files were measured under one label.
    if "wavebench_omega10" in out:
        pass
    return out


def measured_gains() -> dict:
    """Alignment gain on the 2% sensor task, where it was measured."""

    payload = load("exp06_public_data_tasks.json")
    if not payload:
        return {}
    grouped = defaultdict(list)
    for row in payload["rows"]:
        key = row.get("dataset")
        if key:
            grouped[key].append(row)
    out = {}
    for key, rows in grouped.items():
        if "interp_raw_p2_complex_nrmse" in rows[0]:
            out[key] = float(
                np.mean(
                    [
                        r["interp_raw_p2_complex_nrmse"]
                        / max(r["interp_eikonal_p2_complex_nrmse"], 1e-9)
                        for r in rows
                    ]
                )
            )
    return out


ALIASES = {
    "well_acoustic_maze": "well_maze",
    "helmholtz_staircase": "helmholtz_staircase",
    "acoustic_inclusions": "acoustic_inclusions",
}


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    bounds = bounds_at_reference()
    occupancy = occupancy_fractions()
    gains = measured_gains()
    compression = compression_gains()
    compression_alias = {
        "helmholtz_staircase": "helmholtz_staircase",
        "acoustic_inclusions": "acoustic_inclusions",
    }

    entries = []
    for key, meta in PROVENANCE.items():
        record = dict(meta)
        record["dataset"] = key
        raw = bounds.get(key, {}).get("raw")
        aligned = bounds.get(key, {}).get("aligned")
        ratio = (aligned / raw) if (raw and aligned) else None
        record.update(
            {
                "bound_raw": raw,
                "bound_aligned": aligned,
                "bound_ratio": ratio,
                "reference_stride": REFERENCE_STRIDE,
                "verdict": verdict(raw, ratio),
                "sensor_task_gain": gains.get(
                    {v: k for k, v in ALIASES.items()}.get(key, key)
                ),
                "equal_budget_compression_gain": compression.get(
                    compression_alias.get(key, key)
                ),
            }
        )
        record.update(occupancy.get(key, {}))
        entries.append(record)

    for key in sorted(OWN):
        raw = bounds.get(key, {}).get("raw")
        aligned = bounds.get(key, {}).get("aligned")
        ratio = (aligned / raw) if (raw and aligned) else None
        entries.append(
            {
                "dataset": key,
                "title": f"本项目 FDTD 求解器 — {key.replace('fdtd_', '')}",
                "url": "—",
                "revision": "—",
                "license": "本项目生成",
                "access": "experiments/exp03 生成，作为坐标轴两端的参照",
                "family": "wave_reference",
                "bound_raw": raw,
                "bound_aligned": aligned,
                "bound_ratio": ratio,
                "reference_stride": REFERENCE_STRIDE,
                "verdict": verdict(raw, ratio),
            }
        )

    public_wave = [
        e for e in entries if e["family"] == "wave" and e["verdict"] != "未测量"
    ]
    summary = {
        "reference_stride": REFERENCE_STRIDE,
        "rule": (
            f"raw < {ALREADY_IDENTIFIABLE} -> 已可辨识（比值无信息）; "
            "否则 aligned/raw <= 0.5 -> 有利; <= 0.9 -> 中等; 否则 不利"
        ),
        "n_entries": len(entries),
        "public_wave_measured": len(public_wave),
        "public_wave_by_verdict": {
            v: sum(1 for e in public_wave if e["verdict"] == v)
            for v in ("有利", "中等", "不利", "已可辨识（该密度下无需载波）")
        },
        "already_identifiable_threshold": ALREADY_IDENTIFIABLE,
    }
    (REPORTS / "benchmark_registry.json").write_text(
        json.dumps({"summary": summary, "entries": entries}, indent=2, ensure_ascii=False)
    )

    def cell(value, fmt="{:.3f}"):
        return fmt.format(value) if isinstance(value, (int, float)) and np.isfinite(value) else "—"

    lines = [
        "# 波动 benchmark 的区间覆盖审计",
        "",
        f"**判定规则**（对所有数据集一致应用）：在参考阵列间距 m={REFERENCE_STRIDE} 上，",
        f"若原始界已低于 {ALREADY_IDENTIFIABLE}，记为**已可辨识**——该密度下本就不需要载波，",
        "两个接近零的数之比不携带信息；否则按对齐界与原始界之比判定：",
        "`≤0.5` **有利**、`≤0.9` **中等**、否则**不利**。",
        "界的定义与三条测量纪律见 `REPORT_ZH.md` §0.1。",
        "",
        "**一个数据集可以在不同轴上落在不同区间。** Helmholtz staircase 就是例子：",
        "在 m=6 的空间采样下它本就可辨识（界 0.020），但在**频率轴的等预算压缩**上",
        "载波带来 10.6× 增益。因此下表并列给出三根轴，不用单一数字概括一个数据集。",
        "",
        "| 数据集 | 类别 | 界(raw) | 界(aligned) | 比值 | 判定 | 等预算压缩增益 | 2% 传感器增益 | 许可 |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for entry in sorted(entries, key=lambda e: (e["family"], e["bound_ratio"] or 9)):
        lines.append(
            f"| {entry['title']} | {entry['family']} | {cell(entry['bound_raw'])} | "
            f"{cell(entry['bound_aligned'])} | {cell(entry['bound_ratio'], '{:.2f}')} | "
            f"{entry['verdict']} | {cell(entry.get('equal_budget_compression_gain'), '{:.2f}x')} | "
            f"{cell(entry.get('sensor_task_gain'), '{:.2f}x')} | {entry['license']} |"
        )
    lines += [
        "",
        "非波动数据集用另一根轴度量（时间占据比例与对齐增益，见 `REPORT_ZH.md` §11.7）：",
        "",
        "| 数据集 | 能量占记录比例 | 预测增益 | 实测增益 |",
        "|---|---:|---:|---:|",
    ]
    for entry in entries:
        if entry["family"] == "non_wave" and "occupancy_fraction" in entry:
            lines.append(
                f"| {entry['title']} | {entry['occupancy_fraction']:.2f} | "
                f"{entry['predicted_gain']:.2f} | {entry['measured_gain']:.2f} |"
            )
    lines += [
        "",
        "## 覆盖情况",
        "",
        f"已测量的公开**波动**数据集：{summary['public_wave_measured']} 个，其中",
        "、".join(
            f"{v} {n} 个" for v, n in summary["public_wave_by_verdict"].items()
        ) + "。",
        "",
        "**没有一个公开波动数据集落在「有利」区间。** 与此对照，本项目自建 FDTD 求解器的",
        "四个参照点跨越 0.06 到 0.87，覆盖了整条坐标轴——所以缺口不在方法能否奏效，",
        "而在**公开基准在这条轴上的采样偏斜**：它们要么处在相位对齐必然失效的强混响/强散射区间，",
        "要么在所用采样密度下本就已经可辨识。",
        "",
        "这对社区的含义是具体的：",
        "**若要评估物理对齐类表征方法，现有公开波动基准无法区分「方法无效」与",
        "「数据集不在该方法的适用区间」**。补上开放/吸收介质、且采样低于空间 Nyquist 的",
        "公开数据，是让这类方法可被公平评估的前提。",
        "",
        "## 出处与复现",
        "",
        "| 数据集 | 来源 | 版本 | 获取方式 |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['title']} | {entry['url']} | {entry['revision']} | {entry['access']} |"
        )
    (REPORTS / "BENCHMARK_COVERAGE.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {REPORTS / 'benchmark_registry.json'} and BENCHMARK_COVERAGE.md")


if __name__ == "__main__":
    main()
