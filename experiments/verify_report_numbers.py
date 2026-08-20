"""Re-derive every headline number from the result files.

A report assembled over many runs can silently keep a number whose experiment
has since been re-run. This prints the current value of each quoted figure so
the text can be checked against it.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  isort:skip

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def line(label, value):
    print(f"  {label:52s} {value}")


def main() -> None:
    print("=== 0.1 可辨识性界 (exp21) ===")
    d = load("exp21_identifiability_law.json")
    if d:
        f = d["fits"]
        line("样本对数", f["n_bound_pairs"])
        line("打破比例", f"{f['bound_violation_rate']:.2%}")
        line("error_vs_bound R2", f"{f['error_vs_bound']['r2']:.3f}")
        by = defaultdict(lambda: [0, 0])
        for r in d["rows"]:
            fam = "fdtd" if r["dataset"].startswith("fdtd") else r["dataset"]
            for b, e in zip(r["bounds"], r["measured_errors"]):
                if b > 1e-6 and np.isfinite(e):
                    by[fam][1] += 1
                    by[fam][0] += int(e < 0.9 * b)
        for k, (v, n) in sorted(by.items()):
            line(f"  {k}", f"{v}/{n} = {v/n:.1%}")
        for tag in ("t50", "t30"):
            key = f"measured_vs_predicted_{tag}"
            if key in f:
                line(f"设计工具 {tag}", f"R2={f[key]['r2']:.3f} slope={f[key]['slope']:.3f} n={f.get('n_used_'+tag)}")

    print("=== 0.2 不是容量 (exp22) ===")
    d = load("exp22_no_estimator_beats_it.json")
    if d:
        s, rows = d["summary"], d["rows"]
        line("设置数 / 打破界", f"{s['n_settings']} / {s['bound_beaten']} ({s['bound_beaten_rate']:.1%})")
        for k, v in s["capacity_sweep"].items():
            line(f"  {k}", f"params={v['parameters']:>8d} train={v['mean_train']:.2e} test={v['mean_test']:.3f}")
        for k in ("linear", "nearest"):
            line(f"  {k}", f"test={np.mean([r[k+'_test'] for r in rows]):.3f}")
        for c in ("raw", "aligned"):
            sub = [r for r in rows if r["coordinate"] == c]
            line(f"  {c}", f"bound={np.mean([r['bound'] for r in sub]):.3f} best={np.mean([r['best_test'] for r in sub]):.3f}")

    print("=== 0.3 把界当损失 (exp24) ===")
    d = load("exp24_learned_identifiability.json")
    if d:
        for k, v in d["summary"].items():
            line(f"  {k}", f"bound@6={v['mean_bound_s6']:.3f} err@6={v['mean_error_s6']:.3f}")

    print("=== 0.4 公开频域 (exp23) ===")
    d = load("exp23_public_identifiability.json")
    if d:
        for k, v in d["fits"].items():
            if isinstance(v, dict):
                line(f"  {k}", f"slope={v['slope']:.3f} R2={v['r2']:.3f}")
            else:
                line(f"  {k}", f"{v}")

    print("=== 留出验证 (exp25) ===")
    d = load("exp25_heldout_validation.json")
    if d:
        for k, v in d["summary"].items():
            line(f"  {k}", json.dumps(v) if isinstance(v, dict) else v)


if __name__ == "__main__":
    main()
