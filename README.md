# Low-Rank Representation Learning for Wave Fields

波动场低秩表征学习：一个长期、representation-first 的研究仓库。

## 一句话目标

> 先显式消除波场中可解释的快速相位传播，再用低秩模型或生成模型学习剩下的慢变、多路径与不确定成分。

本仓库不再承诺“一个张量分解直接解决复杂波场”。它研究的是更基础的问题：复杂复数波场应该用什么坐标和中间表示，才能让后续的 completion、operator learning、跨频预测、反演或生成建模更容易。

## 当前主线（2026-08-19 起）

主攻方向是 **延迟占据（delay occupancy）秩定律**：复波场 `(x,f)` 展开的数值秩由
**带宽 × 到达时间占据测度** 决定，相位解调的收益等于绝对占据与相对占据之比，
且该比值在训练任何模型之前就能从 `c(x)` 与记录道算出。

- 理论与最终表述：`docs/THEORY_DELAY_OCCUPANCY.md`
- 立项时的原始假设（保留未改）：`docs/PROPOSAL_DELAY_SPREAD_ZH.md`
- 实验结果与结论：`docs/REPORT.md`
- 数据完整性发现（含本地 OpenFWI 配对错误）：`docs/DATA_INTEGRITY.md`

## 当前定位

- **长期研究基础设施，不绑定近期投稿。** 后续可以逐步接收新的波动研究和想法。
- **三角恒等式载波是可复用原语，不是最终模型。** 它精确表达单个 traveling harmonic，但多路径、反射、绕射和移动包络需要更丰富表示。
- **旧 Track 2 的失败被完整保留。** 旧方法在“1% 标签 + 未见几何零样本 + 单首达相位”下严重过拟合；不会用新的命名掩盖负结果。
- **复杂场允许生成式残差。** APEX 所揭示的幅度/相位非对称性支持把简化相位先验作为 conditioning scaffold，而不是要求先验独立生成所有细节。

## 当前研究假设

对复波场 `u(x,ω)`，考虑

\[
u(x,\omega)=e^{i\phi_0(x,\omega)}r(x,\omega),
\]

其中 `φ₀` 是 travel time、Green phase、plane-wave mixture 或学习到的 carrier，`r` 是解调后的复残差。核心问题不是预设 `r` 一定低秩，而是系统测量：

1. 哪种 `φ₀` 能显著降低 `r` 的有效秩和频谱复杂度；
2. 单路径、稀疏多路径、局部 wave atoms 与 learned carrier 的适用区间；
3. 低秩模型应作用于幅度、carrier 参数、复残差还是生成模型的 latent；
4. 表征收益能否转化为真实任务收益，而非只在对齐合成数据上成立。

## 仓库结构

- `src/wave_lr/`：无模型绑定的 phase、demodulation 与 rank diagnostics；
- `experiments/`：先做 representation sanity，再做预测任务；
- `docs/RESEARCH_CHARTER_ZH.md`：研究边界和第一性原理 formulation；
- `docs/APEX_READING_NOTES_ZH.md`：对 arXiv:2605.26732 的详细阅读与本方向关系；
- `docs/LEGACY_TRACK2_AUDIT_ZH.md`：旧方向的正负证据和重启原则；
- `docs/REPRESENTATION_ROADMAP_ZH.md`：从显式 carrier 到生成式/预训练表示的层级；
- `docs/LITERATURE_MAP.md`：相关工作地图与创新边界；
- `docs/DATASETS_AND_PROTOCOLS.md`：共享数据位置、候选 benchmark 和公平协议；
- `docs/ROADMAP.md`：按证据 gate 推进的阶段计划。

## 最小验证

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/python experiments/representation_sanity.py
```

该 sanity 只验证“oracle phase 解调能否揭示低秩”，不是模型结果，也不能作为论文证据。

## 复现主线实验

```bash
.venv/bin/python experiments/exp01_rank_law.py                 # 合成场上的定律证伪测试
.venv/bin/python experiments/exp03_regime_phase_diagram.py     # FDTD 区间相图
.venv/bin/python experiments/exp05_task_gain_vs_regime.py --completion
.venv/bin/python experiments/exp06_public_data_tasks.py        # 公开数据上的同一套任务
.venv/bin/python experiments/exp07_carrier_error_tolerance.py  # 载波精度容限
.venv/bin/python experiments/exp08_bandwidth_not_frequency.py  # 秩由带宽而非中心频率决定
.venv/bin/python experiments/collect_summary.py                # 汇总到 reports/summary.json
.venv/bin/python experiments/make_figures.py                   # 生成 reports/figures/
```

大文件全部位于 NFS（见 `docs/DATASETS_AND_PROTOCOLS.md`）；仓库只保存 `reports/` 下的
小结与图。

## 与中央研究 Hub 的关系

本仓库属于 [Physics-Informed Tensor Learning Hub](https://github.com/xuangu-fang/Geo-Aware-Tensor)，但不与 Track 1/3 共享方法主线。共享的只有数据索引、split/mask discipline、baseline 与结果审计规范。

