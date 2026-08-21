# Low-Rank Representation Learning for Wave Fields

波动场低秩表征学习：一个长期、representation-first 的研究仓库。

## 一句话目标

> 先显式消除波场中可解释的快速相位传播，再用低秩模型或生成模型学习剩下的慢变、多路径与不确定成分。

本仓库不再承诺“一个张量分解直接解决复杂波场”。它研究的是更基础的问题：复杂复数波场应该用什么坐标和中间表示，才能让后续的 completion、operator learning、跨频预测、反演或生成建模更容易。

## 当前主线：可辨识性，而不是容量

> 稀疏观测的物理场上，误差里有多少是**不可学习的缺失信息**，可以在训练前用**一次 FFT**
> 算出来——与用什么模型无关、不需要真值、不需要知道方程。同一次 FFT 还多回答一件事：
> **换一个坐标系能把误差降多少倍**，而这是在做变换**之前**就知道。

核心量：在间距 `m` 的规则传感器阵列上，

```
误差 ≥ sqrt( 能量(|k| > 1/(2m)) / 总能量 )   ≡ 可辨识性界
```

### 最强结果：判据预测了坐标变换的收益，跨物理族

四个物理族、同一条管线、数据全部公开且无一由本项目生成（`exp29`）：

| 物理族 | 原始界 | 预测增益 | 实测增益 | 判据恢复的速度 |
|---|---|---|---|---|
| 地震炮集 OpenFWI | 0.606 | 2.54× | 2.83× | 2538 m/s（速度模型 1500–4500） |
| 圆柱尾流**实测 PIV** | 0.322 | 1.24× | 1.24× | 0.271 m/s = 0.9 × 来流 |
| Kuramoto–Sivashinsky | 0.285 | 1.02× | 1.02× | — |
| 二维湍流 Re40 / Re5000 | 0.35 / 0.37 | 1.00× / 1.01× | 1.00× / 1.01× | 噪声 |

**464 次测量、928 次界检验、0 次违反；预测增益 vs 实测增益 R² = 0.9915，跨越 1.0×–8.7×。**
判据未被告知任何方程，却恢复了物理正确的输运速度；无输运结构时恢复的是噪声。

### 最尖锐的对照：428× 容量做不到的，零参数换个坐标做到了

公开数据、14/70 传感器（`exp31`）：坐标网络扫过 **428×** 参数、训练损失 **4.7e-16**，
测试误差 **1.05**（比输出零还差）；**零参数**带限估计器 0.472；同一估计器**换到学出的坐标**
**0.177**。

> 修正（`exp32`）：监督训练的重构网络会**穿过**原始界（0.114 vs 0.256）。界约束的是只用本
> 实例样本的估计器；跨实例训练带来先验，先验是额外信息。所以跨过界的是**先验**，不是容量。

### 波场上的原始三条结果

| 结论 | 证据 |
|---|---|
| **界成立且接近紧** | 458 组测量（自建求解器 + 4 个公开数据集），打破比例 **1.7%**（时域）/ **8.0%**（频域公开），R²=0.80 / 0.88 |
| **不是容量** | 参数量跨 **128×**，测试误差只动 **3%**，训练误差恒为 **10⁻⁷**；**零参数的线性插值打败每一个网络** |
| **信息可以学回来** | 把界本身当损失：纯学习（无介质、无源、无求解器）在界上追平物理（0.526 vs 0.535），并把被破坏的载波修回（0.927 → 0.525） |
| **坐标可摊销、可迁移** | 仅以界为损失、无标签：未见介质上摊销网络 **0.591** vs 物理 **0.581**；另一介质族上 **0.300** vs **0.330**（超过物理）；**跨介质族** 0.364 vs 0.364（持平）。测试时只需一次前向传播 |
| **跨几何迁移有代价** | 留出边界免费（1.00），但**留出源位置代价 1.23**，几何异质性本身再付 11%——可说"给定几何族的规则"，不能说"与几何无关的规则" |
| **覆盖审计** | 统一判据 + 出处/版本/许可存档：**0 / 5** 个公开波动数据集落在有利区间，自建求解器跨越 0.06–0.87 |
| **留出验证全部复现** | 流程冻结、12 个从未使用的 seed、252 组测量：界打破 4.8%（网络 0.0%）、网络 0.995 vs 线性 0.759、纯学习 0.554 vs 物理 0.552 |

换算成传感器数量：开放无杂波区间 **省 24×**；其余区间无收益——**判据事先说明**。

### 支撑结果（频率轴上的对偶）

| 结论 | 证据 |
|---|---|
| 秩 = 带宽 × 延迟占据测度 | 合成 R²=0.998（n=840）；FDTD 0.944；maze 0.954 |
| 秩由**带宽**而非中心频率决定 | rank~带宽 R²=0.96–0.97 vs rank~中心频率 R²=0.001–0.005 |
| 载波精度判据 `δτ < 1/B` | 粗糙误差在 1/B 处收益归零；平滑标定误差在 2.5/B 仍有 2.7× |
| shifted POD 在每个波场上都没赢过 plain POD | 它撤的是空间刚性平移，波场需要逐点时间弯折 |
| 四个公开波动基准中**三个**落在必然无效区间 | 判据事前预测全中（maze 预测 1.060 / 实测 1.060） |

### 适用边界（已实测）

- **只对"瞬态且相干的输运"有效**：五个公开非波动流场全部统计定常，对齐有害（0.47–0.83），判据提前预警；
- **色散会破坏走时载波**：staircase 的 trapped mode 使跨频外推全线失败——但学出来的非线性相位在那里反超物理 1.74×；
- **跨介质算子学习仍是弱结果**（1.2×，数据受限），明确标注不支持主张。

- 叙事定位：`docs/POSITIONING_CROSS_PHYSICS_ZH.md`（第二版，跨物理族）、`docs/POSITIONING_ZH.md`（第一版，保留）
- 完整报告：`docs/REPORT_ZH.md`
- 理论定稿：`docs/THEORY_DELAY_OCCUPANCY.md`
- 数据完整性发现（含本地 OpenFWI 配对错误）：`docs/DATA_INTEGRITY.md`
- Benchmark 覆盖审计：`reports/BENCHMARK_COVERAGE.md` 与机器可读的 `reports/benchmark_registry.json`

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

