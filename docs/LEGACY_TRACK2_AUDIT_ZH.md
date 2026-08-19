# 旧 Track 2 证据审计：保留什么、放弃什么

历史来源：中央仓库的 [`TRACK2_PHASE_WAVE.md`](https://github.com/xuangu-fang/Geo-Aware-Tensor/blob/main/papers/four_tracks/tech_reports/TRACK2_PHASE_WAVE.md)。本文不复制全部训练日志，只保留对新方向有约束力的结论。

## 1. 经验证的正信号

- paired carriers 的四项实现严格满足 traveling sine/cosine 恒等式；
- 与频率字典对齐的 eikonal harmonic control 达到 `0.0952±0.0144` NRMSE；
- clean traveling harmonic 中，正确 travel time 明显优于错误 Euclidean phase；
- phase-envelope rank-2 扩展能把 moving-envelope NRMSE 从约 `0.897` 降到 `0.644`。

这些结果证明传播坐标和 phase pairing 有表征价值，但不足以证明跨几何预测成功。

## 2. 不能回避的负信号

| 数据/任务 | paired phase | 参照 | 判断 |
|---|---:|---:|---|
| moving envelope | `0.644±0.118` | joint INR `0.624±0.027` | 没有超过联合模型 |
| independent reflected wave | `3.473±0.504` | mean ≈`1.0`, joint INR `1.482` | 严重跨几何过拟合 |
| clean off-grid traveling harmonic | `1.486±0.081` | zero `1.0`, joint INR `1.275` | identity 正确但绝对失败 |
| irregular outer-boundary wave | `1.091` | wrong/joint `1.037` | 几何归因失败 |
| The Well acoustic maze | ≈`0.992` | 多数方法 ≈`1.0` | 所有方法基本无效 |

特别需要保留的诊断是：模型在 observed training labels 上可以拟合到很低误差，却在新几何上远差于零预测。这不是训练步数不够，而是 representation、任务上下文和 transfer 假设不匹配。

## 3. 被放弃的旧主张

- “单首达 geodesic phase + functional CP 可以直接解决复杂反射波场”；
- “测试场 1% observation completion”——旧测试传感器没有进入推理，实际是 sparse-supervision zero-shot transfer；
- “correct phase 比 wrong phase 好就表示方法有效”——绝对 gate 必须先通过；
- “合成频带对齐结果可以作为 headline”。

## 4. 被保留的资产

- 三角载波的纯函数与精确测试；
- 独立 wave solver、travel-time/Euclidean control 和冻结 split 思路；
- `zero/mean → wrong phase → ordinary low rank → joint INR/operator → proposed` 的 baseline discipline；
- absolute gate、逐 case 聚合以及不读取失败后的锁定 test；
- The Well/WaveBench 等外部数据接口经验。

## 5. 新方向的五条硬约束

1. 先用 representation diagnostics 检查秩、平滑性和稳定性，再训练大模型；
2. 单路径只是 R0/R1 sanity，多路径或反射数据必须进入核心 phase diagram；
3. deployable phase estimator 与 oracle phase 分开报告；
4. completion 必须让测试传感器真正进入 inference，否则称 zero-shot prediction；
5. 允许强生成模型学习 residual，不再要求低秩分解独立完成所有细节。

