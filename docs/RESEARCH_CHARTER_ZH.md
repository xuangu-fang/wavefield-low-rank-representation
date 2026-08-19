# 研究章程：波动场为什么需要新的低秩表征

## 1. 问题不是“波场有没有低秩”

一个平移或传播的简单波在原始欧氏网格上也可能呈现很高的矩阵/张量秩。原因不是物理过程真的复杂，而是相位 `φ(x,t)` 快速变化：很小的 travel-time 误差在高频下会变成很大的复数场误差。

因此本方向不从“给原始张量选择 CP、Tucker 还是 TT”开始，而从下面的问题开始：

> 能否找到一个尊重传播的坐标或 carrier，使快速振荡变成显式、可对齐的部分，而把学习能力留给慢变幅度、多路径残差和不确定性？

## 2. 最小数学形式

单路径高频近似可以写成

\[
u(x,\omega)\approx A(x,\omega)e^{i\omega\tau(x)}.
\]

若 `τ(x)` 已知或能近似，则解调

\[
r(x,\omega)=u(x,\omega)e^{-i\omega\tau(x)}
\]

消除了主要振荡。真正需要验证的是：`r` 是否比 `u` 更容易被低秩、神经隐式表示或生成模型表达。

多路径时更合理的形式是

\[
u(x,\omega)=\sum_{m=1}^{M}A_m(x,\omega)e^{i\phi_m(x,\omega)}+r_{\mathrm{diff}}(x,\omega),
\]

其中 `M` 不必已知，`r_diff` 可以容纳绕射、复杂边界交互、未建模介质和噪声。这里的研究空间包括：

- 固定/估计的 travel-time carrier；
- 稀疏、可学习的多路径 phase dictionary；
- 局部 plane-wave/Trefftz atoms；
- transport/shift 对齐后的低秩子空间；
- 以粗相位和稳定幅度为条件的 flow/diffusion latent。

## 3. 三角恒等式处在什么位置

对单个 traveling harmonic，

\[
\cos(kd-kct)=\cos(kd)\cos(kct)+\sin(kd)\sin(kct).
\]

它证明：选对传播坐标后，联合空间—时间相位可以精确写成两个可分项。这个事实仍然重要，因为它提供：

1. 一个无歧义的单元测试和机制 sanity；
2. 显式 phase pairing 的可解释 basis；
3. 多路径 mixture 中每个 carrier 的低秩原子；
4. 更强模型的结构化 conditioning 或初始化。

但它只解决 carrier 的代数可分性，不自动解决：移动包络、未知频率、反射路径发现、跨几何 transfer、极稀疏观测或 posterior multimodality。

## 4. “低秩”可以放在哪一层

本仓库不强制原始场低秩。至少有四种合法位置：

1. **carrier bank 低秩/稀疏**：只激活少数传播方向、路径或频带；
2. **解调残差低秩**：对齐后复包络在空间、频率、source 等 modes 上可压缩；
3. **任务 latent 低秩**：生成模型或 neural operator 的隐变量具有低秩 interaction；
4. **跨样本共享低秩**：不同介质/几何共享 wave atoms，而样本只预测少量系数。

哪一层成立必须由 singular decay、跨数据泛化和下游任务共同验证，不能只用一个合成结果决定。

## 5. 与 APEX 的互补关系

APEX 研究 target-scarce higher-frequency prediction，发现 coarse amplitude anchor 比跨频相位迁移稳定，并用简化 Green-inspired phase prior 条件化 conditional flow matching。它说明：

- 幅度与相位应拆开建模；
- 相位误差的高频放大是真实瓶颈；
- 简化 phase prior 不必精确，只要能为生成式恢复提供有用结构；
- 复杂多路径可以由条件生成模型补全，而不是全压在解析先验里。

本仓库比 APEX 更底层：系统研究哪一种 phase/carrier 表征能形成压缩、稳定或可迁移的中间空间。未来可以把最有效的表征接入 APEX 类生成任务，而不是复制其完整 pipeline。

## 6. 明确不做的主张

- 不声称原始复波场普遍低秩；
- 不声称单首达 travel time 足够描述反射与散射；
- 不把 oracle phase 的 rank 改善当作预测成功；
- 不把与 generator 同频的字典实验当作主证据；
- 不要求低秩模型独立击败所有生成模型或 neural operators；
- 不把 phase retrieval（从幅值测量恢复相位）与本项目的 phase-aware representation 混为一谈。

