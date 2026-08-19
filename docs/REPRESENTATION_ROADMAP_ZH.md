# 表征路线图：从显式相位到生成式 latent

> **进度（2026-08-19）**：R0 与 R1 已完成并给出定量定律，R2 完成"载波库构造"但尚未验证跨介质迁移；
> R3–R5 未启动。详见 `REPORT_ZH.md` 与 `ROADMAP.md`。R1 的 gate 已通过：
> 可部署（非 oracle）的 eikonal 载波在独立求解器与三个公开数据集上都降低有效秩，
> 并在稀疏传感器重建任务上带来最高 13.6× 的提升。

这是一条按复杂度逐级增加的研究路线，不是要求一次全部实现。每一级只有在 representation diagnostics 和下游任务都出现增益时才晋级。

## R0：表示与指标基线

对同一复场比较：

- real/imag；
- amplitude/raw phase；
- log-amplitude + sin/cos phase；
- space-time/frequency unfoldings 的 singular spectrum；
- complex NRMSE、amplitude NRMSE、phase coherence/AWPC。

目的：先确认困难来自幅度、相位、branch cut，还是数据本身没有可压缩结构。

## R1：单 carrier 解调

使用已知或估计的 `φ₀=ωτ(x)`：

\[
r=u e^{-i\phi_0}.
\]

比较 Euclidean、eikonal/travel-time、Green phase、oracle phase。三角恒等式 paired carriers 是实值场的等价可分实现。

Gate：估计 phase（不只 oracle）必须在独立数据上稳定降低 residual effective rank，并改善至少一个真实 completion/operator task。

**已通过。**并且得到了比 gate 更强的结论：秩 = 带宽 × 延迟占据测度，收益 = 占据测度之比，
两者都能在训练前算出。收益消失的区间也被同一公式预测（见 `THEORY_DELAY_OCCUPANCY.md`）。

## R2：稀疏多路径 mixture

\[
u=\sum_{m=1}^{M}a_m e^{i\phi_m}+r.
\]

路径可以来自 image-source/ray candidates、局部 arrival picking 或可学习字典。低秩/稀疏约束放在 carrier coefficients，允许 residual 存在。

Gate：在 direct + reflection、不同反射阶数和频率下，自动估计 mixture 超过单 carrier、普通 SVD/POD 和参数匹配 INR。

**部分通过。**等参数预算下多载波优于单载波（1.41×）与 plain SVD（1.63×），
且优势按理论预测的形状随体散射衰减；载波可以完全从数据估计（拿到 oracle 的 90–96%）。
仍缺：跨介质/跨几何迁移，以及与参数匹配 INR 的正式对照（`exp14` 只对照了逐 case INR）。

## R3：局部、多尺度 wave atoms

复杂介质中全局路径未必可识别，可以在 patch/mesh element 上学习方向、波数和复包络，相当于 learned plane-wave/Trefftz dictionary。再通过 graph/attention/operator 模块组合局部 atoms。

Gate：改善必须来自 wave-aligned atoms；与 Fourier features、SIREN、local INR、plane-wave network 做公平比较。

## R4：结构化条件生成模型

借鉴 APEX：稳定的 coarse amplitude、少量 phase carriers 和介质/几何作为条件，由 flow matching 或 diffusion 学习多路径 residual。低秩可约束 latent interaction，而非强迫输出场本身低秩。

Gate：除了均值误差，还需改善不确定性校准、模式覆盖和高频细节；报告迭代采样成本。

## R5：跨 PDE/频率/任务预训练

预训练一个 wave tokenizer/encoder，将复场映到 amplitude、carrier tokens、residual tokens。下游可包括 sparse completion、cross-frequency prediction、inverse source、material inversion 与 forecasting。

Gate：必须在未见介质/几何/频率上迁移，且优于相同参数量的通用预训练表示；不能只在单数据集重构。

## 推荐的首个新 POC

不直接训练大生成模型。先建立一个 **representation phase diagram**：

- 数据：解析 direct/reflection → 独立 wave solver → OpenFWI/The Well/WaveBench 子集；
- 横轴：frequency、路径数/反射强度、观测率；
- 方法：raw SVD、oracle single demodulation、estimated single demodulation、oracle multipath、learned sparse carriers；
- 输出：有效秩、rank-r residual、跨频/跨几何稳定性、少量传感器重构误差。

这能回答一个比“某个模型是否赢”更持久的问题：相位对齐究竟在哪个物理区间将复杂波场变成可压缩对象。

