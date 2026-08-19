# APEX 阅读笔记及其对本仓库的启发

论文：[APEX: Amplitude Anchors and Phase Priors for Target-Scarce Higher-Frequency Wave Prediction](https://arxiv.org/abs/2605.26732)，Yifan Sun 等，arXiv:2605.26732，2026。

## 1. 它解决什么问题

APEX 研究从数据较丰富的低频波场迁移到目标样本很少的更高频率。困难不是普通的 resolution super-resolution，而是同一介质在频率改变后，复波场的相位对应关系迅速变差。

论文给出的复误差分解非常关键。令真实与预测分别为 `u=A exp(iντ)`、`û=Â exp(iντ̂)`，则

\[
|\hat u-u|^2=(\hat A-A)^2+4A\hat A\sin^2\!\left(\frac{\nu\Delta\tau}{2}\right).
\]

因此同样的 travel-time 偏差 `Δτ` 会随目标频率放大。跨频时，幅度形态可能仍较稳定，直接复制相位却非常脆弱。

## 2. 方法的三个组成

1. 冻结低频 FNO，并只外推较稳定的 coarse log-amplitude，作为 amplitude anchor；
2. 由少量主导路径长度构造 Green-inspired complex phase prior；
3. 用 conditional flow matching 恢复目标 `[log amplitude, sin phase, cos phase]`。

论文在 SimpleWave、Helmholtz 和 Maxwell 上测试 target-scarce 高频迁移。SimpleWave 显式含直达与反射；论文的 phase prior 仍故意保持简化：SimpleWave/Helmholtz 用一条主路径，Maxwell 用两条。这一点很有启发——prior 的角色是 scaffold，不是精确 simulator。

## 3. 对旧 Track 2 的印证

旧 Track 2 尝试用单首达 travel time 和固定 paired carriers 直接完成极稀疏跨几何回归。失败不是意外：

- 一条最短路径没有表达反射、多源相干和绕射；
- 1% 训练标签不足以稳定识别 frequency/path/amplitude；
- 零样本新几何要求 amplitude 和 residual 也跨域泛化；
- 固定 CP 收缩让所有未建模成分都只能扭曲 amplitude factors。

APEX 的结果支持保留 phase prior，却反对让它单独承担完整高频预测。

## 4. 新仓库与 APEX 不重复的研究问题

APEX 已经给出一个特定任务上的生成式系统。本仓库不把“再做一个 APEX”作为贡献，而关注以下更基础的诊断：

- raw real/imag、amplitude + sin/cos phase、oracle-demodulated residual 的 rank/smoothness 差异；
- 单路径、多路径、局部 plane-wave 与 learned carriers 的 phase diagram；
- carrier 错配怎样随频率、反射阶数和观测率放大；
- 低秩约束放在 residual、carrier coefficients 或 generative latent 哪一层最稳；
- 一个 representation gain 何时真正转化为 completion/operator/inverse-task gain。

若这些问题产生稳定结论，后续可以把表征模块作为 APEX 类 conditional flow/diffusion 的输入或 latent regularizer。

## 5. 可直接继承的实验纪律

- 幅度和相位指标必须分开报告；
- phase 用 `(sin φ, cos φ)` 或 complex phasor，避免 `±π` branch cut；
- 不能只报 real-field NRMSE，应增加 complex NRMSE、amplitude error、phase coherence/AWPC；
- target frequency、路径复杂度和 target-sample ratio 应形成 phase diagram；
- 简化 prior 与 oracle prior 要分开：oracle 只给表征上限，不是 deployable baseline；
- 生成式方法要报告采样成本和不确定性，而不只报均值误差。

