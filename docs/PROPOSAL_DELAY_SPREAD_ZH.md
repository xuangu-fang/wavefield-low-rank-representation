# 提案：延迟展宽（delay spread）而非频率，决定波场的可压缩性

状态：主攻方向（2026-08-19 立项）。本文件按 `ROADMAP.md` 的“如何加入新想法”六条要求撰写。

## 0. 一句话

> 复波场 `u(x,ω)` 在 `(x,ω)` 展开下的数值秩由**到达时间的支撑长度**乘以**带宽**决定；相位解调并不“消除振荡”，它把支撑长度从**绝对走时展宽** `L_abs` 换成**相对延迟展宽** `D`。因此相位对齐的收益等于 `G = L_abs / D`，这个量在训练任何模型之前就能从 `c(x)` 与几何算出来。

## 1. 振荡/输运的物理来源

频域波场是时域格林函数的 Fourier 变换：

\[
u(x,\omega)=\int g(x,\tau)\,e^{i\omega\tau}\,d\tau .
\]

`x` 处的快速 `ω` 振荡完全来自 `g(x,\cdot)` 的支撑位置（走时），而不是介质的“复杂度”。这正是为什么一个物理上极简单的传播波在欧氏网格上秩很高。

## 2. 提出的表征与它简化了什么

把 `U[x,\omega]` 看成一个非均匀 Fourier 算子在带宽 `B=\omega_{\max}-\omega_{\min}` 上的截断。band-limited / time-limited 算子的自由度计数（Slepian–Landau–Pollak 型）给出

\[
\operatorname{rank}_\varepsilon(U)\;\approx\;\frac{B\,L}{2\pi}+O\!\big(\log\tfrac1\varepsilon\,\log BL\big),
\qquad
L=\big|\operatorname{supp}_\tau \bigcup_x g(x,\cdot)\big| .
\]

解调 `r(x,\omega)=u(x,\omega)e^{-i\omega\tau_{\mathrm{ref}}(x)}` 等价于把每个 `x` 的脉冲响应在时间轴上平移 `-\tau_{\mathrm{ref}}(x)`。若 `\tau_{\mathrm{ref}}` 取首达 `\tau_1(x)`，支撑变成

\[
L\;\longrightarrow\;D=\max_x\big(\tau_{\max}(x)-\tau_1(x)\big)\quad(\text{延迟展宽 / coda 长度}),
\]

于是有**可证伪的增益律**

\[
\boxed{\;G=\frac{\operatorname{rank}_\varepsilon(U)}{\operatorname{rank}_\varepsilon(R)}\approx\frac{L_{\mathrm{abs}}}{D}\;}
\]

三个直接推论：

- **自由空间/单首达**：`D\to 0`，解调后秩 → 幅度的几何扩散秩（≈1–3），增益极大；
- **强混响（迷宫、闭合腔）**：`D \approx L_{\mathrm{abs}}`，`G\to 1`，**相位对齐无收益**——这正是旧 Track 2 在 The Well acoustic maze 上全线失败的定量解释；
- **跨频外推**：从带 `[\omega_1,\omega_2]` 外推到 `\omega'` 的可解自由度同样正比于 `B L`，因此**可外推带宽随 `1/L` 增长**，解调把它放大 `G` 倍。这与 APEX 的 `|\hat u-u|^2=(\hat A-A)^2+4A\hat A\sin^2(\nu\Delta\tau/2)` 相容，但把“单点误差放大”提升为“表征自由度计数”。

方法侧的推论：既然秩正比于**每个载波的局部延迟支撑**，正确的模型不是“换一个低秩容器”，而是把场拆成少数 co-moving 分量

\[
u(x,\omega)=\sum_{m=1}^{M}e^{i\omega\tau_m(x)}\,r_m(x,\omega)+e,\qquad
\text{参数量}\;\approx\;\sum_m \frac{B d_m}{2\pi}\;\ll\;\frac{B L_{\mathrm{abs}}}{2\pi}.
\]

## 3. 哪部分是 oracle / 估计 / 学习

| 成分 | 来源 | 等级 |
|---|---|---|
| `\tau_1(x)` | 由 `c(x)` 跑 eikonal（fast sweeping） | **可部署**（介质已知的正问题设定） |
| `D(x)` | 时域包络的能量衰减/最后到达阈值 | 估计（诊断用，报告时标注） |
| `\tau_m(x)`, `m>1` | image-source 几何 或 数据驱动多到达拾取 或 可学习 | 估计 / 学习 |
| `r_m` | 低秩 / INR / 生成式 | 学习 |
| “oracle 相位” = 场自身的解缠相位 | 仅用于给表征上限 | **oracle，不可部署** |

## 4. 最小可证伪诊断

在受控合成 M-path 场上扫 `(B, L_{\mathrm{abs}}, D, M)`，检验 `rank_\varepsilon \approx BL/2\pi` 与 `G\approx L_{\mathrm{abs}}/D`。
**证伪条件**：若实测秩与该式在 2 倍以内都对不上（合成场上），整条理论线立即停止。

## 5. 下游任务与等信息 baseline

- **任务 A（跨频外推）**：低频带拟合 → 高频带预测。baseline：raw 复场外推、amplitude/phase 分开外推、POD 外推、FNO。
- **任务 B（稀疏传感器重构）**：1/2/5/10% 传感器**必须进入推理**。baseline：最近邻/插值、POD、shifted POD、SIREN/joint INR、参数量对齐。
- 指标：complex NRMSE、amplitude NRMSE、AWPC，逐 case 后再聚合。

## 6. 停止规则

1. 合成场上增益律偏差 > 2×  → 放弃理论主线，退回纯经验相图；
2. 真实数据上 `G_{\text{预测}}` 与 `G_{\text{实测}}` 的相关性 `R^2 < 0.5` → 只保留“定性区间”结论，不写定律；
3. 表征增益无法在任一真实任务上以**估计（非 oracle）**载波复现 → 只发布负结果与相图；
4. 在 maze 上出现增益（与理论预测矛盾）→ 先查 bug，再考虑理论修正。
