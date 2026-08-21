# 波动 benchmark 的区间覆盖审计

**判定规则**（对所有数据集一致应用）：在参考阵列间距 m=6 上，
若原始界已低于 0.15，记为**已可辨识**——该密度下本就不需要载波，
两个接近零的数之比不携带信息；否则按对齐界与原始界之比判定：
`≤0.5` **有利**、`≤0.9` **中等**、否则**不利**。
界的定义与三条测量纪律见 `REPORT_ZH.md` §0.1。

**一个数据集可以在不同轴上落在不同区间。** Helmholtz staircase 就是例子：
在 m=6 的空间采样下它本就可辨识（界 0.020），但在**频率轴的等预算压缩**上
载波带来 10.6× 增益。因此下表并列给出三根轴，不用单一数字概括一个数据集。

| 数据集 | 类别 | 界(raw) | 界(aligned) | 比值 | 判定 | 等预算压缩增益 | 2% 传感器增益 | 许可 |
|---|---|---:|---:|---:|---|---:|---:|---|
| 2D Navier-Stokes Re40 (Learning Dissipative Dynamics) | non_wave | — | — | — | 未测量 | — | — | CC-BY-4.0 |
| Kuramoto–Sivashinsky（本地生成） | non_wave | — | — | — | 未测量 | — | — | 本项目生成 |
| Cylinder wake（本地预处理） | non_wave | — | — | — | 未测量 | — | — | 本项目生成 |
| Active matter（本地预处理） | non_wave | — | — | — | 未测量 | — | — | 本项目生成 |
| PDEBench — 2D diffusion-reaction | non_wave | — | — | — | 未测量 | — | — | 见官方发布（未在本项目核实） |
| The Well — acoustic_scattering_inclusions | wave | 0.686 | 0.596 | 0.87 | 中等 | 1.18x | 1.18x | CC-BY-4.0 |
| The Well — acoustic_scattering_maze | wave | 0.999 | 0.985 | 0.99 | 不利 | — | 1.02x | CC-BY-4.0 |
| WaveBench — time-harmonic isotropic, omega label 40 | wave | 0.992 | 0.993 | 1.00 | 不利 | — | — | CC-BY-4.0 |
| The Well — helmholtz_staircase | wave | 0.020 | 0.023 | 1.13 | 已可辨识（该密度下无需载波） | 10.63x | 1.43x | CC-BY-4.0 |
| WaveBench — time-harmonic isotropic, omega label 10 | wave | 0.033 | 0.084 | 2.53 | 已可辨识（该密度下无需载波） | — | — | CC-BY-4.0 |
| OpenFWI（本地副本，配对已损坏） | wave | — | — | — | 未测量 | — | — | 见官方发布（未在本项目核实） |
| 本项目 FDTD 求解器 — open_clear | wave_reference | 0.997 | 0.062 | 0.06 | 有利 | — | — | 本项目生成 |
| 本项目 FDTD 求解器 — open_sparse | wave_reference | 0.993 | 0.556 | 0.56 | 中等 | — | — | 本项目生成 |
| 本项目 FDTD 求解器 — partial_clear | wave_reference | 0.997 | 0.663 | 0.66 | 中等 | — | — | 本项目生成 |
| 本项目 FDTD 求解器 — closed_dense | wave_reference | 0.990 | 0.860 | 0.87 | 中等 | — | — | 本项目生成 |

非波动数据集用另一根轴度量（时间占据比例与对齐增益，见 `REPORT_ZH.md` §11.7）：

| 数据集 | 能量占记录比例 | 预测增益 | 实测增益 |
|---|---:|---:|---:|
| 2D Navier-Stokes Re40 (Learning Dissipative Dynamics) | 0.58 | 0.81 | 0.68 |
| Kuramoto–Sivashinsky（本地生成） | 0.85 | 0.99 | 0.65 |
| Cylinder wake（本地预处理） | 0.59 | 0.91 | 0.83 |
| Active matter（本地预处理） | 0.83 | 0.99 | 0.47 |
| PDEBench — 2D diffusion-reaction | 0.86 | 0.97 | 0.47 |

## 覆盖情况

已测量的公开**波动**数据集：5 个，其中
有利 0 个、中等 1 个、不利 2 个、已可辨识（该密度下无需载波） 2 个。

**没有一个公开波动数据集落在「有利」区间。** 与此对照，本项目自建 FDTD 求解器的
四个参照点跨越 0.06 到 0.87，覆盖了整条坐标轴——所以缺口不在方法能否奏效，
而在**公开基准在这条轴上的采样偏斜**：它们要么处在相位对齐必然失效的强混响/强散射区间，
要么在所用采样密度下本就已经可辨识。

这对社区的含义是具体的：
**若要评估物理对齐类表征方法，现有公开波动基准无法区分「方法无效」与
「数据集不在该方法的适用区间」**。补上开放/吸收介质、且采样低于空间 Nyquist 的
公开数据，是让这类方法可被公平评估的前提。

## 出处与复现

| 数据集 | 来源 | 版本 | 获取方式 |
|---|---|---|---|
| The Well — acoustic_scattering_maze | https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_maze | 8df383a3 | 本地 64x64 下采样副本（由同组既有项目预处理） |
| The Well — acoustic_scattering_inclusions | https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_inclusions | c17cd1d1 | test/chunk_36.hdf5，256x256 全分辨率 |
| The Well — helmholtz_staircase | https://huggingface.co/datasets/polymathic-ai/helmholtz_staircase | a2429505 | train + test 全部 16 个 omega |
| WaveBench — time-harmonic isotropic, omega label 10 | https://doi.org/10.5281/zenodo.8015145 | 2023-06-07 | range 请求读取 75 GB zip 内成员的前缀，约 1000 个可读样本 |
| WaveBench — time-harmonic isotropic, omega label 40 | https://doi.org/10.5281/zenodo.8015145 | 2023-06-07 | 同上 |
| 2D Navier-Stokes Re40 (Learning Dissipative Dynamics) | https://doi.org/10.5281/zenodo.7495555 | — | 本地预处理副本 |
| Kuramoto–Sivashinsky（本地生成） | — | — | 同组既有项目生成 |
| Cylinder wake（本地预处理） | — | — | 同组既有项目生成 |
| Active matter（本地预处理） | — | — | 同组既有项目生成 |
| PDEBench — 2D diffusion-reaction | https://github.com/pdebench/PDEBench | — | 本地副本 |
| OpenFWI（本地副本，配对已损坏） | https://github.com/lanl/OpenFWI | — | 本地 seis2_*/vel4_* 副本——两者来自不同 family，见 DATA_INTEGRITY.md |
| 本项目 FDTD 求解器 — closed_dense | — | — | experiments/exp03 生成，作为坐标轴两端的参照 |
| 本项目 FDTD 求解器 — open_clear | — | — | experiments/exp03 生成，作为坐标轴两端的参照 |
| 本项目 FDTD 求解器 — open_sparse | — | — | experiments/exp03 生成，作为坐标轴两端的参照 |
| 本项目 FDTD 求解器 — partial_clear | — | — | experiments/exp03 生成，作为坐标轴两端的参照 |
