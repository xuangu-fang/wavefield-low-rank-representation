# Literature map and novelty boundaries

This map prioritizes primary papers and official project pages. It is a starting index, not a claim of exhaustive coverage.

## 1. Amplitude/phase asymmetry and target-scarce high-frequency transfer

- [APEX: Amplitude Anchors and Phase Priors for Target-Scarce Higher-Frequency Wave Prediction](https://arxiv.org/abs/2605.26732) separates a stable coarse amplitude anchor from a simplified phase prior, then reconstructs the target with conditional flow matching. It directly motivates treating a phase prior as a scaffold rather than a full solution.
- [A Probabilistic Framework for Solving High-Frequency Helmholtz Equations via Diffusion Models](https://arxiv.org/abs/2602.04082) argues that deterministic neural operators struggle with high-frequency sensitivity and spectral bias, and uses a conditional score model. This supports a future probabilistic residual, but diffusion alone is not our novelty.

## 2. Phase preconditioning and model reduction

- [Phase-Preconditioned Rational Krylov Subspaces for Wave Propagation Problems](https://arxiv.org/abs/1711.00942) factors rapid frequency-domain oscillations and compresses the slower amplitudes. Therefore, “remove a known phase before low-rank approximation” is established prior art, not a standalone contribution.
- [Model reduction for transport-dominated problems via online adaptive bases and adaptive sampling](https://arxiv.org/abs/1912.13024) uses transported subspaces because fixed linear spaces poorly represent moving structures.
- [Nonlinear model reduction for transport-dominated problems via dynamical low-rank approximation and neural networks](https://arxiv.org/abs/1911.00565) likewise motivates moving/shifted coordinates when translational invariance destroys ordinary low rank.
- [A robust shifted proper orthogonal decomposition](https://arxiv.org/abs/2403.04313) decomposes multiple transports into co-moving low-rank fields with noise robustness. Shift/transport alignment must therefore be included in the baseline family.

## 3. Wave-aligned neural bases

- [A Neural Network with Plane Wave Activation for Helmholtz Equation](https://arxiv.org/abs/2012.13870) learns plane-wave amplitudes and directions and reports advantages over generic activations and fixed plane-wave bases.
- [A discontinuous plane wave neural network method for Helmholtz and time-harmonic Maxwell equations](https://arxiv.org/abs/2310.09527) uses element-wise complex exponential bases with adaptively determined directions. “Use plane waves in a neural representation” is therefore also established; our opportunity is representation diagnostics, multipath factorization and coupling to sparse/generative tasks.

## 4. Neural operators and wave benchmarks

- [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) is a mandatory regular-grid operator baseline when full input fields are available.
- [MIONet](https://arxiv.org/abs/2202.06137) is relevant when source, medium, geometry and frequency are separate functional inputs.
- [WaveBench](https://openreview.net/forum?id=6wpInwnzs8) provides multiple forward and inverse linear-wave tasks with official [code](https://github.com/wavebench/wavebench). Its operator protocols must not be mixed with sparse-field completion without an explicit adaptation.
- [An analysis of neural operators for learning solution operators of PDEs](https://arxiv.org/abs/2301.11509) includes Helmholtz out-of-distribution behavior and motivates explicit OOD frequency/medium tests.

## 5. Functional and tensor representations

- [F-INR: Functional Tensor Decomposition for Implicit Neural Representations](https://openaccess.thecvf.com/content/WACV2026/html/Vemuri_F-INR_Functional_Tensor_Decomposition_for_Implicit_Neural_Representations_WACV_2026_paper.html) covers functional CP/TT/Tucker-style INR decompositions. Functional tensor factors alone cannot be our novelty.
- Classical POD/SVD, DMD and tensor CP/Tucker/TT remain required compression baselines. The proposed value must come from wave-aligned coordinates, not merely choosing a different low-rank container.

## 6. Working novelty boundary

The plausible long-term contribution is not any single item below:

- phase demodulation;
- plane-wave activations;
- shifted low-rank approximation;
- functional tensor decomposition;
- diffusion/flow prediction.

The research opportunity is a unified, audited **representation hierarchy for complex multipath wave fields** that measures when physics-aligned carriers expose low rank, when they fail, and how the exposed structure should condition stronger generative or pretrained models across tasks.

