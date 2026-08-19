# Datasets, local storage and evaluation protocols

## 1. Storage policy

Git stores only manifests, split specifications, small summaries and figures. Raw or prepared arrays belong under the large NFS mount:

```text
/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/
/mnt/data/xuangu-fang/physics-informed-tensor-learning/cache/
```

Recommended Track 2 root:

```text
/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/
  raw/
  prepared/
  manifests/
```

Code must accept a CLI/config data root; do not hard-code a user home path. Large files, checkpoints and external archives must never be committed to this repository.

## 2. Data already available locally

### OpenFWI CurveFault-A subset

Path:

```text
/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/openfwi_curvefault_a/raw/
```

Currently contains three `seis2_1_*.npy` files (about 700 MB each) and matching velocity files. It is useful for seismic source/receiver waveforms and velocity-conditioned representation studies. Before use, create a manifest recording array axes, units, sample IDs and the official train/test semantics; do not infer axes from shape alone.

### Central shared data

Other local datasets include PDEBench, active-matter, Navier–Stokes and cylinder-flow assets. They are not wave benchmarks and should not be recruited merely to enlarge the table. They can test transport-aligned representation only if the paper question explicitly includes non-wave transport.

## 3. External wave resources

| Resource | Main value | Correct role |
|---|---|---|
| [APEX SimpleWave/Helmholtz/Maxwell](https://arxiv.org/abs/2605.26732) | controlled direct/reflected and cross-frequency settings | reproduce or request/rebuild only with documented split; compare representation layers |
| [WaveBench](https://github.com/wavebench/wavebench) | 24 linear-wave forward/inverse datasets and official baselines | operator benchmark; preserve official task inputs |
| [OpenFWI](https://github.com/lanl/OpenFWI) | large-scale seismic velocity and waveform families | pretraining, sparse receiver completion, inversion representations |
| [The Well acoustic scattering maze](https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_maze) | strong reflections, multiple initial sources, complex geometry | hard multipath stress; do not expect single-arrival model to win |
| [The Well Helmholtz staircase](https://huggingface.co/datasets/polymathic-ai/helmholtz_staircase) | frequency/source variation on staircase geometry | complex harmonic phase/frequency representation |

Downloads must go to NFS. Each prepared dataset requires `manifest.json` with source URL/version, license, checksums, axes, preprocessing, normalization and frozen splits.

## 4. Three distinct tasks—never mix them

### A. Representation compression

Input: complete wave fields. Goal: measure singular decay/effective rank after a transformation. Oracle phase is allowed but must be labeled oracle. This is a diagnostic, not prediction.

### B. Sparse completion

Input at test time must include the allowed sensor values and coordinates. Compare 1%/2%/5%/10% observation ratios with nested masks. If test sensors are not consumed, the task is zero-shot prediction, not completion.

### C. Operator/cross-frequency prediction

All methods receive identical medium/source/boundary/frequency inputs. Full-supervision and target-scarce protocols are separate tables. Official FNO/U-Net/operator checkpoints cannot be compared directly against a 1%-label field regressor.

## 5. Minimum baselines

- zero, observed mean, nearest/interpolation where legal;
- raw complex SVD/POD and tensor CP/Tucker/TT;
- amplitude/phase separated low rank;
- shifted/transported POD;
- Euclidean/wrong carrier, oracle carrier and estimated carrier;
- SIREN/joint INR and parameter-matched functional tensor;
- FNO/U-Net/MIONet/GINO when the task is actually operator learning;
- conditional flow/diffusion when multimodality or high-frequency uncertainty is central.

## 6. Metrics and aggregation

- complex NRMSE and real/imag NRMSE;
- amplitude/log-amplitude error;
- phase coherence or amplitude-weighted phase coherence (ignore phase where amplitude is negligible);
- best-rank-`r` residual and 95%/99% effective rank for multiple unfoldings;
- boundary/reflection-zone and late-arrival errors;
- per-case metrics first, then seed mean/uncertainty; 3–5 seeds for learned models;
- equal updates for early POC, plus convergence audit before a publication claim.

An all-method score near the zero baseline is an invalid experiment for ranking minor differences. Absolute validity gates precede pairwise wins.

