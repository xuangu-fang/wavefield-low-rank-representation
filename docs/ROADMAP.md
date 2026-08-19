# Evidence-gated roadmap

This repository is intentionally not scheduled as an immediate paper. Work proceeds when new data or ideas arrive.

## Stage 0 — repository and audit baseline (complete)

- preserve the legacy Track 2 negative evidence;
- document APEX and adjacent phase/transport literature;
- provide tested carrier, demodulation and rank-diagnostic primitives;
- point all large datasets to NFS storage.

## Stage 1 — representation phase diagram

Build independent direct/reflection/multipath generators and adapters for one public dataset. Compare raw, single-carrier, multipath and transported representations across frequency and path complexity. No large predictive model is needed.

Go gate: a non-oracle representation consistently improves rank/smoothness on independent and public data, not only aligned synthetic fields.

## Stage 2 — one honest downstream task

Choose exactly one: sparse sensor completion, target-scarce cross-frequency prediction, or source/medium-conditioned operator learning. Test whether the representation gain survives with estimated carriers and equal information.

Go gate: absolute validity plus improvement over interpolation, POD/shifted POD, joint INR and the relevant neural-operator baseline.

## Stage 3 — multipath learned representation

Learn a sparse/local carrier bank with explicit residual. Add path-count, direction and phase-mismatch diagnostics. Avoid forcing every component into a globally separable CP model.

Go gate: stable attribution under frequency/path controls and transfer to unseen media or geometries.

## Stage 4 — generative/pretrained coupling

Condition flow/diffusion or a pretrained wave encoder on stable amplitude and carrier tokens. Evaluate uncertainty and multiple plausible phase details, not only point prediction.

Go gate: better task accuracy/calibration than an equally strong unstructured generative baseline, with acceptable sampling cost.

## How to add a new idea

Every proposal should state:

1. the physical source of oscillation or transport;
2. the proposed representation and what becomes simpler after transforming;
3. which part is oracle, estimated or learned;
4. the smallest falsifiable diagnostic;
5. the downstream task and equal-information baselines;
6. a stop rule.

This keeps the repository open to future work without turning it into an accumulation of named components.

