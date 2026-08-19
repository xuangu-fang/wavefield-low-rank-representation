# Evidence-gated roadmap

This repository is intentionally not scheduled as an immediate paper. Work proceeds when new data or ideas arrive.

## Stage 0 — repository and audit baseline (complete)

- preserve the legacy Track 2 negative evidence;
- document APEX and adjacent phase/transport literature;
- provide tested carrier, demodulation and rank-diagnostic primitives;
- point all large datasets to NFS storage.

## Stage 1 — representation phase diagram (complete, 2026-08-19)

Build independent direct/reflection/multipath generators and adapters for one public dataset. Compare raw, single-carrier, multipath and transported representations across frequency and path complexity. No large predictive model is needed.

Go gate: a non-oracle representation consistently improves rank/smoothness on independent and public data, not only aligned synthetic fields.

**Passed.** A batched FDTD solver makes boundary absorption and scatterer
density controlled variables (`exp03`), and the deployable eikonal carrier —
never an oracle — improves rank on that sweep, on The Well acoustic maze, on
acoustic inclusions and on the Helmholtz staircase. More than a phase diagram
came out of it: measured rank follows `bandwidth x delay occupancy` with
slope 0.92-0.99 and R2=0.94-0.998 across all of them, which turns the diagram
into a predictor. See `REPORT_ZH.md` sections 2-4.

## Stage 2 — one honest downstream task (complete, 2026-08-19)

Choose exactly one: sparse sensor completion, target-scarce cross-frequency prediction, or source/medium-conditioned operator learning. Test whether the representation gain survives with estimated carriers and equal information.

Go gate: absolute validity plus improvement over interpolation, POD/shifted POD, joint INR and the relevant neural-operator baseline.

**Passed for sparse sensor reconstruction; failed for cross-frequency
prediction, with a diagnosis.** Sensor reconstruction beats its raw-coordinate
counterpart by 13.6x in open media, and the gain tracks the predicted rank gain
with R2=0.988 in log-log across the regime sweep (`exp05`). A swept
Fourier-feature MLP and a SIREN on the raw field never leave the zero-prediction
level while fitting their sensors to 0.000-0.015 training error (`exp14`).
Cross-frequency extrapolation on the Helmholtz staircase fails for every method
including ours; the cause is guided-mode dispersion and cutoffs, not the
representation (`exp04`, `REPORT_ZH.md` section 10).

## Stage 3 — multipath learned representation (partially complete, 2026-08-19)

Learn a sparse/local carrier bank with explicit residual. Add path-count, direction and phase-mismatch diagnostics. Avoid forcing every component into a globally separable CP model.

Go gate: stable attribution under frequency/path controls and transfer to unseen media or geometries.

**Bank construction done, transfer not yet tested.** Carriers are grown from
the field by scanning the model residual for the virtual source it stacks along
best, and kept only if the fit improves at a fixed total budget (`exp12`). The
estimated bank captures 90-96% of the known-geometry oracle and places each
wavefront within 0.02-0.08 of the `1/B` tolerance. Attribution is stable under
the scatterer-density control: the advantage decays monotonically from 1.41x to
1.06x as volume scattering replaces boundary reflection, as predicted. Transfer
to unseen media or geometries remains open.

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

