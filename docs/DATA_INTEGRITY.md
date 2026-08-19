# Data integrity notes

Findings that change how a dataset may be used. Each entry states the check
that produced it so it can be re-run.

## 1. The local OpenFWI copy pairs seismic and velocity files from different families

Path: `/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/openfwi_curvefault_a/raw/`

The directory holds `seis2_1_{0,1,2}.npy` together with `vel4_1_{0,1,2}.npy`.
OpenFWI names a family's waveform and model files with the *same* numeric
prefix, so `seis2_*` and `vel4_*` come from two different families.

Check performed (2026-08-19): for 40 models we picked the first break on the
centre shot, fitted the moveout slope over receivers 50-69 to get an apparent
surface velocity, and correlated it against the mean surface velocity of the
paired model. Correlation was **0.135** — statistically indistinguishable from
unpaired data. Implied `dt` per model scattered over `0.65-1.6 ms` instead of
concentrating at the documented 1 ms.

Consequence: no eikonal travel time computed from these velocity models
describes the recorded gathers. `wave_lr.fields.load_openfwi` therefore picks
its carrier from the data (first-break picking) and labels the pairing broken.
Any velocity-conditioned experiment on this copy is invalid until matching
files are downloaded from the official OpenFWI release.

## 2. The Well `helmholtz_staircase` stores its y axis reversed

`dimensions/y` ascends from -0.5 to 3.5, but the array's second axis descends:
physical `y = 3.5 - index * dy`. Confirmed by locating the source singularity,
which lands at `(x, y) = (-0.20, 0.11)` under the reversed convention — inside
the documented source grid `x in {-0.4..0.4}, y in {-0.2..0.4}` — and at
`y = 3.39` under the naive one, which is inside the wall.

The staircase wall therefore lies at low physical `y`, with the domain opening
upward, matching the paper's description.

## 3. The Well `helmholtz_staircase` time axis is redundant

`t0_fields/pressure_{re,im}` carries 50 time steps, but the solution is
`u(x) e^{-i omega t}` exactly. We verified `u(t_1) / u(t_0) = exp(-i omega dt)`
to 7 digits, so only `t = 0` is loaded. The stored convention also means an
outgoing arrival carries `exp(+i omega tau)`; fields are conjugated on load to
match the NumPy transform convention used everywhere else in this repository.

## 4. The three test-split staircase trajectories share one source cell

All three peak at array index `(499, 217)` at low frequency, yet their fields
differ by 22% and 64% in relative sup norm. The test split therefore does not
provide three distinct source positions; the train split has 26 trajectories
and is the one to use for per-source statistics.
