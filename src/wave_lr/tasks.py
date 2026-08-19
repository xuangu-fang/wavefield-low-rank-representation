"""Downstream tasks that turn a representation claim into a measurable one.

Both tasks keep the algorithm, the observations and the parameter budget
fixed, and vary only the coordinates in which the field is expressed. Any
difference is therefore attributable to the representation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import summarize
from .spectra import carrier

RANK_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def _as_torch(array: np.ndarray, device: str):
    import torch

    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def complete_low_rank(
    field: NDArray[np.complex128],
    observed: NDArray[np.bool_],
    rank: int,
    iterations: int = 200,
    device: str | None = None,
) -> NDArray[np.complex128]:
    """Rank-``r`` matrix completion by iterative hard thresholding."""

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = _as_torch(field, device)
    mask = _as_torch(observed, device)
    estimate = torch.where(mask, data, torch.zeros_like(data))
    use_randomized = rank + 8 < min(estimate.shape) // 2
    for _ in range(iterations):
        if use_randomized:
            u, s, v = torch.svd_lowrank(estimate, q=rank + 8, niter=2)
            estimate = (u[:, :rank] * s[:rank]) @ v[:, :rank].conj().T
        else:
            u, s, vh = torch.linalg.svd(estimate, full_matrices=False)
            estimate = (u[:, :rank] * s[:rank]) @ vh[:rank]
        estimate = torch.where(mask, data, estimate)
    return estimate.cpu().numpy()


def completion_curve(
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    delays: NDArray[np.float64] | None,
    observed: NDArray[np.bool_],
    ranks: tuple[int, ...] = RANK_GRID,
    iterations: int = 200,
) -> dict:
    """Complete in carrier coordinates and score in the original coordinates.

    ``delays=None`` completes the raw field. Otherwise the field is aligned,
    completed, and mapped back, so the reported error is always a physical
    error on the same target.
    """

    ramp = None if delays is None else np.conj(carrier(frequencies, delays))
    working = field if ramp is None else field * ramp
    hidden = ~observed
    out: dict[str, float] = {}
    best = None
    for rank in ranks:
        if rank > min(working.shape):
            continue
        filled = complete_low_rank(working, observed, rank, iterations=iterations)
        restored = filled if ramp is None else filled * np.conj(ramp)
        scores = summarize(restored[hidden], field[hidden])
        out[f"rank{rank}_complex_nrmse"] = scores["complex_nrmse"]
        out[f"rank{rank}_awpc"] = scores["awpc"]
        if best is None or scores["complex_nrmse"] < best[1]["complex_nrmse"]:
            best = (rank, scores)
    out["best_rank"] = float(best[0])
    for key, value in best[1].items():
        out[f"best_{key}"] = value
    return out


def _fit_polynomial(
    values: NDArray[np.complex128], nodes: NDArray[np.float64], degree: int
) -> NDArray[np.complex128]:
    """Least-squares complex polynomial coefficients along the last axis."""

    scaled = (nodes - nodes.mean()) / max(nodes.std(), 1e-12)
    design = np.stack([scaled**p for p in range(degree + 1)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, values.T, rcond=None)
    return coefficients, scaled.mean(), max(nodes.std(), 1e-12), nodes.mean()


def _evaluate_polynomial(coefficients, nodes, mean, std, degree: int):
    scaled = (nodes - mean) / std
    design = np.stack([scaled**p for p in range(degree + 1)], axis=1)
    return (design @ coefficients).T


def extrapolate_frequency(
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    train: slice | NDArray[np.int_],
    target: NDArray[np.int_],
    delays: NDArray[np.float64] | None,
    degree: int = 2,
    mode: str = "complex",
) -> NDArray[np.complex128]:
    """Predict unseen higher frequencies from the observed low band.

    ``mode`` selects what is extrapolated: the complex values (``"complex"``),
    or log-amplitude and unwrapped phase separately (``"amplitude_phase"``),
    the split advocated for cross-frequency transfer in the APEX line of work.
    """

    ramp = None if delays is None else np.conj(carrier(frequencies, delays))
    working = field if ramp is None else field * ramp
    train_f = frequencies[train]
    target_f = frequencies[target]

    if mode == "complex":
        coefficients, _, std, mean = _fit_polynomial(working[:, train], train_f, degree)
        predicted = _evaluate_polynomial(coefficients, target_f, mean, std, degree)
    elif mode == "amplitude_phase":
        amplitude = np.log(np.abs(working[:, train]) + 1e-12)
        phase = np.unwrap(np.angle(working[:, train]), axis=1)
        coefficients, _, std, mean = _fit_polynomial(
            amplitude.astype(np.complex128), train_f, degree
        )
        log_amp = np.real(_evaluate_polynomial(coefficients, target_f, mean, std, degree))
        coefficients, _, std, mean = _fit_polynomial(
            phase.astype(np.complex128), train_f, degree
        )
        angle = np.real(_evaluate_polynomial(coefficients, target_f, mean, std, degree))
        predicted = np.exp(log_amp) * np.exp(1j * angle)
    else:
        raise ValueError(f"unknown mode {mode}")

    if ramp is not None:
        predicted = predicted * np.conj(ramp[:, target])
    return predicted


def extrapolate_low_rank(
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    train: NDArray[np.int_],
    target: NDArray[np.int_],
    delays: NDArray[np.float64] | None,
    rank: int = 2,
    degree: int = 2,
) -> NDArray[np.complex128]:
    """Continue a field in frequency through a rank-``r`` frequency basis.

    The delay-occupancy law says a band holds only ``bandwidth x occupancy``
    degrees of freedom along the frequency axis, so the field should be
    continued through that many smooth frequency curves rather than through
    one independent fit per location. Alignment is what makes the count small.
    """

    ramp = None if delays is None else np.conj(carrier(frequencies, delays))
    working = field if ramp is None else field * ramp
    left, values, right = np.linalg.svd(working[:, train], full_matrices=False)
    rank = min(rank, values.size)
    basis = left[:, :rank]
    coordinates = (values[:rank, None] * right[:rank])  # (rank, n_train)

    coefficients, _, std, mean = _fit_polynomial(coordinates, frequencies[train], degree)
    extended = _evaluate_polynomial(coefficients, frequencies[target], mean, std, degree)
    predicted = basis @ extended
    if ramp is not None:
        predicted = predicted * np.conj(ramp[:, target])
    return predicted


def low_rank_extrapolation_report(
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    n_train: int,
    delays: NDArray[np.float64] | None,
    ranks: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    degrees: tuple[int, ...] = (0, 1, 2, 3),
) -> dict:
    """Best (rank, degree) frequency continuation for one representation."""

    train = np.arange(n_train)
    target = np.arange(n_train, len(frequencies))
    truth = field[:, target]
    out: dict[str, float] = {}
    best = None
    for rank in ranks:
        if rank > n_train:
            continue
        for degree in degrees:
            if degree + 1 > n_train:
                continue
            predicted = extrapolate_low_rank(
                field, frequencies, train, target, delays, rank=rank, degree=degree
            )
            scores = summarize(predicted, truth)
            out[f"rank{rank}_degree{degree}_complex_nrmse"] = scores["complex_nrmse"]
            if best is None or scores["complex_nrmse"] < best[2]["complex_nrmse"]:
                best = (rank, degree, scores)
    out["best_rank"] = float(best[0])
    out["best_degree"] = float(best[1])
    for key, value in best[2].items():
        out[f"best_{key}"] = value
    return out


def extrapolation_report(
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    n_train: int,
    delays: NDArray[np.float64] | None,
    degrees: tuple[int, ...] = (0, 1, 2, 3),
    mode: str = "complex",
) -> dict:
    """Best-degree extrapolation scores for one representation."""

    train = np.arange(n_train)
    target = np.arange(n_train, len(frequencies))
    truth = field[:, target]
    out: dict[str, float] = {}
    best = None
    for degree in degrees:
        if degree + 1 > n_train:
            continue
        predicted = extrapolate_frequency(
            field, frequencies, train, target, delays, degree=degree, mode=mode
        )
        scores = summarize(predicted, truth)
        for key, value in scores.items():
            out[f"degree{degree}_{key}"] = value
        if best is None or scores["complex_nrmse"] < best[1]["complex_nrmse"]:
            best = (degree, scores)
    out["best_degree"] = float(best[0])
    for key, value in best[1].items():
        out[f"best_{key}"] = value
    return out


def copy_last_baseline(
    field: NDArray[np.complex128], n_train: int
) -> dict:
    """Trivial control: reuse the highest observed frequency for every target."""

    target = np.arange(n_train, field.shape[1])
    predicted = np.repeat(field[:, [n_train - 1]], len(target), axis=1)
    return summarize(predicted, field[:, target])


def random_entry_mask(
    shape: tuple[int, int], fraction: float, seed: int = 0
) -> NDArray[np.bool_]:
    rng = np.random.default_rng(seed)
    return rng.random(shape) < fraction


def sensor_mask(shape: tuple[int, int], fraction: float, seed: int = 0) -> NDArray[np.bool_]:
    """Whole spatial locations observed at every frequency."""

    rng = np.random.default_rng(seed)
    keep = rng.random(shape[0]) < fraction
    mask = np.zeros(shape, dtype=bool)
    mask[keep] = True
    return mask


def interpolate_from_sensors(
    coords: NDArray[np.float64],
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    delays: NDArray[np.float64] | None,
    observed: NDArray[np.bool_],
) -> NDArray[np.complex128]:
    """Reconstruct a field from scattered sensors by linear interpolation.

    A carrier removes the fast spatial oscillation ``exp(-2 pi i f tau(x))``,
    leaving an envelope that varies on the scale of the medium rather than the
    wavelength. Interpolation therefore stops aliasing once sensors are sparser
    than half a wavelength -- the spatial counterpart of the delay-occupancy
    argument.
    """

    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    ramp = None if delays is None else np.conj(carrier(frequencies, delays))
    working = field if ramp is None else field * ramp
    sensors = coords[observed]
    linear = LinearNDInterpolator(sensors, working[observed])
    filled = linear(coords)
    holes = ~np.isfinite(filled).all(axis=1)
    if holes.any():
        nearest = NearestNDInterpolator(sensors, working[observed])
        filled[holes] = nearest(coords[holes])
    if ramp is not None:
        filled = filled * np.conj(ramp)
    return filled


def sensor_interpolation_report(
    coords: NDArray[np.float64],
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    delays: NDArray[np.float64] | None,
    fraction: float,
    seed: int = 0,
) -> dict:
    """Score sensor interpolation on the locations that were not observed."""

    rng = np.random.default_rng(seed)
    observed = rng.random(coords.shape[0]) < fraction
    if observed.sum() < 8:
        raise ValueError("too few sensors")
    predicted = interpolate_from_sensors(coords, field, frequencies, delays, observed)
    hidden = ~observed
    scores = summarize(predicted[hidden], field[hidden])
    scores["n_sensors"] = int(observed.sum())
    return scores
