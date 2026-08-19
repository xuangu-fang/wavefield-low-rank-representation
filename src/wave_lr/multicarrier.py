"""Multi-carrier decomposition: one carrier per resolvable arrival.

The delay-occupancy law says a single carrier can only remove the delay of the
arrival it tracks; whatever else the coda contains still costs ``B`` times its
own occupancy. The remedy the law implies is not a different low-rank
container but *more carriers*:

    u(x, f) = sum_m exp(-2 pi i f tau_m(x)) r_m(x, f),

with each residual ``r_m`` low rank. In a box with reflecting walls the
``tau_m`` are known exactly from image sources, so this can be tested without
any phase optimisation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .spectra import carrier


def image_source_delays(
    coords: NDArray[np.float64],
    source: NDArray[np.float64],
    box: tuple[float, float],
    order: int = 2,
    speed: float = 1.0,
    max_delay: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Arrival times of the image sources of a rectangular Dirichlet box.

    Returns delays of shape ``(n_images, n_x)`` ordered by mean delay, together
    with the reflection order of each image. The solver's outer cells never
    couple to their neighbours and start at zero, so the walls are
    pressure-release and the standard alternating image construction applies.
    """

    length_x, length_y = box
    images, orders = [], []
    for mx in range(-order, order + 1):
        for sx in (1, -1):
            x_image = 2 * mx * length_x + sx * source[0]
            reflections_x = abs(2 * mx) + (0 if sx == 1 else 1)
            for my in range(-order, order + 1):
                for sy in (1, -1):
                    y_image = 2 * my * length_y + sy * source[1]
                    reflections = reflections_x + abs(2 * my) + (0 if sy == 1 else 1)
                    if reflections > 2 * order:
                        continue
                    distance = np.hypot(
                        coords[:, 0] - x_image, coords[:, 1] - y_image
                    )
                    images.append(distance / speed)
                    orders.append(reflections)
    delays = np.stack(images)
    orders = np.asarray(orders)
    # Distinct images can coincide (e.g. a source on an axis); keep unique ones.
    keys = np.round(delays.mean(axis=1), 9)
    _, unique = np.unique(keys, return_index=True)
    delays, orders = delays[np.sort(unique)], orders[np.sort(unique)]
    if max_delay is not None:
        keep = delays.min(axis=1) <= max_delay
        delays, orders = delays[keep], orders[keep]
    order_index = np.argsort(delays.mean(axis=1))
    return delays[order_index], orders[order_index]


def fit_multicarrier(
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    delays: NDArray[np.float64],
    rank: int = 2,
    observed: NDArray[np.bool_] | None = None,
    steps: int = 1500,
    learning_rate: float = 0.05,
    device: str | None = None,
    seed: int = 0,
) -> tuple[NDArray[np.complex128], dict]:
    """Fit ``sum_m carrier_m * (A_m B_m^T)`` by gradient descent.

    Both factors are free complex parameters, so no conjugation is applied.
    ``delays`` is ``(n_carriers, n_x)``. With ``observed`` supplied the fit only
    sees those entries, which turns the routine into carrier-structured matrix
    completion. The parameter count is ``n_carriers * rank * (n_x + n_f)``, so
    comparisons against a plain rank-``R`` model use ``R = n_carriers * rank``.
    """

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device="cpu").manual_seed(seed)

    data = torch.from_numpy(np.ascontiguousarray(field)).to(device)
    carriers = torch.from_numpy(
        np.stack([carrier(frequencies, tau) for tau in delays])
    ).to(device)
    mask = (
        torch.ones_like(data, dtype=torch.bool)
        if observed is None
        else torch.from_numpy(observed).to(device)
    )
    n_carriers, n_x, n_f = carriers.shape
    scale = float(np.abs(field).std()) / max(np.sqrt(rank * n_carriers), 1.0)

    def initial(shape):
        real = torch.randn(shape, generator=generator) * scale
        imaginary = torch.randn(shape, generator=generator) * scale
        return (real + 1j * imaginary).to(device).requires_grad_(True)

    left = initial((n_carriers, n_x, rank))
    right = initial((n_carriers, n_f, rank))
    optimiser = torch.optim.Adam([left, right], lr=learning_rate)
    target = data * mask
    norm = torch.linalg.norm(target)

    history = []
    for step in range(steps):
        optimiser.zero_grad(set_to_none=True)
        estimate = (carriers * (left @ right.transpose(1, 2))).sum(dim=0)
        loss = torch.linalg.norm((estimate - data) * mask) / norm
        loss.backward()
        optimiser.step()
        if step % 100 == 0:
            history.append(float(loss.detach()))
    with torch.no_grad():
        estimate = (carriers * (left @ right.transpose(1, 2))).sum(dim=0)
    info = {
        "parameters": int(n_carriers * rank * (n_x + n_f)),
        "equivalent_rank": int(n_carriers * rank),
        "loss_history": history,
        "final_observed_loss": float(loss.detach()),
    }
    return estimate.detach().cpu().numpy(), info


def fit_multicarrier_als(
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    delays: NDArray[np.float64],
    rank: int = 2,
    observed: NDArray[np.bool_] | None = None,
    sweeps: int = 20,
    ridge: float = 1e-6,
    device: str | None = None,
    seed: int = 0,
) -> tuple[NDArray[np.complex128], dict]:
    """Alternating least squares for the multi-carrier model.

    With one factor held fixed the model is linear in the other, so each half
    sweep is a batch of small normal-equation solves. Unlike a gradient fit
    this is monotone, which matters because the carriers are far from
    orthogonal and a first-order optimiser diverges at larger budgets.
    """

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device="cpu").manual_seed(seed)

    data = torch.from_numpy(np.ascontiguousarray(field)).to(device).to(torch.complex64)
    carriers = (
        torch.from_numpy(np.stack([carrier(frequencies, tau) for tau in delays]))
        .to(device)
        .to(torch.complex64)
    )
    n_carriers, n_x, n_f = carriers.shape
    width = n_carriers * rank
    mask = (
        torch.ones_like(data, dtype=torch.float32)
        if observed is None
        else torch.from_numpy(observed).to(device).to(torch.float32)
    )

    # Seed each carrier with the leading modes of the field aligned to it; ALS
    # is monotone but converges linearly, and a random start needs many more
    # sweeps to reach the same residual.
    factor_b = torch.empty((n_carriers, n_f, rank), dtype=torch.complex64, device=device)
    for index in range(n_carriers):
        aligned = data * carriers[index].conj()
        _, values, right = torch.svd_lowrank(aligned, q=min(rank + 4, min(aligned.shape)))
        weights = torch.sqrt(values[:rank] / n_carriers)
        factor_b[index] = right[:, :rank] * weights
    jitter = (
        torch.randn((n_carriers, n_f, rank), generator=generator)
        + 1j * torch.randn((n_carriers, n_f, rank), generator=generator)
    ).to(device).to(torch.complex64)
    factor_b = factor_b + 1e-3 * factor_b.abs().mean() * jitter

    eye = torch.eye(width, dtype=torch.complex64, device=device)
    norm = torch.linalg.norm(data * mask)
    history = []

    def solve(design, values, weights):
        """Ridge-regularised least squares, batched over the leading axis."""

        weighted = design * weights[..., None]
        gram = torch.einsum("bnp,bnq->bpq", weighted.conj(), design)
        rhs = torch.einsum("bnp,bn->bp", weighted.conj(), values)
        norms = torch.linalg.matrix_norm(gram, keepdim=True)
        # A location with (almost) no observed entries has a zero Gram matrix,
        # so the damping needs an absolute floor as well as a relative term.
        damping = ridge * norms + 1e-6 * norms.mean()
        return torch.linalg.solve(gram + damping * eye, rhs)

    estimate = torch.zeros_like(data)
    for _ in range(sweeps):
        design = torch.einsum("mxf,mfj->xfmj", carriers, factor_b).reshape(n_x, n_f, width)
        factor_a = solve(design, data, mask).reshape(n_x, n_carriers, rank).permute(1, 0, 2)
        design = torch.einsum("mxf,mxj->fxmj", carriers, factor_a).reshape(n_f, n_x, width)
        factor_b = (
            solve(design, data.T, mask.T).reshape(n_f, n_carriers, rank).permute(1, 0, 2)
        )
        estimate = (carriers * (factor_a @ factor_b.transpose(1, 2))).sum(dim=0)
        history.append(float(torch.linalg.norm((estimate - data) * mask) / norm))

    return estimate.cpu().numpy().astype(np.complex128), {
        "parameters": int(width * (n_x + n_f)),
        "equivalent_rank": int(width),
        "loss_history": history,
        "final_observed_loss": history[-1] if history else float("nan"),
    }


def _pick_peak_times(traces: NDArray[np.float64], times: NDArray[np.float64]):
    """Envelope peak time and its amplitude at every location."""

    envelope = np.abs(traces)
    index = np.argmax(envelope, axis=1)
    return times[index], envelope[np.arange(len(index)), index]


def fit_virtual_source(
    coords: NDArray[np.float64],
    picked: NDArray[np.float64],
    weights: NDArray[np.float64],
    speed: float = 1.0,
    span: float = 2.0,
    grid: int = 61,
    refinements: int = 3,
) -> tuple[NDArray[np.float64], float, float]:
    """Fit an arrival surface ``|x - p| / c + t0`` to picked times.

    A specular reflection arrives as though it came from an image source, so a
    two-parameter position plus a time offset describes a whole reflected
    wavefront. Positions are searched on a grid that extends outside the domain
    because image sources live there, then refined locally.
    """

    weights = np.asarray(weights, dtype=float)
    weights = weights / max(weights.sum(), 1e-30)
    lo = coords.min(axis=0) - span
    hi = coords.max(axis=0) + span
    best = None
    for _ in range(refinements):
        axis_x = np.linspace(lo[0], hi[0], grid)
        axis_y = np.linspace(lo[1], hi[1], grid)
        candidates = np.stack(np.meshgrid(axis_x, axis_y, indexing="ij"), axis=-1).reshape(-1, 2)
        for start in range(0, len(candidates), 512):
            block = candidates[start : start + 512]
            distance = np.linalg.norm(coords[None, :, :] - block[:, None, :], axis=2) / speed
            offset = ((picked[None, :] - distance) * weights[None, :]).sum(axis=1)
            residual = picked[None, :] - distance - offset[:, None]
            cost = (weights[None, :] * residual**2).sum(axis=1)
            index = int(np.argmin(cost))
            if best is None or cost[index] < best[2]:
                best = (block[index], float(offset[index]), float(cost[index]))
        step = np.array([axis_x[1] - axis_x[0], axis_y[1] - axis_y[0]])
        lo, hi = best[0] - 2 * step, best[0] + 2 * step
    return best[0], best[1], float(np.sqrt(best[2]))


def scan_virtual_sources(
    field: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    coords: NDArray[np.float64],
    speed: float = 1.0,
    span: float = 2.0,
    grid: int = 61,
    refinements: int = 3,
    sample: int = 512,
    objective: str = "rank1",
    device: str | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Find the virtual source whose wavefront the field stacks along best.

    Picking an arrival per location fails once the coda is dense, because the
    envelope peak of a reverberant residual is noise. Coherent stacking asks a
    different question -- for which wavefront shape does the field add up in
    phase -- and stays informative when no single arrival dominates. In the
    frequency domain the stack is ``sum_x u(x,f) exp(+2 pi i f tau_p(x))``,
    whose magnitude is unaffected by the unknown time offset. ``objective``
    selects between the plain stack and the leading singular value of the
    aligned block, which tolerates amplitude taper and polarity flips.
    """

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = np.linspace(0, len(coords) - 1, min(sample, len(coords))).astype(int)
    values = torch.from_numpy(np.ascontiguousarray(field[rows])).to(device).to(torch.complex64)
    positions = torch.from_numpy(np.ascontiguousarray(coords[rows])).to(device).float()
    freq = torch.from_numpy(np.ascontiguousarray(frequencies)).to(device).float()
    total = float(torch.linalg.norm(values) ** 2) * len(rows)

    lo = coords.min(axis=0) - span
    hi = coords.max(axis=0) + span
    best = (np.zeros(2), -1.0)
    for _ in range(refinements):
        axis_x = np.linspace(lo[0], hi[0], grid)
        axis_y = np.linspace(lo[1], hi[1], grid)
        candidates = np.stack(
            np.meshgrid(axis_x, axis_y, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        block_size = max(1, 2_000_000 // (len(rows) * len(frequencies)) or 1)
        for start in range(0, len(candidates), block_size):
            block = torch.from_numpy(candidates[start : start + block_size]).to(device).float()
            delays = torch.cdist(block, positions) / speed
            phase = torch.exp(2j * np.pi * delays[:, :, None] * freq[None, None, :])
            aligned = values[None] * phase
            if objective == "stack":
                power = (aligned.sum(dim=1).abs() ** 2).sum(dim=1) / total
            else:
                # Leading singular value by two power iterations, started from
                # the uniform vector: unlike a plain stack this survives the
                # amplitude taper and the polarity flip of a wall reflection.
                left = torch.ones(
                    aligned.shape[0], aligned.shape[1], dtype=aligned.dtype, device=device
                )
                for _ in range(2):
                    right = torch.einsum("bxf,bx->bf", aligned.conj(), left)
                    right = right / right.norm(dim=1, keepdim=True).clamp_min(1e-20)
                    left = torch.einsum("bxf,bf->bx", aligned, right.conj())
                    left = left / left.norm(dim=1, keepdim=True).clamp_min(1e-20)
                right = torch.einsum("bxf,bx->bf", aligned.conj(), left)
                power = (right.abs() ** 2).sum(dim=1) / total
            index = int(torch.argmax(power))
            if float(power[index]) > best[1]:
                best = (candidates[start : start + block_size][index], float(power[index]))
        step = np.array([axis_x[1] - axis_x[0], axis_y[1] - axis_y[0]])
        lo, hi = best[0] - 2 * step, best[0] + 2 * step
    return best[0], best[1]


def estimate_carriers(
    spectrum,
    n_carriers: int = 4,
    rank: int = 2,
    speed: float = 1.0,
    seed_delays: NDArray[np.float64] | None = None,
    coords: NDArray[np.float64] | None = None,
    sweeps: int = 20,
    objective: str = "stack",
    budget: int = 24,
    tolerance: float = 0.995,
) -> tuple[NDArray[np.float64], list[dict]]:
    """Grow a carrier bank from the data alone, one arrival at a time.

    Each round refits the whole multi-carrier model, scans the *model* residual
    for the wavefront it stacks along best, and adds that virtual source. A
    carrier is kept only if it improves the fit at a *fixed* total parameter
    budget -- splitting ``budget`` across one more carrier lowers the rank each
    one gets, so a useless carrier makes the model worse and is rejected. No
    geometry, boundary description or image-source construction is used.
    """

    from .spectra import Spectrum, band_limited_traces, shift_spectrum
    from .theory import occupancy_from_traces

    if coords is None:
        raise ValueError("coords are required to fit virtual sources")

    def occupancy_of(values: NDArray[np.complex128], tau: NDArray[np.float64]) -> float:
        block = Spectrum(values, spectrum.frequencies, spectrum.dt, spectrum.n_padded)
        traces, _ = band_limited_traces(shift_spectrum(block, tau - tau.max()))
        return occupancy_from_traces(traces, spectrum.dt, spectrum.bandwidth)

    def next_carrier(residual: NDArray[np.complex128]) -> tuple[NDArray[np.float64], dict]:
        position, power = scan_virtual_sources(
            residual, spectrum.frequencies, coords, speed=speed, objective=objective
        )
        distance = np.linalg.norm(coords - position, axis=1) / speed
        block = Spectrum(residual, spectrum.frequencies, spectrum.dt, spectrum.n_padded)
        traces, times = band_limited_traces(shift_spectrum(block, distance - distance.max()))
        picked, amplitude = _pick_peak_times(traces, times)
        weights = amplitude / max(amplitude.sum(), 1e-30)
        offset = float((picked * weights).sum()) - float(distance.max())
        return distance + offset, {"virtual_source": position.tolist(), "stack_power": power}

    delays: list[NDArray[np.float64]] = []
    diagnostics: list[dict] = []
    residual = spectrum.values
    if seed_delays is not None:
        delays.append(np.asarray(seed_delays, dtype=float))
        diagnostics.append({"carrier": 0, "virtual_source": None, "source": "seed"})
    else:
        tau, info = next_carrier(residual)
        delays.append(tau)
        diagnostics.append({"carrier": 0, "source": "scan", **info})

    norm = np.linalg.norm(spectrum.values)

    def budget_error(bank: list[NDArray[np.float64]]) -> tuple[float, NDArray[np.complex128]]:
        share = max(budget // len(bank), 1)
        estimate, _ = fit_multicarrier_als(
            spectrum.values, spectrum.frequencies, np.stack(bank), rank=share, sweeps=sweeps
        )
        return float(np.linalg.norm(estimate - spectrum.values) / norm), estimate

    best_error, estimate = budget_error(delays)
    diagnostics[0]["budget_error"] = best_error
    diagnostics[0]["residual_occupancy"] = occupancy_of(spectrum.values, delays[0])
    while len(delays) < n_carriers and budget // (len(delays) + 1) >= 1:
        residual = spectrum.values - estimate
        tau, info = next_carrier(residual)
        candidate = delays + [tau]
        error, trial = budget_error(candidate)
        entry = {
            "carrier": len(delays),
            "source": "scan",
            "residual_energy": float(np.linalg.norm(residual) / norm),
            "budget_error": error,
            "previous_budget_error": best_error,
            "residual_occupancy": occupancy_of(residual, tau),
            **info,
        }
        if error > tolerance * best_error:
            entry["accepted"] = False
            diagnostics.append(entry)
            break
        entry["accepted"] = True
        delays, best_error, estimate = candidate, error, trial
        diagnostics.append(entry)
    return np.stack(delays), diagnostics
