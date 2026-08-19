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
