"""Shifted POD: the closest prior art, implemented for a fair comparison.

Reiss et al. decompose a transport-dominated field into a few co-moving frames,

    u(x, t) = sum_k T^{-delta_k(t)} q_k(x, t),

where each ``q_k`` is low rank once the rigid shift ``delta_k(t)`` is undone.
That is a *spatial* warp shared by the whole snapshot. The carrier used in this
repository is instead a *temporal* warp that differs per location. The two
coincide for a rigidly translating pattern and diverge for an expanding or
refracted wavefront, which is what the experiments here measure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _fft_shift(frames, shifts):
    """Sub-pixel rigid shift of each snapshot by its own 2-D offset."""

    import torch

    _, height, width = frames.shape
    ky = torch.fft.fftfreq(height, device=frames.device)[None, :, None]
    kx = torch.fft.rfftfreq(width, device=frames.device)[None, None, :]
    phase = torch.exp(
        -2j * np.pi * (shifts[:, 0, None, None] * ky + shifts[:, 1, None, None] * kx)
    )
    return torch.fft.irfft2(torch.fft.rfft2(frames) * phase, s=(height, width))


def _best_rank(frames, rank: int):
    """Rank-``r`` truncation of the snapshot matrix, and its residual."""

    import torch

    n_t = frames.shape[0]
    matrix = frames.reshape(n_t, -1)
    left, values, right = torch.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, values.numel())
    approx = (left[:, :rank] * values[:rank]) @ right[:rank]
    return approx.reshape(frames.shape)


def estimate_shift_path(frames, reference, search: int = 0, device: str | None = None):
    """Rigid shift per snapshot, by FFT cross-correlation with a template.

    The correlation surface for every lag comes from one transform pair, and a
    parabolic fit around its peak recovers the sub-sample offset. Fractional
    misalignment otherwise dominates the residual and would make this baseline
    look far weaker than it is.
    """

    import torch

    n_t, height, width = frames.shape
    spectrum = torch.fft.rfft2(frames) * torch.conj(torch.fft.rfft2(reference))
    correlation = torch.fft.irfft2(spectrum, s=(height, width))
    flat = correlation.reshape(n_t, -1)
    peak = torch.argmax(flat, dim=1)
    rows = torch.div(peak, width, rounding_mode="floor")
    cols = peak % width

    def parabolic(values_minus, values_zero, values_plus):
        """Vertex of the parabola through three samples, clamped to +-0.5."""

        denominator = values_minus - 2.0 * values_zero + values_plus
        offset = 0.5 * (values_minus - values_plus) / torch.where(
            denominator.abs() < 1e-12, torch.full_like(denominator, 1e-12), denominator
        )
        return torch.clamp(offset, -0.5, 0.5)

    index = torch.arange(n_t, device=frames.device)
    row_offset = parabolic(
        correlation[index, (rows - 1) % height, cols],
        correlation[index, rows, cols],
        correlation[index, (rows + 1) % height, cols],
    )
    col_offset = parabolic(
        correlation[index, rows, (cols - 1) % width],
        correlation[index, rows, cols],
        correlation[index, rows, (cols + 1) % width],
    )
    # Lags beyond half the axis wrap to negative shifts.
    row_shift = torch.where(rows > height // 2, rows - height, rows).float() + row_offset
    col_shift = torch.where(cols > width // 2, cols - width, cols).float() + col_offset
    return torch.stack([row_shift, col_shift], dim=1)


def shifted_pod(
    field: NDArray[np.float64],
    n_frames: int = 2,
    total_rank: int = 8,
    sweeps: int = 6,
    search: int = 12,
    device: str | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """Decompose ``(n_t, H, W)`` into ``n_frames`` co-moving low-rank frames.

    The total rank is split evenly across frames so the parameter budget matches
    a plain rank-``total_rank`` POD. Shift paths are re-estimated between
    sweeps from the current residual, so no transport information is supplied.
    """

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.from_numpy(np.ascontiguousarray(field)).float().to(device)
    n_t = data.shape[0]
    per_frame = max(total_rank // n_frames, 1)

    shifts = [torch.zeros(n_t, 2, device=device) for _ in range(n_frames)]
    components = [torch.zeros_like(data) for _ in range(n_frames)]

    def anchor_of(frames):
        """Use the highest-energy snapshot as the template for that frame."""

        energy = (frames**2).sum(dim=(1, 2))
        return frames[int(torch.argmax(energy))][None]

    for index in range(n_frames):
        # Every frame gets its own transport, estimated from what is left over.
        residual = data - sum(components)
        shifts[index] = estimate_shift_path(residual, anchor_of(residual))

    history = []
    for _ in range(sweeps):
        for index in range(n_frames):
            residual = data - sum(
                components[j] for j in range(n_frames) if j != index
            )
            co_moving = _fft_shift(residual, -shifts[index])
            truncated = _best_rank(co_moving, per_frame)
            components[index] = _fft_shift(truncated, shifts[index])
        reconstruction = sum(components)
        history.append(
            float(torch.linalg.norm(reconstruction - data) / torch.linalg.norm(data))
        )
        for index in range(n_frames):
            residual = data - sum(
                components[j] for j in range(n_frames) if j != index
            )
            shifts[index] = estimate_shift_path(residual, anchor_of(residual))

    reconstruction = sum(components).cpu().numpy()
    return reconstruction, {
        "error_history": history,
        "per_frame_rank": per_frame,
        "equivalent_rank": per_frame * n_frames,
        "shift_ranges": [
            float(torch.abs(shift).max()) for shift in shifts
        ],
    }


def plain_pod(field: NDArray[np.float64], rank: int, device: str | None = None):
    """Rank-``r`` POD of the snapshot matrix, the fixed-frame reference."""

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.from_numpy(np.ascontiguousarray(field)).float().to(device)
    return _best_rank(data, rank).cpu().numpy()


def carrier_pod(
    traces: NDArray[np.float64],
    delays: NDArray[np.float64],
    dt: float,
    rank: int,
    device: str | None = None,
) -> NDArray[np.float64]:
    """Our counterpart: warp each location in time, then take a rank-``r`` POD.

    ``traces`` is ``(n_x, n_t)``. The warp is applied by a Fourier phase ramp so
    it is sub-sample accurate, and undone after truncation.
    """

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.from_numpy(np.ascontiguousarray(traces)).float().to(device)
    n_t = data.shape[1]
    padded = 2 * n_t
    tau = torch.from_numpy(np.ascontiguousarray(delays)).float().to(device)
    freq = torch.fft.rfftfreq(padded, dt, device=device)[None, :]
    ramp = torch.exp(2j * np.pi * freq * (tau[:, None] - tau.max()))

    spectrum = torch.fft.rfft(data, n=padded, dim=1) * ramp
    aligned = torch.fft.irfft(spectrum, n=padded, dim=1)
    left, values, right = torch.linalg.svd(aligned, full_matrices=False)
    rank = min(rank, values.numel())
    truncated = (left[:, :rank] * values[:rank]) @ right[:rank]
    restored = torch.fft.irfft(
        torch.fft.rfft(truncated, n=padded, dim=1) * torch.conj(ramp), n=padded, dim=1
    )
    return restored[:, :n_t].cpu().numpy()
