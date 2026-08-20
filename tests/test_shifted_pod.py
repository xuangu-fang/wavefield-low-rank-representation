import numpy as np

from wave_lr.shifted_pod import plain_pod, shifted_pod


def _translating_pulse(n_t: int = 48, size: int = 48) -> np.ndarray:
    """A rigidly translating Gaussian: the case shifted POD is designed for."""

    axis = np.arange(size)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    frames = []
    for step in range(n_t):
        centre_y = 10 + 0.5 * step
        centre_x = 10 + 0.4 * step
        frames.append(np.exp(-((yy - centre_y) ** 2 + (xx - centre_x) ** 2) / 12.0))
    return np.stack(frames)


def test_shifted_pod_beats_plain_pod_on_a_translating_pulse() -> None:
    field = _translating_pulse()
    norm = np.linalg.norm(field)
    plain = np.linalg.norm(plain_pod(field, rank=4) - field) / norm
    moved, info = shifted_pod(field, n_frames=1, total_rank=2, sweeps=4)
    moved_error = np.linalg.norm(moved - field) / norm
    # One co-moving frame captures a rigid translation to numerical precision,
    # so the baseline is at full strength where its assumption holds.
    assert moved_error < 1e-3
    assert plain > 0.4
    assert info["equivalent_rank"] == 2
