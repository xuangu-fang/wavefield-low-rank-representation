import numpy as np

from wave_lr.phase import paired_phase_carriers, phase_embedding, wrap_phase


def test_paired_carriers_reconstruct_traveling_sine_and_cosine() -> None:
    distance = np.linspace(0.1, 1.2, 9)[:, None]
    time = np.linspace(0.0, 0.7, 7)[None, :]
    k = np.array([2.3, 5.1])
    c = np.array([0.8, 1.4])
    carriers = paired_phase_carriers(distance, time, k, c)
    phase = k[:, None] * (
        distance[..., None, None] - c[None, None, None, :] * time[..., None, None]
    )
    np.testing.assert_allclose(carriers[..., 0] + carriers[..., 1], np.cos(phase))
    np.testing.assert_allclose(carriers[..., 3] - carriers[..., 2], np.sin(phase))


def test_phase_embedding_is_branch_cut_safe() -> None:
    left = phase_embedding(np.pi - 1e-7)
    right = phase_embedding(-np.pi + 1e-7)
    assert np.linalg.norm(left - right) < 1e-6
    np.testing.assert_allclose(wrap_phase([3 * np.pi, -3 * np.pi]), [-np.pi, -np.pi])

