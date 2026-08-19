import numpy as np

from wave_lr.eikonal import batched_travel_time, fast_sweeping


def _exact_distance(n: int, spacing: float) -> np.ndarray:
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return spacing * np.hypot(rows - n // 2, cols - n // 2)


def test_reference_solver_is_first_order_accurate() -> None:
    n = 41
    spacing = 1.0 / (n - 1)
    tau = fast_sweeping(np.ones((n, n)), [(n // 2, n // 2)], spacing=spacing)
    exact = _exact_distance(n, spacing)
    mask = (exact > 0.05) & (exact < 0.4)
    # Unseeded first-order Godunov sweeping overestimates diagonal paths.
    assert np.max((np.abs(tau - exact) / exact)[mask]) < 0.16
    axis = tau[n // 2, n // 2 :]
    assert np.allclose(axis, exact[n // 2, n // 2 :], atol=1e-9)


def test_seeded_batched_solver_is_accurate() -> None:
    n = 64
    spacing = 1.0 / (n - 1)
    masks = np.zeros((1, n, n), dtype=bool)
    masks[0, n // 2, n // 2] = True
    tau = batched_travel_time(
        np.ones((1, n, n)), masks, spacing=spacing, iterations=200, device="cpu"
    )[0]
    exact = _exact_distance(n, spacing)
    mask = (exact > 0.05) & (exact < 0.45)
    assert np.max((np.abs(tau - exact) / exact)[mask]) < 0.02


def test_batched_solver_matches_reference_without_seeding() -> None:
    rng = np.random.default_rng(0)
    speed = 0.5 + rng.random((2, 25, 25))
    masks = np.zeros((2, 25, 25), dtype=bool)
    masks[:, 12, 12] = True
    batched = batched_travel_time(
        speed, masks, spacing=0.04, device="cpu", seed_radius=0
    )
    for index in range(2):
        reference = fast_sweeping(speed[index], [(12, 12)], spacing=0.04)
        assert np.allclose(batched[index], reference, rtol=1e-4, atol=1e-6)
