import numpy as np

from wave_lr.spectra import carrier
from wave_lr.tasks import (
    complete_low_rank,
    extrapolate_frequency,
    random_entry_mask,
)


def test_completion_recovers_a_low_rank_matrix() -> None:
    rng = np.random.default_rng(0)
    left = rng.standard_normal((60, 2)) + 1j * rng.standard_normal((60, 2))
    right = rng.standard_normal((2, 40)) + 1j * rng.standard_normal((2, 40))
    field = left @ right
    mask = random_entry_mask(field.shape, 0.6, seed=1)
    filled = complete_low_rank(field, mask, rank=2, iterations=300, device="cpu")
    error = np.linalg.norm(filled - field) / np.linalg.norm(field)
    assert error < 1e-3


def test_carrier_extrapolation_is_exact_for_a_single_arrival() -> None:
    frequencies = np.linspace(2.0, 10.0, 16)
    delays = np.linspace(0.0, 1.5, 40)
    amplitude = 1.0 / (1.0 + delays)
    field = amplitude[:, None] * carrier(frequencies, delays)
    target = np.arange(8, 16)
    predicted = extrapolate_frequency(
        field, frequencies, np.arange(8), target, delays, degree=0
    )
    error = np.linalg.norm(predicted - field[:, target]) / np.linalg.norm(field[:, target])
    assert error < 1e-10
