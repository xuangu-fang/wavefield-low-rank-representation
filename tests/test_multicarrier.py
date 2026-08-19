import itertools

import numpy as np

from wave_lr.multicarrier import (
    fit_multicarrier,
    fit_multicarrier_als,
    image_source_delays,
)
from wave_lr.spectra import carrier


def test_image_sources_include_the_direct_arrival() -> None:
    coords = np.array([[0.5, 0.5], [0.25, 0.75]])
    delays, orders = image_source_delays(coords, np.array([0.5, 0.25]), (1.0, 1.0), order=1)
    assert orders[0] == 0
    assert np.isclose(delays[0, 0], 0.25)
    assert delays.shape[0] > 4


def test_multicarrier_fit_recovers_a_two_arrival_field() -> None:
    rng = np.random.default_rng(0)
    frequencies = np.linspace(5.0, 15.0, 32)
    tau = np.stack([np.linspace(0.0, 0.4, 64), np.linspace(0.6, 0.9, 64)])
    amplitudes = rng.standard_normal((2, 64)) + 1j * rng.standard_normal((2, 64))
    field = sum(
        amplitudes[m][:, None] * carrier(frequencies, tau[m]) for m in range(2)
    )
    estimate, info = fit_multicarrier(field, frequencies, tau, rank=1, steps=1500)
    error = np.linalg.norm(estimate - field) / np.linalg.norm(field)
    assert error < 0.02
    assert info["equivalent_rank"] == 2


def test_als_fit_is_monotone_and_accurate() -> None:
    rng = np.random.default_rng(1)
    frequencies = np.linspace(5.0, 15.0, 32)
    tau = np.stack([np.linspace(0.0, 0.4, 64), np.linspace(0.6, 0.9, 64)])
    amplitudes = rng.standard_normal((2, 64)) + 1j * rng.standard_normal((2, 64))
    field = sum(
        amplitudes[m][:, None] * carrier(frequencies, tau[m]) for m in range(2)
    )
    estimate, info = fit_multicarrier_als(field, frequencies, tau, rank=1, sweeps=30)
    history = info["loss_history"]
    assert history[-1] < 1e-4
    assert all(later <= earlier + 1e-5 for earlier, later in itertools.pairwise(history))
    error = np.linalg.norm(estimate - field) / np.linalg.norm(field)
    assert error < 1e-4
