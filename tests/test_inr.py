import numpy as np

from wave_lr.inr import TrainConfig, fit_field_network
from wave_lr.spectra import carrier


def test_carrier_network_beats_direct_network_on_an_oscillatory_field() -> None:
    rng = np.random.default_rng(0)
    grid = np.linspace(0.0, 1.0, 48)
    coords = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    frequencies = np.linspace(20.0, 26.0, 8)
    delays = np.linalg.norm(coords - 0.5, axis=1)
    envelope = 1.0 / (1.0 + 3.0 * delays)
    field = envelope[:, None] * carrier(frequencies, delays)
    observed = rng.random(len(coords)) < 0.08

    config = TrainConfig(steps=400, seed=0)
    direct = fit_field_network(coords, field, observed, "fourier", config=config)
    guided = fit_field_network(
        coords, field, observed, "fourier", delays=delays,
        frequencies=frequencies, config=config,
    )
    hidden = ~observed
    error = lambda p: np.linalg.norm(p[hidden] - field[hidden]) / np.linalg.norm(field[hidden])
    assert error(guided) < 0.5 * error(direct)
