import numpy as np

from wave_lr.demodulation import demodulate, remodulate


def test_demodulation_round_trip() -> None:
    phase = np.linspace(-8.0, 11.0, 31)
    envelope = 1.0 + 0.2 * np.cos(np.linspace(0.0, 2.0, 31)) + 0.1j
    field = remodulate(envelope, phase)
    np.testing.assert_allclose(demodulate(field, phase), envelope)
    np.testing.assert_allclose(remodulate(demodulate(field, phase), phase), field)

