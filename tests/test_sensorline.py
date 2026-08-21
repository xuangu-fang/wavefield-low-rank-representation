"""The sensor-line criterion has to behave like a bound, not just a number."""

from __future__ import annotations

import numpy as np

from wave_lr.sensorline import (
    apply_warp,
    bound_line,
    reconstruct_bandlimited,
    tapered,
    transport_delays,
)


def _travelling(n_freq=24, n_sensor=64, speed=3.0, seed=0):
    """A transient transported along the line: the case a warp should collapse."""

    rng = np.random.default_rng(seed)
    freqs = np.linspace(0.05, 0.45, n_freq)
    coords = np.arange(n_sensor, dtype=np.float64)
    envelope = rng.normal(size=(n_freq, 1)) + 1j * rng.normal(size=(n_freq, 1))
    return freqs, coords, envelope * np.exp(-2j * np.pi * freqs[:, None] * coords / speed)


def test_warp_collapses_a_travelling_transient():
    freqs, coords, spectrum = _travelling()
    weights = np.full(freqs.size, 1.0 / freqs.size)
    raw = bound_line(spectrum, weights, 0.2)
    delays = transport_delays(coords, 0.0, 3.0)
    aligned = bound_line(apply_warp(spectrum, freqs, delays), weights, 0.2)
    assert aligned < 0.05 * raw


def test_the_taper_puts_a_floor_under_the_bound():
    """The correct warp cannot drive the bound to zero, and that is not a bug.

    Once a transient is perfectly aligned the field is constant along the line,
    so what the transform sees is the window itself -- and the window's
    sidelobes reach past the Nyquist of a very sparse array. The floor is real:
    it is the price of the taper that keeps the bound from being inflated by
    leakage in the first place. It only binds where the array is sparse enough
    that the resolvable band is a handful of bins.
    """

    freqs, coords, spectrum = _travelling()
    weights = np.full(freqs.size, 1.0 / freqs.size)
    delays = transport_delays(coords, 0.0, 3.0)
    aligned = apply_warp(spectrum, freqs, delays)
    floors = [bound_line(aligned, weights, fraction) for fraction in (0.05, 0.1, 0.2)]
    assert floors[0] > floors[1] > floors[2]
    assert floors[0] > 0.1  # sparse arrays: the floor dominates
    assert floors[2] < 0.05  # by 1-in-5 sensors it is negligible


def test_bound_is_never_beaten_by_the_estimator():
    """The whole claim rests on this inequality holding on both footings."""

    rng = np.random.default_rng(1)
    weights = np.full(24, 1.0 / 24)
    for seed in range(8):
        _, _, spectrum = _travelling(seed=seed)
        spectrum = spectrum + 0.3 * (
            rng.normal(size=spectrum.shape) + 1j * rng.normal(size=spectrum.shape)
        )
        for fraction in (0.1, 0.2, 0.34):
            bound = bound_line(spectrum, weights, fraction)
            error = reconstruct_bandlimited(spectrum, weights, fraction)
            assert error >= bound - 1e-9


def test_bound_is_invariant_to_a_pointwise_phase():
    """A unimodular per-sensor factor is a relabelling, not information."""

    freqs, _, spectrum = _travelling(seed=3)
    weights = np.full(freqs.size, 1.0 / freqs.size)
    rng = np.random.default_rng(7)
    phase = np.exp(2j * np.pi * rng.random(spectrum.shape[-1]))
    plain = bound_line(spectrum, weights, 0.2)
    rotated = bound_line(spectrum * phase, weights, 0.2)
    assert abs(plain - rotated) > 0  # a *sensor*-dependent phase does change the bound
    same = bound_line(spectrum * np.exp(1j * 0.7), weights, 0.2)
    assert abs(plain - same) < 1e-12  # a global phase does not


def test_taper_is_applied_once_and_only_once():
    _, _, spectrum = _travelling(seed=5)
    once = tapered(spectrum)
    assert once.shape == spectrum.shape
    assert np.abs(once[:, 0]).max() < np.abs(spectrum[:, 0]).max()
