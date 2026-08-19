import numpy as np

from wave_lr.spectra import Spectrum, shift_spectrum, to_spectrum


def _ricker(t: np.ndarray, peak: float, centre: float) -> np.ndarray:
    arg = (np.pi * peak * (t - centre)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def test_shift_aligns_arrivals() -> None:
    dt = 0.002
    t = np.arange(1024) * dt
    delays = np.array([0.2, 0.35, 0.5])
    traces = np.stack([_ricker(t, 20.0, d) for d in delays])
    spectrum = to_spectrum(traces, dt, 2.0, 60.0)
    aligned = shift_spectrum(spectrum, delays)
    from wave_lr.spectra import band_limited_traces

    shifted, times = band_limited_traces(aligned)
    peaks = times[np.argmax(np.abs(shifted), axis=1)]
    assert np.allclose(peaks, peaks[0], atol=2 * dt)


def test_shift_is_invertible() -> None:
    rng = np.random.default_rng(0)
    values = rng.standard_normal((4, 16)) + 1j * rng.standard_normal((4, 16))
    spectrum = Spectrum(values, np.linspace(2.0, 10.0, 16), 0.01, 64)
    delays = rng.standard_normal(4) * 0.05
    roundtrip = shift_spectrum(shift_spectrum(spectrum, delays), -delays)
    assert np.allclose(roundtrip.values, values)
