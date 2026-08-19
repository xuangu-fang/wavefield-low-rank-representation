import numpy as np

from wave_lr.fdtd import MediumSpec, build_medium, simulate


def test_free_space_arrival_matches_travel_time() -> None:
    spec = MediumSpec(name="free", grid=96, absorption=6.0)
    speed, damping = build_medium(spec)
    frames, record_dt, spacing = simulate(
        speed[None], damping[None], (48, 48), peak_frequency=12.0,
        duration=0.6, record_every=4, device="cpu",
    )
    traces = frames[0].reshape(frames.shape[1], -1).T
    envelope = np.abs(traces)
    peak_time = np.argmax(envelope, axis=1) * record_dt
    rows, cols = np.meshgrid(np.arange(96), np.arange(96), indexing="ij")
    distance = spacing * np.hypot(rows - 48, cols - 48).ravel()
    band = (distance > 0.15) & (distance < 0.35)
    # Peak time is the travel time plus the fixed Ricker delay; the slope of
    # peak time against distance therefore recovers the wave speed.
    slope, _ = np.polyfit(distance[band], peak_time[band], 1)
    assert abs(1.0 / slope - 1.0) < 0.06


def test_scatterers_change_the_medium() -> None:
    plain, _ = build_medium(MediumSpec(name="plain"))
    scattered, _ = build_medium(MediumSpec(name="scattered", scatterer_fraction=0.2, seed=1))
    assert np.isclose(plain.min(), 1.0)
    assert scattered.min() < 0.9
    assert 0.05 < (scattered < 1.0).mean() < 0.6
