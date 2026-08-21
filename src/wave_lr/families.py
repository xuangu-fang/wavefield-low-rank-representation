"""Loaders that put different physics families on a common footing.

Every family is reduced to the same object: a complex spectrum ``U`` of shape
``(n_freq, n_sensor)`` on a *regular* sensor line, the sensor coordinates, the
frequency axis, and an energy weight per frequency. That is the only structure
the identifiability criterion needs, and it is what makes waves, wakes and
turbulence comparable without any per-family special casing.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .spectra import tukey

CYLINDER_ROOT = (
    "/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/"
    "realpde_cylinder_subset/raw/hf_dataset/real"
)
OPENFWI_ROOT = (
    "/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/openfwi_curvefault_a/raw"
)
KOLMOGOROV_ROOT = "/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/kolmogorov_mno"
KS_ROOT = "/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/ks_forecast_object"


@dataclass
class Case:
    """One sampled field, reduced to a sensor-line spectrum."""

    family: str
    name: str
    spectrum: NDArray[np.complex128]  # (n_freq, n_sensor)
    coords: NDArray[np.float64]  # sensor positions, metres
    freqs: NDArray[np.float64]  # Hz
    weights: NDArray[np.float64]  # energy share per frequency, sums to 1
    source: float  # transport origin along the sensor line, metres
    speed_hint: float  # characteristic speed, only used to centre the scan


def _band(spectrum, freqs, lo_frac=0.001, f_lo=0.0, f_hi=np.inf):
    energy = (np.abs(spectrum) ** 2).sum(1)
    keep = np.where(energy > lo_frac * energy.max())[0]
    keep = keep[(freqs[keep] > f_lo) & (freqs[keep] < f_hi)]
    return keep, energy[keep] / energy[keep].sum()


def openfwi_cases(limit=24, taper=0.25):
    """Seismic shot gathers: broadband transients with large moveout."""

    paths = sorted(glob.glob(f"{OPENFWI_ROOT}/seis2_*.npy"))
    dt, dr = 1e-3, 10.0
    out = []
    for path in paths:
        block = np.load(path, mmap_mode="r")
        n_sample, n_shot, n_t, n_r = block.shape
        coords = np.arange(n_r) * dr
        # OpenFWI places the five shots at the ends, quarters and centre.
        shot_x = np.array([0, n_r // 4, n_r // 2, 3 * n_r // 4, n_r - 1]) * dr
        for index in range(n_sample):
            for shot in range(n_shot):
                gather = np.asarray(block[index, shot], dtype=np.float64)
                spectrum = np.fft.rfft(gather * tukey(n_t, taper)[:, None], axis=0)
                freqs = np.fft.rfftfreq(n_t, dt)
                keep, weights = _band(spectrum, freqs, f_lo=1.0, f_hi=60.0)
                out.append(
                    Case(
                        "seismic",
                        f"{path.split('/')[-1][:-4]}#{index}s{shot}",
                        spectrum[keep],
                        coords,
                        freqs[keep],
                        weights,
                        float(shot_x[shot]),
                        3000.0,
                    )
                )
                if len(out) >= limit:
                    return out
    return out


def _cylinder_rows():
    import pyarrow as pa

    for path in sorted(glob.glob(f"{CYLINDER_ROOT}/*.arrow")):
        with pa.memory_map(path, "rb") as src:
            table = pa.ipc.open_stream(src).read_all()
        for index in range(table.num_rows):
            yield table.slice(index, 1).to_pylist()[0]


def cylinder_cases(limit=12, field="v", n_time=2048, taper=0.25):
    """Real PIV of a cylinder wake: convected vortices, measured not simulated.

    The sensor line is the wake centreline row, so the family lands in the same
    ``(n_freq, n_sensor)`` shape as a shot gather.
    """

    out = []
    seen = set()
    for row in _cylinder_rows():
        if row["sim_id"] in seen:
            continue
        seen.add(row["sim_id"])
        n_t, n_h, n_w = row["shape_t"], row["shape_h"], row["shape_w"]
        values = np.frombuffer(row[field], dtype=np.float32)
        if values.size != n_t * n_h * n_w:
            continue
        values = values.reshape(n_t, n_h, n_w).astype(np.float64)
        x = np.frombuffer(row["x"], dtype=np.float64).reshape(n_h, n_w)
        t = np.frombuffer(row["t"], dtype=np.float64)
        dt = float(t[1] - t[0])
        fluct = values - values.mean(0, keepdims=True)
        energy = (fluct**2).mean(0)
        cols = np.where(energy.mean(0) > 0.2 * energy.mean(0).max())[0]
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        # Centreline: the row carrying most fluctuation energy in the wake.
        row_index = int(np.argmax(energy[:, c0:c1].mean(1)))
        line = fluct[:n_time, row_index, c0:c1]
        spectrum = np.fft.rfft(line * tukey(line.shape[0], taper)[:, None], axis=0)
        freqs = np.fft.rfftfreq(line.shape[0], dt)
        keep, weights = _band(spectrum, freqs, f_lo=0.2)
        out.append(
            Case(
                "wake",
                f"{row['sim_id'][:-3]}/{field}",
                spectrum[keep],
                x[row_index, c0:c1],
                freqs[keep],
                weights,
                float(x[row_index, c0]),
                0.25,
            )
        )
        if len(out) >= limit:
            break
    return out


def _line_case(family, name, line, coords, dt, source, speed_hint, taper=0.25, f_lo=0.0):
    """Common reduction: a (time, sensor) block becomes a sensor-line spectrum."""

    block = line - line.mean(0, keepdims=True)
    spectrum = np.fft.rfft(block * tukey(block.shape[0], taper)[:, None], axis=0)
    freqs = np.fft.rfftfreq(block.shape[0], dt)
    keep, weights = _band(spectrum, freqs, f_lo=f_lo)
    return Case(family, name, spectrum[keep], coords, freqs[keep], weights, source, speed_hint)


def kolmogorov_cases(limit=12, reynolds=40, n_time=501):
    """Forced 2-D turbulence: broadband, but its fine structure is incoherent.

    The negative control the criterion has to get right. Nothing about the
    loader treats it differently from a shot gather.
    """

    path = f"{KOLMOGOROV_ROOT}/raw/2D_NS_Re{reynolds}.npy"
    block = np.load(path, mmap_mode="r")
    n_traj, _, n_y, n_x = block.shape
    coords = np.arange(n_x) * (2 * np.pi / n_x)
    out = []
    for index in range(min(limit, n_traj)):
        line = np.asarray(block[index, :n_time, n_y // 2], dtype=np.float64)
        out.append(
            _line_case(
                f"turbulence_re{reynolds}",
                f"traj{index}",
                line,
                coords,
                1.0,
                0.0,
                1.0,
                f_lo=1e-9,
            )
        )
    return out


def ks_cases(limit=12, version="v1"):
    """Kuramoto-Sivashinsky: travelling but chaotic, so an intermediate case."""

    data = np.load(f"{KS_ROOT}/{version}/train.npz")
    fields = data["fields"]
    n_traj, _, n_x = fields.shape
    coords = np.arange(n_x, dtype=np.float64)
    out = []
    for index in range(min(limit, n_traj)):
        out.append(
            _line_case(
                "ks",
                f"traj{index}",
                np.asarray(fields[index], dtype=np.float64),
                coords,
                1.0,
                0.0,
                1.0,
                f_lo=1e-9,
            )
        )
    return out
