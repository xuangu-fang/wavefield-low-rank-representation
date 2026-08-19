"""No-training sanity check: does a phase coordinate expose low rank?"""

from __future__ import annotations

import json

import numpy as np

from wave_lr.demodulation import demodulate
from wave_lr.diagnostics import best_rank_error, effective_rank


def main() -> None:
    x = np.linspace(0.0, 1.0, 192)
    frequencies = np.linspace(18.0, 46.0, 96)
    travel_time = x + 0.08 * np.sin(2.0 * np.pi * x)
    amplitude = (1.0 + 0.15 * np.cos(3.0 * np.pi * x))[:, None]
    phase = 2.0 * np.pi * travel_time[:, None] * frequencies[None, :]
    field = amplitude * np.exp(1j * phase)
    residual = demodulate(field, phase)

    summary = {
        "purpose": "representation diagnostic, not a predictive result",
        "raw_effective_rank_99": effective_rank(field, energy=0.99),
        "demodulated_effective_rank_99": effective_rank(residual, energy=0.99),
        "raw_best_rank1_error": best_rank_error(field, rank=1),
        "demodulated_best_rank1_error": best_rank_error(residual, rank=1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
