"""Small, auditable primitives for wavefield representation experiments."""

from .demodulation import demodulate, remodulate
from .diagnostics import best_rank_error, effective_rank, singular_spectrum
from .phase import paired_phase_carriers, phase_embedding, wrap_phase

__all__ = [
    "best_rank_error",
    "demodulate",
    "effective_rank",
    "paired_phase_carriers",
    "phase_embedding",
    "remodulate",
    "singular_spectrum",
    "wrap_phase",
]

