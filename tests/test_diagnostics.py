import numpy as np

from wave_lr.diagnostics import best_rank_error, effective_rank


def test_rank_one_diagnostic() -> None:
    matrix = np.outer(np.arange(1.0, 5.0), np.linspace(0.3, 1.1, 6))
    assert best_rank_error(matrix, rank=1) < 1e-12
    assert effective_rank(matrix, energy=0.999) == 1

