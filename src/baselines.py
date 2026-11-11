"""Trivial, model-free baselines."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def first_difference_scores(
    values: np.ndarray,
    columns: Sequence[int] | None = None,
) -> np.ndarray:
    """Anomaly score ``|x_t - x_{t-1}|`` per timestep.

    For multivariate input the score is the maximum absolute difference over the
    selected ``columns`` (all columns when ``None``). The first timestep has no
    predecessor and gets a score of 0.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D array, got shape {values.shape}.")
    if columns is not None:
        values = values[:, list(columns)]
    if len(values) == 0:
        return np.zeros(0, dtype=float)

    diffs = np.abs(np.diff(values, axis=0)).max(axis=1)
    return np.concatenate(([0.0], diffs))
