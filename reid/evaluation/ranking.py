"""Deterministic ranking helpers used by all evaluation paths."""

from __future__ import annotations

from typing import Optional

import numpy as np


def stable_rank_indices(scores: np.ndarray, *, descending: bool = True, tie_break_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Rank rows by score and use original column index as a stable tie-breaker."""
    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError(f"scores must be a 2D array, got shape={values.shape}")
    n_columns = values.shape[1]
    tie_break = np.arange(n_columns, dtype=np.int64) if tie_break_indices is None else np.asarray(tie_break_indices, dtype=np.int64)
    if tie_break.shape != (n_columns,):
        raise ValueError(f"tie_break_indices must have shape {(n_columns,)}, got {tie_break.shape}")
    ranked = np.empty((values.shape[0], n_columns), dtype=np.int64)
    for row_index, row in enumerate(values):
        ranked[row_index] = np.lexsort((tie_break, -row if descending else row))
    return ranked


def stable_rank_1d(scores: np.ndarray, *, descending: bool = True) -> np.ndarray:
    """Rank one score vector with original position as the tie-breaker."""
    values = np.asarray(scores)
    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional scores, got shape {values.shape}")
    primary = -values if descending else values
    return np.lexsort((np.arange(values.shape[0]), primary))
