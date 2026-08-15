"""Dependency-light helpers for shortlist-based retrieval methods."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def build_shortlist_score_matrix(
    stage_similarity: Optional[np.ndarray],
    candidate_indices: Optional[np.ndarray],
) -> np.ndarray:
    """Return a shortlist-only score matrix initialized with ``-inf``.

    Candidate positions are intentionally left for the reranker to overwrite.
    If no shortlist is supplied, the caller is expected to overwrite every
    position during full matching.
    """
    if stage_similarity is None:
        if candidate_indices is None:
            raise ValueError("Either stage_similarity or candidate_indices is required")
        candidates = np.asarray(candidate_indices)
        if candidates.ndim != 2:
            raise ValueError("candidate_indices must be a two-dimensional array")
        width = int(candidates.max()) + 1 if candidates.size else 0
        return np.full((candidates.shape[0], width), -np.inf, dtype=np.float32)

    scores = np.asarray(stage_similarity)
    if scores.ndim != 2:
        raise ValueError("stage_similarity must be a two-dimensional array")
    return np.full(scores.shape, -np.inf, dtype=np.float32)


def shortlist_pair_counts(
    num_queries: int,
    num_database: int,
    candidate_indices: Optional[np.ndarray],
) -> Dict[str, float]:
    """Return auditable scored/unscored pair counts for a shortlist matrix."""
    total_pairs = int(num_queries) * int(num_database)
    if candidate_indices is None:
        candidate_pairs = total_pairs
    else:
        try:
            candidate_rows = list(candidate_indices)
        except TypeError as exc:
            raise ValueError("candidate_indices must be a sequence of candidate rows") from exc
        if len(candidate_rows) != int(num_queries):
            raise ValueError("candidate_indices must have one row per query")
        candidate_pairs = int(sum(len(row) for row in candidate_rows))
    unscored_pairs = total_pairs - candidate_pairs
    return {
        "num_candidate_pairs": float(candidate_pairs),
        "num_unscored_pairs": float(unscored_pairs),
        "candidate_fraction": float(candidate_pairs / total_pairs) if total_pairs else float("nan"),
    }


def normalize_shortlist_score(value: Any) -> float:
    """Convert invalid matcher output into the shortlist exclusion score."""
    score = float(value)
    return score if np.isfinite(score) else -np.inf


def candidate_recall_metrics(
    query_labels: Any,
    database_labels: Any,
    candidate_indices: Optional[np.ndarray],
) -> Dict[str, float]:
    """Measure whether each query identity survived Stage-A candidate selection."""
    query = np.asarray(query_labels)
    database = np.asarray(database_labels)
    if candidate_indices is None:
        hits = np.ones(len(query), dtype=bool)
    else:
        candidates = np.asarray(candidate_indices)
        if candidates.shape[0] != len(query):
            raise ValueError("candidate_indices row count must match query labels")
        hits = np.asarray(
            [np.any(database[row] == query[i]) if len(row) else False for i, row in enumerate(candidates)],
            dtype=bool,
        )
    hit_rate = float(np.mean(hits)) if len(hits) else float("nan")
    return {
        "candidate_hit_rate": hit_rate,
        "candidate_recall_at_k": hit_rate,
        "num_queries_without_candidate_identity": float(np.count_nonzero(~hits)),
    }
