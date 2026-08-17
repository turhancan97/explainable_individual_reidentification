"""Dependency-light helpers for shortlist-based retrieval methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

DEFAULT_MAX_PERSISTED_SCORES = 5_000_000


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


def save_score_matrix(
    path: Any,
    similarity: Any,
    max_entries: int = DEFAULT_MAX_PERSISTED_SCORES,
) -> Optional[Path]:
    """Persist the scored entries of a score matrix in sparse COO form.

    Only finite entries are stored, so a shortlist matrix costs about one row per
    scored pair instead of the full dense grid. This lets metrics be recomputed
    later without repeating a matcher run. Dense matrices above ``max_entries``
    are skipped and return ``None`` rather than writing a multi-gigabyte file.
    """
    scores = np.asarray(similarity)
    if scores.ndim != 2:
        raise ValueError(f"similarity must be a two-dimensional array, got shape {scores.shape}")
    rows, cols = np.nonzero(np.isfinite(scores))
    if rows.size > int(max_entries):
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        shape=np.asarray(scores.shape, dtype=np.int64),
        rows=rows.astype(np.int64),
        cols=cols.astype(np.int64),
        values=scores[rows, cols].astype(np.float32),
    )
    return destination


def load_score_matrix(path: Any) -> np.ndarray:
    """Rebuild a dense score matrix written by :func:`save_score_matrix`."""
    with np.load(Path(path)) as data:
        shape = tuple(int(value) for value in data["shape"])
        scores = np.full(shape, -np.inf, dtype=np.float32)
        scores[data["rows"], data["cols"]] = data["values"]
    return scores


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
