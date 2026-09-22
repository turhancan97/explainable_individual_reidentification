from typing import Any, Dict, List, Optional, Tuple

import numpy as np


from reid.evaluation.ranking import stable_rank_indices

DEFAULT_MAP_AT_K = 100

def _balanced_accuracy_top1(query_labels: np.ndarray, predicted_top1_labels: np.ndarray) -> float:
    classes = np.unique(query_labels)
    if classes.size == 0:
        return float("nan")

    recalls: List[float] = []
    for cls in classes:
        mask = query_labels == cls
        denom = int(mask.sum())
        if denom == 0:
            continue
        tp = int((predicted_top1_labels[mask] == cls).sum())
        recalls.append(tp / denom)
    return float(np.mean(recalls)) if recalls else float("nan")


def _balanced_hit_rate(query_labels: np.ndarray, hits: np.ndarray) -> float:
    """Macro-average of the per-identity top-k hit rate.

    The plain ``top_k`` averages over queries, so identities with many query images
    dominate it; this averages the hit rate per identity first, which is what
    ``balanced_top_1`` does for k=1 (a top-1 hit is exactly a correct top-1 label).
    """
    classes = np.unique(query_labels)
    if classes.size == 0:
        return float("nan")
    recalls = [float(hits[query_labels == cls].mean()) for cls in classes if int((query_labels == cls).sum())]
    return float(np.mean(recalls)) if recalls else float("nan")


def _average_precision(relevant: np.ndarray) -> float:
    relevant = np.asarray(relevant, dtype=np.float32)
    n_relevant = int(relevant.sum())
    if n_relevant == 0:
        return 0.0
    cumulative = np.cumsum(relevant)
    ranks = np.arange(1, len(relevant) + 1, dtype=np.float32)
    return float(((cumulative / ranks) * relevant).sum() / n_relevant)


def _truncated_average_precision(
    relevant: np.ndarray,
    scored: np.ndarray,
    num_relevant_total: int,
    k: int,
) -> Tuple[float, float, int]:
    """Return ``(AP@k, rerank AP@k, hits)`` for one already-ranked query.

    ``relevant`` and ``scored`` are ordered by rank. A position the method never
    scored cannot be a hit, so it earns no credit while still occupying its rank.
    This keeps shortlist-constrained methods from being graded on the unscored
    tail, whose order is an artifact of the original database index rather than a
    property of the method.

    ``AP@k`` divides by ``min(num_relevant_total, k)``, so a query whose identity
    never reached the shortlist scores 0 and the metric stays end-to-end. The
    rerank variant divides by the hits actually present in the scored top-k, which
    isolates ordering quality from shortlist reach; it is ``nan`` when there is
    nothing to order.
    """
    k = int(k)
    hits = (
        np.asarray(relevant[:k], dtype=bool) & np.asarray(scored[:k], dtype=bool)
    ).astype(np.float32)
    num_hits = int(hits.sum())
    if num_hits == 0:
        return 0.0, float("nan"), 0
    cumulative = np.cumsum(hits)
    ranks = np.arange(1, hits.shape[0] + 1, dtype=np.float32)
    precision_mass = float(((cumulative / ranks) * hits).sum())
    end_to_end_denominator = min(int(num_relevant_total), k)
    end_to_end = (
        precision_mass / float(end_to_end_denominator) if end_to_end_denominator else 0.0
    )
    return end_to_end, precision_mass / float(num_hits), num_hits


def _label_retrieval_metrics(
    query_labels: np.ndarray,
    database_labels: np.ndarray,
    similarity: np.ndarray,
    top_k_values: List[int],
    compute_map: bool,
    map_at_k: Optional[int] = None,
) -> Dict[str, float]:
    if similarity.shape != (len(query_labels), len(database_labels)):
        raise ValueError(
            f"Invalid similarity shape {similarity.shape}, "
            f"expected {(len(query_labels), len(database_labels))}"
        )
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if len(database_labels) == 0:
        raise ValueError("The database must contain at least one item")

    ranked_idx = stable_rank_indices(similarity)
    scored = np.isfinite(np.asarray(similarity, dtype=np.float64))
    num_scored = int(scored.sum())
    fully_scored = bool(scored.size) and num_scored == scored.size
    metrics: Dict[str, float] = {
        "num_queries": float(len(query_labels)),
        "score_coverage": float(num_scored / scored.size) if scored.size else float("nan"),
    }
    max_k = max(top_k_values)
    if max_k > len(database_labels):
        raise ValueError(
            f"Requested top-k includes {max_k}, "
            f"but database has only {len(database_labels)} samples"
        )

    for k in top_k_values:
        hits = np.asarray(
            [
                query_labels[q_idx] in database_labels[ranked_idx[q_idx, :k]]
                for q_idx in range(len(query_labels))
            ],
            dtype=bool,
        )
        metrics[f"top_{k}"] = float(hits.mean()) if hits.size else float("nan")
        metrics[f"balanced_top_{k}"] = (
            _balanced_hit_rate(query_labels, hits) if hits.size else float("nan")
        )

    if 1 not in top_k_values:  # balanced_top_1 is reported even when top-1 is not requested
        metrics["balanced_top_1"] = (
            _balanced_accuracy_top1(query_labels, database_labels[ranked_idx[:, 0]])
            if len(query_labels)
            else float("nan")
        )

    cutoff = min(int(map_at_k), len(database_labels)) if map_at_k else 0
    relevant_counts = np.zeros(len(query_labels), dtype=np.int64)
    eligible_aps: List[float] = []
    all_aps: List[float] = []
    truncated_aps: List[float] = []
    rerank_aps: List[float] = []
    queries_with_hits = 0
    for q_idx in range(len(query_labels)):
        ranked_row = ranked_idx[q_idx]
        relevant = (database_labels[ranked_row] == query_labels[q_idx]).astype(np.float32)
        relevant_counts[q_idx] = int(relevant.sum())
        ap = _average_precision(relevant)
        all_aps.append(ap)
        if relevant_counts[q_idx] > 0:
            eligible_aps.append(ap)
        if cutoff:
            truncated_ap, rerank_ap, hits = _truncated_average_precision(
                relevant, scored[q_idx][ranked_row], int(relevant_counts[q_idx]), cutoff
            )
            truncated_aps.append(truncated_ap)
            if hits:
                queries_with_hits += 1
                rerank_aps.append(rerank_ap)

    eligible_queries = int(np.count_nonzero(relevant_counts))
    metrics["num_queries_with_gallery_match"] = float(eligible_queries)
    metrics["num_queries_without_gallery_match"] = float(
        len(query_labels) - eligible_queries
    )
    metrics["mAP_query_coverage"] = (
        float(eligible_queries / len(query_labels)) if len(query_labels) else float("nan")
    )
    if compute_map:
        # Full-matrix mAP ranks every database entry, so it is only meaningful when
        # every entry carries a real score. On a shortlist-constrained matrix the
        # unscored tail dominates the average and its order comes from the original
        # database index, so the number would measure metadata row order rather than
        # the method. Report `nan` instead of a value that invites false comparison;
        # `mAP_at_k` is the comparable metric in that case.
        metrics["mAP"] = (
            float(np.mean(all_aps)) if all_aps and fully_scored else float("nan")
        )
        metrics["mAP_eligible"] = (
            float(np.mean(eligible_aps)) if eligible_aps and fully_scored else float("nan")
        )
        if cutoff:
            metrics["map_at_k"] = float(cutoff)
            metrics["mAP_at_k"] = (
                float(np.mean(truncated_aps)) if truncated_aps else float("nan")
            )
            metrics["rerank_mAP_at_k"] = (
                float(np.mean(rerank_aps)) if rerank_aps else float("nan")
            )
            metrics["recall_at_k"] = (
                float(queries_with_hits / len(query_labels)) if len(query_labels) else float("nan")
            )
            metrics["num_queries_with_relevant_in_top_k"] = float(queries_with_hits)
    return metrics


def compute_metrics(
    dataset_query: Any,
    dataset_database: Any,
    similarity: np.ndarray,
    top_k_values: List[int],
    compute_map: bool,
    map_at_k: Optional[int] = None,
) -> Dict[str, float]:
    query_labels = dataset_query.df[dataset_query.col_label].to_numpy()
    db_labels = dataset_database.df[dataset_database.col_label].to_numpy()
    return _label_retrieval_metrics(
        query_labels=query_labels,
        database_labels=db_labels,
        similarity=similarity,
        top_k_values=top_k_values,
        compute_map=compute_map,
        map_at_k=map_at_k,
    )


def compute_identity_metrics(
    query_labels: np.ndarray,
    identity_labels: np.ndarray,
    similarity: np.ndarray,
    top_k_values: List[int],
    compute_map: bool,
    map_at_k: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate retrieval where each database column represents one identity."""
    return _label_retrieval_metrics(
        query_labels=np.asarray(query_labels),
        database_labels=np.asarray(identity_labels),
        similarity=similarity,
        top_k_values=top_k_values,
        compute_map=compute_map,
        map_at_k=map_at_k,
    )
