from typing import Any, Dict, List

import numpy as np


from reid.evaluation.ranking import stable_rank_indices

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


def _average_precision(relevant: np.ndarray) -> float:
    relevant = np.asarray(relevant, dtype=np.float32)
    n_relevant = int(relevant.sum())
    if n_relevant == 0:
        return 0.0
    cumulative = np.cumsum(relevant)
    ranks = np.arange(1, len(relevant) + 1, dtype=np.float32)
    return float(((cumulative / ranks) * relevant).sum() / n_relevant)


def _label_retrieval_metrics(
    query_labels: np.ndarray,
    database_labels: np.ndarray,
    similarity: np.ndarray,
    top_k_values: List[int],
    compute_map: bool,
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
    metrics: Dict[str, float] = {
        "num_queries": float(len(query_labels)),
    }
    max_k = max(top_k_values)
    if max_k > len(database_labels):
        raise ValueError(
            f"Requested top-k includes {max_k}, "
            f"but database has only {len(database_labels)} samples"
        )

    for k in top_k_values:
        hits = [
            query_labels[q_idx] in database_labels[ranked_idx[q_idx, :k]]
            for q_idx in range(len(query_labels))
        ]
        metrics[f"top_{k}"] = float(np.mean(hits)) if hits else float("nan")

    if len(query_labels):
        top1_pred_labels = database_labels[ranked_idx[:, 0]]
        metrics["balanced_top_1"] = _balanced_accuracy_top1(
            query_labels, top1_pred_labels
        )
    else:
        metrics["balanced_top_1"] = float("nan")

    relevant_counts = np.zeros(len(query_labels), dtype=np.int64)
    eligible_aps: List[float] = []
    all_aps: List[float] = []
    for q_idx in range(len(query_labels)):
        relevant = (
            database_labels[ranked_idx[q_idx]] == query_labels[q_idx]
        ).astype(np.float32)
        relevant_counts[q_idx] = int(relevant.sum())
        ap = _average_precision(relevant)
        all_aps.append(ap)
        if relevant_counts[q_idx] > 0:
            eligible_aps.append(ap)

    eligible_queries = int(np.count_nonzero(relevant_counts))
    metrics["num_queries_with_gallery_match"] = float(eligible_queries)
    metrics["num_queries_without_gallery_match"] = float(
        len(query_labels) - eligible_queries
    )
    metrics["mAP_query_coverage"] = (
        float(eligible_queries / len(query_labels)) if len(query_labels) else float("nan")
    )
    if compute_map:
        metrics["mAP"] = float(np.mean(all_aps)) if all_aps else float("nan")
        metrics["mAP_eligible"] = (
            float(np.mean(eligible_aps)) if eligible_aps else float("nan")
        )
    return metrics


def compute_metrics(
    dataset_query: Any,
    dataset_database: Any,
    similarity: np.ndarray,
    top_k_values: List[int],
    compute_map: bool,
) -> Dict[str, float]:
    query_labels = dataset_query.df[dataset_query.col_label].to_numpy()
    db_labels = dataset_database.df[dataset_database.col_label].to_numpy()
    return _label_retrieval_metrics(
        query_labels=query_labels,
        database_labels=db_labels,
        similarity=similarity,
        top_k_values=top_k_values,
        compute_map=compute_map,
    )


def compute_identity_metrics(
    query_labels: np.ndarray,
    identity_labels: np.ndarray,
    similarity: np.ndarray,
    top_k_values: List[int],
    compute_map: bool,
) -> Dict[str, float]:
    """Evaluate retrieval where each database column represents one identity."""
    return _label_retrieval_metrics(
        query_labels=np.asarray(query_labels),
        database_labels=np.asarray(identity_labels),
        similarity=similarity,
        top_k_values=top_k_values,
        compute_map=compute_map,
    )
