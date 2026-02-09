from typing import Any, Dict, List

import numpy as np


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


def compute_metrics(
    dataset_query: Any,
    dataset_database: Any,
    similarity: np.ndarray,
    top_k_values: List[int],
    compute_map: bool,
) -> Dict[str, float]:
    if similarity.shape != (len(dataset_query), len(dataset_database)):
        raise ValueError(
            f"Invalid similarity shape {similarity.shape}, expected {(len(dataset_query), len(dataset_database))}"
        )

    ranked_idx = np.argsort(similarity, axis=1)[:, ::-1]
    query_labels = dataset_query.df[dataset_query.col_label].to_numpy()
    db_labels = dataset_database.df[dataset_database.col_label].to_numpy()

    metrics: Dict[str, float] = {}
    max_k = max(top_k_values)
    if max_k > len(dataset_database):
        raise ValueError(f"Requested top-k includes {max_k}, but database has only {len(dataset_database)} samples")

    for k in top_k_values:
        hits = []
        for q_idx in range(len(dataset_query)):
            top_db_idx = ranked_idx[q_idx, :k]
            top_labels = db_labels[top_db_idx]
            hits.append(query_labels[q_idx] in top_labels)
        metrics[f"top_{k}"] = float(np.mean(hits))

    top1_pred_labels = db_labels[ranked_idx[:, 0]]
    metrics["balanced_top_1"] = _balanced_accuracy_top1(query_labels, top1_pred_labels)

    if compute_map:
        aps: List[float] = []
        for q_idx in range(len(dataset_query)):
            relevant = (db_labels[ranked_idx[q_idx]] == query_labels[q_idx]).astype(np.float32)
            n_relevant = int(relevant.sum())
            if n_relevant == 0:
                continue
            cum_rel = np.cumsum(relevant)
            ranks = np.arange(1, len(relevant) + 1, dtype=np.float32)
            precision = cum_rel / ranks
            ap = float((precision * relevant).sum() / n_relevant)
            aps.append(ap)
        metrics["mAP"] = float(np.mean(aps)) if aps else float("nan")

    return metrics
