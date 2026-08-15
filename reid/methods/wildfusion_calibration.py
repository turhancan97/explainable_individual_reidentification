"""Calibration policy compatible with official WildFusion all-pairs fitting."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _feature_paths(dataset: Any) -> Optional[Tuple[str, ...]]:
    frame = getattr(dataset, "df", getattr(dataset, "metadata", None))
    if frame is None:
        return None
    for column in ("path", "filepath", "file", "image_path"):
        if column in frame.columns:
            return tuple(frame[column].astype(str).tolist())
    return None


def _is_same_image_set(dataset_a: Any, dataset_b: Any) -> bool:
    if dataset_a is dataset_b:
        return True
    paths_a = _feature_paths(dataset_a)
    paths_b = _feature_paths(dataset_b)
    return paths_a is not None and paths_a == paths_b


def fit_pipeline_calibration(
    pipeline: Any,
    dataset_a: Any,
    dataset_b: Any,
    *,
    exclude_self_pairs: bool = True,
) -> Dict[str, Any]:
    """Fit one pipeline using all pairs, optionally excluding same-image diagonals."""
    features_a = pipeline.get_feature_dataset(dataset_a)
    features_b = pipeline.get_feature_dataset(dataset_b)
    scores = np.asarray(pipeline.matcher(features_a, features_b))
    labels_a = np.asarray(getattr(features_a, "labels_string"))
    labels_b = np.asarray(getattr(features_b, "labels_string"))
    hits = labels_a[:, None] == labels_b[None, :]
    use = np.ones(scores.shape, dtype=bool)
    same_set = _is_same_image_set(dataset_a, dataset_b)
    excluded = 0
    if exclude_self_pairs and same_set and scores.shape[0] == scores.shape[1]:
        diagonal = np.eye(scores.shape[0], dtype=bool)
        use &= ~diagonal
        excluded = int(diagonal.sum())
    calibration_scores = scores[use]
    calibration_hits = hits[use]
    if calibration_scores.size == 0:
        raise ValueError("WildFusion calibration has no usable pairs after self-pair exclusion")
    pipeline.calibration.fit(calibration_scores, calibration_hits)
    pipeline.calibration_done = True
    return {
        "source": "same_set_excluding_self_pairs" if same_set and exclude_self_pairs else "official_all_pairs",
        "total_pairs": int(scores.size),
        "used_pairs": int(calibration_scores.size),
        "excluded_self_pairs": excluded,
    }


def fit_wildfusion_calibration(
    wildfusion: Any,
    dataset_a: Any,
    dataset_b: Any,
    *,
    exclude_self_pairs: bool = True,
    official_same_set: bool = False,
) -> Dict[str, Any]:
    """Fit all calibrated WildFusion pipelines and return provenance."""
    if official_same_set:
        exclude_self_pairs = False
    same_set = _is_same_image_set(dataset_a, dataset_b)
    if same_set and exclude_self_pairs and not official_same_set:
        warnings.warn(
            "WildFusion calibration uses a same-set fallback; self-pairs are excluded. "
            "Configure a disjoint calibration split for paper-grade evaluation.",
            RuntimeWarning,
            stacklevel=2,
        )
    diagnostics = []
    for pipeline in list(getattr(wildfusion, "calibrated_pipelines", [])):
        diagnostics.append(
            fit_pipeline_calibration(
                pipeline,
                dataset_a,
                dataset_b,
                exclude_self_pairs=exclude_self_pairs,
            )
        )
    if not diagnostics:
        raise ValueError("WildFusion has no calibrated pipelines")
    return {
        "source": "official_same_set_all_pairs" if official_same_set else diagnostics[0]["source"],
        "pipelines": diagnostics,
        "total_pairs": int(sum(item["total_pairs"] for item in diagnostics)),
        "used_pairs": int(sum(item["used_pairs"] for item in diagnostics)),
        "excluded_self_pairs": int(sum(item["excluded_self_pairs"] for item in diagnostics)),
    }
