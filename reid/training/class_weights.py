"""Dependency-light class-weight construction for classifier probes."""

from __future__ import annotations

from collections import Counter
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np


def compute_identity_class_weights(
    labels: Sequence[Any],
    class_order: Sequence[Any],
    *,
    weighting: str = "inverse_frequency",
    normalize: bool = True,
    max_weight: float | None = 5.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return class weights in a deterministic ``class_order``.

    The frequencies are intentionally computed only from ``labels``.  In the
    linear probe this is the database/training split, so query labels can never
    influence the loss.  Capping happens after mean normalization and is not
    followed by a second normalization pass.
    """

    if weighting not in {"inverse_frequency", "none"}:
        raise ValueError(
            "class_weighting must be 'inverse_frequency' or 'none'; "
            f"got {weighting!r}"
        )
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError(f"class_weight_normalize must be boolean; got {normalize!r}")
    if max_weight is not None:
        if not isinstance(max_weight, Real) or not np.isfinite(float(max_weight)) or float(max_weight) <= 0:
            raise ValueError(f"class_weight_max must be a positive finite number; got {max_weight!r}")
        max_weight = float(max_weight)

    ordered_classes = list(class_order)
    if len(ordered_classes) != len(set(ordered_classes)):
        raise ValueError("class_order must contain each identity exactly once")
    if not ordered_classes:
        raise ValueError("class_order must contain at least one identity")

    counts = Counter(labels)
    missing = [identity for identity in ordered_classes if counts[identity] <= 0]
    if missing:
        raise ValueError(f"class_order contains identities absent from training labels: {missing[:5]}")

    metadata: dict[str, Any] = {
        "mode": weighting,
        "formula": "1 / n_identity" if weighting == "inverse_frequency" else "none",
        "normalize_to_mean_one": bool(normalize),
        "cap_after_normalization": True,
        "max_weight": max_weight,
        "num_identities": len(ordered_classes),
        "identity_order": [str(identity) for identity in ordered_classes],
        "counts": [int(counts[identity]) for identity in ordered_classes],
    }
    if weighting == "none":
        metadata.update({"min": None, "max": None, "mean": None, "median": None})
        return None, metadata

    weights = np.asarray([1.0 / float(counts[identity]) for identity in ordered_classes], dtype=np.float64)
    if normalize:
        weights *= float(len(weights)) / float(weights.sum())
    if max_weight is not None:
        weights = np.minimum(weights, max_weight)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("computed class weights must be positive and finite")

    weights = weights.astype(np.float32)
    metadata.update(
        {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
            "median": float(np.median(weights)),
        }
    )
    return weights, metadata
