from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class FeatureContainer:
    features: np.ndarray
    labels_string: Optional[np.ndarray] = None


def get_labels_string(dataset: Any, label_col: str) -> np.ndarray:
    if hasattr(dataset, "labels_string"):
        labels = getattr(dataset, "labels_string")
        return np.asarray(labels)
    if hasattr(dataset, "df") and label_col in dataset.df.columns:
        return dataset.df[label_col].astype(str).to_numpy()
    if hasattr(dataset, "metadata") and label_col in dataset.metadata.columns:
        return dataset.metadata[label_col].astype(str).to_numpy()
    return np.array([], dtype=str)


def normalize_features(raw: Any) -> np.ndarray:
    value = raw
    if isinstance(value, dict):
        for key in ("features", "embeddings", "vectors", "x"):
            if key in value:
                value = value[key]
                break

    if isinstance(value, tuple) and len(value) >= 1:
        first = value[0]
        if isinstance(first, (np.ndarray, list, tuple)) or torch.is_tensor(first):
            value = first

    def _to_numpy(v: Any) -> np.ndarray:
        if torch.is_tensor(v):
            return v.detach().cpu().numpy()
        return np.asarray(v)

    def _extract_from_sequence_rows(seq: Any):
        if not isinstance(seq, (list, tuple)) or len(seq) == 0:
            return None
        first = seq[0]
        if isinstance(first, (list, tuple)) and len(first) >= 1:
            first_item = first[0]
            if isinstance(first_item, (np.ndarray, list, tuple)) or torch.is_tensor(first_item):
                rows = [_to_numpy(row[0]) for row in seq]
                return np.stack(rows, axis=0)
        return None

    maybe_rows = _extract_from_sequence_rows(value)
    if maybe_rows is not None:
        arr = maybe_rows
    elif isinstance(value, (list, tuple)):
        rows = [_to_numpy(v) for v in value]
        try:
            arr = np.stack(rows, axis=0)
        except ValueError:
            arr = np.asarray(rows, dtype=np.float32)
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        try:
            arr = np.asarray(value)
        except ValueError:
            try:
                seq = list(value)
            except Exception as exc:
                raise ValueError(f"Could not convert features to numpy. type={type(raw)}") from exc
            maybe_rows = _extract_from_sequence_rows(seq)
            if maybe_rows is None:
                raise
            arr = maybe_rows

    if arr.dtype == object:
        arr = np.stack([np.asarray(x) for x in arr], axis=0)

    if arr.ndim == 1:
        raise ValueError(f"Features must be 2D (N, D). Got shape={arr.shape}")
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    return arr.astype(np.float32, copy=False)
