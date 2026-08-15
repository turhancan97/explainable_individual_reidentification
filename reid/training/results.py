"""Dependency-light serialization helpers for training model selection."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def build_final_training_metrics(
    selected_metrics: Mapping[str, Any],
    final_epoch_metrics: Mapping[str, Any],
    *,
    best_epoch: int,
    best_metric: str,
    selected_checkpoint: str,
) -> Dict[str, Any]:
    """Make selected-best metrics primary while retaining final-epoch metrics."""
    result = dict(selected_metrics)
    result.update(
        {
            "best_epoch": int(best_epoch),
            "best_metric": str(best_metric),
            "best_metric_value": selected_metrics.get(best_metric),
            "selected_checkpoint": str(selected_checkpoint),
            "best_checkpoint_metrics": dict(selected_metrics),
            "final_epoch_metrics": dict(final_epoch_metrics),
        }
    )
    return result
