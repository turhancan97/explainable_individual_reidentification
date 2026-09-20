"""Readable, deterministic default names for tracked experiment runs."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any


_SAFE = re.compile(r"[^A-Za-z0-9.-]+")
_RUN_HASH = re.compile(r"[0-9a-f]{8}")
_MAX_NAME_LENGTH = 128


def _get(value: Any, key: str, default: Any = "") -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _part(value: Any, default: str = "unknown") -> str:
    text = str(value or default).strip().lower()
    text = _SAFE.sub("-", text).strip("-.")
    return text or default


def _run_token(run_id: str) -> str:
    last_component = str(run_id).rsplit("_", 1)[-1]
    if _RUN_HASH.fullmatch(last_component):
        return last_component
    return hashlib.sha1(str(run_id).encode("utf-8")).hexdigest()[:8]


def _finish(parts: list[Any], run_id: str) -> str:
    full = "-".join(_part(value) for value in parts if value not in (None, ""))
    token = _run_token(run_id)
    name = f"{full}-{token}" if full else f"run-{token}"
    if len(name) > _MAX_NAME_LENGTH:
        suffix = f"-{token}"
        name = f"{name[:_MAX_NAME_LENGTH - len(suffix)]}{suffix}"
    return name


def probe_wandb_name(cfg: Any, run_id: str) -> str:
    """Build a name identifying the probe's data, method, and key settings."""
    dataset = _get(cfg, "dataset", {})
    model = _get(cfg, "model", {})
    benchmark = _get(cfg, "benchmark", {})
    method = str(_get(benchmark, "method", "unknown"))
    method_cfg = _get(_get(benchmark, "methods", {}), method, {})
    parts: list[Any] = [
        "probe",
        _get(dataset, "name", "dataset"),
        _get(dataset, "animal", "animal"),
        _get(dataset, "split_col", "split"),
        _get(model, "type", "model"),
        _get(model, "mode", "pretrained"),
        method,
    ]
    if method == "vismatch":
        parts.extend(
            [
                _get(method_cfg, "matcher", "matcher"),
                "finetuned"
                if str(_get(method_cfg, "checkpoint_source", "default")) == "custom"
                else "default",
                f"k{_get(benchmark, 'candidate_k', '')}",
            ]
        )
    elif method in {"wildfusion", "local_lightglue"}:
        parts.append(f"k{_get(benchmark, 'candidate_k', '')}")
    elif method in {"linear_probe", "efficient_probe"}:
        weighting = str(_get(method_cfg, "class_weighting", "inverse_frequency"))
        parts.extend(
            [
                _get(method_cfg, "train_mode", "classifier"),
                "weighted" if weighting == "inverse_frequency" else "unweighted",
            ]
        )
    parts.append("masked" if bool(_get(dataset, "no_background", False)) else "background")
    return _finish(parts, run_id)


def finetune_wandb_name(cfg: Any, run_id: str) -> str:
    """Build a name identifying finetune data, split, model, and schedule."""
    dataset = _get(cfg, "dataset", {})
    model = _get(cfg, "model", {})
    train = _get(cfg, "train", {})
    return _finish(
        [
            "finetune",
            _get(dataset, "name", "dataset"),
            _get(dataset, "animal", "animal"),
            _get(dataset, "split_col", "split"),
            _get(model, "type", "model"),
            f"{_get(train, 'epochs', '')}ep",
            f"lr{_get(train, 'lr', '')}",
        ],
        run_id,
    )
