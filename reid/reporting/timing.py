"""Dependency-light timing semantics shared by probe and reporting code."""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping


MATCHER_METHODS = frozenset({"vismatch", "wildfusion", "local_lightglue"})


def finite_seconds(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def primary_runtime_key(method: str) -> str:
    """Return the timing field used as the primary compute runtime."""

    return "matcher_runtime_sec" if method in MATCHER_METHODS else "method_compute_runtime_sec"


def set_primary_compute_runtime(method: str, timings: MutableMapping[str, Any]) -> float | None:
    """Derive the primary runtime without inventing values for legacy runs."""

    value = finite_seconds(timings.get(primary_runtime_key(method)))
    if value is not None:
        timings["primary_compute_runtime_sec"] = value
    return value


def primary_runtime_seconds(method: str, timings: Mapping[str, Any]) -> float | None:
    """Read the standardized primary runtime from a timing mapping."""

    return finite_seconds(timings.get("primary_compute_runtime_sec"))
