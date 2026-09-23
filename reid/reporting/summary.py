"""Read and format the central experiment run index."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence


SUMMARY_COLUMNS = [
    "run_id",
    "workflow",
    "dataset",
    "model",
    "method",
    "variant",
    "status",
    "top_1",
    "top_5",
    "top_10",
    "mAP",
    "total_runtime_sec",
    "manifest_path",
]


def load_run_rows(index_path: Path, experiment_root: Optional[Path] = None) -> List[dict]:
    if index_path.is_file():
        with index_path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if experiment_root is None or not experiment_root.is_dir():
        return []
    rows: List[dict] = []
    for manifest_path in sorted(experiment_root.rglob("run_manifest.json")):
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows.append(
            {
                "run_id": manifest.get("run_id", manifest_path.parent.name),
                "status": manifest.get("status", "unknown"),
                "workflow": manifest.get("workflow", ""),
                "run_utc": manifest.get("run_utc", ""),
                "dataset": manifest.get("dataset", ""),
                "animal": manifest.get("animal", ""),
                "split_protocol": manifest.get("split_protocol", ""),
                "model": manifest.get("model", ""),
                "method": manifest.get("method", ""),
                "variant": manifest.get("variant", ""),
                "config_hash": manifest.get("config_hash", ""),
                "run_dir": manifest_path.parent.as_posix(),
                "manifest_path": manifest_path.as_posix(),
                **dict(manifest.get("metrics", {})),
                **dict(manifest.get("timings", {})),
            }
        )
    return rows


def filter_run_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: Optional[str] = None,
    workflow: Optional[str] = None,
    method: Optional[str] = None,
    matcher: Optional[str] = None,
) -> List[dict]:
    def matches(row: Mapping[str, Any]) -> bool:
        if dataset and str(row.get("dataset", "")) != dataset:
            return False
        if workflow and str(row.get("workflow", "")) != workflow:
            return False
        if method and str(row.get("method", "")) != method:
            return False
        if matcher and str(row.get("variant", "")) != matcher:
            return False
        return True

    return [dict(row) for row in rows if matches(row)]


def sort_run_rows(rows: Sequence[Mapping[str, Any]], key: str, descending: bool = True) -> List[dict]:
    # Total order: numbers first in the requested direction, then text, then
    # empty and NaN cells in their original order. Mixing float and str keys
    # raises TypeError, and NaN compares false both ways, so the sort would
    # otherwise either crash or silently leave rows unsorted.
    numeric: List[tuple] = []
    text: List[tuple] = []
    missing: List[dict] = []
    for row in rows:
        value = row.get(key, "")
        try:
            number = float(value)
        except (TypeError, ValueError):
            label = "" if value is None else str(value).strip()
            if label:
                text.append((label, dict(row)))
            else:
                missing.append(dict(row))
            continue
        if math.isnan(number):
            missing.append(dict(row))
        else:
            numeric.append((number, dict(row)))
    numeric.sort(key=lambda item: item[0], reverse=descending)
    text.sort(key=lambda item: item[0], reverse=descending)
    return [row for _, row in numeric] + [row for _, row in text] + missing


def format_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = list(SUMMARY_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = []
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def format_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = [column for column in SUMMARY_COLUMNS if any(column in row for row in rows)]
    if not columns:
        columns = SUMMARY_COLUMNS[:7]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"
