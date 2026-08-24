#!/usr/bin/env python3
"""Summarize and index organized Slurm probe-task logs."""

from __future__ import annotations

import argparse
import csv
import json
import fcntl
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


FIELDS = [
    "job_id",
    "task_id",
    "submission_id",
    "manifest_path",
    "profile_id",
    "dataset",
    "animal",
    "method",
    "matcher",
    "checkpoint",
    "checkpoint_owner",
    "checkpoint_sha256",
    "validation_status",
    "validation_error",
    "candidate_k",
    "status",
    "start_time",
    "end_time",
    "experiment_run_directory",
    "stdout_path",
    "stderr_path",
    "combined_path",
    "error_summary",
    "command",
]


def load_records(logs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not logs_root.is_dir():
        return records
    for path in sorted(logs_root.rglob("task-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        record["_metadata_path"] = str(path)
        records.append(record)
    return records


def _matches(record: dict[str, Any], args: argparse.Namespace) -> bool:
    for key in ("dataset", "animal", "method", "matcher", "checkpoint", "status"):
        expected = getattr(args, key)
        if expected and str(record.get(key, "")) != expected:
            return False
    return True


def _sort_key(record: dict[str, Any], field: str) -> tuple[int, Any]:
    value = record.get(field, "")
    if value in (None, ""):
        return (1, "")
    if field == "candidate_k":
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, "")
    return (0, str(value).lower())


def sort_records(records: Iterable[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: _sort_key(record, field))


def _display_path(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(value)


def _index_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {field: record.get(field, "") for field in FIELDS}
    for field in ("stdout_path", "stderr_path", "combined_path", "experiment_run_directory"):
        row[field] = _display_path(row[field])
    return row


def write_index(logs_root: Path, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_name(index_path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        # Read while holding the lock. Otherwise two finishing tasks can each
        # scan an incomplete view and the later atomic replace can omit a row.
        records = load_records(logs_root)
        fd, temporary = tempfile.mkstemp(prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                for record in sort_records(records, "start_time"):
                    writer.writerow(_index_row(record))
            os.replace(temporary, index_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _markdown(records: list[dict[str, Any]]) -> str:
    columns = [
        "job_id",
        "task_id",
        "dataset",
        "animal",
        "method",
        "matcher",
        "checkpoint",
        "candidate_k",
        "status",
        "error_summary",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in records:
        values = []
        for column in columns:
            value = str(record.get(column, "")).replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _csv(records: list[dict[str, Any]]) -> str:
    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    for record in records:
        writer.writerow(_index_row(record))
    return output.getvalue().rstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=Path("logs/parallel_run"))
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--animal")
    parser.add_argument("--method")
    parser.add_argument("--matcher")
    parser.add_argument("--checkpoint")
    parser.add_argument("--status")
    parser.add_argument("--sort-by", default="start_time", choices=FIELDS)
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--write-index", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    all_records = load_records(args.logs_root)
    if args.write_index:
        index_path = args.index_path or args.logs_root.parent / "index.csv"
        write_index(args.logs_root, index_path)
    records = sort_records([record for record in all_records if _matches(record, args)], args.sort_by)
    if not args.quiet:
        print(_csv(records) if args.format == "csv" else _markdown(records))


if __name__ == "__main__":
    main()
