#!/usr/bin/env python3
"""Create and update dependency-light metadata records for parallel probe tasks."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _error_summary(error_file: str | None) -> str:
    if not error_file:
        return ""
    path = Path(error_file)
    if not path.is_file():
        return ""
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return ""
    pattern = re.compile(r"(traceback|error|exception|oom|out.of.memory|killed|failed|permission)", re.IGNORECASE)
    matching = [line for line in lines if pattern.search(line)]
    return (matching[-1] if matching else lines[-1])[:1000]


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--path", required=True, type=Path)


def init(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": args.status,
        "job_id": args.job_id,
        "task_id": int(args.task_id),
        "dataset": args.dataset,
        "animal": args.animal,
        "split_protocol": args.split_protocol,
        "method": args.method,
        "matcher": args.matcher,
        "train_mode": args.train_mode,
        "class_weighting": args.class_weighting,
        "checkpoint": args.checkpoint,
        "checkpoint_path": None if args.checkpoint_path in {"", "-"} else args.checkpoint_path,
        "candidate_k": int(args.candidate_k),
        "command": args.command,
        "start_time": args.start_time,
        "end_time": "",
        "experiment_run_directory": "",
        "stdout_path": args.stdout_path,
        "stderr_path": args.stderr_path,
        "combined_path": args.combined_path,
        "error_summary": "",
        "submission_id": args.submission_id,
        "manifest_path": args.manifest_path,
        "profile_id": args.profile_id,
        "checkpoint_owner": args.checkpoint_owner,
        "checkpoint_sha256": args.checkpoint_sha256,
        "validation_status": args.validation_status,
        "validation_error": "",
    }
    _write_json(args.path, payload)


def update(args: argparse.Namespace) -> None:
    if not args.path.is_file():
        raise SystemExit(f"metadata file does not exist: {args.path}")
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    payload["status"] = args.status
    payload["end_time"] = args.end_time
    payload["experiment_run_directory"] = args.experiment_run_directory or ""
    payload["error_summary"] = _error_summary(args.error_file)
    if args.validation_status:
        payload["validation_status"] = args.validation_status
    if args.validation_error:
        payload["validation_error"] = args.validation_error
    _write_json(args.path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    _common_parser(init_parser)
    init_parser.add_argument("--job-id", required=True)
    init_parser.add_argument("--task-id", required=True)
    init_parser.add_argument("--dataset", required=True)
    init_parser.add_argument("--animal", required=True)
    init_parser.add_argument("--split-protocol", default="")
    init_parser.add_argument("--method", required=True)
    init_parser.add_argument("--matcher", required=True)
    init_parser.add_argument("--train-mode", default="-")
    init_parser.add_argument("--class-weighting", default="-")
    init_parser.add_argument("--checkpoint", required=True)
    init_parser.add_argument("--checkpoint-path", required=True)
    init_parser.add_argument("--candidate-k", required=True)
    init_parser.add_argument("--command", required=True)
    init_parser.add_argument("--start-time", required=True)
    init_parser.add_argument("--stdout-path", required=True)
    init_parser.add_argument("--stderr-path", required=True)
    init_parser.add_argument("--combined-path", required=True)
    init_parser.add_argument("--submission-id", default="")
    init_parser.add_argument("--manifest-path", default="")
    init_parser.add_argument("--profile-id", default="")
    init_parser.add_argument("--checkpoint-owner", default="")
    init_parser.add_argument("--checkpoint-sha256", default="")
    init_parser.add_argument("--validation-status", default="")
    init_parser.add_argument("--status", choices=("running", "completed", "failed"), default="running")
    init_parser.set_defaults(handler=init)

    update_parser = subparsers.add_parser("update")
    _common_parser(update_parser)
    update_parser.add_argument("--status", choices=("running", "completed", "failed"), required=True)
    update_parser.add_argument("--end-time", required=True)
    update_parser.add_argument("--experiment-run-directory", default="")
    update_parser.add_argument("--error-file")
    update_parser.add_argument("--validation-status", default="")
    update_parser.add_argument("--validation-error", default="")
    update_parser.set_defaults(handler=update)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
