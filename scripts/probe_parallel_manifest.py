#!/usr/bin/env python3
"""Create, inspect, and validate immutable parallel probe manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TASK_FIELDS = (
    "profile_id",
    "dataset_name",
    "animal",
    "root",
    "metadata_file",
    "label_col",
    "mask_col",
    "no_background",
    "image_variant",
    "split_col",
    "database_split_value",
    "query_split_value",
    "calibration_size",
    "method",
    "matcher",
    "checkpoint_label",
    "checkpoint_path",
    "checkpoint_owner",
    "checkpoint_components",
    "loma_arch",
    "train_mode",
    "class_weighting",
    "candidate_k",
    "evaluation_animal",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or a directory deterministically."""
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"checkpoint path does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _path_owner(path: str) -> str | None:
    match = re.search(r"/wildlife-reid-10k/([^/]+)/(?:rdd|loma)-finetuned/", path)
    return match.group(1) if match else None


def _parse_task_line(line: str, line_number: int) -> dict[str, str]:
    values = line.rstrip("\n").split("|")
    legacy_task_format = len(values) < len(TASK_FIELDS)
    # Keep accepting task tables produced before explicit train-mode and
    # class-weighting fields were added.
    if len(values) == len(TASK_FIELDS) - 3:
        values.insert(-1, "-")
        values.insert(-1, "-")
    elif len(values) == len(TASK_FIELDS) - 2:
        values.insert(-1, "-")
    if len(values) == len(TASK_FIELDS) - 1:
        values.append(values[2])
    if len(values) != len(TASK_FIELDS):
        raise ValueError(
            f"task line {line_number} has {len(values)} fields; expected {len(TASK_FIELDS)}"
        )
    task = dict(zip(TASK_FIELDS, values))
    if legacy_task_format and task["method"] == "linear_probe" and task["class_weighting"] == "-":
        # Before weighting was explicit, linear probes used the unweighted
        # objective. Preserve that behavior for already-created task tables.
        task["class_weighting"] = "unweighted"
    if not task["profile_id"] or not task["dataset_name"] or not task["animal"]:
        raise ValueError(f"task line {line_number} is missing a dataset profile")
    if not task["candidate_k"].isdigit() or int(task["candidate_k"]) <= 0:
        raise ValueError(f"task line {line_number} has invalid candidate_k")
    if task["method"] in {"linear_probe", "efficient_probe"}:
        if task["train_mode"] not in {"classifier", "partial", "all"}:
            raise ValueError(f"task line {line_number} has invalid {task['method']} train_mode")
        if task["class_weighting"] not in {"weighted", "unweighted"}:
            raise ValueError(
                f"task line {line_number} has invalid {task['method']} class_weighting"
            )
    elif task["train_mode"] != "-":
        raise ValueError(f"task line {line_number} has train_mode for non-linear probe method")
    elif task["class_weighting"] != "-":
        raise ValueError(f"task line {line_number} has class_weighting for non-linear probe method")
    if not task["evaluation_animal"]:
        task["evaluation_animal"] = task["animal"]
    return task


def _read_task_file(path: Path) -> list[dict[str, str]]:
    tasks = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            tasks.append(_parse_task_line(line, line_number))
    if not tasks:
        raise ValueError("task table is empty")
    return tasks


def _validate_checkpoint(task: dict[str, str]) -> dict[str, Any]:
    source = "default" if task["checkpoint_label"] == "default" else "custom"
    components = task.get("checkpoint_components", "-")
    if task["checkpoint_label"] == "descriptor-fine-tuned" and components != "descriptor_only":
        raise ValueError("descriptor-fine-tuned checkpoints must use checkpoint_components=descriptor_only")
    if task["checkpoint_label"] == "custom" and components == "descriptor_only":
        raise ValueError("descriptor_only checkpoints must use checkpoint_label=descriptor-fine-tuned")
    owner = task["checkpoint_owner"] if source == "custom" else ""
    path_text = task["checkpoint_path"] if source == "custom" else ""
    if source == "default":
        return {"source": source, "path": None, "owner": None, "sha256": None}
    if not owner:
        raise ValueError(f"custom checkpoint owner is missing for {task['profile_id']}")
    path = Path(path_text)
    if not path.exists():
        raise ValueError(f"{task['matcher']} custom checkpoint does not exist: {path}")
    path_owner = _path_owner(path_text)
    if path_owner and path_owner != owner:
        raise ValueError(
            f"checkpoint path owner mismatch: declared={owner} path={path_owner} path={path}"
        )
    if not path_owner and owner != task["animal"]:
        raise ValueError(
            "dataset/checkpoint owner mismatch: "
            f"dataset={task['animal']} checkpoint_owner={owner}; "
            "cross-species checkpoints must use an owner-identifiable path"
        )
    return {
        "source": source,
        "path": str(path.resolve()),
        "owner": owner,
        "sha256": sha256_path(path),
    }


def create_manifest(args: argparse.Namespace) -> None:
    submission_dir = args.submission_dir.resolve()
    if submission_dir.exists():
        if not submission_dir.is_dir() or any(child.name != "tasks.tsv" for child in submission_dir.iterdir()):
            raise ValueError(f"submission directory already exists and is not empty: {submission_dir}")
    else:
        submission_dir.mkdir(parents=True, exist_ok=False)
    config_file = args.config_file.resolve()
    if not config_file.is_file():
        raise ValueError(f"probe config does not exist: {config_file}")
    config_snapshot = submission_dir / "probe.yaml"
    shutil.copy2(config_file, config_snapshot)

    tasks = []
    for index, raw in enumerate(_read_task_file(args.task_file)):
        checkpoint = _validate_checkpoint(raw)
        task = {
            "index": index,
            "profile_id": raw["profile_id"],
            "dataset": {
                "name": raw["dataset_name"],
                "animal": raw["animal"],
                "evaluation_animal": raw["evaluation_animal"],
                "root": raw["root"],
                "metadata_file": raw["metadata_file"],
                "label_col": raw["label_col"],
                "mask_col": raw["mask_col"],
                "no_background": raw["no_background"].lower() == "true",
                "image_variant": raw["image_variant"],
                "split_col": raw["split_col"],
                "split_protocol": raw["split_col"],
                "database_split_value": raw["database_split_value"],
                "query_split_value": raw["query_split_value"],
                "calibration_size": int(raw["calibration_size"]),
            },
            "benchmark": {
                "method": raw["method"],
                "matcher": raw["matcher"],
                "candidate_k": int(raw["candidate_k"]),
                "checkpoint_label": raw["checkpoint_label"],
                "checkpoint_components": raw["checkpoint_components"],
                "loma_arch": raw["loma_arch"],
                "train_mode": raw["train_mode"],
                "class_weighting": raw["class_weighting"],
            },
            "checkpoint": checkpoint,
        }
        tasks.append(task)

    launcher_path = args.launcher_path.resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "submission_id": args.submission_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_dir": str(args.repository_dir.resolve()),
        "launcher_path": str(launcher_path),
        "launcher_sha256": _sha256_file(launcher_path),
        "config_snapshot": str(config_snapshot),
        "config_sha256": _sha256_file(config_snapshot),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    manifest_path = submission_dir / "manifest.json"
    _atomic_json(manifest_path, payload)
    print(manifest_path)


def _load_task(manifest_path: Path, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported parallel probe manifest schema")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or index < 0 or index >= len(tasks):
        raise ValueError(f"task index {index} is outside manifest task range")
    return payload, tasks[index]


def validate_task(args: argparse.Namespace) -> None:
    payload, task = _load_task(args.manifest, args.index)
    config_snapshot = Path(payload["config_snapshot"])
    if not config_snapshot.is_file() or _sha256_file(config_snapshot) != payload["config_sha256"]:
        raise ValueError("immutable probe config snapshot is missing or changed")
    checkpoint = task["checkpoint"]
    if checkpoint["source"] == "custom":
        path = Path(checkpoint["path"])
        if not path.exists():
            raise ValueError(f"custom checkpoint is missing: {path}")
        if sha256_path(path) != checkpoint["sha256"]:
            raise ValueError(f"custom checkpoint content changed after submission: {path}")
        path_owner = _path_owner(str(path))
        if path_owner and path_owner != checkpoint["owner"]:
            raise ValueError(
                f"checkpoint path owner mismatch: declared={checkpoint['owner']} path={path_owner}"
            )
        if not path_owner and checkpoint["owner"] != task["dataset"]["animal"]:
            raise ValueError(
                "dataset/checkpoint owner mismatch: "
                f"dataset={task['dataset']['animal']} checkpoint_owner={checkpoint['owner']}; "
                "cross-species checkpoints must use an owner-identifiable path"
            )
    print("valid")


def emit_shell(args: argparse.Namespace) -> None:
    payload, task = _load_task(args.manifest, args.index)
    dataset = task["dataset"]
    benchmark = task["benchmark"]
    checkpoint = task["checkpoint"]
    values = {
        "SUBMISSION_ID": payload["submission_id"],
        "MANIFEST_PATH": str(args.manifest.resolve()),
        "CONFIG_SNAPSHOT_PATH": payload["config_snapshot"],
        "TASK_INDEX": str(task["index"]),
        "PROFILE_ID": task["profile_id"],
        "DATASET_NAME": dataset["name"],
        "ANIMAL": dataset["animal"],
        "EVALUATION_ANIMAL": dataset.get("evaluation_animal", dataset["animal"]),
        "DATASET_ROOT": dataset["root"],
        "METADATA_FILE": dataset["metadata_file"],
        "LABEL_COL": dataset["label_col"],
        "MASK_COL": dataset["mask_col"],
        "NO_BACKGROUND": str(dataset["no_background"]).lower(),
        "IMAGE_VARIANT": dataset["image_variant"],
        "SPLIT_COL": dataset["split_col"],
        "SPLIT_PROTOCOL": dataset.get("split_protocol", dataset["split_col"]),
        "DATABASE_SPLIT_VALUE": dataset["database_split_value"],
        "QUERY_SPLIT_VALUE": dataset["query_split_value"],
        "CALIBRATION_SIZE": str(dataset["calibration_size"]),
        "METHOD": benchmark["method"],
        "MATCHER": benchmark["matcher"],
        "CANDIDATE_K": str(benchmark["candidate_k"]),
        "CHECKPOINT_LABEL": benchmark["checkpoint_label"],
        "CHECKPOINT_COMPONENTS": benchmark["checkpoint_components"],
        "LOMA_ARCH": benchmark["loma_arch"],
        "TRAIN_MODE": benchmark["train_mode"],
        "CLASS_WEIGHTING": benchmark.get("class_weighting", "-"),
        "CHECKPOINT_SOURCE": checkpoint["source"],
        "CHECKPOINT_PATH": checkpoint["path"] or "",
        "CHECKPOINT_OWNER": checkpoint["owner"] or "",
        "CHECKPOINT_SHA256": checkpoint["sha256"] or "",
        "CONFIG_SHA256": payload["config_sha256"],
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--submission-dir", type=Path, required=True)
    create.add_argument("--submission-id", required=True)
    create.add_argument("--config-file", type=Path, required=True)
    create.add_argument("--task-file", type=Path, required=True)
    create.add_argument("--launcher-path", type=Path, required=True)
    create.add_argument("--repository-dir", type=Path, required=True)
    create.set_defaults(handler=create_manifest)

    for name in ("validate", "emit-shell"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--index", type=int, required=True)
        command.set_defaults(handler=validate_task if name == "validate" else emit_shell)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
