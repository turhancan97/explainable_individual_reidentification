#!/usr/bin/env python3
"""Build a deterministic gallery/query split for unseen identities.

The generated CSV is intended to be passed to the existing probe entrypoint
with ``dataset.split_col`` set to the generated split column.  This utility is
deliberately independent of the probe implementation and does not modify the
source metadata file.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
HASH_ALGORITHM = "sha256"
DEFAULT_OUTPUT_NAME = "metadata_unseen_eval.csv"
DEFAULT_MANIFEST_NAME = "unseen_eval_manifest.json"
DEFAULT_SPLIT_COLUMN = "unseen_eval_split"
DEFAULT_GROUP_COLUMNS = ("encounter",)
DEFAULT_ORDER_COLUMNS = ("date",)
PATH_COLUMNS = ("path", "filepath", "file", "image_path", "img_path")
DATE_FORMATS = (
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%Y/%m/%d",
)


class UnseenSplitError(ValueError):
    """Raised when a safe unseen-identity split cannot be constructed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise_text(value: Any, *, field: str, row_number: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise UnseenSplitError(f"row {row_number}: {field} must not be empty")
    return text


def _parse_order_value(value: Any, *, column: str, row_number: int) -> tuple[int, Any]:
    """Return a comparable date or numeric order key.

    Dates are normalized to ordinal values. Numeric encounter/order values are
    represented by Decimal. Mixing date and numeric values is deterministic,
    but malformed free-form values are rejected rather than sorted silently.
    """
    text = _normalise_text(value, field=column, row_number=row_number)
    iso_text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_text)
        return 0, parsed.date().toordinal()
    except ValueError:
        pass
    for fmt in DATE_FORMATS:
        try:
            return 0, datetime.strptime(text, fmt).date().toordinal()
        except ValueError:
            continue
    try:
        return 1, Decimal(text)
    except InvalidOperation as exc:
        raise UnseenSplitError(
            f"row {row_number}: {column} value {text!r} is not a supported date or number"
        ) from exc


def _resolve_path(value: Any, *, root: Path, row_number: int, path_col: str) -> Path:
    raw = _normalise_text(value, field=path_col, row_number=row_number)
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.expanduser().resolve()


def _detect_path_column(fieldnames: Sequence[str], requested: str | None) -> str:
    if requested:
        if requested not in fieldnames:
            raise UnseenSplitError(f"path column {requested!r} is not present in metadata")
        return requested
    for candidate in PATH_COLUMNS:
        if candidate in fieldnames:
            return candidate
    expected = ", ".join(PATH_COLUMNS)
    raise UnseenSplitError(f"could not detect an image path column; expected one of: {expected}")


def _read_metadata(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise UnseenSplitError(f"metadata file does not exist: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise UnseenSplitError(f"could not read metadata file {path}: {exc}") from exc
    if not fieldnames:
        raise UnseenSplitError(f"metadata file has no header: {path}")
    if not rows:
        raise UnseenSplitError(f"metadata file has no data rows: {path}")
    return fieldnames, rows


def _require_columns(fieldnames: Sequence[str], columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(fieldnames))
    if missing:
        raise UnseenSplitError(f"required metadata columns are missing: {', '.join(missing)}")


def _group_sort_key(group: Mapping[str, Any]) -> tuple[Any, tuple[str, ...]]:
    return group["first_order"], tuple(group["key"])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _base_manifest(
    *,
    metadata_path: Path,
    output_dir: Path,
    output_csv: Path,
    manifest_path: Path,
    label_col: str,
    source_split_col: str,
    dataset: str | None,
    database_value: str,
    query_value: str,
    group_cols: Sequence[str],
    order_cols: Sequence[str],
    split_col: str,
    root: Path,
) -> dict[str, Any]:
    source_hash: str | None = None
    if metadata_path.is_file():
        try:
            source_hash = sha256_file(metadata_path)
        except OSError:
            source_hash = None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "created_at_utc": _utc_now(),
        "source_metadata": {
            "path": metadata_path.as_posix(),
            "sha256": source_hash,
        },
        "output": {
            "directory": output_dir.as_posix(),
            "metadata_path": output_csv.as_posix(),
            "manifest_path": manifest_path.as_posix(),
            "metadata_sha256": None,
        },
        "configuration": {
            "dataset": dataset or metadata_path.parent.name,
            "root": root.as_posix(),
            "label_col": label_col,
            "source_split_col": source_split_col,
            "database_value": database_value,
            "query_value": query_value,
            "group_cols": list(group_cols),
            "order_cols": list(order_cols),
            "generated_split_col": split_col,
            "generated_database_value": "database",
            "generated_query_value": "query",
        },
        "source_counts": {},
        "selected_counts": {},
        "selected_identities": [],
        "excluded_identities": [],
        "validation": {},
        "errors": [],
    }


def generate_unseen_split(
    *,
    metadata: Path,
    output_dir: Path,
    label_col: str,
    source_split_col: str,
    dataset: str | None = None,
    database_value: str = "train",
    query_value: str = "test",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    order_cols: Sequence[str] = DEFAULT_ORDER_COLUMNS,
    split_col: str = DEFAULT_SPLIT_COLUMN,
    root: Path | None = None,
    path_col: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create and validate a deterministic unseen-identity metadata split."""
    metadata_path = metadata.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_csv = output_dir / DEFAULT_OUTPUT_NAME
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME
    root_path = (root or metadata_path.parent).expanduser().resolve()
    group_cols = tuple(group_cols)
    order_cols = tuple(order_cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _base_manifest(
        metadata_path=metadata_path,
        output_dir=output_dir,
        output_csv=output_csv,
        manifest_path=manifest_path,
        label_col=label_col,
        source_split_col=source_split_col,
        dataset=dataset,
        database_value=database_value,
        query_value=query_value,
        group_cols=group_cols,
        order_cols=order_cols,
        split_col=split_col,
        root=root_path,
    )

    try:
        if output_csv.exists() and not force:
            raise UnseenSplitError(
                f"output already exists: {output_csv}; choose another directory or pass --force"
            )
        fieldnames, rows = _read_metadata(metadata_path)
        _require_columns(fieldnames, (label_col, source_split_col, *group_cols, *order_cols))
        path_column = _detect_path_column(fieldnames, path_col)
        if split_col in fieldnames:
            raise UnseenSplitError(f"generated split column already exists in source metadata: {split_col}")

        indexed_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=2):
            indexed_rows.append({"row": row, "row_number": index})
        database_rows = [item for item in indexed_rows if str(item["row"].get(source_split_col, "")).strip() == database_value]
        query_rows = [item for item in indexed_rows if str(item["row"].get(source_split_col, "")).strip() == query_value]
        if not database_rows or not query_rows:
            raise UnseenSplitError(
                f"source split values must select non-empty database/query sets; "
                f"database={len(database_rows)}, query={len(query_rows)}"
            )

        database_labels = {
            _normalise_text(item["row"].get(label_col), field=label_col, row_number=item["row_number"])
            for item in database_rows
        }
        query_labels = {
            _normalise_text(item["row"].get(label_col), field=label_col, row_number=item["row_number"])
            for item in query_rows
        }
        unseen_labels = sorted(query_labels - database_labels)
        manifest["source_counts"] = {
            "total_rows": len(rows),
            "database_rows": len(database_rows),
            "query_rows": len(query_rows),
            "database_identities": len(database_labels),
            "query_identities": len(query_labels),
            "unseen_query_identities": len(unseen_labels),
        }
        if not unseen_labels:
            raise UnseenSplitError("source query split contains no identities absent from the source database split")
        manifest["configuration"]["path_col"] = path_column

        selected: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for label in unseen_labels:
            identity_rows = [
                item for item in query_rows
                if str(item["row"].get(label_col, "")).strip() == label
            ]
            groups: dict[tuple[str, ...], dict[str, Any]] = {}
            for item in identity_rows:
                row = item["row"]
                key = tuple(_normalise_text(row.get(col), field=col, row_number=item["row_number"]) for col in group_cols)
                order_key = tuple(
                    _parse_order_value(row.get(col), column=col, row_number=item["row_number"])
                    for col in order_cols
                )
                group = groups.setdefault(key, {"key": key, "rows": [], "first_order": order_key})
                group["rows"].append(item)
                if order_key < group["first_order"]:
                    group["first_order"] = order_key

            if len(groups) < 2:
                excluded.append({"identity": label, "reason": "fewer_than_two_groups", "num_groups": len(groups)})
                continue

            ordered_groups = sorted(groups.values(), key=_group_sort_key)
            gallery_group = ordered_groups[0]
            gallery_rows = gallery_group["rows"]
            query_group_rows = [item for group in ordered_groups[1:] for item in group["rows"]]
            if not gallery_rows or not query_group_rows:
                excluded.append({"identity": label, "reason": "empty_gallery_or_query_group", "num_groups": len(groups)})
                continue

            for item in gallery_rows:
                selected.append({"item": item, "generated_split": "database", "group_key": gallery_group["key"]})
            for group in ordered_groups[1:]:
                for item in group["rows"]:
                    selected.append({"item": item, "generated_split": "query", "group_key": group["key"]})

        manifest["excluded_identities"] = excluded
        if not selected:
            raise UnseenSplitError("no unseen identities have both a gallery and query group")

        selected.sort(
            key=lambda item: (
                0 if item["generated_split"] == "database" else 1,
                str(item["item"]["row"].get(label_col, "")),
                tuple(item["group_key"]),
                str(item["item"]["row"].get(path_column, "")),
                item["item"]["row_number"],
            )
        )

        missing_or_unreadable: list[str] = []
        resolved_paths: dict[str, str] = {}
        for selected_item in selected:
            item = selected_item["item"]
            resolved = _resolve_path(item["row"].get(path_column), root=root_path, row_number=item["row_number"], path_col=path_column)
            try:
                sha256_file(resolved)
            except (OSError, IOError):
                missing_or_unreadable.append(resolved.as_posix())
                continue
            resolved_paths[str(item["row_number"])] = resolved.as_posix()
        if missing_or_unreadable:
            manifest["validation"] = {
                "path_col": path_column,
                "missing_or_unreadable_files": missing_or_unreadable,
                "content_hash_algorithm": HASH_ALGORITHM,
            }
            raise UnseenSplitError(
                f"{len(missing_or_unreadable)} selected image files are missing or unreadable"
            )

        gallery_paths = {
            resolved_paths[str(item["item"]["row_number"])]
            for item in selected
            if item["generated_split"] == "database"
        }
        query_paths = {
            resolved_paths[str(item["item"]["row_number"])]
            for item in selected
            if item["generated_split"] == "query"
        }
        path_overlap = sorted(gallery_paths.intersection(query_paths))
        gallery_hashes: dict[str, list[str]] = {}
        query_hashes: dict[str, list[str]] = {}
        for selected_item in selected:
            row_number = str(selected_item["item"]["row_number"])
            path = resolved_paths[row_number]
            digest = sha256_file(Path(path))
            target = gallery_hashes if selected_item["generated_split"] == "database" else query_hashes
            target.setdefault(digest, []).append(path)
        duplicate_hashes = sorted(set(gallery_hashes).intersection(query_hashes))
        duplicate_content = [
            {
                "sha256": digest,
                "database_paths": gallery_hashes[digest][:10],
                "query_paths": query_hashes[digest][:10],
            }
            for digest in duplicate_hashes
        ]
        manifest["validation"] = {
            "path_col": path_column,
            "missing_or_unreadable_files": missing_or_unreadable,
            "num_overlapping_paths": len(path_overlap),
            "sample_overlapping_paths": path_overlap[:20],
            "content_hash_algorithm": HASH_ALGORITHM,
            "num_duplicate_content_hashes": len(duplicate_hashes),
            "sample_duplicate_content": duplicate_content[:20],
        }
        if path_overlap:
            raise UnseenSplitError(f"{len(path_overlap)} image paths overlap between generated database/query sets")
        if duplicate_hashes:
            raise UnseenSplitError(
                f"{len(duplicate_hashes)} duplicate image content hashes cross generated database/query sets"
            )

        output_fieldnames = [*fieldnames, split_col]
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
            writer.writeheader()
            for selected_item in selected:
                row = dict(selected_item["item"]["row"])
                row[split_col] = selected_item["generated_split"]
                writer.writerow(row)

        selected_database = [item for item in selected if item["generated_split"] == "database"]
        selected_query = [item for item in selected if item["generated_split"] == "query"]
        selected_labels = sorted({str(item["item"]["row"][label_col]).strip() for item in selected})
        selected_query_labels = sorted({str(item["item"]["row"][label_col]).strip() for item in selected_query})
        selected_database_labels = sorted({str(item["item"]["row"][label_col]).strip() for item in selected_database})
        manifest["selected_counts"] = {
            "database_rows": len(selected_database),
            "query_rows": len(selected_query),
            "database_identities": len(selected_database_labels),
            "query_identities": len(selected_query_labels),
            "identities": len(selected_labels),
        }
        manifest["selected_identities"] = selected_labels
        manifest["validation"].update({
            "query_identities_all_have_gallery": set(selected_query_labels).issubset(set(selected_database_labels)),
            "selected_identities_absent_from_source_database": not bool(set(selected_labels).intersection(database_labels)),
        })
        manifest["output"]["metadata_sha256"] = sha256_file(output_csv)
        manifest["status"] = "completed"
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["errors"] = [str(exc)]
        _write_json(manifest_path, manifest)
        if isinstance(exc, UnseenSplitError):
            raise
        raise UnseenSplitError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Source metadata CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for CSV and audit manifest")
    parser.add_argument("--label-col", required=True, help="Identity-label column")
    parser.add_argument("--source-split-col", required=True, help="Existing train/test split column")
    parser.add_argument("--dataset", default=None, help="Dataset name recorded in the audit manifest")
    parser.add_argument("--database-value", default="train")
    parser.add_argument("--query-value", default="test")
    parser.add_argument("--group-col", action="append", dest="group_cols", default=None, help="Encounter/group column; repeat for a composite group")
    parser.add_argument("--order-col", action="append", dest="order_cols", default=None, help="Date/order column; repeat for tie-breaking")
    parser.add_argument("--split-col", default=DEFAULT_SPLIT_COLUMN, help="Generated split column")
    parser.add_argument("--root", type=Path, default=None, help="Root for relative image paths; defaults to metadata directory")
    parser.add_argument("--path-col", default=None, help="Image path column; auto-detected when omitted")
    parser.add_argument("--force", action="store_true", help="Overwrite generated CSV if it already exists")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = generate_unseen_split(
            metadata=args.metadata,
            output_dir=args.output_dir,
            label_col=args.label_col,
            source_split_col=args.source_split_col,
            dataset=args.dataset,
            database_value=args.database_value,
            query_value=args.query_value,
            group_cols=tuple(args.group_cols or DEFAULT_GROUP_COLUMNS),
            order_cols=tuple(args.order_cols or DEFAULT_ORDER_COLUMNS),
            split_col=args.split_col,
            root=args.root,
            path_col=args.path_col,
            force=args.force,
        )
    except UnseenSplitError as exc:
        print(f"[unseen-eval][error] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"metadata": manifest["output"]["metadata_path"], "manifest": manifest["output"]["manifest_path"], "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
