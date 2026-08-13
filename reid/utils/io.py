import csv
from pathlib import Path
from typing import Any, Dict, List


def ensure_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def ensure_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")


def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.is_file():
        fieldnames = list(row.keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        return

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames or []
        existing_rows = list(reader)

    if not existing_header:
        fieldnames = list(row.keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        return

    new_keys = [k for k in row.keys() if k not in existing_header]
    if new_keys:
        print(
            f"CSV schema normalized: {csv_path} "
            f"(added {len(new_keys)} column{'s' if len(new_keys) != 1 else ''})"
        )
        _rewrite_csv_with_header(
            csv_path=csv_path,
            rows=existing_rows,
            header=existing_header + new_keys,
        )
        existing_header = existing_header + new_keys

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=existing_header, extrasaction="ignore")
        writer.writerow(row)


def update_csv_rows(
    csv_path: Path,
    match: Dict[str, Any],
    updates: Dict[str, Any],
) -> int:
    """Update rows matching all key/value pairs and rewrite the CSV atomically."""

    if not csv_path.is_file():
        return 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if not header:
        return 0

    new_keys = [key for key in updates if key not in header]
    header.extend(new_keys)
    updated_count = 0
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in match.items()):
            row.update({key: value for key, value in updates.items()})
            updated_count += 1
    if updated_count:
        _rewrite_csv_with_header(csv_path=csv_path, rows=rows, header=header)
    return updated_count


def _rewrite_csv_with_header(csv_path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_sanitize_row(row))
    tmp_path.replace(csv_path)


def _sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # csv.DictReader uses None as a key when a row has more values than the header.
    # We drop these orphan values so schema rewrites stay robust.
    return {k: v for k, v in row.items() if k is not None}
