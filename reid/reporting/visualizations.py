"""Index and contact-sheet helpers for run-local qualitative artifacts."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from reid.reporting.artifacts import RunContext


def _label(dataset: Any, index: int, label_col: str) -> str:
    try:
        return str(dataset.df.iloc[int(index)][label_col])
    except Exception:
        return ""


def _path(dataset: Any, index: int, path_col: str = "path") -> str:
    try:
        return str(dataset.df.iloc[int(index)][path_col])
    except Exception:
        return ""


def _query_index_from_path(path: Path) -> Optional[int]:
    match = re.search(r"(?:predictions_|query_)(\d+)", path.stem)
    return int(match.group(1)) if match else None


def prediction_index_rows(
    *,
    prediction_paths: Sequence[str],
    similarity: Any,
    dataset_query: Any,
    dataset_database: Any,
    label_col: str,
    top_k: int,
    path_col: str = "path",
) -> Tuple[List[dict], List[str]]:
    """Build searchable rows and identify incorrect top-1 prediction images."""

    rows: List[dict] = []
    failures: List[str] = []
    for raw_path in prediction_paths:
        path = Path(raw_path)
        query_index = _query_index_from_path(path)
        if query_index is None or query_index >= len(similarity):
            continue
        ranked = similarity[query_index].argsort()[::-1][: int(top_k)]
        query_label = _label(dataset_query, query_index, label_col)
        top1_correct = False
        for rank, database_index in enumerate(ranked, start=1):
            database_index = int(database_index)
            database_label = _label(dataset_database, database_index, label_col)
            correct = bool(query_label and query_label == database_label)
            if rank == 1:
                top1_correct = correct
            rows.append(
                {
                    "artifact_type": "prediction_grid",
                    "query_index": query_index,
                    "database_index": database_index,
                    "rank": rank,
                    "query_identity": query_label,
                    "database_identity": database_label,
                    "score": float(similarity[query_index, database_index]),
                    "correct": correct,
                    "query_path": _path(dataset_query, query_index, path_col),
                    "database_path": _path(dataset_database, database_index, path_col),
                    "artifact_path": path.as_posix(),
                }
            )
        if not top1_correct and path.is_file():
            failures.append(path.as_posix())
    return rows, failures


def append_artifact_rows(rows: List[dict], paths: Iterable[str], artifact_type: str) -> None:
    for path in paths:
        rows.append(
            {
                "artifact_type": artifact_type,
                "query_index": "",
                "database_index": "",
                "rank": "",
                "query_identity": "",
                "database_identity": "",
                "score": "",
                "correct": "",
                "query_path": "",
                "database_path": "",
                "artifact_path": str(path),
            }
        )


def write_visualization_index(context: RunContext, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = context.visualization_dir / "index.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "artifact_type",
        "query_index",
        "database_index",
        "rank",
        "query_identity",
        "database_identity",
        "score",
        "correct",
        "query_path",
        "database_path",
        "artifact_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def create_contact_sheet(
    paths: Sequence[str],
    output_path: Path,
    *,
    title: str,
    columns: int = 2,
) -> Optional[Path]:
    """Create a lightweight contact sheet, skipping cleanly when no images exist."""

    existing = [Path(path) for path in paths if Path(path).is_file()]
    if not existing:
        return None
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    columns = max(1, int(columns))
    thumb_width, thumb_height = 480, 360
    rows = (len(existing) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "white")
    draw = ImageDraw.Draw(sheet)
    for position, image_path in enumerate(existing):
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((thumb_width - 20, thumb_height - 45))
        except Exception:
            continue
        x = (position % columns) * thumb_width
        y = (position // columns) * thumb_height
        sheet.paste(image, (x + (thumb_width - image.width) // 2, y + 25))
        draw.text((x + 8, y + 5), image_path.stem, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def finalize_visualizations(
    context: RunContext,
    *,
    prediction_paths: Sequence[str],
    similarity: Any,
    dataset_query: Any,
    dataset_database: Any,
    label_col: str,
    top_k: int,
    path_col: str = "path",
    extra_paths: Optional[Mapping[str, Sequence[str]]] = None,
) -> dict:
    rows, failures = prediction_index_rows(
        prediction_paths=prediction_paths,
        similarity=similarity,
        dataset_query=dataset_query,
        dataset_database=dataset_database,
        label_col=label_col,
        top_k=top_k,
        path_col=path_col,
    )
    for artifact_type, paths in (extra_paths or {}).items():
        append_artifact_rows(rows, paths, artifact_type)
    index_path = write_visualization_index(context, rows)
    top1_paths = [str(row["artifact_path"]) for row in rows if row.get("rank") == 1]
    top1_sheet = create_contact_sheet(
        top1_paths,
        context.visualization_dir / "contact_sheet_top1.png",
        title="Top-1 predictions",
    )
    failure_sheet = create_contact_sheet(
        failures,
        context.visualization_dir / "contact_sheet_failures.png",
        title="Top-1 failures",
    )
    artifacts = {"index": index_path.as_posix()}
    if top1_sheet is not None:
        artifacts["contact_sheet_top1"] = top1_sheet.as_posix()
    if failure_sheet is not None:
        artifacts["contact_sheet_failures"] = failure_sheet.as_posix()
    return artifacts
