#!/usr/bin/env python
"""Add balanced_top_5 / balanced_top_10 to probe runs made before those metrics existed.

The shortlist scores of a completed run are stored in ``scores.npz`` (sparse COO: every
scored query/database pair), so the ranking the run produced can be rebuilt exactly and the
missing macro-averaged metrics computed without repeating the matcher. Labels come from the
run's own ``config.snapshot.yaml`` + ``result.json`` (metadata CSV, split column, split
values), in the order ``load_dataset_splits`` selects them.

Nothing is written unless the metrics that the run already reports (top_1, top_5, top_10,
balanced_top_1) are reproduced exactly from the stored scores, which is what makes the
reconstruction trustworthy; runs that fail the check are reported and skipped.

    python scripts/fewshot_backfill_balanced.py --animal CzechLynx --dry-run
    python scripts/fewshot_backfill_balanced.py --animal CzechLynx
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TOP_K = (1, 5, 10)
CHECKED = ("top_1", "top_5", "top_10", "balanced_top_1")
ADDED = ("balanced_top_5", "balanced_top_10")
TOLERANCE = 1e-9


def _labels(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(query labels, database labels) in the order the run evaluated them."""
    import pandas as pd
    import yaml

    try:
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        config = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return None
    dataset = (config or {}).get("dataset") or {}
    label_col = dataset.get("label_col")
    metadata_path = Path(result.get("dataset_root", "")) / str(result.get("metadata_file", ""))
    split_col = result.get("split_col")
    if not (label_col and split_col and metadata_path.is_file()):
        return None
    metadata = pd.read_csv(metadata_path)
    if split_col not in metadata.columns or label_col not in metadata.columns:
        return None
    query = metadata[metadata[split_col] == result.get("query_split_value")][label_col].to_numpy()
    database = metadata[metadata[split_col] == result.get("database_split_value")][label_col].to_numpy()
    return query, database


def _ranking(run_dir: Path, num_query: int, num_database: int) -> list[np.ndarray] | None:
    """Per query, the database indices ordered as ``stable_rank_indices`` orders them:
    scored entries by descending score, ties and the unscored tail by column index."""
    path = run_dir / "scores.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(v) for v in data["shape"])
        rows, cols, values = data["rows"], data["cols"], data["values"]
    if shape != (num_query, num_database):
        return None
    order = np.lexsort((cols, -values, rows))  # per query: score desc, then column index
    rows, cols = rows[order], cols[order]
    boundaries = np.searchsorted(rows, np.arange(num_query + 1))
    ranking = []
    for q_idx in range(num_query):
        scored = cols[boundaries[q_idx]:boundaries[q_idx + 1]]
        if scored.size >= max(TOP_K):
            ranking.append(scored)
            continue
        tail = np.setdiff1d(np.arange(num_database, dtype=np.int64), scored, assume_unique=False)
        ranking.append(np.concatenate([scored, tail]))
    return ranking


def _metrics(ranking: list[np.ndarray], query_labels: np.ndarray, database_labels: np.ndarray) -> dict[str, float]:
    from reid.evaluation.metrics import _balanced_hit_rate

    metrics: dict[str, float] = {}
    for k in TOP_K:
        hits = np.asarray([query_labels[q] in database_labels[row[:k]] for q, row in enumerate(ranking)], dtype=bool)
        metrics[f"top_{k}"] = float(hits.mean()) if hits.size else float("nan")
        metrics[f"balanced_top_{k}"] = _balanced_hit_rate(query_labels, hits) if hits.size else float("nan")
    return metrics


def _matches(computed: float, stored: Any) -> bool:
    stored_value = None if stored is None else float(stored)
    if stored_value is None or math.isnan(stored_value):
        return math.isnan(computed)
    return abs(computed - stored_value) <= TOLERANCE


def backfill(run_dir: Path, *, dry_run: bool) -> str:
    """-> one of 'added', 'present', 'skipped: <reason>'."""
    metrics_path = run_dir / "metrics.json"
    try:
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "skipped: no metrics.json"
    if all(key in stored for key in ADDED):
        return "present"
    try:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "skipped: no run_manifest.json"
    if manifest.get("status") != "completed":
        return "skipped: run not completed"
    labels = _labels(run_dir)
    if labels is None:
        return "skipped: labels not reconstructable"
    query_labels, database_labels = labels
    if (len(query_labels), len(database_labels)) != (manifest.get("num_query"), manifest.get("num_database")):
        return "skipped: metadata no longer matches the run"
    ranking = _ranking(run_dir, len(query_labels), len(database_labels))
    if ranking is None:
        return "skipped: no usable scores.npz"
    computed = _metrics(ranking, query_labels, database_labels)
    mismatched = [key for key in CHECKED if not _matches(computed[key], stored.get(key))]
    if mismatched:
        return "skipped: " + ", ".join(f"{key} {computed[key]:.6f} != {stored.get(key)}" for key in mismatched)
    if dry_run:
        return "added (dry-run): " + ", ".join(f"{key}={computed[key]:.4f}" for key in ADDED)
    stored.update({key: computed[key] for key in ADDED})
    metrics_path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return "added: " + ", ".join(f"{key}={computed[key]:.4f}" for key in ADDED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--animal", help="only runs of this animal (default: all)")
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = parser.parse_args()

    counts: dict[str, int] = {}
    for manifest_path in sorted((args.experiments / "probe").rglob("run_manifest.json")):
        if args.animal:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if manifest.get("animal") != args.animal:
                continue
        status = backfill(manifest_path.parent, dry_run=args.dry_run)
        counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
        if not status.startswith("present"):
            print(f"{manifest_path.parent}: {status}")
    print("; ".join(f"{key}: {value}" for key, value in sorted(counts.items())) or "no runs found")


if __name__ == "__main__":
    main()
