#!/usr/bin/env python
"""Export per-identity image counts for the paper's evaluation splits.

Writes one CSV per dataset/split with exact database and query image counts for
every individual, a cross-dataset ``summary.csv``, and ``manifest.json`` with
source metadata hashes. Counts use the same metadata files, identity columns,
and split values as the parallel probe launchers. Rows whose split value is
neither the database nor the query value (for example unlabeled ZindiTurtleRecall
rows) are excluded from the counts and reported in the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.reporting.paper_datasets import PAPER_PROFILES, PaperProfile as Profile
from reid.utils.fingerprints import sha256_file

PROFILES = PAPER_PROFILES

IDENTITY_COLUMNS = [
    "rank",
    "identity",
    "total_images",
    "database_images",
    "query_images",
    "database_share",
    "query_share",
    "in_database",
    "in_query",
    "database_singleton",
]


def gini(counts: np.ndarray) -> float:
    values = np.sort(np.asarray(counts, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(values) / (n * values.sum()))


def identity_table(profile: Profile, frame: pd.DataFrame) -> pd.DataFrame:
    split = frame[profile.split_col].astype(str)
    identity = frame[profile.identity_col].astype(str)
    database = identity[split == profile.database_value].value_counts()
    query = identity[split == profile.query_value].value_counts()
    table = pd.DataFrame({"database_images": database, "query_images": query}).fillna(0).astype(int)
    table.index.name = "identity"
    table = table.reset_index()
    table["total_images"] = table["database_images"] + table["query_images"]
    table["database_share"] = table["database_images"] / max(int(table["database_images"].sum()), 1)
    table["query_share"] = table["query_images"] / max(int(table["query_images"].sum()), 1)
    table["in_database"] = table["database_images"] > 0
    table["in_query"] = table["query_images"] > 0
    table["database_singleton"] = table["database_images"] == 1
    # Deterministic order: most images first, identity string as tie-breaker.
    table = table.sort_values(["total_images", "database_images", "identity"], ascending=[False, False, True], kind="mergesort")
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table[IDENTITY_COLUMNS]


def summary_row(profile: Profile, frame: pd.DataFrame, table: pd.DataFrame) -> Dict[str, object]:
    split = frame[profile.split_col].astype(str)
    excluded = int((~split.isin([profile.database_value, profile.query_value])).sum())
    database = table.loc[table["in_database"], "database_images"].to_numpy()
    query = table.loc[table["in_query"], "query_images"].sort_values(ascending=False).to_numpy()
    top_decile = max(1, int(np.ceil(0.1 * len(query))))
    unseen = table["in_query"] & ~table["in_database"]
    return {
        "dataset": profile.key,
        "label": profile.label,
        "source": profile.source,
        "split_protocol": profile.split_col,
        "total_images": int(table["total_images"].sum()),
        "database_images": int(database.sum()),
        "query_images": int(query.sum()),
        "total_identities": int(len(table)),
        "database_identities": int(len(database)),
        "query_identities": int(len(query)),
        "query_identities_unseen": int(unseen.sum()),
        "query_images_unseen": int(table.loc[unseen, "query_images"].sum()),
        "database_images_per_identity_min": int(database.min()),
        "database_images_per_identity_median": float(np.median(database)),
        "database_images_per_identity_mean": float(database.mean()),
        "database_images_per_identity_max": int(database.max()),
        "database_max_to_median": float(database.max() / np.median(database)),
        "database_gini": gini(database),
        "database_singleton_identities": int((database == 1).sum()),
        "database_singleton_fraction": float((database == 1).mean()),
        "query_images_per_identity_median": float(np.median(query)),
        "query_images_per_identity_max": int(query.max()),
        "query_gini": gini(query),
        "query_share_top10pct_identities": float(query[:top_decile].sum() / query.sum()),
        "excluded_rows_other_split": excluded,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/class-balance"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []
    sources: List[Dict[str, object]] = []
    for profile in PROFILES:
        frame = pd.read_csv(profile.metadata, low_memory=False)
        missing = {profile.identity_col, profile.split_col} - set(frame.columns)
        if missing:
            raise SystemExit(f"{profile.metadata}: missing columns {sorted(missing)}")
        table = identity_table(profile, frame)
        output = args.output_dir / f"{profile.key}.csv"
        table.to_csv(output, index=False, float_format="%.6f")
        summaries.append(summary_row(profile, frame, table))
        sources.append({
            "dataset": profile.key,
            "output": output.name,
            "metadata": str(profile.metadata),
            "metadata_sha256": sha256_file(profile.metadata),
            "identity_col": profile.identity_col,
            "split_col": profile.split_col,
            "database_value": profile.database_value,
            "query_value": profile.query_value,
        })
        print(f"[class-balance] {output} ({len(table)} identities)")
    write_csv(args.output_dir / "summary.csv", summaries)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/export_class_balance.py",
        "identity_columns": IDENTITY_COLUMNS,
        "sources": sources,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[class-balance] {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
