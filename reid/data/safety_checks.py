from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reid.utils.fingerprints import HASH_ALGORITHM, sha256_file


def _detect_path_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for col in ("path", "filepath", "file", "image_path", "img_path"):
        if col in df.columns:
            return col
    raise KeyError(
        "Could not find an image path column in dataframe. "
        "Expected one of: path, filepath, file, image_path, img_path."
    )


def _normalize_paths(df: pd.DataFrame, root: Path, path_col: str) -> pd.Series:
    def _norm(v: Any) -> str:
        raw = str(v).strip()
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        return p.as_posix()

    return df[path_col].map(_norm)


def _save_histogram(
    counts_a: pd.Series,
    counts_b: pd.Series,
    split_a_name: str,
    split_b_name: str,
    out_path: Path,
) -> None:
    n_ids = max(int(counts_a.size), int(counts_b.size), 1)
    bins = max(10, min(100, int(round(math.sqrt(n_ids) * 2))))
    max_count = max(float(counts_a.max()) if not counts_a.empty else 0.0, float(counts_b.max()) if not counts_b.empty else 0.0)
    if max_count <= 1:
        bins = min(bins, 10)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(counts_a.to_numpy(dtype=np.int64), bins=bins, color="#1f77b4", alpha=0.9)
    ax[0].set_title(f"{split_a_name}: images per identity")
    ax[0].set_xlabel("images per identity")
    ax[0].set_ylabel("number of identities")
    ax[0].grid(alpha=0.2)

    ax[1].hist(counts_b.to_numpy(dtype=np.int64), bins=bins, color="#ff7f0e", alpha=0.9)
    ax[1].set_title(f"{split_b_name}: images per identity")
    ax[1].set_xlabel("images per identity")
    ax[1].set_ylabel("number of identities")
    ax[1].grid(alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_split_safety_checks(
    *,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    split_a_name: str,
    split_b_name: str,
    root: Path,
    label_col: str,
    run_dir: Path,
    preferred_path_col: Optional[str] = None,
    fail_on_overlap: bool = True,
    require_b_labels_in_a: bool = False,
    check_content_hashes: bool = True,
) -> Dict[str, Any]:
    """Validate two splits and report identity coverage.

    ``require_b_labels_in_a`` selects how unseen identities are treated: closed-set
    classification fails on them, while open-set retrieval warns and continues. The
    former ``warn_only_unseen`` flag was unreachable, because it was only ever passed
    together with ``require_b_labels_in_a=False``, which skipped the check entirely.
    """
    if label_col not in df_a.columns:
        raise KeyError(f"label_col '{label_col}' not found in {split_a_name} split")
    if label_col not in df_b.columns:
        raise KeyError(f"label_col '{label_col}' not found in {split_b_name} split")

    path_col = _detect_path_col(df_a, preferred=preferred_path_col)
    if path_col not in df_b.columns:
        raise KeyError(f"path column '{path_col}' found in {split_a_name} but missing in {split_b_name}")

    normalized_a = _normalize_paths(df_a, root=root, path_col=path_col)
    normalized_b = _normalize_paths(df_b, root=root, path_col=path_col)
    overlap = sorted(set(normalized_a.tolist()).intersection(set(normalized_b.tolist())))

    hashes_a: Dict[str, List[str]] = {}
    hashes_b: Dict[str, List[str]] = {}
    missing_files: List[str] = []
    if check_content_hashes:
        for path in list(dict.fromkeys(normalized_a.tolist())):
            resolved = Path(path)
            if not resolved.is_file():
                missing_files.append(resolved.as_posix())
                continue
            hashes_a.setdefault(sha256_file(resolved), []).append(resolved.as_posix())
        for path in list(dict.fromkeys(normalized_b.tolist())):
            resolved = Path(path)
            if not resolved.is_file():
                missing_files.append(resolved.as_posix())
                continue
            hashes_b.setdefault(sha256_file(resolved), []).append(resolved.as_posix())
    duplicate_hashes = sorted(set(hashes_a).intersection(hashes_b))
    duplicate_content = [
        {
            "sha256": digest,
            "split_a_paths": hashes_a[digest][:10],
            "split_b_paths": hashes_b[digest][:10],
        }
        for digest in duplicate_hashes[:20]
    ]

    labels_a = set(df_a[label_col].astype(str).tolist())
    labels_b = set(df_b[label_col].astype(str).tolist())
    unseen_b = sorted(labels_b - labels_a)
    seen_b = sorted(labels_b.intersection(labels_a))

    counts_a = df_a.groupby(label_col).size().sort_values(ascending=False)
    counts_b = df_b.groupby(label_col).size().sort_values(ascending=False)

    checks_dir = run_dir / "safety_checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    _save_histogram(
        counts_a=counts_a,
        counts_b=counts_b,
        split_a_name=split_a_name,
        split_b_name=split_b_name,
        out_path=checks_dir / "class_count_histogram.png",
    )

    counts_df = pd.concat(
        [
            counts_a.rename("count").reset_index().rename(columns={label_col: "label"}).assign(split=split_a_name),
            counts_b.rename("count").reset_index().rename(columns={label_col: "label"}).assign(split=split_b_name),
        ],
        ignore_index=True,
    )
    counts_df = counts_df[["split", "label", "count"]]
    counts_df.to_csv(checks_dir / "class_counts.csv", index=False)

    summary: Dict[str, Any] = {
        "path_col": path_col,
        "num_rows_a": int(len(df_a)),
        "num_rows_b": int(len(df_b)),
        "num_unique_labels_a": int(len(labels_a)),
        "num_unique_labels_b": int(len(labels_b)),
        "num_seen_labels_b_in_a": int(len(seen_b)),
        "num_unseen_labels_b_in_a": int(len(unseen_b)),
        "num_overlapping_files": int(len(overlap)),
        "sample_overlapping_files": overlap[:20],
        "content_hash_algorithm": HASH_ALGORITHM if check_content_hashes else None,
        "num_duplicate_content_hashes": int(len(duplicate_hashes)),
        "sample_duplicate_content": duplicate_content,
        "num_missing_or_unreadable_files": int(len(set(missing_files))),
        "sample_missing_or_unreadable_files": sorted(set(missing_files))[:20],
        "sample_unseen_labels_b_in_a": unseen_b[:20],
    }
    with (checks_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[safety] path_col={path_col} "
        f"{split_a_name}: rows={len(df_a)} ids={len(labels_a)} | "
        f"{split_b_name}: rows={len(df_b)} ids={len(labels_b)}"
    )
    print(
        f"[safety] identity coverage ({split_b_name} in {split_a_name}): "
        f"seen={len(seen_b)} unseen={len(unseen_b)}"
    )
    if unseen_b:
        print(f"[safety] unseen sample ({split_b_name} not in {split_a_name}): {unseen_b[:10]}")
    print(f"[safety] overlapping files between {split_a_name}/{split_b_name}: {len(overlap)}")
    if overlap:
        print(f"[safety] overlap sample: {overlap[:5]}")
    print(
        f"[safety] duplicate content hashes between {split_a_name}/{split_b_name}: "
        f"{len(duplicate_hashes)}"
    )
    if missing_files:
        print(f"[safety][warning] missing/unreadable files: {len(set(missing_files))}")
    print(f"[safety] saved artifacts under: {checks_dir}")

    if fail_on_overlap and overlap:
        raise ValueError(
            f"Safety check failed: {len(overlap)} overlapping files between "
            f"{split_a_name} and {split_b_name}. See {checks_dir / 'summary.json'}"
        )

    if check_content_hashes and duplicate_hashes:
        raise ValueError(
            f"Safety check failed: {len(duplicate_hashes)} duplicate image content hashes between "
            f"{split_a_name} and {split_b_name}. See {checks_dir / 'summary.json'}"
        )

    if unseen_b:
        detail = (
            f"{len(unseen_b)} identities in {split_b_name} are not present in {split_a_name}."
        )
        if require_b_labels_in_a:
            raise ValueError(
                f"Safety check failed: {detail} This violates closed-set classification assumptions."
            )
        print(f"[safety][warning] {detail} Open-set retrieval is expected to rank these as misses.")

    return summary
