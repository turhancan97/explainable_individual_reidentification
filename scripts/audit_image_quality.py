#!/usr/bin/env python
"""Flag candidate low-quality images in the paper datasets for manual review.

Measures the exact model input: pre-masked files for WildlifeReID-10k and the
load-time COCO-RLE mask for CzechLynx, applied with the probe's own
``BenchmarkDatasetView`` code. Per image it records foreground area, mask
fragmentation, exposure, contrast, and sharpness, and raises heuristic flags.
Flags only rank candidates: every example cited in the paper must be confirmed
by eye from the generated contact sheets.

When completed probe runs with ``scores.npz`` exist, each query image also gets
its top-1 correctness rate across those runs, so flagged and clean queries can
be compared. Outputs go to ``experiments/image-quality/`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.data.dataset_view import BenchmarkDatasetView
from reid.reporting.paper_datasets import PAPER_PROFILES, PaperProfile
from reid.utils.fingerprints import sha256_file

ANALYSIS_LONG_SIDE = 512
# Pre-masked JPEGs leave near-black compression noise in the background.
PREMASKED_FOREGROUND_THRESHOLD = 12
MIN_COMPONENT_FRACTION = 0.005

# Heuristic thresholds. They rank candidates for review, not ground truth.
FLAG_RULES = {
    "empty_foreground": "foreground_fraction < 0.005",
    "tiny_foreground": "foreground_fraction < 0.05 or foreground_pixels < 64*64",
    "fragmented_mask": "mask_components >= 5 and largest_component_fraction < 0.5",
    "overexposed": "saturated_fraction > 0.30",
    "underexposed": "mean_luma < 25",
    "low_contrast": "std_luma < 10",
    "blurry": "sharpness below the dataset's 2nd percentile (review ranking only; excluded from any_flag)",
}
SHEET_ORDER = {
    "empty_foreground": ("foreground_fraction", True),
    "tiny_foreground": ("foreground_fraction", True),
    "fragmented_mask": ("largest_component_fraction", True),
    "overexposed": ("saturated_fraction", False),
    "underexposed": ("mean_luma", True),
    "low_contrast": ("std_luma", True),
    "blurry": ("sharpness", True),
}

_WORKER_FRAME: Optional[pd.DataFrame] = None
_WORKER_PROFILE: Optional[PaperProfile] = None
_WORKER_VIEW: Optional[BenchmarkDatasetView] = None


class _FrameAdapter:
    def __init__(self, frame: pd.DataFrame, label_col: str):
        self.df = frame
        self.metadata = frame
        self.col_label = label_col

    def __len__(self) -> int:
        return len(self.df)


def load_model_input(profile: PaperProfile, frame: pd.DataFrame, view: Optional[BenchmarkDatasetView], idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the RGB image the probe sees and its foreground mask."""
    path = Path(str(frame.iloc[idx]["path"]))
    if not path.is_absolute():
        path = profile.root / path
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if view is not None:
        masked = np.asarray(view._apply_no_background(image, idx))
        foreground = view._decode_mask(frame.iloc[idx], idx).astype(bool)
    else:
        masked = np.asarray(image)
        foreground = masked.max(axis=2) > PREMASKED_FOREGROUND_THRESHOLD
    return masked, foreground


def image_metrics(image: np.ndarray, foreground: np.ndarray) -> Dict[str, float]:
    height, width = foreground.shape
    fraction = float(foreground.mean())
    metrics: Dict[str, float] = {
        "width": width,
        "height": height,
        "foreground_fraction": fraction,
        "foreground_pixels": int(foreground.sum()),
    }
    scale = ANALYSIS_LONG_SIDE / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    small_fg = cv2.resize(foreground.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    luma = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(small_fg.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.zeros(0)
    total = float(areas.sum())
    significant = areas[areas >= MIN_COMPONENT_FRACTION * total] if total else areas
    metrics["mask_components"] = int(len(significant))
    metrics["largest_component_fraction"] = float(areas.max() / total) if total else 0.0

    values = luma[small_fg]
    if values.size == 0:
        metrics.update(mean_luma=0.0, std_luma=0.0, saturated_fraction=0.0, sharpness=0.0, grayscale=True)
        return metrics
    metrics["mean_luma"] = float(values.mean())
    metrics["std_luma"] = float(values.std())
    metrics["saturated_fraction"] = float((values >= 245).mean())
    # Erode so the artificial mask boundary against black does not count as detail.
    interior = cv2.erode(small_fg.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    laplacian = cv2.Laplacian(luma, cv2.CV_32F)
    metrics["sharpness"] = float(laplacian[interior].var()) if interior.sum() >= 64 else 0.0
    rgb = small[small_fg].astype(np.int16)
    metrics["grayscale"] = bool(np.abs(rgb - rgb.mean(axis=1, keepdims=True)).mean() < 3)
    return metrics


def _init_worker(frame: pd.DataFrame, profile: PaperProfile) -> None:
    global _WORKER_FRAME, _WORKER_PROFILE, _WORKER_VIEW
    _WORKER_FRAME, _WORKER_PROFILE = frame, profile
    _WORKER_VIEW = None
    if profile.mask_col:
        _WORKER_VIEW = BenchmarkDatasetView(
            _FrameAdapter(frame, profile.identity_col), label_col=profile.identity_col,
            no_background=True, mask_col=profile.mask_col,
        )


def _measure(idx: int) -> Dict[str, object]:
    try:
        image, foreground = load_model_input(_WORKER_PROFILE, _WORKER_FRAME, _WORKER_VIEW, idx)
        return {"row_index": idx, "error": "", **image_metrics(image, foreground)}
    except Exception as exc:  # Unreadable or malformed inputs are findings too.
        return {"row_index": idx, "error": f"{type(exc).__name__}: {exc}"}


def measure_frame(profile: PaperProfile, frame: pd.DataFrame, indices: Sequence[int], workers: int) -> pd.DataFrame:
    context = mp.get_context("fork")
    with context.Pool(workers, initializer=_init_worker, initargs=(frame, profile)) as pool:
        rows = []
        for done, row in enumerate(pool.imap(_measure, indices, chunksize=32), start=1):
            rows.append(row)
            if done % 2000 == 0 or done == len(indices):
                print(f"[image-quality] {profile.key}: {done}/{len(indices)}", flush=True)
    return pd.DataFrame(rows)


def add_flags(table: pd.DataFrame) -> pd.DataFrame:
    ok = table["error"] == ""
    blur_cut = float(table.loc[ok, "sharpness"].quantile(0.02)) if ok.any() else 0.0
    flags = {
        "empty_foreground": table["foreground_fraction"] < 0.005,
        "tiny_foreground": (table["foreground_fraction"] < 0.05) | (table["foreground_pixels"] < 64 * 64),
        "fragmented_mask": (table["mask_components"] >= 5) & (table["largest_component_fraction"] < 0.5),
        "overexposed": table["saturated_fraction"] > 0.30,
        "underexposed": table["mean_luma"] < 25,
        "low_contrast": table["std_luma"] < 10,
        "blurry": table["sharpness"] <= blur_cut,
    }
    for name, series in flags.items():
        table[f"flag_{name}"] = series.fillna(False).astype(bool) & ok
    flag_cols = [f"flag_{name}" for name in flags]
    table["flag_unreadable"] = ~ok
    table["flags"] = table[flag_cols + ["flag_unreadable"]].apply(
        lambda row: ";".join(col[5:] for col, value in row.items() if value), axis=1
    )
    # Blur is dataset-relative (always ~2%), so it ranks review candidates but
    # is kept out of any_flag and the flagged-vs-clean accuracy comparison.
    absolute = [col for col in flag_cols if col != "flag_blurry"] + ["flag_unreadable"]
    table["any_flag"] = table[absolute].any(axis=1)
    table.attrs["blur_cut"] = blur_cut
    return table


def query_top1_rates(profile: PaperProfile, frame: pd.DataFrame, experiment_root: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Per-query top-1 correctness across completed runs of this split."""
    split = frame[profile.split_col].astype(str)
    identity = frame[profile.identity_col].astype(str)
    query_rows = np.flatnonzero((split == profile.query_value).to_numpy())
    database_rows = np.flatnonzero((split == profile.database_value).to_numpy())
    query_ids = identity.to_numpy()[query_rows]
    database_ids = identity.to_numpy()[database_rows]
    base = experiment_root / "probe" / profile.dataset_name / profile.animal / profile.split_col
    correct = np.zeros(len(query_rows))
    runs: List[str] = []
    for scores_path in sorted(base.glob("*/*/*/*/scores.npz")):
        manifest_path = scores_path.with_name("run_manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("status") != "completed":
            continue
        with np.load(scores_path) as data:
            shape = tuple(int(v) for v in data["shape"])
            rows, cols, values = data["rows"], data["cols"], data["values"]
        if shape != (len(query_rows), len(database_rows)):
            print(f"[image-quality] skip {scores_path}: shape {shape} does not match split", flush=True)
            continue
        # Shared ranking rule: descending score, lowest database index on ties.
        order = np.lexsort((cols, -values, rows))
        rows, cols = rows[order], cols[order]
        first = np.r_[True, rows[1:] != rows[:-1]]
        hit = np.zeros(len(query_rows), dtype=bool)
        hit[rows[first]] = database_ids[cols[first]] == query_ids[rows[first]]
        correct += hit
        runs.append(str(scores_path.parent))
    rates = pd.DataFrame({
        "row_index": query_rows,
        "query_runs": len(runs),
        "query_top1_rate": correct / len(runs) if runs else np.nan,
    })
    return rates, runs


def verify_query_order(profile: PaperProfile, frame: pd.DataFrame, runs: Sequence[str]) -> None:
    """Fail closed if a run's visualization index disagrees with metadata order."""
    split = frame[profile.split_col].astype(str)
    query_paths = frame.loc[split == profile.query_value, "path"].astype(str).to_numpy()
    for run in runs[:3]:
        index = Path(run) / "visualizations" / "index.csv"
        if not index.is_file():
            continue
        sample = pd.read_csv(index, usecols=["query_index", "query_path"]).dropna().drop_duplicates().head(200)
        sample["query_index"] = sample["query_index"].astype(int)
        mismatched = sample[query_paths[sample["query_index"].to_numpy()] != sample["query_path"].astype(str).to_numpy()]
        if len(mismatched):
            raise SystemExit(f"{run}: query order differs from metadata order; refusing to join correctness")


def contact_sheet(profile: PaperProfile, frame: pd.DataFrame, table: pd.DataFrame, flag: str, output: Path, count: int) -> int:
    metric, ascending = SHEET_ORDER[flag]
    picked = table[table[f"flag_{flag}"]].sort_values([metric, "row_index"], ascending=[ascending, True]).head(count)
    if picked.empty:
        return 0
    view = None
    if profile.mask_col:
        view = BenchmarkDatasetView(_FrameAdapter(frame, profile.identity_col), label_col=profile.identity_col, no_background=True, mask_col=profile.mask_col)
    thumb, caption, columns = 200, 30, 8
    rows = (len(picked) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + caption)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for slot, (_, record) in enumerate(picked.iterrows()):
        idx = int(record["row_index"])
        image, _ = load_model_input(profile, frame, view, idx)
        tile = Image.fromarray(image)
        tile.thumbnail((thumb, thumb))
        x, y = (slot % columns) * thumb, (slot // columns) * (thumb + caption)
        sheet.paste(tile, (x + (thumb - tile.width) // 2, y + (thumb - tile.height) // 2))
        draw.text((x + 4, y + thumb + 2), f"row {idx} {record['side']}", fill=(0, 0, 0))
        draw.text((x + 4, y + thumb + 15), f"{metric}={record[metric]:.3g}", fill=(90, 90, 90))
    sheet.save(output, quality=90)
    return len(picked)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", choices=[p.key for p in PAPER_PROFILES], help="Dataset key; repeat for several (default: all)")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/image-quality"))
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Measure only the first N used images (development)")
    parser.add_argument("--sheet-count", type=int, default=48, help="Images per contact sheet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = [p for p in PAPER_PROFILES if not args.dataset or p.key in args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = args.output_dir / "contact_sheets"
    sheets_dir.mkdir(exist_ok=True)
    measured: Dict[Path, pd.DataFrame] = {}
    summaries: List[Dict[str, object]] = []
    sources: List[Dict[str, object]] = []
    for profile in profiles:
        frame = pd.read_csv(profile.metadata, low_memory=False)
        split = frame[profile.split_col].astype(str)
        used = np.flatnonzero(split.isin([profile.database_value, profile.query_value]).to_numpy())
        if args.limit:
            used = used[: args.limit]
        # Closed and open Lynx share images; measure each metadata file once.
        cached = measured.get(profile.metadata)
        missing = sorted(set(used) - set(cached["row_index"])) if cached is not None else list(used)
        if missing:
            fresh = measure_frame(profile, frame, missing, args.workers)
            cached = fresh if cached is None else pd.concat([cached, fresh], ignore_index=True)
            measured[profile.metadata] = cached
        table = cached[cached["row_index"].isin(used)].sort_values("row_index").reset_index(drop=True)
        table.insert(1, "side", np.where(split.to_numpy()[table["row_index"]] == profile.query_value, "query", "database"))
        table.insert(2, "identity", frame[profile.identity_col].astype(str).to_numpy()[table["row_index"]])
        table.insert(3, "path", frame["path"].astype(str).to_numpy()[table["row_index"]])
        table = add_flags(table)
        blur_cut = table.attrs["blur_cut"]

        rates, runs = query_top1_rates(profile, frame, args.experiment_root)
        verify_query_order(profile, frame, runs)
        table = table.merge(rates, on="row_index", how="left")
        table.to_csv(args.output_dir / f"{profile.key}.csv", index=False, float_format="%.6g")

        row: Dict[str, object] = {"dataset": profile.key, "label": profile.label, "images": len(table), "scored_runs": len(runs)}
        queries = table[table["side"] == "query"]
        for flag in list(FLAG_RULES) + ["unreadable"]:
            column = f"flag_{flag}"
            row[f"{flag}_images"] = int(table[column].sum())
            if flag in SHEET_ORDER:
                contact_sheet(profile, frame, table, flag, sheets_dir / f"{profile.key}__{flag}.jpg", args.sheet_count)
        row["any_flag_images"] = int(table["any_flag"].sum())
        row["any_flag_fraction"] = float(table["any_flag"].mean())
        row["blur_sharpness_cut"] = blur_cut
        if runs:
            row["query_top1_flagged"] = float(queries.loc[queries["any_flag"], "query_top1_rate"].mean())
            row["query_top1_clean"] = float(queries.loc[~queries["any_flag"], "query_top1_rate"].mean())
            row["flagged_queries"] = int(queries["any_flag"].sum())
            row["queries_never_correct"] = int((queries["query_top1_rate"] == 0).sum())
        summaries.append(row)
        sources.append({"dataset": profile.key, "metadata": str(profile.metadata), "metadata_sha256": sha256_file(profile.metadata), "runs": runs})
        print(f"[image-quality] {profile.key}: {row['any_flag_images']} flagged of {len(table)}", flush=True)

    columns = list(dict.fromkeys(key for row in summaries for key in row))
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/audit_image_quality.py",
        "limit": args.limit,
        "analysis_long_side": ANALYSIS_LONG_SIDE,
        "premasked_foreground_threshold": PREMASKED_FOREGROUND_THRESHOLD,
        "flag_rules": FLAG_RULES,
        "sources": sources,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
