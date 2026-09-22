"""Few-shot result collection: accuracy as a function of the training images per individual.

Two retrieval settings are collected for one animal:

* ``reduced`` gallery — probe runs whose ``split_protocol`` is a few-shot column
  ``split_frac<f>_seed<s>`` (written by rdd-parallel-benchmark's ``wildlife_fewshot.py``):
  the kept training images are both the fine-tuning data and the retrieval database;
* ``full`` gallery — probe runs on the same few-shot metadata with ``split_protocol ==
  "split"``: the database is the whole training split while the fine-tuned checkpoint
  comes from a few-shot view (recognised from its path ``.../frac<f>-seed<s>/...``).
  Default (pretrained) checkpoints do not depend on the fraction there, so they are a
  single fraction-independent reference.

The x axis comes from ``$FEWSHOT_ROOT/views/<animal>/<protocol>/<view>/fewshot.json``
(images kept per identity, effective fraction); the benchmark-repository evaluations
``$FEWSHOT_ROOT/eval/<animal>/<protocol>/<view>/*-full.json`` are collected as well.
Probe metrics are stored as fractions in ``metrics.json``; every table and figure here
reports percentages.
"""

from __future__ import annotations

import csv
import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reid.reporting.paper_tables import METHOD_LABELS, _finite_float, _record_from_manifest

SPLIT_PATTERN = re.compile(r"^split_frac(?P<fraction>[0-9.]+)_seed(?P<seed>\d+)$")
VIEW_PATTERN = re.compile(r"/(?P<view>frac(?P<fraction>[0-9.]+)-seed(?P<seed>\d+))/")
# checkpoint directories of the few-shot pipeline: <backend>-finetuned (the matcher, label
# ``custom``) or <backend>-<component>-finetuned (descriptor runs, label ``custom-<component>``
# as submitted by probe-fewshot-wildlife.sh)
COMPONENT_PATTERN = re.compile(
    r"/(?:rdd|loma)-(?:(?P<component>[a-z][a-z-]*?)-)?finetuned(?:-(?P<tag>[a-z]+\d+))?/"
)
TAG_PATTERN = re.compile(r"^[a-z]+\d+$")  # experiment tag such as ep30 (a run of another length)
FINE_TUNED_STATES = {
    "custom": "fine-tuned",
    "custom-descriptor": "fine-tuned descriptor",
    "custom-lg-descriptor": "fine-tuned LG+descriptor",
    "custom-descriptor-matcher": "fine-tuned descriptor+matcher",
}
PLOT_METRICS = ("top_1", "top_5", "top_10", "balanced_top_1", "balanced_top_5", "balanced_top_10",
                "mAP_at_k", "mAP")
DEFAULT_PLOT_METRICS = ("top_1", "top_5", "balanced_top_1")
# the accuracy-vs-k figures: plain and balanced top-k, the balanced ones macro-averaged over
# identities (probe runs made before they existed carry them only after
# scripts/fewshot_backfill_balanced.py has filled them in from the stored scores)
K_SWEEP_METRICS = ("top_1", "top_5", "top_10", "balanced_top_1", "balanced_top_5", "balanced_top_10")
DEFAULT_K_VALUES = (10, 50, 100, 250)
PERCENT_METRICS = ("top_1", "top_5", "top_10", "balanced_top_1", "balanced_top_5", "balanced_top_10",
                   "mAP", "mAP_at_k")
GALLERIES = ("reduced", "full")
CSV_COLUMNS = (
    "animal", "gallery", "view", "fraction", "effective_fraction", "train_images", "train_identities",
    "images_per_identity", "images_per_identity_with_positives", "method", "matcher",
    "checkpoint", "checkpoint_path", "candidate_k", "top_1", "top_5", "top_10", "balanced_top_1",
    "balanced_top_5", "balanced_top_10", "mAP",
    "mAP_at_k", "num_query", "num_database", "runtime_min", "run_id", "manifest_path",
)
METRIC_LABELS = {
    "top_1": "Top-1 (%)", "top_5": "Top-5 (%)", "top_10": "Top-10 (%)",
    "balanced_top_1": "Balanced Top-1 (%)", "balanced_top_5": "Balanced Top-5 (%)",
    "balanced_top_10": "Balanced Top-10 (%)", "mAP_at_k": "mAP@k (%)", "mAP": "mAP (%)",
}
MATCHER_LABELS = {"rdd-lightglue": "RDD-LightGlue", "loma": "LoMa"}
MATCHER_SLUGS = {"rdd-lightglue": "rdd", "loma": "loma"}


def fraction_label(fraction: float | None) -> str:
    """1/8, 1/4, 1/2, full — or the decimal when it is none of those."""
    if fraction is None:
        return "any"
    for denominator in (2, 4, 8, 16, 32):
        if abs(fraction * denominator - 1) < 1e-9:
            return f"1/{denominator}"
    return "full" if abs(fraction - 1) < 1e-9 else f"{fraction:g}"


def parse_split_column(split_col: str) -> tuple[float, int] | None:
    match = SPLIT_PATTERN.match(str(split_col))
    if not match:
        return None
    return float(match.group("fraction")), int(match.group("seed"))


def parse_view_from_path(path: str | None) -> tuple[str, float, int] | None:
    match = VIEW_PATTERN.search(str(path or ""))
    if not match:
        return None
    return match.group("view"), float(match.group("fraction")), int(match.group("seed"))


def view_name(fraction: float, seed: int) -> str:
    text = f"{fraction:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return f"frac{text}-seed{seed}"


def load_views(fewshot_root: Path, animal: str, protocol: str, seed: int) -> dict[str, dict[str, Any]]:
    """view name -> x-axis facts from fewshot.json, for every view of the animal/seed."""
    views: dict[str, dict[str, Any]] = {}
    base = fewshot_root / "views" / animal / protocol
    if not base.is_dir():
        return views
    for path in sorted(base.glob(f"frac*-seed{seed}/fewshot.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        selection = summary.get("fewshot") or {}
        kept = int(selection.get("train_frames_kept") or 0)
        identities = int(selection.get("train_identities") or 0)
        with_positives = int(selection.get("train_identities_with_positives") or 0)
        per_identity = selection.get("per_identity") or {}
        kept_with_positives = sum(int(v["kept"]) for v in per_identity.values() if int(v["kept"]) >= 2)
        views[path.parent.name] = {
            "view": path.parent.name,
            "fraction": float(selection.get("fraction")),
            "seed": int(selection.get("seed", seed)),
            "effective_fraction": _finite_float(selection.get("effective_fraction")),
            "budget_feasible": bool(selection.get("budget_feasible", True)),
            "train_images": kept,
            "train_identities": identities,
            "images_per_identity": kept / identities if identities else None,
            "images_per_identity_with_positives": kept_with_positives / with_positives if with_positives else None,
        }
    return views


def fewshot_metadata_suffix(animal: str) -> str:
    """Path suffix of the few-shot metadata copy written for ``animal``."""
    if animal == "CzechLynx":
        return "metadata_fewshot/CzechLynxDataset-Metadata-Real.csv"
    return f"metadata_fewshot/metadata_{animal}.csv"


def _result_metadata_file(manifest_path: Path) -> str | None:
    """The metadata CSV of a run (only result.json records it)."""
    try:
        result = json.loads(manifest_path.with_name("result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return result.get("metadata_file")


def _checkpoint_path(manifest: Mapping[str, Any]) -> str | None:
    checkpoint = manifest.get("vismatch_checkpoint")
    if isinstance(checkpoint, Mapping):
        return checkpoint.get("requested_path") or checkpoint.get("path")
    return None


def checkpoint_label(checkpoint: str, path: str | None) -> str:
    """Launcher label of a run: ``default``, ``custom`` (fine-tuned matcher),
    ``custom-<component>`` for a descriptor checkpoint and ``…-<tag>`` for a run of another
    length (``ep30``), recognised from the checkpoint directory name
    (``<backend>[-<component>]-finetuned[-<tag>]``; the run manifest only records ``custom``)."""
    if checkpoint != "custom":
        return checkpoint
    match = COMPONENT_PATTERN.search(path or "")
    if match is None:
        return "custom"
    return "-".join(part for part in ("custom", match.group("component"), match.group("tag")) if part)


def split_label(label: str) -> tuple[str, str | None]:
    """``custom-descriptor-ep30`` -> (``custom-descriptor``, ``ep30``)."""
    head, _, last = label.rpartition("-")
    if head and TAG_PATTERN.match(last):
        return head, last
    return label, None


def is_fine_tuned(label: str) -> bool:
    return label == "custom" or label.startswith("custom-")


def probe_records(
    experiment_root: Path, animal: str, seed: int, gallery: str = "reduced", metadata_suffix: str | None = None
) -> list[dict[str, Any]]:
    """Latest completed probe run per (gallery, view, method, matcher, checkpoint, candidate_k).

    ``reduced``: runs on a few-shot split column. ``full``: runs with ``split_protocol ==
    "split"`` on the few-shot metadata file (``metadata_suffix`` defaults to
    ``metadata_fewshot/metadata_<animal>.csv``); a custom checkpoint is attributed to the
    view found in its path, default checkpoints get ``view=None`` (fraction independent).
    """
    if gallery not in GALLERIES:
        raise ValueError(f"gallery must be one of {GALLERIES}")
    suffix = metadata_suffix or fewshot_metadata_suffix(animal)
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for manifest_path in sorted((experiment_root / "probe").rglob("run_manifest.json")):
        record = _record_from_manifest(manifest_path)
        if record is None or record["animal"] != animal:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        split_protocol = str(manifest.get("split_protocol") or "")
        record["checkpoint_path"] = _checkpoint_path(manifest) or ""
        record["checkpoint"] = checkpoint_label(record["checkpoint"], record["checkpoint_path"])
        if gallery == "reduced":
            parsed = parse_split_column(split_protocol)
            if parsed is None or parsed[1] != seed:
                continue
            record["view"] = view_name(parsed[0], seed)
            record["fraction"] = parsed[0]
        else:
            if parse_split_column(split_protocol) is not None:
                continue  # a few-shot column, i.e. the reduced-gallery setting
            metadata_file = _result_metadata_file(manifest_path) or ""
            if not metadata_file.endswith(suffix):
                continue
            if is_fine_tuned(record["checkpoint"]):
                parsed_view = parse_view_from_path(_checkpoint_path(manifest))
                if parsed_view is None or parsed_view[2] != seed:
                    continue
                record["view"], record["fraction"] = parsed_view[0], parsed_view[1]
            else:
                record["view"], record["fraction"] = None, None
        record["gallery"] = gallery
        record["num_query"] = manifest.get("num_query")
        record["num_database"] = manifest.get("num_database")
        key = (gallery, record["view"], record["method_key"], record["matcher"], record["checkpoint"], record["candidate_k"])
        previous = latest.get(key)
        if previous is None or record["_sort_token"] > previous["_sort_token"]:
            latest[key] = record
    return list(latest.values())


def evaluation_records(fewshot_root: Path, animal: str, protocol: str, seed: int) -> list[dict[str, Any]]:
    """rdd-parallel-benchmark evaluations (wildlife_evaluate.py) for the few-shot views."""
    rows = []
    base = fewshot_root / "eval" / animal / protocol
    if not base.is_dir():
        return rows
    for path in sorted(base.glob(f"frac*-seed{seed}/*-full.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        backend, checkpoint, _ = path.stem.split("-", 2)
        rows.append({
            "view": path.parent.name,
            "backend": backend,
            "checkpoint": "default" if checkpoint == "pretrained" else checkpoint,
            "mode": data.get("mode"),
            "frames_per_collection": data.get("frames_per_collection"),
            "n_queries": data.get("n_queries"),
            "n_gallery_collections": data.get("n_gallery_collections"),
            "top1_acc": _finite_float(data.get("top1_acc")),
            "top5_acc": _finite_float(data.get("top5_acc")),
            "frame_accuracy": _finite_float(data.get("frame_accuracy")),
            "mAP": _finite_float(data.get("mAP")),
            "path": path.as_posix(),
        })
    return rows


def series_label(record: Mapping[str, Any]) -> str:
    method = record["method_key"]
    if method != "vismatch":
        return METHOD_LABELS.get(method, method)
    matcher = {"rdd-lightglue": "RDD-LightGlue", "loma": "LoMa"}.get(record["matcher"], record["matcher"])
    label = record["checkpoint"]
    if is_fine_tuned(label):
        base, tag = split_label(label)
        state = FINE_TUNED_STATES.get(base, "fine-tuned " + base[len("custom-"):])
        if tag:
            state = f"{state}, {tag}"
    else:
        state = "default"
    return f"{matcher} ({state})"


def promote_fallback_checkpoints(
    rows: Sequence[dict[str, Any]], fallback_checkpoints: Iterable[str]
) -> list[dict[str, Any]]:
    """Let a stand-in training run take the place of the regular one where that one is missing.

    A checkpoint trained outside the standard pipeline (the single-GPU emergency run of
    ``train_czechlynx_loma_1gpu.sh``, tag ``gpu1``) lives in its own directory and is probed
    under its own label (``custom-gpu1``), so it never overwrites the regular series. Naming
    its directory here relabels its runs to the untagged series — but only where the regular
    run has produced nothing for that view, matcher and candidate budget, so the moment the
    regular checkpoint is probed it wins and the stand-in falls back to its own series.
    Dropping the directory from the call therefore restores the regular checkpoint.
    """
    prefixes = [str(Path(path)) for path in fallback_checkpoints if str(path).strip()]
    if not prefixes:
        return list(rows)
    occupied = {
        (row["gallery"], row["view"], row["method_key"], row["matcher"], row["checkpoint"], row["candidate_k"])
        for row in rows
    }
    promoted = []
    for row in rows:
        path = str(row.get("checkpoint_path") or "")
        base, tag = split_label(str(row["checkpoint"]))
        if not (tag and path and any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)):
            promoted.append(row)
            continue
        key = (row["gallery"], row["view"], row["method_key"], row["matcher"], base, row["candidate_k"])
        if key in occupied:  # the regular run exists for this point: keep the stand-in aside
            promoted.append(row)
            continue
        stand_in = dict(row)
        stand_in["checkpoint"] = base
        stand_in["method"] = series_label({**row, "checkpoint": base})
        stand_in["stand_in_for"] = base
        promoted.append(stand_in)
    return promoted


def _percent(value: Any) -> float | None:
    number = _finite_float(value)
    return None if number is None else 100.0 * number


def build_rows(
    records: Iterable[Mapping[str, Any]], views: Mapping[str, Mapping[str, Any]], animal: str
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        view = views.get(record.get("view") or "", {})
        rows.append({
            "animal": animal,
            "gallery": record.get("gallery", "reduced"),
            "view": record.get("view"),
            "fraction": record.get("fraction"),
            "effective_fraction": view.get("effective_fraction"),
            "train_images": view.get("train_images"),
            "train_identities": view.get("train_identities"),
            "images_per_identity": view.get("images_per_identity"),
            "images_per_identity_with_positives": view.get("images_per_identity_with_positives"),
            "method": series_label(record),
            "method_key": record["method_key"],
            "matcher": record["matcher"],
            "checkpoint": record["checkpoint"],
            "checkpoint_path": record.get("checkpoint_path") or "",
            "candidate_k": record["candidate_k"],
            **{metric: _percent(record.get(metric)) for metric in PERCENT_METRICS},
            "num_query": record.get("num_query"),
            "num_database": record.get("num_database"),
            "runtime_min": record["runtime_min"],
            "run_id": record["run_id"],
            "manifest_path": record["manifest_path"],
        })
    rows.sort(key=lambda r: (r["gallery"], r["fraction"] if r["fraction"] is not None else -1, r["method"], r["candidate_k"] or 0))
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str] = CSV_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    number = _finite_float(value)
    return "–" if number is None else f"{number:.{digits}f}"


def _metric_table(rows: Sequence[Mapping[str, Any]], ordered_views: Sequence[Mapping[str, Any]], metric: str) -> list[str]:
    header = "| method | " + " | ".join(
        f"{view_label(v)} ({_fmt(v['images_per_identity'], 1)} img/id)" for v in ordered_views
    ) + " |"
    lines = [header, "|---|" + "---|" * len(ordered_views)]
    for method in sorted({row["method"] for row in rows}):
        cells = []
        for view in ordered_views:
            match = [r for r in rows if r["method"] == method and (r["view"] == view["view"] or r["view"] is None)]
            cells.append(_fmt(match[0][metric]) if match else "–")
        suffix = " (fraction independent)" if any(r["method"] == method and r["view"] is None for r in rows) else ""
        lines.append(f"| {method}{suffix} | " + " | ".join(cells) + " |")
    return lines


def markdown_summary(
    rows: Sequence[Mapping[str, Any]],
    views: Mapping[str, Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    animal: str,
    metrics: Sequence[str] = ("top_1", "top_5", "balanced_top_1", "mAP_at_k"),
) -> str:
    ordered_views = sorted(views.values(), key=lambda v: v["fraction"])
    lines = [f"# Few-shot results — {animal}", "", "Probe metrics in percent (candidate budget k as listed).", ""]
    lines += ["| view | fraction | effective | train images | identities | images / identity | feasible |",
              "|---|---|---|---|---|---|---|"]
    for view in ordered_views:
        nominal = ", ".join(fraction_label(f) for f in (view.get("nominal_fractions") or [view["fraction"]]))
        lines.append(
            f"| {view['view']} | {nominal} | {_fmt(view['effective_fraction'], 3)} | "
            f"{view['train_images']} | {view['train_identities']} | {_fmt(view['images_per_identity'], 1)} | "
            f"{'yes' if view['budget_feasible'] else 'no'} |"
        )
    lines.append("")
    for gallery in GALLERIES:
        gallery_rows = [r for r in rows if r["gallery"] == gallery]
        if not gallery_rows:
            continue
        title = ("reduced gallery: the kept training images are the retrieval database" if gallery == "reduced"
                 else "full gallery: the whole training split is the retrieval database, checkpoints fine-tuned on the view")
        lines += [f"## Probe — {title}", ""]
        by_k = defaultdict(list)
        for row in gallery_rows:
            by_k[row["candidate_k"]].append(row)
        for candidate_k, k_rows in sorted(by_k.items(), key=lambda item: (item[0] is None, item[0] or 0)):
            for metric in metrics:
                lines.append(f"### {METRIC_LABELS[metric]}, k={candidate_k if candidate_k is not None else '– (full ranking)'}")
                lines.append("")
                lines += _metric_table(k_rows, ordered_views, metric)
                lines.append("")
    if evaluations:
        lines.append("## rdd-parallel-benchmark evaluation (wildlife_evaluate.py, full gallery, ≤20 frames per collection)")
        lines.append("")
        lines += ["| view | backend | checkpoint | queries (collections) | collection top-1 (%) | frame accuracy (%) | mAP (%) |",
                  "|---|---|---|---|---|---|---|"]
        for row in sorted(evaluations, key=lambda r: (views.get(r["view"], {}).get("fraction", 0), r["backend"], r["checkpoint"])):
            lines.append(
                f"| {row['view']} | {row['backend']} | {row['checkpoint']} | {row['n_queries']} | "
                f"{_fmt(_percent(row['top1_acc']))} | {_fmt(_percent(row['frame_accuracy']))} | {_fmt(_percent(row['mAP']))} |"
            )
        lines.append("")
    return "\n".join(lines)


# --- figures -------------------------------------------------------------------------------

# Categorical slots in the validated reference order (dataviz palette); one hue per entity
# (matcher/method), the checkpoint state is carried by line style and marker.
ENTITY_STYLE = {
    "RDD-LightGlue": {"color": "#2a78d6", "slot": 1},
    "LoMa": {"color": "#eb6834", "slot": 2},
    "WildFusion": {"color": "#1baf7a", "slot": 3},
    "Cosine": {"color": "#eda100", "slot": 4},
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
X_LABEL = "training images per individual (mean; fraction of the training split)"


def _entity(label: str) -> str:
    return label.split(" (")[0]


# line style / marker per checkpoint state: defaults solid, every fine-tuned variant dashed
# in its own pattern so the matcher-, descriptor- and joint runs stay apart within one hue
STATE_STYLE = {
    "default": {"linestyle": "-", "marker": "o", "order": 3},
    "fine-tuned": {"linestyle": (0, (5, 2)), "marker": "s", "order": 0},
    "fine-tuned descriptor": {"linestyle": (0, (1, 2)), "marker": "^", "order": 1},
    "fine-tuned LG+descriptor": {"linestyle": (0, (4, 1, 1, 1)), "marker": "D", "order": 2},
    "fine-tuned descriptor+matcher": {"linestyle": (0, (4, 1, 1, 1)), "marker": "D", "order": 2},
}


def _state(label: str) -> str:
    """Checkpoint state of a series label without its experiment tag: ``RDD-LightGlue
    (fine-tuned descriptor, ep30)`` -> ``fine-tuned descriptor``."""
    if " (" not in label or not label.endswith(")"):
        return "default"
    return label[label.find(" (") + 2:-1].split(", ")[0]


def _series_order(label: str) -> tuple[int, int]:
    state = STATE_STYLE.get(_state(label), {"order": 2 if "fine-tuned" in label else 3})
    return (ENTITY_STYLE.get(_entity(label), {"slot": 99})["slot"], state["order"])


def _style(label: str) -> dict[str, Any]:
    """Colour = entity (matcher/method); fine-tuned checkpoints dashed, defaults solid."""
    entity = _entity(label)
    state = STATE_STYLE.get(_state(label), STATE_STYLE["fine-tuned" if "fine-tuned" in label else "default"])
    return {
        "color": ENTITY_STYLE.get(entity, {"color": "#4a3aa7"})["color"],
        "linestyle": state["linestyle"],
        "marker": state["marker"],
    }


def view_label(view: Mapping[str, Any]) -> str:
    """Fraction label of a view: nominal when the budget was met, otherwise the effective
    fraction with the nominal fraction(s) it stands for, e.g. ``0.275 (1/8, 1/4)``."""
    nominal = view.get("nominal_fractions") or [view["fraction"]]
    labels = ", ".join(fraction_label(f) for f in nominal)
    if view.get("budget_feasible", True) and len(nominal) == 1:
        return labels
    effective = _finite_float(view.get("effective_fraction"))
    return f"{effective:.3g} ({labels})" if effective is not None else labels


def views_alias_members(view: Mapping[str, Any]) -> list[str]:
    """Names of all views folded into a merged view (its own name included)."""
    seed = int(view.get("seed", 0))
    return [view_name(f, seed) for f in (view.get("nominal_fractions") or [view["fraction"]])]


def merge_identical_views(
    views: Mapping[str, Mapping[str, Any]], rows: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Fold views that keep exactly the same training images (minimum-per-identity rule
    makes e.g. 1/8 and 1/4 identical) into one x point; the view that has probe runs is
    kept as the canonical name and the rows of its aliases are re-pointed to it."""
    groups: dict[tuple[int, float | None], list[dict[str, Any]]] = defaultdict(list)
    for view in sorted(views.values(), key=lambda v: v["fraction"]):
        groups[(view["train_images"], view.get("effective_fraction"))].append(dict(view))
    used = {row["view"] for row in rows if row.get("view")}
    merged: dict[str, dict[str, Any]] = {}
    for members in groups.values():
        canonical = next((m for m in members if m["view"] in used), members[0])
        canonical["nominal_fractions"] = [m["fraction"] for m in members]
        aliases = {m["view"] for m in members if m is not canonical}
        for row in rows:
            if row.get("view") in aliases:
                row["view"] = canonical["view"]
        merged[canonical["view"]] = canonical
    return merged


def _new_axis(plt, title: str, ylabel: str):
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    axis.set_xscale("log", base=2)
    axis.grid(True, axis="y", color=GRID, linewidth=1, linestyle="-")
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axis.set_xlabel(X_LABEL, color=TEXT_SECONDARY, fontsize=9)
    axis.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    axis.set_title("\n".join(textwrap.wrap(title, 72)), color=TEXT_PRIMARY, fontsize=9, loc="left")
    return figure, axis


def _legend_below(axis, handles=None) -> None:
    kwargs = {"handles": handles} if handles else {}
    axis.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24),
                ncol=2, labelcolor=TEXT_PRIMARY, handlelength=3.2, **kwargs)


def _set_view_ticks(axis, ordered_views: Sequence[Mapping[str, Any]], x_key: str) -> list[float]:
    from matplotlib.ticker import FixedLocator, NullFormatter

    ticks = [(v[x_key], view_label(v)) for v in ordered_views if _finite_float(v.get(x_key)) is not None]
    if ticks:
        axis.xaxis.set_major_locator(FixedLocator([t[0] for t in ticks]))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xticklabels([f"{x:.1f}\n({label})" for x, label in ticks], fontsize=9)
    return [t[0] for t in ticks]


def render_fewshot_figure(
    rows: Sequence[Mapping[str, Any]],
    views: Mapping[str, Mapping[str, Any]],
    *,
    animal: str,
    metric: str,
    candidate_k: int | None,
    gallery: str = "reduced",
    x_key: str = "images_per_identity",
):
    """Metric vs. training images per individual, one line per method/checkpoint.

    In the ``full`` gallery setting the default checkpoints are fraction independent and
    are drawn as horizontal reference lines.
    """
    if metric not in PLOT_METRICS:
        raise ValueError(f"unsupported metric {metric!r}; choose one of {', '.join(PLOT_METRICS)}")
    import matplotlib.pyplot as plt

    selected = [
        r for r in rows
        if r["gallery"] == gallery and (r["candidate_k"] == candidate_k or r["candidate_k"] is None)
        and _finite_float(r[metric]) is not None
    ]
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    references: dict[str, float] = {}
    for row in selected:
        if row["view"] is None:
            references[row["method"]] = float(row[metric])
            continue
        x = _finite_float(row.get(x_key))
        if x is not None:
            series[row["method"]].append((x, float(row[metric])))
    ordered_views = sorted(views.values(), key=lambda v: v["fraction"])
    setting = "reduced gallery" if gallery == "reduced" else "full training gallery"
    budget = f", k={candidate_k}" if candidate_k is not None else ""
    figure, axis = _new_axis(plt, f"{animal} — {setting}{budget}: {METRIC_LABELS[metric]} vs. training images per individual",
                             METRIC_LABELS[metric])
    xs = _set_view_ticks(axis, ordered_views, x_key)
    handles = []
    for label in sorted(series, key=_series_order):
        points = sorted(series[label])
        style = _style(label)
        (line,) = axis.plot(
            [p[0] for p in points], [p[1] for p in points], color=style["color"], linewidth=2,
            linestyle=style["linestyle"], marker=style["marker"], markersize=6.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, solid_joinstyle="round", solid_capstyle="round", label=label,
        )
        handles.append(line)
    if references and xs:
        for label in sorted(references, key=_series_order):
            style = _style(label)
            (line,) = axis.plot(
                [min(xs) / 1.15, max(xs) * 1.15], [references[label]] * 2, color=style["color"], linewidth=2,
                linestyle=style["linestyle"], label=f"{label}, fraction independent",
            )
            handles.append(line)
    if len(series) + len(references) <= 4:  # direct end-labels only while they cannot collide badly
        for label, points in series.items():
            x_last, y_last = sorted(points)[-1]
            axis.annotate(label, (x_last, y_last), xytext=(6, 0), textcoords="offset points",
                          fontsize=8, color=TEXT_PRIMARY, va="center")
    if handles:
        _legend_below(axis, handles)
    figure.tight_layout()
    return figure


# fraction series of the accuracy-vs-k figures: an ordinal ramp of the matcher's own hue
# (light -> dark as the training fraction grows; the blue steps are the validated sequential
# ramp, the orange ones its lightness levels at the slot-2 hue, every step >= 2:1 on the
# surface). The other methods keep their categorical hue from ENTITY_STYLE.
FRACTION_RAMP = {
    "rdd-lightglue": ("#86b6ef", "#5598e7", "#2a78d6", "#184f95"),
    "loma": ("#ff895a", "#eb6834", "#c64d18", "#902500"),
}


def k_sweep_series(
    rows: Sequence[Mapping[str, Any]],
    views: Mapping[str, Mapping[str, Any]],
    *,
    matcher: str,
    metric: str,
    gallery: str = "full",
    k_values: Sequence[int] | None = None,
) -> tuple[list[tuple[str, list[tuple[int, float]]]], dict[str, float]]:
    """(series, references) of one matcher: the fine-tuned views and the matcher's default run
    as ``(label, [(k, value)])``, plus the k-independent methods (cosine) as flat references.

    WildFusion and the default checkpoints do not depend on the fraction, so they are single
    series; the fine-tuned runs give one series per view, ordered by fraction.
    """
    wanted = set(k_values) if k_values else None
    selected = [
        row for row in rows
        if row["gallery"] == gallery and _finite_float(row.get(metric)) is not None
        and (row["candidate_k"] is None or wanted is None or row["candidate_k"] in wanted)
    ]
    ordered_views = [view["view"] for view in sorted(views.values(), key=lambda v: v["fraction"])]
    points: dict[str, list[tuple[int, float]]] = defaultdict(list)
    references: dict[str, float] = {}
    for row in selected:
        value = float(row[metric])
        if row["method_key"] == "vismatch" and row["matcher"] != matcher:
            continue  # the other matcher belongs on its own figure
        if row["candidate_k"] is None:  # no shortlist (cosine): one value for every k
            references[row["method"]] = value
            continue
        checkpoint = str(row["checkpoint"])
        if is_fine_tuned(checkpoint):
            # one line per fraction, from the regular fine-tuned runs only: a component or
            # tagged run (custom-descriptor, custom-gpu1) is a different experiment and would
            # silently double a fraction's line. promote_fallback_checkpoints() is what makes
            # a stand-in run count as the regular one.
            if checkpoint != "custom" or not row["view"]:
                continue
            view = views.get(row["view"])
            label = f"fine-tuned {view_label(view)}" if view else f"fine-tuned {row['view']}"
        else:
            label = row["method"]
        points[label].append((int(row["candidate_k"]), value))
    fraction_labels = [view_label(views[view]) for view in ordered_views if view in views]

    def order(item: tuple[str, list[tuple[int, float]]]) -> tuple[int, int]:
        label = item[0]
        if label.startswith("fine-tuned "):
            name = label[len("fine-tuned "):]
            return (0, fraction_labels.index(name) if name in fraction_labels else 99)
        return (1, _series_order(label)[0])
    series = sorted(((label, sorted(set(values))) for label, values in points.items()), key=order)
    return series, references


def _annotate_without_overlap(axis, labels: Sequence[tuple[float, float, str]]) -> None:
    """Label line ends directly, pushing labels apart when their values nearly coincide
    (the fine-tuned fractions often land within a fraction of a percent of each other)."""
    if not labels:
        return
    axis.autoscale_view()
    low, high = axis.get_ylim()
    minimum_gap = 0.045 * (high - low)
    placed = float("inf")
    for x, y, text in sorted(labels, key=lambda item: -item[1]):
        placed = min(y, placed - minimum_gap)
        axis.annotate(text, (x, placed), xytext=(7, 0), textcoords="offset points", fontsize=8,
                      color=TEXT_PRIMARY, va="center", annotation_clip=False)


def render_k_sweep_figure(
    rows: Sequence[Mapping[str, Any]],
    views: Mapping[str, Mapping[str, Any]],
    *,
    animal: str,
    metric: str,
    matcher: str,
    gallery: str = "full",
    k_values: Sequence[int] | None = None,
):
    """Metric vs. the candidate budget k, one line per fine-tuned fraction plus the default
    checkpoint of the same matcher, WildFusion and (as a flat line) cosine."""
    if metric not in PLOT_METRICS:
        raise ValueError(f"unsupported metric {metric!r}; choose one of {', '.join(PLOT_METRICS)}")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter

    series, references = k_sweep_series(rows, views, matcher=matcher, metric=metric,
                                        gallery=gallery, k_values=k_values)
    setting = "reduced gallery" if gallery == "reduced" else "full training gallery"
    name = MATCHER_LABELS.get(matcher, matcher)
    figure, axis = _new_axis(
        plt, f"{animal} — {name}, {setting}: {METRIC_LABELS[metric]} vs. candidate budget k",
        METRIC_LABELS[metric])
    axis.set_xlabel("candidate budget k (database entries re-ranked per query)",
                    color=TEXT_SECONDARY, fontsize=9)
    axis.set_xscale("log")
    ticks = sorted({k for _, values in series for k, _ in values} | set(k_values or ()))
    if ticks:
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xticklabels([str(k) for k in ticks], fontsize=9)
    ramp = FRACTION_RAMP.get(matcher, FRACTION_RAMP["rdd-lightglue"])
    fractions = [label for label, _ in series if label.startswith("fine-tuned ")]
    handles: list[Any] = []
    end_labels: list[tuple[float, float, str]] = []
    for label, values in series:
        if label.startswith("fine-tuned "):
            index = fractions.index(label)
            step = round(index * (len(ramp) - 1) / max(len(fractions) - 1, 1))  # spread over the ramp
            style = {"color": ramp[step], "linestyle": (0, (5, 2)), "marker": "s"}
        else:
            style = _style(label)
        (line,) = axis.plot(
            [k for k, _ in values], [v for _, v in values], color=style["color"], linewidth=2,
            linestyle=style["linestyle"], marker=style["marker"], markersize=6.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, solid_joinstyle="round",
            solid_capstyle="round", label=label,
        )
        handles.append(line)
        if label.startswith("fine-tuned ") and values:
            end_labels.append((values[-1][0], values[-1][1], label[len("fine-tuned "):]))
    if references and ticks:
        for label in sorted(references, key=_series_order):
            style = _style(label)
            (line,) = axis.plot([min(ticks), max(ticks)], [references[label]] * 2, color=style["color"],
                                linewidth=2, linestyle=(0, (1, 2)), label=f"{label}, k independent")
            handles.append(line)
    _annotate_without_overlap(axis, end_labels)  # after every series: the y range is final
    if handles:
        _legend_below(axis, handles)
    figure.tight_layout()
    return figure


def render_evaluation_figure(evaluations: Sequence[Mapping[str, Any]], views: Mapping[str, Mapping[str, Any]], *, animal: str):
    """Frame accuracy of wildlife_evaluate.py (pretrained vs fine-tuned) vs. images per individual."""
    import matplotlib.pyplot as plt

    figure, axis = _new_axis(plt, f"{animal} — wildlife_evaluate.py frame accuracy (full gallery, ≤20 frames/collection)", "frame accuracy (%)")
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in evaluations:
        view = views.get(row["view"])
        if not view or row["frame_accuracy"] is None or view.get("images_per_identity") is None:
            continue
        name = {"rdd": "RDD-LightGlue", "loma": "LoMa"}.get(row["backend"], row["backend"])
        state = "default" if row["checkpoint"] == "default" else "fine-tuned"
        series[f"{name} ({state})"].append((view["images_per_identity"], 100 * row["frame_accuracy"]))
    for label in sorted(series, key=_series_order):
        points = sorted(series[label])
        style = _style(label)
        axis.plot([p[0] for p in points], [p[1] for p in points], color=style["color"], linewidth=2,
                  linestyle=style["linestyle"], marker=style["marker"], markersize=6.5,
                  markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    _set_view_ticks(axis, sorted(views.values(), key=lambda v: v["fraction"]), "images_per_identity")
    if series:
        _legend_below(axis)
    figure.tight_layout()
    return figure


def collect(
    *,
    experiment_root: Path,
    fewshot_root: Path,
    animal: str,
    protocol: str = "legacy",
    seed: int = 0,
    output_dir: Path,
    metrics: Sequence[str] = DEFAULT_PLOT_METRICS,
    candidate_k: int | None = 50,
    formats: Sequence[str] = ("png", "pdf"),
    k_sweep: bool = False,
    k_sweep_metrics: Sequence[str] = K_SWEEP_METRICS,
    k_values: Sequence[int] | None = DEFAULT_K_VALUES,
    fallback_checkpoints: Sequence[str] = (),
) -> list[Path]:
    views = load_views(fewshot_root, animal, protocol, seed)
    if not views:
        raise ValueError(f"no few-shot views for {animal} under {fewshot_root / 'views' / animal / protocol}")
    records = probe_records(experiment_root, animal, seed, "reduced") + probe_records(experiment_root, animal, seed, "full")
    rows = promote_fallback_checkpoints(build_rows(records, views, animal), fallback_checkpoints)
    evaluations = evaluation_records(fewshot_root, animal, protocol, seed)
    aliases = {v: v for v in views}
    views = merge_identical_views(views, rows)
    for canonical, view in views.items():
        for member in views_alias_members(view):
            aliases[member] = canonical
    for evaluation in evaluations:
        evaluation["view"] = aliases.get(evaluation["view"], evaluation["view"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    csv_path = output_dir / "fewshot_results.csv"
    write_csv(rows, csv_path)
    outputs.append(csv_path)
    if evaluations:
        eval_csv = output_dir / "fewshot_wildlife_evaluate.csv"
        write_csv(evaluations, eval_csv, columns=list(evaluations[0].keys()))
        outputs.append(eval_csv)
    summary_path = output_dir / "fewshot_summary.md"
    summary_path.write_text(markdown_summary(rows, views, evaluations, animal), encoding="utf-8")
    outputs.append(summary_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for gallery in GALLERIES:
        if not any(r["gallery"] == gallery for r in rows):
            continue
        prefix = "fewshot" if gallery == "reduced" else "fewshot_fullgallery"
        for metric in metrics:
            figure = render_fewshot_figure(rows, views, animal=animal, metric=metric, candidate_k=candidate_k, gallery=gallery)
            for fmt in formats:
                path = output_dir / f"{prefix}_{metric}.{fmt}"
                figure.savefig(path, dpi=200, facecolor=figure.get_facecolor())
                outputs.append(path)
            plt.close(figure)
    if k_sweep:
        # one figure per metric and matcher: the fine-tuned fractions, the matcher's default
        # checkpoint, WildFusion and cosine as a function of the candidate budget
        for matcher, slug in MATCHER_SLUGS.items():
            for metric in k_sweep_metrics:
                series, references = k_sweep_series(rows, views, matcher=matcher, metric=metric,
                                                    gallery="full", k_values=k_values)
                if not series and not references:
                    print(f"no full-gallery runs for {matcher} / {metric}: figure skipped")
                    continue
                figure = render_k_sweep_figure(rows, views, animal=animal, metric=metric,
                                               matcher=matcher, gallery="full", k_values=k_values)
                for fmt in formats:
                    path = output_dir / f"fewshot_fullgallery_k_{metric}_{slug}.{fmt}"
                    figure.savefig(path, dpi=200, facecolor=figure.get_facecolor())
                    outputs.append(path)
                plt.close(figure)
    if evaluations:
        figure = render_evaluation_figure(evaluations, views, animal=animal)
        for fmt in formats:
            path = output_dir / f"fewshot_frame_accuracy.{fmt}"
            figure.savefig(path, dpi=200, facecolor=figure.get_facecolor())
            outputs.append(path)
        plt.close(figure)
    return outputs


# --- submission helpers ----------------------------------------------------------------------

def completed_variants(experiment_root: Path, animal: str, split_col: str, candidate_k: int | None) -> set[tuple[str, str, str, str]]:
    """(method, matcher, checkpoint label, checkpoint path) of completed probe runs.

    ``checkpoint path`` is the requested custom checkpoint (``""`` for default runs), so a
    full-gallery run (``split_col == "split"``) is identified by the view it was fine-tuned on.
    Methods without a candidate shortlist (cosine) match any ``candidate_k``.
    """
    done = set()
    for manifest_path in sorted((experiment_root / "probe").rglob("run_manifest.json")):
        record = _record_from_manifest(manifest_path)
        if record is None or record["animal"] != animal:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(manifest.get("split_protocol") or "") != split_col:
            continue
        if parse_split_column(split_col) is None and not (_result_metadata_file(manifest_path) or "").endswith(fewshot_metadata_suffix(animal)):
            continue  # a run of the original metadata, not the few-shot copy
        if record["candidate_k"] is not None and candidate_k is not None and record["candidate_k"] != candidate_k:
            continue
        label = checkpoint_label(record["checkpoint"], _checkpoint_path(manifest))
        path = _checkpoint_path(manifest) if is_fine_tuned(label) else ""
        done.add((record["method_key"], record["matcher"] if record["method_key"] == "vismatch" else "-",
                  label, str(path or "")))
    return done


def missing_variants(experiment_root: Path, animal: str, split_col: str, candidate_k: int | None, variants: Iterable[str]) -> list[str]:
    """Filter launcher variant strings ``method|matcher|label|path`` to those without a completed run."""
    done = completed_variants(experiment_root, animal, split_col, candidate_k)
    missing = []
    for variant in variants:
        method, matcher, label, path = (variant.split("|") + ["", "", "", ""])[:4]
        key = (method, matcher if method == "vismatch" else "-", label, path if is_fine_tuned(label) else "")
        if key not in done:
            missing.append(variant)
    return missing
