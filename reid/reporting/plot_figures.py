"""Prepare and render per-animal candidate-budget accuracy figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reid.reporting.paper_tables import (
    discover_animals,
    discover_records,
    discover_splits,
    select_latest_records,
)

DEFAULT_PLOT_BUDGETS = (10, 50, 100, 250, 500, 1000)
DEFAULT_PLOT_METRICS = ("top_1", "top_5", "top_10", "balanced_top_1")
PLOT_METRICS = {
    "top_1": "Top-1 accuracy",
    "top_5": "Top-5 accuracy",
    "top_10": "Top-10 accuracy",
    "balanced_top_1": "Balanced Top-1 accuracy",
}

# The order and visual encoding match the paper figures. A series is included
# only when at least one completed run exists for the selected animal.
PLOT_SERIES = (
    {
        "name": "WildFusion",
        "method_key": "wildfusion",
        "matcher": "-",
        "checkpoint": "default",
        "color": "#555555",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "name": "LoMa default",
        "method_key": "vismatch",
        "matcher": "loma",
        "checkpoint": "default",
        "color": "#2b6cb0",
        "marker": "s",
        "linestyle": "-",
    },
    {
        "name": "LoMa fine-tuned",
        "method_key": "vismatch",
        "matcher": "loma",
        "checkpoint": "custom",
        "color": "#ed7d16",
        "marker": "^",
        "linestyle": "--",
    },
    {
        "name": "RDD-LightGlue default",
        "method_key": "vismatch",
        "matcher": "rdd-lightglue",
        "checkpoint": "default",
        "color": "#2f855a",
        "marker": "D",
        "linestyle": "-",
    },
    {
        "name": "RDD-LightGlue fine-tuned",
        "method_key": "vismatch",
        "matcher": "rdd-lightglue",
        "checkpoint": "custom",
        "color": "#c53030",
        "marker": "*",
        "linestyle": "--",
    },
)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalise_checkpoint(value: Any) -> str:
    value = str(value or "").lower()
    return "custom" if value == "fine-tuned" else value


def _series_key(series: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(series["method_key"]),
        str(series["matcher"]),
        _normalise_checkpoint(series["checkpoint"]),
    )


def prepare_series_data(
    records: Iterable[Mapping[str, Any]],
    *,
    animal: str,
    metric: str,
    budgets: Sequence[int] = DEFAULT_PLOT_BUDGETS,
    split_protocol: str | None = None,
) -> list[dict[str, Any]]:
    """Return plotting-ready series for one animal and split.

    ``split_protocol`` is part of the selection identity.  Without this
    filter, closed and open split records with the same method and budget
    could overwrite one another while preparing the plot.
    """
    if metric not in PLOT_METRICS:
        valid = ", ".join(PLOT_METRICS)
        raise ValueError(f"unsupported metric {metric!r}; choose one of: {valid}")
    budgets = tuple(int(budget) for budget in budgets)
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must contain positive integers")

    selected = select_latest_records(records, animal, split_protocol)
    by_key = {
        (
            str(record.get("method_key") or ""),
            str(record.get("matcher") or ""),
            _normalise_checkpoint(record.get("checkpoint")),
            record.get("candidate_k"),
        ): record
        for record in selected
    }
    result: list[dict[str, Any]] = []
    for series in PLOT_SERIES:
        key = _series_key(series)
        values = [
            _finite_float(by_key.get((*key, budget), {}).get(metric))
            if by_key.get((*key, budget)) is not None
            else None
            for budget in budgets
        ]
        if any(value is not None for value in values):
            result.append({**series, "budgets": budgets, "values": values})
    return result


def _panel_ylim(series_data: Sequence[Mapping[str, Any]]) -> tuple[float, float] | None:
    values = [value for series in series_data for value in series["values"] if value is not None]
    if not values:
        return None
    lower, upper = min(values) * 100.0, max(values) * 100.0
    span = max(upper - lower, 1.0)
    margin = max(span * 0.12, 0.5)
    return lower - margin, upper + margin


def render_metric_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    animals: Sequence[str],
    metric: str,
    budgets: Sequence[int] = DEFAULT_PLOT_BUDGETS,
    split_protocol: str | None = None,
):
    """Build a matplotlib figure without saving it.

    Matplotlib is imported lazily so data-preparation tests and environments
    that only inspect artifacts do not need to import the plotting stack.
    """
    if metric not in PLOT_METRICS:
        valid = ", ".join(PLOT_METRICS)
        raise ValueError(f"unsupported metric {metric!r}; choose one of: {valid}")
    import matplotlib.pyplot as plt

    records = list(records)
    animals = list(animals)
    if not animals:
        raise ValueError("at least one animal is required")
    columns = min(3, max(1, len(animals)))
    rows = math.ceil(len(animals) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.3 * columns, 4.25 * rows),
        squeeze=False,
    )
    axes_flat = [axis for row in axes for axis in row]
    handles = []
    labels = []
    budget_values = tuple(int(budget) for budget in budgets)
    x_values = list(range(len(budget_values)))
    for index, animal in enumerate(animals):
        axis = axes_flat[index]
        series_data = prepare_series_data(
            records,
            animal=animal,
            metric=metric,
            budgets=budget_values,
            split_protocol=split_protocol,
        )
        for series in series_data:
            plotted_x = [x for x, value in zip(x_values, series["values"]) if value is not None]
            plotted_y = [value * 100.0 for value in series["values"] if value is not None]
            line = axis.plot(
                plotted_x,
                plotted_y,
                label=series["name"],
                color=series["color"],
                marker=series["marker"],
                linestyle=series["linestyle"],
                linewidth=2.0,
                markersize=6.5,
                markerfacecolor="white",
                markeredgewidth=1.5,
            )[0]
            if series["name"] not in labels:
                handles.append(line)
                labels.append(series["name"])
        axis.set_title(animal, fontsize=13, fontweight="bold")
        axis.set_xlabel("k")
        axis.set_ylabel("Accuracy (%)")
        axis.set_xticks(x_values, [str(budget) for budget in budget_values])
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        axis.grid(axis="x", color="#eeeeee", linewidth=0.6)
        axis.set_axisbelow(True)
        limits = _panel_ylim(series_data)
        if limits is not None:
            axis.set_ylim(*limits)
        if not series_data:
            axis.text(0.5, 0.5, "No completed data", ha="center", va="center", transform=axis.transAxes)

    for axis in axes_flat[len(animals):]:
        axis.set_visible(False)
    title = f"{PLOT_METRICS[metric]} versus k"
    if split_protocol:
        title += f" ({split_protocol})"
    figure.suptitle(title, fontsize=16, fontweight="bold")
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(3, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )
    figure.tight_layout(rect=(0, 0.10 if handles else 0.02, 1, 0.94))
    return figure


def plot_metrics(
    experiment_root: Path,
    output_dir: Path,
    *,
    animals: Sequence[str] | None = None,
    split_protocols: Sequence[str] | None = None,
    metrics: Sequence[str] = DEFAULT_PLOT_METRICS,
    budgets: Sequence[int] = DEFAULT_PLOT_BUDGETS,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Render selected metrics for the requested or automatically discovered animals."""
    records = discover_records(experiment_root)
    discovered = discover_animals(records)
    selected_animals = list(animals) if animals else discovered
    missing_animals = sorted(set(selected_animals) - set(discovered))
    if missing_animals:
        raise ValueError(f"no completed probe records found for animal(s): {', '.join(missing_animals)}")
    selected_metrics = tuple(metrics)
    if "all" in selected_metrics:
        selected_metrics = tuple(PLOT_METRICS)
    unknown_metrics = sorted(set(selected_metrics) - set(PLOT_METRICS))
    if unknown_metrics:
        raise ValueError(f"unsupported metric(s): {', '.join(unknown_metrics)}")
    formats = tuple(str(fmt).lower().lstrip(".") for fmt in formats)
    allowed_formats = {"png", "pdf", "svg"}
    unknown_formats = sorted(set(formats) - allowed_formats)
    if unknown_formats:
        raise ValueError(f"unsupported output format(s): {', '.join(unknown_formats)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    available_splits = discover_splits(records)
    if split_protocols is not None:
        unknown_splits = sorted(set(split_protocols) - set(available_splits))
        if unknown_splits:
            raise ValueError(
                f"no completed probe records found for split(s): {', '.join(unknown_splits)}"
            )
        plot_groups: list[tuple[str | None, list[str]]] = [
            (
                split,
                [
                    animal
                    for animal in selected_animals
                    if any(
                        record.get("animal") == animal
                        and record.get("split_protocol") == split
                        for record in records
                    )
                ],
            )
            for split in split_protocols
        ]
    elif available_splits:
        # Split-aware artifacts get one figure per split.  This is essential
        # for CzechLynx, where closed and open protocols must never share a
        # panel or overwrite one another.
        plot_groups = [
            (
                split,
                [
                    animal
                    for animal in selected_animals
                    if any(
                        record.get("animal") == animal
                        and record.get("split_protocol") == split
                        for record in records
                    )
                ],
            )
            for split in available_splits
        ]
        # Keep old artifacts without split provenance visible when they are
        # present alongside split-aware runs.
        legacy_animals = [
            animal
            for animal in selected_animals
            if any(
                record.get("animal") == animal and not record.get("split_protocol")
                for record in records
            )
        ]
        if legacy_animals:
            plot_groups.append((None, legacy_animals))
    else:
        plot_groups = [(None, selected_animals)]

    plot_groups = [(split, group) for split, group in plot_groups if group]
    if not plot_groups:
        raise ValueError("no completed probe records found for the requested split(s)")
    outputs: list[Path] = []
    for split_protocol, group_animals in plot_groups:
        suffix = "" if split_protocol is None and not available_splits else (
            f"_{split_protocol}" if split_protocol else "_unspecified"
        )
        for metric in selected_metrics:
            figure = render_metric_figure(
                records,
                animals=group_animals,
                metric=metric,
                budgets=budgets,
                split_protocol=split_protocol,
            )
            try:
                for fmt in formats:
                    output_path = output_dir / f"{metric}_vs_k{suffix}.{fmt}"
                    figure.savefig(output_path, dpi=300, bbox_inches="tight")
                    outputs.append(output_path)
            finally:
                import matplotlib.pyplot as plt

                plt.close(figure)
    return outputs
