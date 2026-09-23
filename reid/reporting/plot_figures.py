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
UNSEEN_EVAL_PLOT_BUDGETS = (10, 50, 100, 160)
DEFAULT_PLOT_METRICS = ("top_1", "top_5", "top_10", "balanced_top_1")
PLOT_METRICS = {
    "top_1": "Top-1 accuracy",
    "top_5": "Top-5 accuracy",
    "top_10": "Top-10 accuracy",
    "balanced_top_1": "Balanced Top-1 accuracy",
}

PLOT_STYLES = ("paper", "presentation", "diagnostic")
PLOT_METHODS = ("wildfusion", "rdd", "loma")

# The palette is deliberately color-blind friendly.  Line style and marker
# shape also carry meaning, so the figure remains interpretable when printed
# in grayscale or viewed by someone with color-vision deficiency.
PLOT_STYLE_DEFAULTS = {
    "paper": {
        "x_scale": "log",
        "shared_y": True,
        "figsize": (5.15, 3.75),
        "line_width": 2.25,
        "marker_size": 6.5,
        "title_size": 11.5,
        "label_size": 10,
        "tick_size": 8.5,
    },
    "presentation": {
        "x_scale": "categorical",
        "shared_y": True,
        "figsize": (5.8, 4.3),
        "line_width": 2.8,
        "marker_size": 8,
        "title_size": 14,
        "label_size": 12,
        "tick_size": 10,
    },
    "diagnostic": {
        "x_scale": "categorical",
        "shared_y": False,
        "figsize": (5.3, 4.25),
        "line_width": 2.0,
        "marker_size": 6.5,
        "title_size": 13,
        "label_size": 10,
        "tick_size": 9,
    },
}

PLOT_RC_PARAMS = {
    "paper": {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.linewidth": 0.8,
        "axes.labelweight": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    },
    "presentation": {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    },
    "diagnostic": {},
}

# The order and visual encoding match the paper figures. A series is included
# only when at least one completed run exists for the selected animal.
PLOT_SERIES = (
    {
        "name": "WildFusion",
        "method_key": "wildfusion",
        "matcher": "-",
        "checkpoint": "default",
        "color": "#3c3c3c",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "name": "LoMa default",
        "method_key": "vismatch",
        "matcher": "loma",
        "checkpoint": "default",
        "color": "#0072b2",
        "marker": "s",
        "linestyle": "-",
    },
    {
        "name": "LoMa fine-tuned",
        "method_key": "vismatch",
        "matcher": "loma",
        "checkpoint": "custom",
        "color": "#d55e00",
        "marker": "^",
        "linestyle": "--",
    },
    {
        "name": "RDD-LightGlue default",
        "method_key": "vismatch",
        "matcher": "rdd-lightglue",
        "checkpoint": "default",
        "color": "#009e73",
        "marker": "D",
        "linestyle": "-",
    },
    {
        "name": "RDD-LightGlue fine-tuned",
        "method_key": "vismatch",
        "matcher": "rdd-lightglue",
        "checkpoint": "custom",
        "color": "#cc79a7",
        "marker": "*",
        "linestyle": "--",
    },
)

DESCRIPTOR_PLOT_SERIES = {
    "rdd": (
        PLOT_SERIES[0],
        PLOT_SERIES[3],
        {"name": "RDD-LightGlue descriptor fine-tuned", "method_key": "vismatch", "matcher": "rdd-lightglue", "checkpoint": "descriptor-fine-tuned", "color": "#805ad5", "marker": "X", "linestyle": "--"},
    ),
    "loma": (
        PLOT_SERIES[0],
        PLOT_SERIES[1],
        {"name": "LoMa descriptor fine-tuned", "method_key": "vismatch", "matcher": "loma", "checkpoint": "descriptor-fine-tuned", "color": "#dd6b20", "marker": "^", "linestyle": "--"},
    ),
}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalise_checkpoint(value: Any) -> str:
    value = str(value or "").lower()
    if value in {
        "custom",
        "fine-tuned",
        "matcher-fine-tuned",
        "extractor-fine-tuned",
        "full-fine-tuned",
    }:
        return "custom"
    return value


def _series_key(series: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(series["method_key"]),
        str(series["matcher"]),
        _normalise_checkpoint(series["checkpoint"]),
    )


def _series_method_family(series: Mapping[str, Any]) -> str:
    """Return the public method name used by ``--exclude-method``."""
    if str(series.get("method_key", "")).lower() == "wildfusion":
        return "wildfusion"
    matcher = str(series.get("matcher", "")).lower()
    if matcher == "rdd-lightglue":
        return "rdd"
    if matcher == "loma":
        return "loma"
    return matcher


def _normalise_excluded_methods(exclude_methods: Sequence[str] | None) -> set[str]:
    excluded = {str(method).strip().lower() for method in (exclude_methods or ()) if str(method).strip()}
    unknown = sorted(excluded - set(PLOT_METHODS))
    if unknown:
        valid = ", ".join(PLOT_METHODS)
        raise ValueError(f"unsupported excluded method(s): {', '.join(unknown)}; choose from: {valid}")
    return excluded


def _effective_plot_budgets(
    budgets: Sequence[int],
    split_protocol: str | None,
) -> tuple[int, ...]:
    """Resolve candidate budgets that are valid for a plotted split.

    The unseen-identity CzechLynx metadata has 160 gallery images, so budgets
    above 160 are not valid for that split. The default budget grid is replaced
    with the complete valid grid; an explicit custom grid is intersected with
    it so callers can still request a smaller diagnostic plot.
    """
    requested = tuple(int(budget) for budget in budgets)
    if split_protocol == "unseen_eval_split":
        if requested == DEFAULT_PLOT_BUDGETS:
            requested = UNSEEN_EVAL_PLOT_BUDGETS
        else:
            requested = tuple(budget for budget in requested if budget in UNSEEN_EVAL_PLOT_BUDGETS)
    if not requested or any(budget <= 0 for budget in requested):
        raise ValueError("budgets must contain positive integers valid for the selected split")
    return requested


def prepare_series_data(
    records: Iterable[Mapping[str, Any]],
    *,
    animal: str,
    metric: str,
    budgets: Sequence[int] = DEFAULT_PLOT_BUDGETS,
    split_protocol: str | None = None,
    descriptor_family: str | None = None,
    exclude_methods: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return plotting-ready series for one animal and split.

    ``split_protocol`` is part of the selection identity.  Without this
    filter, closed and open split records with the same method and budget
    could overwrite one another while preparing the plot.
    """
    if metric not in PLOT_METRICS:
        valid = ", ".join(PLOT_METRICS)
        raise ValueError(f"unsupported metric {metric!r}; choose one of: {valid}")
    budgets = _effective_plot_budgets(budgets, split_protocol)
    excluded_methods = _normalise_excluded_methods(exclude_methods)

    selected = select_latest_records(records, animal, split_protocol)
    registry = PLOT_SERIES if descriptor_family is None else DESCRIPTOR_PLOT_SERIES[descriptor_family]
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
    for series in registry:
        if _series_method_family(series) in excluded_methods:
            continue
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


def _resolve_plot_style(
    style: str,
    *,
    x_scale: str | None,
    shared_y: bool | None,
) -> dict[str, Any]:
    if style not in PLOT_STYLES:
        valid = ", ".join(PLOT_STYLES)
        raise ValueError(f"unsupported plot style {style!r}; choose one of: {valid}")
    if x_scale not in {None, "log", "categorical"}:
        raise ValueError("x_scale must be 'log' or 'categorical'")
    resolved = dict(PLOT_STYLE_DEFAULTS[style])
    if x_scale is not None:
        resolved["x_scale"] = x_scale
    if shared_y is not None:
        resolved["shared_y"] = bool(shared_y)
    return resolved


def render_metric_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    animals: Sequence[str],
    metric: str,
    budgets: Sequence[int] = DEFAULT_PLOT_BUDGETS,
    split_protocol: str | None = None,
    descriptor_family: str | None = None,
    style: str = "paper",
    x_scale: str | None = None,
    shared_y: bool | None = None,
    label_endpoints: bool = False,
    exclude_methods: Sequence[str] | None = None,
):
    """Build a matplotlib figure without saving it.

    Matplotlib is imported lazily so data-preparation tests and environments
    that only inspect artifacts do not need to import the plotting stack.
    """
    if metric not in PLOT_METRICS:
        valid = ", ".join(PLOT_METRICS)
        raise ValueError(f"unsupported metric {metric!r}; choose one of: {valid}")
    import matplotlib.pyplot as plt

    style_config = _resolve_plot_style(style, x_scale=x_scale, shared_y=shared_y)
    excluded_methods = _normalise_excluded_methods(exclude_methods)

    records = list(records)
    animals = list(animals)
    if not animals:
        raise ValueError("at least one animal is required")
    columns = min(3, max(1, len(animals)))
    rows = math.ceil(len(animals) / columns)
    with plt.rc_context(PLOT_RC_PARAMS[style]):
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(style_config["figsize"][0] * columns, style_config["figsize"][1] * rows),
            squeeze=False,
        )
        figure.patch.set_facecolor("white")
    axes_flat = [axis for row in axes for axis in row]
    handles = []
    labels = []
    budget_values = _effective_plot_budgets(budgets, split_protocol)
    if style_config["x_scale"] == "log":
        x_values = list(budget_values)
    else:
        x_values = list(range(len(budget_values)))
    panel_data: list[tuple[Any, list[dict[str, Any]]]] = []
    for index, animal in enumerate(animals):
        axis = axes_flat[index]
        series_data = prepare_series_data(
            records,
            animal=animal,
            metric=metric,
            budgets=budget_values,
            split_protocol=split_protocol,
            descriptor_family=descriptor_family,
            exclude_methods=excluded_methods,
        )
        panel_data.append((axis, series_data))

    all_series_data = [series for _, series_data in panel_data for series in series_data]
    shared_limits = _panel_ylim(all_series_data) if style_config["shared_y"] else None
    for index, (axis, series_data) in enumerate(panel_data):
        animal = animals[index]
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
                linewidth=style_config["line_width"],
                markersize=style_config["marker_size"],
                markerfacecolor="white",
                markeredgewidth=1.5,
            )[0]
            if series["name"] not in labels:
                handles.append(line)
                labels.append(series["name"])
            if label_endpoints:
                valid_points = [
                    (x, value * 100.0)
                    for x, value in zip(x_values, series["values"])
                    if value is not None
                ]
                if valid_points:
                    last_x, last_y = valid_points[-1]
                    axis.annotate(
                        series["name"],
                        xy=(last_x, last_y),
                        xytext=(5, 0),
                        textcoords="offset points",
                        color=series["color"],
                        fontsize=max(style_config["tick_size"] - 1, 7),
                        va="center",
                    )
        axis.set_title(animal, fontsize=style_config["title_size"], fontweight="bold", pad=8)
        axis.set_xlabel("Candidate budget ($k$)", fontsize=style_config["label_size"])
        axis.set_ylabel("Accuracy (%)", fontsize=style_config["label_size"])
        if style_config["x_scale"] == "log":
            axis.set_xscale("log", base=10)
            axis.set_xlim(min(budget_values) * 0.8, max(budget_values) * 1.25)
        axis.set_xticks(x_values, [str(budget) for budget in budget_values],
                        fontsize=style_config["tick_size"])
        axis.tick_params(axis="y", labelsize=style_config["tick_size"])
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.75)
        if style_config["x_scale"] == "log":
            axis.grid(axis="x", which="major", color="#eeeeee", linewidth=0.6)
        else:
            axis.grid(axis="x", color="#eeeeee", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        limits = shared_limits or _panel_ylim(series_data)
        if limits is not None:
            axis.set_ylim(*limits)
        if not series_data:
            axis.text(0.5, 0.5, "No completed data", ha="center", va="center", transform=axis.transAxes)

    for axis in axes_flat[len(animals):]:
        axis.set_visible(False)
    title = f"{PLOT_METRICS[metric]} versus k"
    if descriptor_family:
        title += f" ({descriptor_family} descriptor fine-tuning)"
    if split_protocol:
        title += f" ({split_protocol})"
    figure.suptitle(title, fontsize=style_config["title_size"] + 2, fontweight="bold")
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(3, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )
    figure.tight_layout(rect=(0, 0.13 if handles else 0.04, 1, 0.93))
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
    descriptor_families: Sequence[str] | None = None,
    style: str = "paper",
    x_scale: str | None = None,
    shared_y: bool | None = None,
    label_endpoints: bool = False,
    exclude_methods: Sequence[str] | None = None,
) -> list[Path]:
    """Render selected metrics for the requested or automatically discovered animals."""
    records = discover_records(experiment_root)
    excluded_methods = _normalise_excluded_methods(exclude_methods)
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
                style=style,
                x_scale=x_scale,
                shared_y=shared_y,
                label_endpoints=label_endpoints,
                exclude_methods=excluded_methods,
            )
            try:
                for fmt in formats:
                    output_path = output_dir / f"{metric}_vs_k{suffix}.{fmt}"
                    figure.savefig(
                        output_path,
                        dpi=600 if fmt == "png" else 300,
                        bbox_inches="tight",
                        facecolor="white",
                        metadata={"Creator": "scripts/plot_paper_figures.py"},
                    )
                    outputs.append(output_path)
            finally:
                import matplotlib.pyplot as plt

                plt.close(figure)
        available_descriptor_families = [
            family for family in (descriptor_families or DESCRIPTOR_PLOT_SERIES)
            if family in DESCRIPTOR_PLOT_SERIES
            and family not in excluded_methods
            and any(
                record.get("animal") in group_animals
                and record.get("split_protocol") == (split_protocol or "")
                and _normalise_checkpoint(record.get("checkpoint")) == "descriptor-fine-tuned"
                and str(record.get("matcher", "")).lower() == ("rdd-lightglue" if family == "rdd" else "loma")
                for record in records
            )
        ]
        for family in available_descriptor_families:
            for metric in selected_metrics:
                figure = render_metric_figure(
                    records,
                    animals=group_animals,
                    metric=metric,
                    budgets=budgets,
                    split_protocol=split_protocol,
                    descriptor_family=family,
                    style=style,
                    x_scale=x_scale,
                    shared_y=shared_y,
                    label_endpoints=label_endpoints,
                    exclude_methods=excluded_methods,
                )
                try:
                    for fmt in formats:
                        output_path = output_dir / f"descriptor_{family}_{metric}_vs_k{suffix}.{fmt}"
                        figure.savefig(
                            output_path,
                            dpi=600 if fmt == "png" else 300,
                            bbox_inches="tight",
                            facecolor="white",
                            metadata={"Creator": "scripts/plot_paper_figures.py"},
                        )
                        outputs.append(output_path)
                finally:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
    return outputs
