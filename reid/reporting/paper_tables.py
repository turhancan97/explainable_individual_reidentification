"""Paper-ready per-animal result table discovery and rendering."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_ABLATION_BUDGETS = (10, 50, 100, 250, 500, 1000)
SHORTLIST_METHODS = {"wildfusion", "vismatch", "local_lightglue"}
METHOD_LABELS = {
    "cosine": "Cosine",
    "wildfusion": "WildFusion",
    "local_lightglue": "Local LightGlue",
    "linear_probe": "Linear Probe",
    "efficient_probe": "Efficient Probe",
    "vismatch": "Vismatch",
}
METHOD_ORDER = {
    "cosine": 0,
    "wildfusion": 1,
    "local_lightglue": 2,
    "linear_probe": 3,
    "efficient_probe": 4,
    "vismatch": 5,
}
TABLE_COLUMNS = (
    "method",
    "matcher",
    "checkpoint",
    "candidate_k",
    "top_1",
    "top_5",
    "top_10",
    "balanced_top_1",
    "mAP",
    "mAP_at_k",
    "runtime_min",
    "total_runtime_min",
    "run_id",
    "manifest_path",
)
METRIC_COLUMNS = ("top_1", "top_5", "top_10", "balanced_top_1", "mAP", "mAP_at_k")
ABLATION_LATEX_METRIC_COLUMNS = ("top_1", "top_5", "top_10", "balanced_top_1")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _checkpoint_source(manifest: Mapping[str, Any]) -> str:
    checkpoint = manifest.get("vismatch_checkpoint")
    if isinstance(checkpoint, Mapping) and checkpoint.get("source"):
        return str(checkpoint["source"])
    return str(manifest.get("variant") or "default")


def _candidate_k(method: str, metrics: Mapping[str, Any], timings: Mapping[str, Any]) -> int | None:
    if method not in SHORTLIST_METHODS:
        return None
    value = metrics.get("map_at_k")
    if value is None:
        value = timings.get("benchmark_candidate_k", timings.get("vismatch_candidate_k"))
    number = _finite_float(value)
    if number is None or number <= 0 or int(number) != number:
        return None
    return int(number)


def _load_json_mapping(path: Path, fallback: Any) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        value = fallback
    return value if isinstance(value, Mapping) else {}


def _record_from_manifest(manifest_path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("status") != "completed" or manifest.get("workflow", "probe") != "probe":
        return None
    method = str(manifest.get("method") or "")
    if method not in METHOD_LABELS:
        return None
    metrics = _load_json_mapping(manifest_path.with_name("metrics.json"), manifest.get("metrics") or {})
    timings = _load_json_mapping(manifest_path.with_name("timings.json"), manifest.get("timings") or {})
    animal = str(manifest.get("animal") or "")
    if not animal:
        return None
    matcher = str(manifest.get("variant") or "-") if method == "vismatch" else "-"
    checkpoint = _checkpoint_source(manifest)
    run_id = str(manifest.get("run_id") or manifest_path.parent.name)
    candidate = _candidate_k(method, metrics, timings)
    primary_runtime_sec = _finite_float(timings.get("primary_compute_runtime_sec"))
    total_runtime_sec = _finite_float(timings.get("total_run_sec"))
    if total_runtime_sec is None:
        legacy_total_min = _finite_float(timings.get("total_run_min"))
        total_runtime_sec = None if legacy_total_min is None else legacy_total_min * 60.0
    record = {
        "animal": animal,
        "method_key": method,
        "method": METHOD_LABELS[method],
        "matcher": matcher,
        "checkpoint": checkpoint,
        "candidate_k": candidate,
        "top_1": _finite_float(metrics.get("top_1")),
        "top_5": _finite_float(metrics.get("top_5")),
        "top_10": _finite_float(metrics.get("top_10")),
        "balanced_top_1": _finite_float(metrics.get("balanced_top_1")),
        "mAP": _finite_float(metrics.get("mAP")) if method not in SHORTLIST_METHODS else None,
        "mAP_at_k": _finite_float(metrics.get("mAP_at_k")) if method in SHORTLIST_METHODS else None,
        "runtime_min": None if primary_runtime_sec is None else primary_runtime_sec / 60.0,
        "total_runtime_min": None if total_runtime_sec is None else total_runtime_sec / 60.0,
        "run_id": run_id,
        "manifest_path": manifest_path.as_posix(),
        "_sort_token": (str(manifest.get("run_utc") or ""), run_id),
    }
    return record


def discover_records(experiment_root: Path) -> list[dict[str, Any]]:
    """Return completed probe records from modern experiment artifacts."""
    records = []
    for manifest_path in sorted((experiment_root / "probe").rglob("run_manifest.json")):
        record = _record_from_manifest(manifest_path)
        if record is not None:
            records.append(record)
    return records


def discover_animals(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return sorted({str(record["animal"]) for record in records if record.get("animal")})


def _selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["animal"],
        record["method_key"],
        record["matcher"],
        record["checkpoint"],
        record["candidate_k"],
    )


def select_latest_records(records: Iterable[Mapping[str, Any]], animal: str) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record.get("animal") != animal:
            continue
        key = _selection_key(record)
        candidate = dict(record)
        previous = selected.get(key)
        if previous is None or candidate["_sort_token"] > previous["_sort_token"]:
            selected[key] = candidate
    return list(selected.values())


def _sort_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda record: (
            METHOD_ORDER.get(record.get("method_key", ""), 99),
            record.get("method", ""),
            record.get("matcher", ""),
            record.get("checkpoint", ""),
            -1 if record.get("candidate_k") is None else record["candidate_k"],
        ),
    )


def _placeholder(template: Mapping[str, Any], candidate_k: int | None) -> dict[str, Any]:
    row = {column: None for column in TABLE_COLUMNS}
    row.update(
        {
            "animal": template["animal"],
            "method_key": template["method_key"],
            "method": template["method"],
            "matcher": template["matcher"],
            "checkpoint": template["checkpoint"],
            "candidate_k": candidate_k,
            "run_id": None,
            "manifest_path": None,
            "_sort_token": ("", ""),
        }
    )
    return row


def build_main_rows(records: Iterable[Mapping[str, Any]], animal: str, candidate_k: int) -> list[dict[str, Any]]:
    selected = select_latest_records(records, animal)
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in selected:
        groups.setdefault((record["method_key"], record["matcher"], record["checkpoint"]), []).append(record)
    for group in groups.values():
        budgeted = group[0]["method_key"] in SHORTLIST_METHODS
        match = (
            next((record for record in group if record.get("candidate_k") == candidate_k), None)
            if budgeted
            else group[0]
        )
        rows.append(match or _placeholder(group[0], candidate_k if budgeted else None))
    return _sort_records(rows)


def build_ablation_rows(
    records: Iterable[Mapping[str, Any]], animal: str, budgets: Sequence[int]
) -> list[dict[str, Any]]:
    selected = select_latest_records(records, animal)
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in selected:
        groups.setdefault((record["method_key"], record["matcher"], record["checkpoint"]), []).append(record)
    for group in groups.values():
        if group[0]["method_key"] in SHORTLIST_METHODS:
            by_budget = {record.get("candidate_k"): record for record in group}
            rows.extend(by_budget.get(budget, _placeholder(group[0], budget)) for budget in budgets)
        else:
            rows.append(group[0])
    return _sort_records(rows)


def _latex_escape(value: Any) -> str:
    text = "--" if value is None or value == "" else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _display_number(value: Any, bold: bool = False) -> str:
    number = _finite_float(value)
    if number is None:
        return "--"
    text = f"{number * 100.0:.2f}"
    return rf"\textbf{{{text}}}" if bold else text


def _best_values(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] = METRIC_COLUMNS,
) -> dict[str, float]:
    best: dict[str, float] = {}
    for column in columns:
        values = [_finite_float(row.get(column)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            best[column] = max(values)
    return best


def _checkpoint_display(value: Any) -> Any:
    """Use paper-friendly labels without changing run-selection identities."""
    return "fine-tuned" if str(value).lower() == "custom" else value


def _row_csv(row: Mapping[str, Any]) -> dict[str, Any]:
    rendered = {column: row.get(column) for column in TABLE_COLUMNS}
    rendered["checkpoint"] = _checkpoint_display(rendered.get("checkpoint"))
    return rendered


def render_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_row_csv(row) for row in rows)
    return output.getvalue()


def _ablation_section_label(row: Mapping[str, Any]) -> str:
    method_key = str(row.get("method_key") or "")
    if method_key == "cosine":
        return "Baselines"
    if method_key == "wildfusion":
        return "WildFusion"
    if method_key == "vismatch":
        matcher = str(row.get("matcher") or "").lower()
        if matcher == "loma":
            return "LoMa"
        if matcher == "rdd-lightglue":
            return "RDD-LightGlue"
    return str(row.get("method") or row.get("method_key") or "Other")


def _ablation_delta_suffix(
    row: Mapping[str, Any],
    column: str,
    baselines: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> str:
    if str(row.get("checkpoint") or "").lower() not in {"custom", "fine-tuned"}:
        return ""
    key = (row.get("method_key"), row.get("matcher"), row.get("candidate_k"))
    baseline = baselines.get(key)
    if baseline is None:
        return ""
    value = _finite_float(row.get(column))
    default_value = _finite_float(baseline.get(column))
    if value is None or default_value is None:
        return ""
    delta = (value - default_value) * 100.0
    if abs(delta) < 0.005:
        return r"\,\textcolor{gray}{\scriptsize$(=)$}"
    arrow = r"\uparrow" if delta > 0 else r"\downarrow"
    color = "green!45!black" if delta > 0 else "red!55!black"
    return rf"\,\textcolor{{{color}}}{{\scriptsize${arrow}$}}{abs(delta):.2f}"


def _sort_ablation_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep default/fine-tuned rows adjacent and put default first at each budget."""
    def key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        checkpoint = str(record.get("checkpoint") or "").lower()
        checkpoint_order = {"default": 0, "custom": 1, "fine-tuned": 1}.get(checkpoint, 2)
        budget = record.get("candidate_k")
        return (
            METHOD_ORDER.get(record.get("method_key", ""), 99),
            record.get("method", ""),
            record.get("matcher", ""),
            -1 if budget is None else budget,
            checkpoint_order,
        )

    return sorted((dict(record) for record in records), key=key)


def _render_compact_ablation_latex(
    rows: Sequence[Mapping[str, Any]],
    *,
    animal: str,
    table_name: str,
    candidate_k: int | None,
    generated_at: str,
    detailed_comments: bool,
) -> str:
    """Render a compact, CVPR-style fixed-budget or ablation table."""
    ordered_rows = _sort_ablation_records(rows)
    best = _best_values(ordered_rows, ABLATION_LATEX_METRIC_COLUMNS)
    baselines: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in ordered_rows:
        if str(row.get("checkpoint") or "").lower() not in {"custom", "fine-tuned"}:
            key = (row.get("method_key"), row.get("matcher"), row.get("candidate_k"))
            baselines[key] = row

    header: list[str] = []
    if detailed_comments:
        source_comments = [
            f"% run_id={row.get('run_id') or '--'} manifest={row.get('manifest_path') or '--'}"
            for row in ordered_rows
        ]
        header.extend([
            "% Generated by scripts/export_paper_tables.py",
            f"% generated_at={generated_at}",
            *source_comments,
            f"% Compact {table_name} LaTeX omits mAP, mAP@k, and total runtime; see companion CSV.",
        ])
    header.extend([
        "\\begin{table}[t]",
        "\\centering",
        (
            f"\\caption{{{_latex_escape(animal)} results ({_latex_escape(table_name)}"
            + (f", candidate budget $k={candidate_k}$" if candidate_k is not None else "")
            + "). Best values in each metric column are boldfaced. Default rows are light gray "
            "and fine-tuned rows are light green. Arrows show the change from the default "
            "checkpoint at the same $k$.}}"
        ),
        f"\\label{{tab:{re.sub(r'[^A-Za-z0-9:.-]+', '-', animal.lower())}-{table_name}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lll rrrrrr}",
        "\\toprule",
        'Method & Matcher & Checkpoint & $k$ & Top-1 (\\%) & Top-5 (\\%) & Top-10 (\\%) & Balanced Top-1 (\\%) & Primary Compute (min) \\\\',
        "\\midrule",
    ])
    body: list[str] = []
    current_section: str | None = None
    for row in ordered_rows:
        section = _ablation_section_label(row)
        if section != current_section:
            if current_section is not None:
                body.append("\\addlinespace")
            body.extend([
                rf"\\multicolumn{{9}}{{l}}{{\\textbf{{{_latex_escape(section)}}}}} \\\\",
                "\\midrule",
            ])
            current_section = section
        checkpoint = str(row.get("checkpoint") or "").lower()
        if checkpoint in {"custom", "fine-tuned"}:
            body.append("\\rowcolor{green!10}")
        elif checkpoint == "default":
            body.append("\\rowcolor{gray!10}")
        cells = [
            _latex_escape(row.get("method")),
            _latex_escape(row.get("matcher")),
            _latex_escape(_checkpoint_display(row.get("checkpoint"))),
            _latex_escape(row.get("candidate_k")),
        ]
        for column in ABLATION_LATEX_METRIC_COLUMNS:
            value = _finite_float(row.get(column))
            rendered = _display_number(value, value is not None and best.get(column) == value)
            cells.append(rendered + _ablation_delta_suffix(row, column, baselines))
        runtime = _finite_float(row.get("runtime_min"))
        cells.append("--" if runtime is None else f"{runtime:.2f}")
        body.append(" & ".join(cells) + r" \\")
    footer = [
        "\\bottomrule",
        "\\end{tabular}",
        "}%",
        "\\end{table}",
    ]
    if detailed_comments:
        footer.append("% Percentages are source fractions multiplied by 100 for display.")
    rendered = "\n".join([*header, *body, *footer, ""])
    # The section template is raw so its braces remain readable above; normalize
    # its command slashes before returning a fragment for LaTeX.
    return (
        rendered
        .replace("$k$.}}", "$k$.}")
        .replace("\\\\multicolumn", "\\multicolumn")
        .replace("\\\\textbf", "\\textbf")
        .replace(" " + "\\" * 4, " " + "\\" * 2)
    )


def render_latex(
    rows: Sequence[Mapping[str, Any]],
    *,
    animal: str,
    table_name: str,
    candidate_k: int | None,
    generated_at: str | None = None,
    detailed_comments: bool = False,
    compact_ablation: bool = False,
) -> str:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    if compact_ablation:
        return _render_compact_ablation_latex(
            rows,
            animal=animal,
            table_name=table_name,
            candidate_k=candidate_k,
            generated_at=generated_at,
            detailed_comments=detailed_comments,
        )
    best = _best_values(rows)
    header = []
    if detailed_comments:
        source_comments = [
            f"% run_id={row.get('run_id') or '--'} manifest={row.get('manifest_path') or '--'}"
            for row in rows
        ]
        header.extend([
            "% Generated by scripts/export_paper_tables.py",
            f"% generated_at={generated_at}",
            *source_comments,
        ])
    header.extend([
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(animal)} results ({_latex_escape(table_name)}).}}",
        f"\\label{{tab:{re.sub(r'[^A-Za-z0-9:.-]+', '-', animal.lower())}-{table_name}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lll rrrrrrrrr}",
        "\\toprule",
        'Method & Matcher & Checkpoint & $k$ & Top-1 (\\%) & Top-5 (\\%) & Top-10 (\\%) & Balanced Top-1 (\\%) & mAP (\\%) & mAP@k (\\%) & Primary Compute (min) & Total Runtime (min) \\\\',
        "\\midrule",
    ])
    body = []
    for row in rows:
        cells = [
            _latex_escape(row.get("method")),
            _latex_escape(row.get("matcher")),
            _latex_escape(_checkpoint_display(row.get("checkpoint"))),
            _latex_escape(row.get("candidate_k")),
        ]
        for column in METRIC_COLUMNS:
            value = _finite_float(row.get(column))
            cells.append(_display_number(value, value is not None and best.get(column) == value))
        runtime = _finite_float(row.get("runtime_min"))
        cells.append("--" if runtime is None else f"{runtime:.2f}")
        total_runtime = _finite_float(row.get("total_runtime_min"))
        cells.append("--" if total_runtime is None else f"{total_runtime:.2f}")
        body.append(" & ".join(cells) + r" \\")
    footer = [
        "\\bottomrule",
        "\\end{tabular}",
        "}%",
        "\\end{table}",
    ]
    if detailed_comments:
        footer.append("% Percentages are source fractions multiplied by 100 for display.")
        if candidate_k is not None:
            footer.append(f"% Shortlist methods use candidate_k={candidate_k}; full-matrix methods use full mAP.")
    return "\n".join([*header, *body, *footer, ""]) 


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def write_animal_tables(
    records: Iterable[Mapping[str, Any]],
    *,
    animal: str,
    output_dir: Path,
    main_candidate_k: int = 50,
    budgets: Sequence[int] = DEFAULT_ABLATION_BUDGETS,
    generated_at: str | None = None,
    detailed_comments: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    main_rows = build_main_rows(records, animal, main_candidate_k)
    ablation_rows = build_ablation_rows(records, animal, budgets)
    stem = _safe_filename(animal)
    outputs = {
        output_dir / f"{stem}_main.tex": render_latex(
            main_rows,
            animal=animal,
            table_name="main",
            candidate_k=main_candidate_k,
            generated_at=generated_at,
            detailed_comments=detailed_comments,
            compact_ablation=True,
        ),
        output_dir / f"{stem}_main.csv": render_csv(main_rows),
        output_dir / f"{stem}_ablation.tex": render_latex(
            ablation_rows,
            animal=animal,
            table_name="ablation",
            candidate_k=None,
            generated_at=generated_at,
            detailed_comments=detailed_comments,
            compact_ablation=True,
        ),
        output_dir / f"{stem}_ablation.csv": render_csv(ablation_rows),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    return list(outputs)


def export_tables(
    experiment_root: Path,
    output_dir: Path,
    *,
    animals: Sequence[str] | None = None,
    main_candidate_k: int = 50,
    budgets: Sequence[int] = DEFAULT_ABLATION_BUDGETS,
    generated_at: str | None = None,
    detailed_comments: bool = False,
) -> list[Path]:
    if main_candidate_k <= 0:
        raise ValueError("main_candidate_k must be positive")
    budgets = tuple(int(budget) for budget in budgets)
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must contain positive integers")
    records = discover_records(experiment_root)
    selected_animals = sorted(set(animals or discover_animals(records)))
    if not selected_animals:
        raise ValueError(f"no completed probe animals found under {experiment_root / 'probe'}")
    unknown = sorted(set(selected_animals) - set(discover_animals(records)))
    if unknown:
        raise ValueError(f"no completed probe records found for animal(s): {', '.join(unknown)}")
    outputs: list[Path] = []
    for animal in selected_animals:
        outputs.extend(
            write_animal_tables(
                records,
                animal=animal,
                output_dir=output_dir,
                main_candidate_k=main_candidate_k,
                budgets=budgets,
                generated_at=generated_at,
                detailed_comments=detailed_comments,
            )
        )
    return outputs
