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
UNSEEN_EVAL_TABLE_BUDGETS = (10, 50, 100, 160)
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
    "animal",
    "split_protocol",
    "method",
    "matcher",
    "backbone",
    "checkpoint",
    "checkpoint_component",
    "checkpoint_owner",
    "evaluation_animal",
    "checkpoint_protocol",
    "applied_prefixes",
    "ignored_prefixes",
    "candidate_k",
    "class_weighting",
    "top_1",
    "top_5",
    "top_10",
    "balanced_top_1",
    "mAP",
    "mAP_at_k",
    "runtime_min",
    "total_runtime_min",
    "classifier_open_set_policy",
    "classification_seen_top_1",
    "classification_seen_top_5",
    "classification_seen_top_10",
    "classification_seen_balanced_top_1",
    "classification_open_top_1",
    "classification_open_top_5",
    "classification_open_top_10",
    "classification_open_balanced_top_1",
    "classification_top_1",
    "classification_top_5",
    "classification_top_10",
    "classification_balanced_top_1",
    "classification_num_query_images",
    "classification_num_seen_query_images",
    "classification_num_unseen_query_images",
    "classification_num_query_identities",
    "classification_num_seen_query_identities",
    "classification_num_unseen_query_identities",
    "classification_query_seen_coverage",
    "classification_embedding_retrieval_enabled",
    "embedding_top_1",
    "embedding_top_5",
    "embedding_top_10",
    "embedding_balanced_top_1",
    "embedding_mAP",
    "embedding_mAP_at_k",
    "embedding_num_queries",
    "embedding_num_queries_with_gallery_match",
    "embedding_num_queries_without_gallery_match",
    "embedding_mAP_query_coverage",
    "embedding_score_coverage",
    "run_id",
    "manifest_path",
)
METRIC_COLUMNS = ("top_1", "top_5", "top_10", "balanced_top_1", "mAP", "mAP_at_k")
ABLATION_LATEX_METRIC_COLUMNS = ("top_1", "top_5", "top_10", "balanced_top_1")
FINE_TUNED_CHECKPOINTS = {
    "custom",
    "fine-tuned",
    "matcher-fine-tuned",
    "descriptor-fine-tuned",
    "extractor-fine-tuned",
    "full-fine-tuned",
}


def _backbone_name(manifest: Mapping[str, Any]) -> str:
    """Return the model identity used by a run.

    The reporting manifest stores this as ``model``.  The additional aliases
    keep discovery compatible with small synthetic/legacy manifests that used
    a more explicit key.  A missing identity is intentionally represented as
    ``unknown`` rather than being guessed from a directory name.
    """
    for key in ("model", "backbone", "model_type", "model_name"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            for nested_key in ("type", "name", "model", "backbone", "identifier"):
                nested = value.get(nested_key)
                if nested:
                    value = nested
                    break
        if value:
            return str(value)
    return "unknown"


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


def _checkpoint_variant(manifest: Mapping[str, Any]) -> str:
    checkpoint = manifest.get("vismatch_checkpoint")
    if isinstance(checkpoint, Mapping):
        value = checkpoint.get("checkpoint_variant")
        if value and str(value).lower() != "auto":
            return str(value)
        mode = str(checkpoint.get("resolved_component_mode") or checkpoint.get("component_mode") or "").lower()
        if mode in {"matcher_only", "extractor_only", "descriptor_only", "full"}:
            return {
                "matcher_only": "matcher-fine-tuned",
                "extractor_only": "extractor-fine-tuned",
                "descriptor_only": "descriptor-fine-tuned",
                "full": "full-fine-tuned",
            }[mode]
    return _checkpoint_source(manifest)


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


def _classifier_probe_train_mode(manifest_path: Path, manifest: Mapping[str, Any], method: str) -> str:
    """Read a classifier-probe mode so distinct training scopes remain distinct."""
    for source in (manifest, manifest.get("metrics", {}), manifest.get("timings", {})):
        if not isinstance(source, Mapping):
            continue
        for key in (f"{method}_train_mode", "train_mode"):
            if source.get(key):
                return str(source[key]).lower()
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.load(manifest_path.with_name("config.snapshot.yaml"))
        value = OmegaConf.select(config, f"benchmark.methods.{method}.train_mode")
    except (ImportError, OSError, ValueError):
        value = None
    return str(value).lower() if value else "unknown"


def _classifier_probe_weighting(manifest: Mapping[str, Any], metrics: Mapping[str, Any], method: str) -> str:
    key = f"{method}_class_weighting"
    value = metrics.get(key, manifest.get(key))
    if isinstance(value, Mapping):
        value = value.get("mode")
    if value in {"inverse_frequency", "weighted"}:
        return "weighted"
    if value in {"none", "unweighted"}:
        return "unweighted"
    return "unknown" if method in {"linear_probe", "efficient_probe"} else ""


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
    train_mode = (
        _classifier_probe_train_mode(manifest_path, manifest, method)
        if method in {"linear_probe", "efficient_probe"}
        else ""
    )
    class_weighting = _classifier_probe_weighting(manifest, metrics, method)
    checkpoint = _checkpoint_variant(manifest)
    checkpoint_info = manifest.get("vismatch_checkpoint")
    if not isinstance(checkpoint_info, Mapping):
        checkpoint_info = {}
    component_mode = str(
        checkpoint_info.get("resolved_component_mode")
        or checkpoint_info.get("component_mode")
        or ""
    ).lower()
    classifier_payload = manifest.get("classifier_evaluation")
    if not isinstance(classifier_payload, Mapping):
        classifier_payload = {}
    run_id = str(manifest.get("run_id") or manifest_path.parent.name)
    candidate = _candidate_k(method, metrics, timings)
    primary_runtime_sec = _finite_float(timings.get("primary_compute_runtime_sec"))
    total_runtime_sec = _finite_float(timings.get("total_run_sec"))
    if total_runtime_sec is None:
        legacy_total_min = _finite_float(timings.get("total_run_min"))
        total_runtime_sec = None if legacy_total_min is None else legacy_total_min * 60.0
    record = {
        "animal": animal,
        "split_protocol": str(manifest.get("split_protocol") or manifest.get("split_col") or ""),
        "method_key": method,
        "method": METHOD_LABELS[method],
        "matcher": matcher,
        "backbone": _backbone_name(manifest),
        "train_mode": train_mode,
        "checkpoint": checkpoint,
        "checkpoint_component": "descriptor" if component_mode == "descriptor_only" else component_mode or "",
        "checkpoint_owner": manifest.get("checkpoint_owner", checkpoint_info.get("checkpoint_owner", "")),
        "evaluation_animal": manifest.get("evaluation_animal", animal),
        "checkpoint_protocol": checkpoint_info.get("protocol_metadata", {}),
        "applied_prefixes": checkpoint_info.get("applied_prefixes", []),
        "ignored_prefixes": checkpoint_info.get("ignored_prefixes", []),
        "candidate_k": candidate,
        "class_weighting": class_weighting,
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
    classifier_fields = (
        "classification_seen_top_1",
        "classification_seen_top_5",
        "classification_seen_top_10",
        "classification_seen_balanced_top_1",
        "classification_open_top_1",
        "classification_open_top_5",
        "classification_open_top_10",
        "classification_open_balanced_top_1",
        "classification_top_1",
        "classification_top_5",
        "classification_top_10",
        "classification_balanced_top_1",
        "classification_num_query_images",
        "classification_num_seen_query_images",
        "classification_num_unseen_query_images",
        "classification_num_query_identities",
        "classification_num_seen_query_identities",
        "classification_num_unseen_query_identities",
        "classification_query_seen_coverage",
        "embedding_top_1",
        "embedding_top_5",
        "embedding_top_10",
        "embedding_balanced_top_1",
        "embedding_mAP",
        "embedding_mAP_at_k",
        "embedding_num_queries",
        "embedding_num_queries_with_gallery_match",
        "embedding_num_queries_without_gallery_match",
        "embedding_mAP_query_coverage",
        "embedding_score_coverage",
    )
    for field in classifier_fields:
        record[field] = _finite_float(metrics.get(field, classifier_payload.get(field)))
    record["classifier_open_set_policy"] = metrics.get(
        "classifier_open_set_policy",
        manifest.get("classifier_open_set_policy", classifier_payload.get("classifier_open_set_policy")),
    )
    record["classification_embedding_retrieval_enabled"] = metrics.get(
        "classification_embedding_retrieval_enabled",
        manifest.get("classification_embedding_retrieval_enabled"),
    )
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


def discover_splits(records: Iterable[Mapping[str, Any]], animal: str | None = None) -> list[str]:
    return sorted(
        {
            str(record.get("split_protocol"))
            for record in records
            if record.get("split_protocol") and (animal is None or record.get("animal") == animal)
        }
    )


def _selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["animal"],
        record.get("split_protocol", ""),
        record["method_key"],
        record["matcher"],
        record.get("backbone", "unknown"),
        record.get("train_mode", ""),
        record.get("class_weighting", ""),
        record["checkpoint"],
        record["candidate_k"],
    )


def _is_descriptor_record(record: Mapping[str, Any]) -> bool:
    return (
        str(record.get("checkpoint_component") or "").lower() == "descriptor"
        or str(record.get("checkpoint") or "").lower() == "descriptor-fine-tuned"
    )


def select_latest_records(
    records: Iterable[Mapping[str, Any]], animal: str, split_protocol: str | None = None
) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record.get("animal") != animal:
            continue
        if split_protocol is not None and record.get("split_protocol", "") != split_protocol:
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
            record.get("backbone", "unknown"),
            {"all": 0, "partial": 1, "classifier": 2}.get(record.get("train_mode", ""), 3),
            record.get("class_weighting", ""),
            record.get("checkpoint", ""),
            -1 if record.get("candidate_k") is None else record["candidate_k"],
        ),
    )


def _placeholder(template: Mapping[str, Any], candidate_k: int | None) -> dict[str, Any]:
    row = {column: None for column in TABLE_COLUMNS}
    row.update(
        {
            "animal": template["animal"],
            "split_protocol": template.get("split_protocol", ""),
            "method_key": template["method_key"],
            "method": template["method"],
            "matcher": template["matcher"],
            "backbone": template.get("backbone", "unknown"),
            "train_mode": template.get("train_mode", ""),
            "class_weighting": template.get("class_weighting", ""),
            "checkpoint": template["checkpoint"],
            "candidate_k": candidate_k,
            "run_id": None,
            "manifest_path": None,
            "_sort_token": ("", ""),
        }
    )
    return row


def _effective_table_budgets(
    budgets: Sequence[int],
    split_protocol: str | None,
) -> tuple[int, ...]:
    """Resolve the candidate grid that is valid for a table's split."""
    requested = tuple(int(budget) for budget in budgets)
    if split_protocol == "unseen_eval_split":
        if requested == DEFAULT_ABLATION_BUDGETS:
            requested = UNSEEN_EVAL_TABLE_BUDGETS
        else:
            requested = tuple(
                budget for budget in requested if budget in UNSEEN_EVAL_TABLE_BUDGETS
            )
    if not requested or any(budget <= 0 for budget in requested):
        raise ValueError("budgets must contain positive integers valid for the selected split")
    return requested


def _effective_main_candidate(candidate_k: int, split_protocol: str | None) -> int:
    candidate_k = int(candidate_k)
    if candidate_k <= 0:
        raise ValueError("main_candidate_k must be positive")
    if split_protocol == "unseen_eval_split" and candidate_k not in UNSEEN_EVAL_TABLE_BUDGETS:
        valid = ", ".join(str(budget) for budget in UNSEEN_EVAL_TABLE_BUDGETS)
        raise ValueError(
            f"main_candidate_k={candidate_k} is invalid for unseen_eval_split; choose from: {valid}"
        )
    return candidate_k


def build_main_rows(
    records: Iterable[Mapping[str, Any]], animal: str, candidate_k: int, split_protocol: str | None = None,
    *, include_descriptor: bool = False,
) -> list[dict[str, Any]]:
    candidate_k = _effective_main_candidate(candidate_k, split_protocol)
    selected = [
        record for record in select_latest_records(records, animal, split_protocol)
        if include_descriptor or not _is_descriptor_record(record)
    ]
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for record in selected:
        key = (
            record["method_key"],
            record["matcher"],
            record.get("backbone", "unknown"),
            record.get("train_mode", ""),
            record.get("class_weighting", ""),
            record["checkpoint"],
        )
        groups.setdefault(key, []).append(record)
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
    records: Iterable[Mapping[str, Any]],
    animal: str,
    budgets: Sequence[int],
    split_protocol: str | None = None,
    *,
    include_descriptor: bool = False,
) -> list[dict[str, Any]]:
    budgets = _effective_table_budgets(budgets, split_protocol)
    selected = [
        record for record in select_latest_records(records, animal, split_protocol)
        if include_descriptor or not _is_descriptor_record(record)
    ]
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for record in selected:
        key = (
            record["method_key"],
            record["matcher"],
            record.get("backbone", "unknown"),
            record.get("train_mode", ""),
            record.get("class_weighting", ""),
            record["checkpoint"],
        )
        groups.setdefault(key, []).append(record)
    for group in groups.values():
        if group[0]["method_key"] in SHORTLIST_METHODS:
            by_budget = {record.get("candidate_k"): record for record in group}
            rows.extend(by_budget.get(budget, _placeholder(group[0], budget)) for budget in budgets)
        else:
            rows.append(group[0])
    return _sort_records(rows)


def build_descriptor_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    animal: str,
    matcher: str,
    budgets: Sequence[int],
    candidate_k: int | None = None,
    split_protocol: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build descriptor-specific rows with the matching default/context baselines."""
    selected = select_latest_records(records, animal, split_protocol)
    matcher = str(matcher).lower()
    relevant = [
        record for record in selected
        if (
            (record.get("method_key") == "vismatch" and str(record.get("matcher", "")).lower() == matcher
             and (_is_descriptor_record(record) or str(record.get("checkpoint", "")).lower() == "default"))
            or record.get("method_key") in {"cosine", "wildfusion"}
        )
    ]
    return (
        build_main_rows(relevant, animal, candidate_k or 50, split_protocol, include_descriptor=True),
        build_ablation_rows(relevant, animal, budgets, split_protocol, include_descriptor=True),
    )


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
    labels = {
        "custom": "fine-tuned",
        "matcher-fine-tuned": "fine-tuned",
        "descriptor-fine-tuned": "descriptor fine-tuned",
        "extractor-fine-tuned": "extractor fine-tuned",
        "full-fine-tuned": "full fine-tuned",
    }
    return labels.get(str(value).lower(), value)


def _backbone_display(value: Any) -> Any:
    """Use compact, readable model labels in manuscript-facing LaTeX."""
    labels = {
        "megadescriptor-t": "MegaDescriptor-T",
        "megadescriptor-l": "MegaDescriptor-L",
        "dinov2": "DINOv2",
        "dinov2-l": "DINOv2-L",
        "dinov3": "DINOv3",
        "dinov3-l": "DINOv3-L",
        "lynx_megadescriptorv3": "Lynx MegaDescriptorV3",
        "lynx_megadescriptorv4": "Lynx MegaDescriptorV4",
        "miewid": "MiewID",
    }
    text = str(value or "unknown")
    return labels.get(text.lower(), value or "unknown")


def _paper_checkpoint_display(row: Mapping[str, Any]) -> Any:
    """Display probe training scope and loss weighting in the checkpoint column.

    The audit CSV still has a dedicated ``class_weighting`` field.  Including the
    policy in the paper-facing label prevents weighted and unweighted classifier
    rows from looking identical after the train mode is mapped to ``frozen``.
    """
    if row.get("method_key") in {"linear_probe", "efficient_probe"}:
        labels = {
            "classifier": "frozen",
            "partial": "partial fine-tuned",
            "all": "full fine-tuned",
        }
        scope = labels.get(str(row.get("train_mode") or "").lower(), "unknown")
        weighting = str(row.get("class_weighting") or "unknown").lower()
        weighting_label = {
            "weighted": "weighted",
            "unweighted": "unweighted",
        }.get(weighting, "weighting unknown")
        return f"{scope} ({weighting_label})"
    return _checkpoint_display(row.get("checkpoint"))


def _row_csv(row: Mapping[str, Any]) -> dict[str, Any]:
    rendered = {column: row.get(column) for column in TABLE_COLUMNS}
    rendered["checkpoint"] = _paper_checkpoint_display(row)
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
    if str(row.get("checkpoint") or "").lower() not in FINE_TUNED_CHECKPOINTS:
        return ""
    key = (
        row.get("method_key"),
        row.get("matcher"),
        row.get("backbone", "unknown"),
        row.get("train_mode", ""),
        row.get("class_weighting", ""),
        row.get("candidate_k"),
    )
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
        checkpoint_order = {
            "default": 0,
            "custom": 1,
            "fine-tuned": 1,
            "matcher-fine-tuned": 1,
            "descriptor-fine-tuned": 1,
            "extractor-fine-tuned": 1,
            "full-fine-tuned": 1,
        }.get(checkpoint, 2)
        budget = record.get("candidate_k")
        return (
            METHOD_ORDER.get(record.get("method_key", ""), 99),
            record.get("method", ""),
            record.get("matcher", ""),
            record.get("backbone", "unknown"),
            {"all": 0, "partial": 1, "classifier": 2}.get(record.get("train_mode", ""), 3),
            record.get("class_weighting", ""),
            -1 if budget is None else budget,
            checkpoint_order,
        )

    return sorted((dict(record) for record in records), key=key)


def _render_compact_ablation_latex(
    rows: Sequence[Mapping[str, Any]],
    *,
    animal: str,
    split_protocol: str | None,
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
        if str(row.get("checkpoint") or "").lower() not in FINE_TUNED_CHECKPOINTS:
            key = (
                row.get("method_key"),
                row.get("matcher"),
                row.get("backbone", "unknown"),
                row.get("train_mode", ""),
                row.get("class_weighting", ""),
                row.get("candidate_k"),
            )
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
            f"\\caption{{{_latex_escape(animal)}"
            + (f" ({_latex_escape(split_protocol)})" if split_protocol else "")
            + f" results ({_latex_escape(table_name)}"
            + (f", candidate budget $k={candidate_k}$" if candidate_k is not None else "")
            + "). Best values in each metric column are boldfaced. Default rows are light gray "
            "and fine-tuned rows are light green. Arrows show the change from the default "
            "checkpoint at the same $k$.}}"
        ),
        f"\\label{{tab:{re.sub(r'[^A-Za-z0-9:.-]+', '-', animal.lower())}-{re.sub(r'[^A-Za-z0-9:.-]+', '-', split_protocol.lower()) + '-' if split_protocol else ''}{table_name}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llll rrrrrr}",
        "\\toprule",
        'Method & Matcher & Backbone & Checkpoint & $k$ & Top-1 (\\%) & Top-5 (\\%) & Top-10 (\\%) & Balanced Top-1 (\\%) & Primary Compute (min) \\\\',
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
                rf"\\multicolumn{{10}}{{l}}{{\\textbf{{{_latex_escape(section)}}}}} \\\\",
                "\\midrule",
            ])
            current_section = section
        checkpoint = str(row.get("checkpoint") or "").lower()
        if checkpoint in FINE_TUNED_CHECKPOINTS:
            body.append("\\rowcolor{green!10}")
        elif checkpoint == "default":
            body.append("\\rowcolor{gray!10}")
        cells = [
            _latex_escape(row.get("method")),
            _latex_escape(row.get("matcher")),
            _latex_escape(_backbone_display(row.get("backbone"))),
            _latex_escape(_paper_checkpoint_display(row)),
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
    split_protocol: str | None = None,
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
            split_protocol=split_protocol,
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
        f"\\caption{{{_latex_escape(animal)}"
        + (f" ({_latex_escape(split_protocol)})" if split_protocol else "")
        + f" results ({_latex_escape(table_name)}).}}",
        f"\\label{{tab:{re.sub(r'[^A-Za-z0-9:.-]+', '-', animal.lower())}-{re.sub(r'[^A-Za-z0-9:.-]+', '-', split_protocol.lower()) + '-' if split_protocol else ''}{table_name}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llll rrrrrrrrr}",
        "\\toprule",
        'Method & Matcher & Backbone & Checkpoint & $k$ & Top-1 (\\%) & Top-5 (\\%) & Top-10 (\\%) & Balanced Top-1 (\\%) & mAP (\\%) & mAP@k (\\%) & Primary Compute (min) & Total Runtime (min) \\\\',
        "\\midrule",
    ])
    body = []
    for row in rows:
        cells = [
            _latex_escape(row.get("method")),
            _latex_escape(row.get("matcher")),
            _latex_escape(_backbone_display(row.get("backbone"))),
            _latex_escape(_paper_checkpoint_display(row)),
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
    split_protocol: str | None = None,
    output_dir: Path,
    main_candidate_k: int = 50,
    budgets: Sequence[int] = DEFAULT_ABLATION_BUDGETS,
    generated_at: str | None = None,
    detailed_comments: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    main_rows = build_main_rows(records, animal, main_candidate_k, split_protocol)
    ablation_rows = build_ablation_rows(records, animal, budgets, split_protocol)
    stem = _safe_filename(animal)
    if split_protocol:
        stem = f"{stem}_{_safe_filename(split_protocol)}"
    outputs = {
        output_dir / f"{stem}_main.tex": render_latex(
            main_rows,
            animal=animal,
            split_protocol=split_protocol,
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
            split_protocol=split_protocol,
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
    selected = select_latest_records(records, animal, split_protocol)
    for descriptor_matcher, family_label in (("rdd-lightglue", "rdd"), ("loma", "loma")):
        if not any(
            _is_descriptor_record(record)
            and record.get("method_key") == "vismatch"
            and str(record.get("matcher", "")).lower() == descriptor_matcher
            for record in selected
        ):
            continue
        descriptor_main, descriptor_ablation = build_descriptor_rows(
            records,
            animal=animal,
            matcher=descriptor_matcher,
            budgets=budgets,
            candidate_k=main_candidate_k,
            split_protocol=split_protocol,
        )
        descriptor_outputs = {
            output_dir / f"{stem}_descriptor_{family_label}_main.tex": render_latex(
                descriptor_main,
                animal=animal,
                split_protocol=split_protocol,
                table_name=f"descriptor {family_label} main",
                candidate_k=main_candidate_k,
                generated_at=generated_at,
                detailed_comments=detailed_comments,
                compact_ablation=True,
            ),
            output_dir / f"{stem}_descriptor_{family_label}_main.csv": render_csv(descriptor_main),
            output_dir / f"{stem}_descriptor_{family_label}_ablation.tex": render_latex(
                descriptor_ablation,
                animal=animal,
                split_protocol=split_protocol,
                table_name=f"descriptor {family_label} ablation",
                candidate_k=None,
                generated_at=generated_at,
                detailed_comments=detailed_comments,
                compact_ablation=True,
            ),
            output_dir / f"{stem}_descriptor_{family_label}_ablation.csv": render_csv(descriptor_ablation),
        }
        for path, content in descriptor_outputs.items():
            path.write_text(content, encoding="utf-8")
        outputs.update(descriptor_outputs)
    return list(outputs)


def export_tables(
    experiment_root: Path,
    output_dir: Path,
    *,
    animals: Sequence[str] | None = None,
    split_protocols: Sequence[str] | None = None,
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
        available_splits = discover_splits(records, animal)
        if split_protocols is not None:
            selected_splits = [split for split in split_protocols if split in available_splits]
            unknown_splits = sorted(set(split_protocols) - set(available_splits))
            if unknown_splits:
                raise ValueError(
                    f"no completed probe records found for split(s) of {animal}: {', '.join(unknown_splits)}"
                )
        else:
            selected_splits = available_splits
        # Preserve legacy animal-only filenames when artifacts have no split
        # provenance; split-aware artifacts always get one output set per split.
        if not selected_splits:
            selected_splits = [None]
        for split in selected_splits:
            outputs.extend(
                write_animal_tables(
                    records,
                    animal=animal,
                    split_protocol=split,
                    output_dir=output_dir,
                    main_candidate_k=main_candidate_k,
                    budgets=budgets,
                    generated_at=generated_at,
                    detailed_comments=detailed_comments,
                )
            )
    return outputs
