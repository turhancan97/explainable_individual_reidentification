from __future__ import annotations

import json
from pathlib import Path

from reid.reporting.fewshot import (
    build_rows,
    fraction_label,
    k_sweep_series,
    load_views,
    markdown_summary,
    parse_split_column,
    probe_records,
    promote_fallback_checkpoints,
    series_label,
    view_name,
)


def test_split_column_round_trip():
    assert parse_split_column("split_frac0.125_seed0") == (0.125, 0)
    assert parse_split_column("split_frac1.0_seed3") == (1.0, 3)
    assert parse_split_column("split") is None
    assert view_name(0.125, 0) == "frac0.125-seed0" and view_name(1.0, 0) == "frac1.0-seed0"
    assert [fraction_label(f) for f in (0.125, 0.25, 0.5, 1.0, 0.3)] == ["1/8", "1/4", "1/2", "full", "0.3"]


def _write_view(root: Path, fraction: float, kept: int, identities: int) -> None:
    view = root / "views" / "Toy" / "legacy" / view_name(fraction, 0)
    view.mkdir(parents=True)
    (view / "fewshot.json").write_text(json.dumps({
        "fewshot": {
            "fraction": fraction, "seed": 0, "effective_fraction": kept / 100, "budget_feasible": True,
            "train_frames_kept": kept, "train_identities": identities,
            "train_identities_with_positives": identities,
            "per_identity": {f"id{i}": {"full": 10, "kept": kept // identities} for i in range(identities)},
        }
    }))


def _write_run(experiments: Path, split: str, method: str, matcher: str, source: str, top_1: float, run_utc: str,
               metadata_file: str = "metadata_fewshot/metadata_Toy.csv", checkpoint_path: str | None = None,
               candidate_k: int = 50) -> None:
    run = experiments / "probe" / "WildlifeReID-10k" / "Toy" / split / "megadescriptor-l" / method / (matcher if method == "vismatch" else "default") / f"{run_utc}_{source}_k{candidate_k}"
    run.mkdir(parents=True)
    manifest = {
        "status": "completed", "workflow": "probe", "method": method, "animal": "Toy",
        "split_protocol": split, "variant": matcher if method == "vismatch" else "default",
        "run_id": run.name, "run_utc": run_utc, "num_query": 20, "num_database": 50,
        "vismatch_checkpoint": {"source": source, "requested_path": checkpoint_path} if method == "vismatch" else None,
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    (run / "metrics.json").write_text(json.dumps({
        "top_1": top_1 / 100, "top_5": 0.99, "top_10": 0.995, "balanced_top_1": (top_1 - 1) / 100,
        "balanced_top_5": (top_1 + 1) / 100, "balanced_top_10": (top_1 + 2) / 100,
        "map_at_k": candidate_k, "mAP_at_k": 0.80, "mAP": 0.81}))
    (run / "result.json").write_text(json.dumps({"metadata_file": metadata_file, "split_col": split}))
    (run / "timings.json").write_text(json.dumps({"benchmark_candidate_k": candidate_k, "primary_compute_runtime_sec": 60}))


def test_probe_records_join_views_and_keep_latest(tmp_path: Path):
    _write_view(tmp_path, 0.125, kept=20, identities=10)
    _write_view(tmp_path, 1.0, kept=100, identities=10)
    experiments = tmp_path / "experiments"
    _write_run(experiments, "split_frac0.125_seed0", "vismatch", "rdd-lightglue", "custom", 90.0, "2026-09-15T10:00:00Z")
    _write_run(experiments, "split_frac0.125_seed0", "vismatch", "rdd-lightglue", "custom", 91.0, "2026-09-15T11:00:00Z")  # rerun wins
    _write_run(experiments, "split_frac0.125_seed0", "vismatch", "rdd-lightglue", "default", 85.0, "2026-09-15T10:00:00Z")
    _write_run(experiments, "split_frac1.0_seed0", "cosine", "-", "default", 70.0, "2026-09-15T10:00:00Z")
    _write_run(experiments, "split", "cosine", "-", "default", 71.0, "2026-09-15T10:00:00Z")  # full gallery, fraction independent
    _write_run(experiments, "split", "vismatch", "rdd-lightglue", "custom", 88.0, "2026-09-15T10:00:00Z",
               checkpoint_path="/x/checkpoints/Toy/legacy/frac0.125-seed0/rdd-finetuned/epoch_299/model.safetensors")
    _write_run(experiments, "split", "cosine", "-", "default", 50.0, "2026-09-15T12:00:00Z", metadata_file="metadata_mdsplit_no_background/metadata_Toy.csv")  # other metadata: ignored

    views = load_views(tmp_path, "Toy", "legacy", 0)
    assert views["frac0.125-seed0"]["images_per_identity"] == 2.0
    records = probe_records(experiments, "Toy", 0, "reduced")
    rows = build_rows(records, views, "Toy")
    assert [(r["view"], r["method"], r["top_1"]) for r in rows] == [
        ("frac0.125-seed0", "RDD-LightGlue (default)", 85.0),
        ("frac0.125-seed0", "RDD-LightGlue (fine-tuned)", 91.0),
        ("frac1.0-seed0", "Cosine", 70.0),
    ]
    assert rows[0]["images_per_identity"] == 2.0 and rows[2]["images_per_identity"] == 10.0
    assert rows[0]["candidate_k"] == 50 and rows[2]["candidate_k"] is None
    full = build_rows(probe_records(experiments, "Toy", 0, "full"), views, "Toy")
    assert [(r["view"], r["method"], r["top_1"]) for r in full] == [
        (None, "Cosine", 71.0),
        ("frac0.125-seed0", "RDD-LightGlue (fine-tuned)", 88.0),
    ]
    summary = markdown_summary(rows + full, views, [], "Toy")
    assert "| RDD-LightGlue (fine-tuned) | 91.00 | – |" in summary
    assert "| Cosine (fraction independent) | 71.00 | 71.00 |" in summary
    assert series_label({"method_key": "wildfusion", "matcher": "-", "checkpoint": "default"}) == "WildFusion"


def test_descriptor_checkpoints_get_their_own_label_series_and_missing_key(tmp_path: Path):
    from reid.reporting.fewshot import _style, checkpoint_label, missing_variants, series_label

    _write_view(tmp_path, 1.0, kept=100, identities=10)
    experiments = tmp_path / "experiments"
    base = "/fewshot/checkpoints/Toy/legacy/frac1.0-seed0"
    matcher_ckpt = f"{base}/rdd-finetuned/epoch_299/model.safetensors"
    descriptor_ckpt = f"{base}/rdd-descriptor-finetuned/epoch_299/model.safetensors"
    joint_ckpt = f"{base}/rdd-lg-descriptor-finetuned/epoch_299"  # directory: RDD + LightGlue files
    loma_ckpt = f"{base}/loma-descriptor-matcher-finetuned/epoch_299/model.safetensors"
    _write_run(experiments, "split", "vismatch", "rdd-lightglue", "custom", 88.0, "2026-09-22T10:00:00Z", checkpoint_path=matcher_ckpt)
    _write_run(experiments, "split", "vismatch", "rdd-lightglue", "custom", 86.0, "2026-09-22T10:01:00Z", checkpoint_path=descriptor_ckpt)
    _write_run(experiments, "split", "vismatch", "rdd-lightglue", "custom", 89.0, "2026-09-22T10:02:00Z", checkpoint_path=joint_ckpt)
    _write_run(experiments, "split", "vismatch", "loma", "custom", 80.0, "2026-09-22T10:03:00Z", checkpoint_path=loma_ckpt)

    assert checkpoint_label("custom", descriptor_ckpt) == "custom-descriptor"
    assert checkpoint_label("custom", f"{base}/rdd-finetuned-ep30/epoch_29/model.safetensors") == "custom-ep30"
    assert checkpoint_label("custom", f"{base}/loma-descriptor-matcher-finetuned-ep30/epoch_029/model.safetensors") == "custom-descriptor-matcher-ep30"
    assert series_label({"method_key": "vismatch", "matcher": "rdd-lightglue", "checkpoint": "custom-descriptor-ep30"}) == "RDD-LightGlue (fine-tuned descriptor, ep30)"
    assert series_label({"method_key": "vismatch", "matcher": "loma", "checkpoint": "custom-ep30"}) == "LoMa (fine-tuned, ep30)"
    assert _style("RDD-LightGlue (fine-tuned descriptor, ep30)") == _style("RDD-LightGlue (fine-tuned descriptor)")
    assert checkpoint_label("custom", matcher_ckpt) == "custom" and checkpoint_label("default", None) == "default"
    views = load_views(tmp_path, "Toy", "legacy", 0)
    rows = build_rows(probe_records(experiments, "Toy", 0, "full"), views, "Toy")
    assert sorted((r["method"], r["top_1"]) for r in rows) == [
        ("LoMa (fine-tuned descriptor+matcher)", 80.0),
        ("RDD-LightGlue (fine-tuned LG+descriptor)", 89.0),
        ("RDD-LightGlue (fine-tuned descriptor)", 86.0),
        ("RDD-LightGlue (fine-tuned)", 88.0),
    ]
    assert all(r["view"] == "frac1.0-seed0" for r in rows)  # attributed to the view like the matcher run
    assert series_label({"method_key": "vismatch", "matcher": "loma", "checkpoint": "custom-descriptor"}) == "LoMa (fine-tuned descriptor)"
    styles = {label: _style(label) for label in ("RDD-LightGlue (default)", "RDD-LightGlue (fine-tuned)", "RDD-LightGlue (fine-tuned descriptor)")}
    assert styles["RDD-LightGlue (default)"]["linestyle"] == "-"
    assert len({str(s["linestyle"]) for s in styles.values()}) == 3  # the three states are told apart
    variants = [
        f"vismatch|rdd-lightglue|custom|{matcher_ckpt}",
        f"vismatch|rdd-lightglue|custom-descriptor|{descriptor_ckpt}",
        f"vismatch|rdd-lightglue|custom-lg-descriptor|{joint_ckpt}",
        f"vismatch|loma|custom-descriptor|{base}/loma-descriptor-finetuned/epoch_299/model.safetensors",
    ]
    assert missing_variants(experiments, "Toy", "split", 50, variants) == [variants[3]]


def _k_sweep_fixture(tmp_path: Path) -> tuple[Path, dict]:
    """Full-gallery runs at two budgets: two fine-tuned views, the matcher default,
    WildFusion and the budget-independent cosine."""
    for fraction, kept in ((0.5, 50), (1.0, 100)):
        _write_view(tmp_path, fraction, kept=kept, identities=10)
    experiments = tmp_path / "experiments"
    base = "/fewshot/checkpoints/Toy/legacy"
    for candidate_k, offset in ((10, 0.0), (50, 2.0)):
        for index, fraction in enumerate((0.5, 1.0)):
            _write_run(experiments, "split", "vismatch", "loma", "custom", 70.0 + offset + fraction,
                       f"2026-09-22T1{candidate_k}:0{index}:00Z", candidate_k=candidate_k,
                       checkpoint_path=f"{base}/frac{fraction}-seed0/loma-finetuned/epoch_299/model.safetensors")
        _write_run(experiments, "split", "vismatch", "loma", "default", 60.0 + offset, f"2026-09-22T2{candidate_k}:00:00Z", candidate_k=candidate_k)
        _write_run(experiments, "split", "vismatch", "rdd-lightglue", "default", 55.0 + offset, f"2026-09-22T3{candidate_k}:00:00Z", candidate_k=candidate_k)
        _write_run(experiments, "split", "wildfusion", "-", "default", 65.0 + offset, f"2026-09-22T4{candidate_k}:00:00Z", candidate_k=candidate_k)
    _write_run(experiments, "split", "cosine", "-", "default", 40.0, "2026-09-22T05:00:00Z")
    views = load_views(tmp_path, "Toy", "legacy", 0)
    return experiments, views


def test_k_sweep_series_are_per_matcher_and_keep_cosine_as_a_reference(tmp_path: Path):
    experiments, views = _k_sweep_fixture(tmp_path)
    rows = build_rows(probe_records(experiments, "Toy", 0, "full"), views, "Toy")

    series, references = k_sweep_series(rows, views, matcher="loma", metric="top_1", k_values=(10, 50))
    assert [label for label, _ in series] == ["fine-tuned 1/2", "fine-tuned full", "LoMa (default)", "WildFusion"]
    assert dict(series)["fine-tuned full"] == [(10, 71.0), (50, 73.0)]
    assert dict(series)["LoMa (default)"] == [(10, 60.0), (50, 62.0)]
    assert references == {"Cosine": 40.0}  # no shortlist: one value for every budget
    assert all(k in (10, 50) for _, values in series for k, _ in values)

    rdd_series, _ = k_sweep_series(rows, views, matcher="rdd-lightglue", metric="top_1", k_values=(10, 50))
    assert [label for label, _ in rdd_series] == ["RDD-LightGlue (default)", "WildFusion"]  # no LoMa lines

    balanced = dict(k_sweep_series(rows, views, matcher="loma", metric="balanced_top_10", k_values=(10, 50))[0])
    assert balanced["fine-tuned full"] == [(10, 73.0), (50, 75.0)]
    narrow = dict(k_sweep_series(rows, views, matcher="loma", metric="top_1", k_values=(50,))[0])
    assert narrow["fine-tuned full"] == [(50, 73.0)]


def test_fallback_checkpoint_stands_in_only_where_the_regular_run_is_missing(tmp_path: Path):
    for fraction, kept in ((0.5, 50), (1.0, 100)):
        _write_view(tmp_path, fraction, kept=kept, identities=10)
    experiments = tmp_path / "experiments"
    base = "/fewshot/checkpoints/Toy/legacy"
    stand_in = f"{base}/frac1.0-seed0/loma-finetuned-gpu1"
    _write_run(experiments, "split", "vismatch", "loma", "custom", 70.5, "2026-09-22T10:00:00Z",
               checkpoint_path=f"{base}/frac0.5-seed0/loma-finetuned/epoch_299/model.safetensors")
    _write_run(experiments, "split", "vismatch", "loma", "custom", 69.0, "2026-09-22T11:00:00Z",
               checkpoint_path=f"{stand_in}/epoch_299/model.safetensors")
    views = load_views(tmp_path, "Toy", "legacy", 0)
    rows = build_rows(probe_records(experiments, "Toy", 0, "full"), views, "Toy")
    assert sorted(r["checkpoint"] for r in rows) == ["custom", "custom-gpu1"]

    promoted = promote_fallback_checkpoints(rows, [stand_in])
    full = [r for r in promoted if r["view"] == "frac1.0-seed0"]
    assert [(r["checkpoint"], r["method"]) for r in full] == [("custom", "LoMa (fine-tuned)")]
    series = dict(k_sweep_series(promoted, views, matcher="loma", metric="top_1", k_values=(50,))[0])
    assert series["fine-tuned full"] == [(50, 69.0)]  # the stand-in carries the missing point

    # the regular 4-GPU run lands: it wins and the stand-in keeps its own label again
    _write_run(experiments, "split", "vismatch", "loma", "custom", 71.0, "2026-09-22T12:00:00Z",
               checkpoint_path=f"{base}/frac1.0-seed0/loma-finetuned/epoch_299/model.safetensors")
    rows = build_rows(probe_records(experiments, "Toy", 0, "full"), views, "Toy")
    promoted = promote_fallback_checkpoints(rows, [stand_in])
    assert sorted((r["checkpoint"], r["top_1"]) for r in promoted if r["view"] == "frac1.0-seed0") == [
        ("custom", 71.0), ("custom-gpu1", 69.0),
    ]
    series = dict(k_sweep_series(promoted, views, matcher="loma", metric="top_1", k_values=(50,))[0])
    assert series["fine-tuned full"] == [(50, 71.0)]  # the stand-in no longer feeds the fraction line
    assert "LoMa (fine-tuned, gpu1)" not in series  # nor does it open a series of its own there
    # and without naming the directory nothing is promoted at all
    assert [r["checkpoint"] for r in promote_fallback_checkpoints(rows, [])] == [r["checkpoint"] for r in rows]


def test_k_sweep_figure_draws_every_series(tmp_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    from reid.reporting.fewshot import render_k_sweep_figure

    experiments, views = _k_sweep_fixture(tmp_path)
    rows = build_rows(probe_records(experiments, "Toy", 0, "full"), views, "Toy")
    figure = render_k_sweep_figure(rows, views, animal="Toy", metric="top_1", matcher="loma", k_values=(10, 50))
    axis = figure.axes[0]
    labels = [line.get_label() for line in axis.lines]
    assert labels == ["fine-tuned 1/2", "fine-tuned full", "LoMa (default)", "WildFusion", "Cosine, k independent"]
    colours = [line.get_color() for line in axis.lines[:2]]
    assert colours == ["#ff895a", "#902500"]  # the fraction ramp runs light -> dark
    assert [text.get_text() for text in axis.texts] == ["full", "1/2"]  # direct labels, top value first
    assert axis.get_xscale() == "log" and [t for t in axis.get_xticks()] == [10, 50]
