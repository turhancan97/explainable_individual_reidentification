import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from reid.reporting.paper_tables import (
    DEFAULT_ABLATION_BUDGETS,
    build_ablation_rows,
    build_main_rows,
    discover_animals,
    discover_records,
    export_tables,
    render_latex,
    write_animal_tables,
)


def write_run(
    root: Path,
    *,
    animal: str,
    run_id: str,
    method: str,
    variant: str = "default",
    status: str = "completed",
    candidate_k=None,
    top_1=0.5,
    map_value=0.4,
    map_at_k=0.3,
):
    run_dir = root / "probe" / "Dataset" / animal / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "run_utc": run_id,
        "workflow": "probe",
        "status": status,
        "animal": animal,
        "method": method,
        "variant": variant,
        "metrics": {
            "top_1": top_1,
            "top_5": top_1 + 0.1,
            "top_10": top_1 + 0.2,
            "balanced_top_1": top_1 - 0.05,
            "mAP": map_value,
            "mAP_at_k": map_at_k,
        },
        "timings": {"total_run_min": 12.5},
    }
    if candidate_k is not None:
        manifest["metrics"]["map_at_k"] = candidate_k
        manifest["timings"]["benchmark_candidate_k"] = candidate_k
    if method == "vismatch":
        manifest["vismatch_checkpoint"] = {"source": variant}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(manifest["metrics"]), encoding="utf-8")
    (run_dir / "timings.json").write_text(json.dumps(manifest["timings"]), encoding="utf-8")
    return run_dir


class PaperTableTests(unittest.TestCase):
    def test_discovery_filters_status_and_selects_newest_run(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiments"
            write_run(root, animal="Lynx", run_id="20260101_old", method="cosine", top_1=0.2)
            write_run(root, animal="Lynx", run_id="20260102_new", method="cosine", top_1=0.8)
            direct_metrics = root / "probe" / "Dataset" / "Lynx" / "20260102_new" / "metrics.json"
            payload = json.loads(direct_metrics.read_text(encoding="utf-8"))
            payload["top_1"] = 0.91
            direct_metrics.write_text(json.dumps(payload), encoding="utf-8")
            write_run(root, animal="Lynx", run_id="20260103_failed", method="cosine", status="failed", top_1=0.99)
            write_run(root, animal="Whale", run_id="20260101_run", method="cosine")
            records = discover_records(root)

            self.assertEqual(discover_animals(records), ["Lynx", "Whale"])
            rows = build_main_rows(records, "Lynx", 100)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "20260102_new")
            self.assertEqual(rows[0]["top_1"], 0.91)

    def test_checkpoint_variants_and_shortlist_map_are_separate(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiments"
            write_run(root, animal="Lynx", run_id="20260101_default", method="vismatch", variant="default", candidate_k=100, map_value=float("nan"), map_at_k=0.31)
            write_run(root, animal="Lynx", run_id="20260102_custom", method="vismatch", variant="custom", candidate_k=100, map_value=float("nan"), map_at_k=0.72)
            records = discover_records(root)
            rows = build_main_rows(records, "Lynx", 100)

            self.assertEqual({row["checkpoint"] for row in rows}, {"default", "custom"})
            self.assertEqual({row["mAP"] for row in rows}, {None})
            self.assertEqual({row["mAP_at_k"] for row in rows}, {0.31, 0.72})

    def test_ablation_grid_preserves_missing_budgets(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiments"
            write_run(root, animal="Lynx", run_id="20260101_k10", method="vismatch", variant="custom", candidate_k=10)
            write_run(root, animal="Lynx", run_id="20260102_k100", method="vismatch", variant="custom", candidate_k=100)
            write_run(root, animal="Lynx", run_id="20260103_cosine", method="cosine")
            records = discover_records(root)
            rows = build_ablation_rows(records, "Lynx", DEFAULT_ABLATION_BUDGETS)
            matcher_rows = [row for row in rows if row["method_key"] == "vismatch"]

            self.assertEqual([row["candidate_k"] for row in matcher_rows], list(DEFAULT_ABLATION_BUDGETS))
            self.assertIsNone(next(row for row in matcher_rows if row["candidate_k"] == 50)["run_id"])
            self.assertEqual(len([row for row in rows if row["method_key"] == "cosine"]), 1)

    def test_latex_and_csv_outputs_have_provenance_and_display_format(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiments"
            output = Path(temp_dir) / "reports" / "paper_tables"
            write_run(root, animal="Czech_Lynx", run_id="20260101_run", method="cosine", top_1=0.8765)
            outputs = write_animal_tables(
                discover_records(root),
                animal="Czech_Lynx",
                output_dir=output,
                generated_at="2026-08-24T00:00:00+00:00",
            )
            self.assertEqual({path.suffix for path in outputs}, {".tex", ".csv"})
            latex = (output / "Czech_Lynx_main.tex").read_text(encoding="utf-8")
            self.assertIn("% generated_at=2026-08-24T00:00:00+00:00", latex)
            self.assertIn("% run_id=20260101_run", latex)
            self.assertIn(r"\resizebox{\linewidth}{!}{%", latex)
            self.assertIn("}%", latex)
            self.assertIn(r"\textbf{87.65}", latex)
            self.assertIn("12.50", latex)
            self.assertIn(r"\_", latex)
            with (output / "Czech_Lynx_main.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["top_1"], "0.8765")

    def test_export_all_animals_writes_four_files_each(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiments"
            output = Path(temp_dir) / "reports" / "paper_tables"
            write_run(root, animal="Lynx", run_id="20260101_run", method="cosine")
            write_run(root, animal="Whale", run_id="20260101_run", method="cosine")
            outputs = export_tables(root, output)
            self.assertEqual(len(outputs), 8)
            self.assertTrue((output / "Lynx_main.tex").is_file())
            self.assertTrue((output / "Whale_ablation.csv").is_file())

    def test_render_latex_missing_and_nan_values(self):
        row = {
            "method": "Vismatch",
            "matcher": "rdd-lightglue",
            "checkpoint": "custom_x",
            "candidate_k": 100,
            "top_1": float("nan"),
            "top_5": None,
            "top_10": 0.5,
            "balanced_top_1": None,
            "mAP": None,
            "mAP_at_k": 0.4,
            "runtime_min": 2.0,
            "run_id": None,
            "manifest_path": None,
        }
        latex = render_latex([row], animal="Lynx", table_name="main", candidate_k=100, generated_at="now")
        self.assertIn("--", latex)
        self.assertIn(r"custom\_x", latex)
        self.assertIn("40.00", latex)


if __name__ == "__main__":
    unittest.main()
