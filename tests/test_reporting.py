import csv
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from omegaconf import OmegaConf

from reid.reporting.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    build_run_context,
    configuration_hash,
    safe_component,
    upsert_run_index,
)
from reid.reporting.summary import filter_run_rows, format_csv, format_markdown, sort_run_rows
from reid.reporting.visualizations import finalize_visualizations


def _config(root: str = "experiments"):
    return OmegaConf.create(
        {
            "dataset": {
                "name": "CzechLynx_v2",
                "animal": "Czech/Lynx",
                "split_col": "split-time_closed",
                "label_col": "identity",
            },
            "model": {"type": "megadescriptor-l"},
            "benchmark": {
                "method": "vismatch",
                "methods": {"vismatch": {"matcher": "loma"}},
            },
            "output": {"experiment_root": root},
            "reporting": {"index_path": "reports/runs.csv"},
        }
    )


class ReportingArtifactTests(unittest.TestCase):
    def test_run_path_and_hash_are_stable_and_safe(self):
        with TemporaryDirectory() as temp_dir:
            cfg = _config(str(Path(temp_dir) / "experiments"))
            started = datetime(2026, 8, 13, 14, 25, 30, tzinfo=timezone.utc)
            first = build_run_context(cfg, "probe", run_started=started)
            second = build_run_context(cfg, "probe", run_started=started)

            self.assertEqual(first.config_hash, second.config_hash)
            self.assertIn("Czech_Lynx", first.run_dir.as_posix())
            self.assertIn("vismatch/loma", first.run_dir.as_posix())
            self.assertRegex(first.run_id, r"^20260813T142530Z_[0-9a-f]{8}$")
            first.run_dir.mkdir(parents=True)
            collision = build_run_context(cfg, "probe", run_started=started)
            self.assertRegex(collision.run_id, r"^20260813T142530Z_[0-9a-f]{8}_01$")
            self.assertEqual(safe_component("a path/with spaces"), "a_path_with_spaces")
            self.assertNotEqual(configuration_hash(cfg), configuration_hash(OmegaConf.merge(cfg, {"seed": 1})))

    def test_manifest_metrics_and_timings_use_relative_artifacts(self):
        with TemporaryDirectory() as temp_dir:
            cfg = _config(str(Path(temp_dir) / "experiments"))
            context = build_run_context(cfg, "probe")
            context.write_config(cfg)
            context.write_metrics({"top_1": 0.75})
            context.write_timings({"total_runtime_sec": 12.5})
            context.write_manifest({"dataset": "CzechLynx_v2"}, status="completed")

            manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], ARTIFACT_SCHEMA_VERSION)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["artifacts"]["metrics"], "metrics.json")
            self.assertEqual(json.loads(context.metrics_path.read_text())["top_1"], 0.75)

    def test_run_index_upsert_replaces_same_run(self):
        with TemporaryDirectory() as temp_dir:
            index = Path(temp_dir) / "reports" / "runs.csv"
            upsert_run_index(index, {"run_id": "run-a", "status": "running", "top_1": ""})
            upsert_run_index(index, {"run_id": "run-a", "status": "completed", "top_1": 0.9})
            with index.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(rows[0]["top_1"], "0.9")

    def test_visualization_index_and_contact_sheets(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow unavailable")

        class Dataset:
            df = __import__("pandas").DataFrame(
                {
                    "identity": ["lynx-a", "lynx-b"],
                    "path": ["query.jpg", "database.jpg"],
                }
            )

        with TemporaryDirectory() as temp_dir:
            cfg = _config(str(Path(temp_dir) / "experiments"))
            context = build_run_context(cfg, "probe")
            prediction = context.visualization_dir / "predictions" / "query_000000.png"
            prediction.parent.mkdir(parents=True)
            Image.new("RGB", (20, 20), "white").save(prediction)
            similarity = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
            artifacts = finalize_visualizations(
                context,
                prediction_paths=[str(prediction)],
                similarity=similarity,
                dataset_query=Dataset(),
                dataset_database=Dataset(),
                label_col="identity",
                top_k=2,
            )
            self.assertTrue(Path(artifacts["index"]).is_file())
            self.assertTrue(Path(artifacts["contact_sheet_top1"]).is_file())
            with Path(artifacts["index"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["query_index"], "0")
            self.assertEqual(rows[0]["rank"], "1")

    def test_visualization_index_ranking_matches_the_drawn_grid(self):
        # The index annotates the prediction grid, so it must resolve ties exactly as
        # `stable_rank_indices` does. A shortlist matrix is almost entirely `-inf` ties,
        # where a reversed argsort would order them backwards and describe other images.
        from reid.evaluation.ranking import stable_rank_indices
        from reid.reporting.visualizations import prediction_index_rows

        class Dataset:
            df = __import__("pandas").DataFrame(
                {
                    "identity": [f"lynx-{i}" for i in range(5)],
                    "path": [f"db_{i}.jpg" for i in range(5)],
                }
            )

        # One scored candidate; the remaining four are unscored shortlist positions.
        similarity = np.asarray([[-np.inf, -np.inf, 0.7, -np.inf, -np.inf]], dtype=np.float32)
        grid_order = stable_rank_indices(similarity)[0, :4].tolist()

        with TemporaryDirectory() as temp_dir:
            prediction = Path(temp_dir) / "query_000000.png"
            prediction.write_bytes(b"")
            rows, _failures = prediction_index_rows(
                prediction_paths=[str(prediction)],
                similarity=similarity,
                dataset_query=Dataset(),
                dataset_database=Dataset(),
                label_col="identity",
                top_k=4,
            )

        self.assertEqual([row["database_index"] for row in rows], grid_order)
        self.assertEqual(grid_order, [2, 0, 1, 3])
        self.assertEqual([row["rank"] for row in rows], [1, 2, 3, 4])

    def test_summary_filter_sort_and_formats(self):
        rows = [
            {"run_id": "a", "dataset": "lynx", "workflow": "probe", "method": "vismatch", "variant": "loma", "top_1": "0.7"},
            {"run_id": "b", "dataset": "lynx", "workflow": "probe", "method": "cosine", "variant": "default", "top_1": "0.9"},
        ]
        filtered = filter_run_rows(rows, dataset="lynx", method="vismatch", matcher="loma")
        sorted_rows = sort_run_rows(filtered, "top_1")
        self.assertEqual(sorted_rows[0]["run_id"], "a")
        self.assertIn("| run_id |", format_markdown(sorted_rows))
        self.assertIn("run_id", format_csv(sorted_rows).splitlines()[0])


if __name__ == "__main__":
    unittest.main()
