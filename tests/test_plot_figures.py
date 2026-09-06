import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from reid.reporting.paper_tables import discover_records
from reid.reporting.plot_figures import (
    DEFAULT_PLOT_BUDGETS,
    prepare_series_data,
    plot_metrics,
    render_metric_figure,
)


def write_plot_run(
    root: Path,
    *,
    animal: str,
    run_id: str,
    method: str,
    matcher: str = "-",
    checkpoint: str = "default",
    candidate_k: int | None = 10,
    top_1: float = 0.5,
    top_5: float = 0.6,
    top_10: float = 0.7,
    balanced_top_1: float = 0.4,
) -> None:
    run_dir = root / "probe" / "Dataset" / animal / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "run_utc": run_id,
        "workflow": "probe",
        "status": "completed",
        "animal": animal,
        "method": method,
        "variant": matcher if method == "vismatch" else checkpoint,
        "metrics": {
            "top_1": top_1,
            "top_5": top_5,
            "top_10": top_10,
            "balanced_top_1": balanced_top_1,
        },
        "timings": {"benchmark_candidate_k": candidate_k},
    }
    if method == "vismatch":
        manifest["vismatch_checkpoint"] = {"source": checkpoint}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(manifest["metrics"]), encoding="utf-8")
    (run_dir / "timings.json").write_text(json.dumps(manifest["timings"]), encoding="utf-8")


class PlotFigureTests(unittest.TestCase):
    def test_prepare_series_data_maps_runs_and_leaves_missing_budgets(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiments"
            write_plot_run(root, animal="Lynx", run_id="20260101_loma_10", method="vismatch", matcher="loma", candidate_k=10, top_1=0.50)
            write_plot_run(root, animal="Lynx", run_id="20260102_loma_100", method="vismatch", matcher="loma", candidate_k=100, top_1=0.70)
            write_plot_run(root, animal="Lynx", run_id="20260103_wildfusion_10", method="wildfusion", candidate_k=10, top_1=0.40)
            records = discover_records(root)

            series = prepare_series_data(records, animal="Lynx", metric="top_1")
            loma_default = next(item for item in series if item["name"] == "LoMa default")
            self.assertEqual(loma_default["budgets"], DEFAULT_PLOT_BUDGETS)
            self.assertEqual(loma_default["values"][0], 0.50)
            self.assertIsNone(loma_default["values"][1])
            self.assertEqual(loma_default["values"][2], 0.70)
            self.assertEqual({item["name"] for item in series}, {"WildFusion", "LoMa default"})

    def test_prepare_series_data_rejects_unknown_metric(self):
        with self.assertRaisesRegex(ValueError, "unsupported metric"):
            prepare_series_data([], animal="Lynx", metric="mAP")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is optional for dependency-light tests")
    def test_render_and_save_figures(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiments"
            output = Path(temp_dir) / "reports" / "figures"
            write_plot_run(root, animal="Lynx", run_id="20260101_loma_10", method="vismatch", matcher="loma", candidate_k=10)
            write_plot_run(root, animal="Whale", run_id="20260101_wildfusion_10", method="wildfusion", candidate_k=10)
            records = discover_records(root)
            figure = render_metric_figure(records, animals=["Lynx", "Whale"], metric="top_1")
            self.assertEqual(len(figure.axes), 2)
            figure.clf()
            outputs = plot_metrics(root, output, metrics=("top_1",), formats=("png", "pdf"))
            self.assertEqual({path.suffix for path in outputs}, {".png", ".pdf"})
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))


if __name__ == "__main__":
    unittest.main()
