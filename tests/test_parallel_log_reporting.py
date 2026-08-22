import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "scripts" / "probe_log_metadata.py"
SUMMARY = ROOT / "scripts" / "summarize_logs.py"


class ParallelLogReportingTests(unittest.TestCase):
    def test_metadata_lifecycle_and_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs_root = root / "logs" / "parallel_run"
            task_dir = logs_root / "WildlifeReID-10k" / "WhaleSharkID" / "job-123"
            task_dir.mkdir(parents=True)
            metadata = task_dir / "task-000__vismatch-rdd-lightglue__custom__k100.json"
            stdout = task_dir / "task.out"
            stderr = task_dir / "task.err"
            combined = task_dir / "task.combined.log"
            command = [
                sys.executable,
                str(METADATA),
                "init",
                "--path",
                str(metadata),
                "--job-id",
                "123",
                "--task-id",
                "0",
                "--dataset",
                "WildlifeReID-10k",
                "--animal",
                "WhaleSharkID",
                "--method",
                "vismatch",
                "--matcher",
                "rdd-lightglue",
                "--checkpoint",
                "custom",
                "--checkpoint-path",
                "/tmp/checkpoint.safetensors",
                "--candidate-k",
                "100",
                "--command",
                "python train/probe.py benchmark.method=vismatch",
                "--start-time",
                "2026-08-22T10:00:00Z",
                "--stdout-path",
                str(stdout),
                "--stderr-path",
                str(stderr),
                "--combined-path",
                str(combined),
                "--status",
                "running",
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(metadata.read_text())["status"], "running")

            stderr.write_text("ordinary warning\nRuntimeError: synthetic failure\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(METADATA),
                    "update",
                    "--path",
                    str(metadata),
                    "--status",
                    "failed",
                    "--end-time",
                    "2026-08-22T10:01:00Z",
                    "--experiment-run-directory",
                    "experiments/probe/example",
                    "--error-file",
                    str(stderr),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(metadata.read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error_summary"], "RuntimeError: synthetic failure")

            index = root / "logs" / "index.csv"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUMMARY),
                    "--logs-root",
                    str(logs_root),
                    "--index-path",
                    str(index),
                    "--write-index",
                    "--method",
                    "vismatch",
                    "--status",
                    "failed",
                    "--format",
                    "markdown",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("RuntimeError: synthetic failure", result.stdout)
            with index.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_k"], "100")
            self.assertEqual(rows[0]["experiment_run_directory"], "experiments/probe/example")

    def test_summary_filters_and_csv_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs_root = root / "logs"
            task_dir = logs_root / "dataset" / "animal" / "job-1"
            task_dir.mkdir(parents=True)
            for task_id, method in (("000", "cosine"), ("001", "vismatch")):
                payload = {
                    "schema_version": 1,
                    "job_id": "1",
                    "task_id": int(task_id),
                    "dataset": "dataset",
                    "animal": "animal",
                    "method": method,
                    "matcher": "-" if method == "cosine" else "loma",
                    "checkpoint": "default",
                    "candidate_k": 100,
                    "status": "completed",
                    "start_time": f"2026-08-22T10:0{task_id[-1]}:00Z",
                    "end_time": "",
                }
                (task_dir / f"task-{task_id}__{method}__default__k100.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUMMARY),
                    "--logs-root",
                    str(logs_root),
                    "--method",
                    "vismatch",
                    "--format",
                    "csv",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = list(csv.DictReader(result.stdout.splitlines()))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["matcher"], "loma")


if __name__ == "__main__":
    unittest.main()
