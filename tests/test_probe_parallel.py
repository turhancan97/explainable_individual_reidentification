import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "probe-parallel.sh"


class ParallelProbeLauncherTests(unittest.TestCase):
    def run_script(self, *args, env=None):
        merged_env = os.environ.copy()
        merged_env.pop("SLURM_ARRAY_TASK_ID", None)
        if env:
            merged_env.update(env)
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=ROOT,
            env=merged_env,
            text=True,
            capture_output=True,
            check=False,
        )

    def checkpoint_env(self, loma, rdd):
        return {
            "LOMA_CUSTOM_CHECKPOINT_PATH": str(loma),
            "RDD_CUSTOM_CHECKPOINT_PATH": str(rdd),
        }

    def task_rows(self):
        result = self.run_script("--list-tasks")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("index=")]
        return [dict(item.split("=", 1) for item in line.split()) for line in lines]

    def test_shell_syntax_is_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --output=logs/parallel_run/%x_%A_%a.out", text)
        self.assertIn("#SBATCH --error=logs/parallel_run/%x_%A_%a.err", text)

    def test_task_table_has_expected_grid(self):
        result = self.run_script("--list-tasks")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("index=")]
        self.assertGreater(len(lines), 0)

        parsed = []
        for line in lines:
            fields = dict(item.split("=", 1) for item in line.split())
            parsed.append(fields)

        self.assertEqual([int(row["index"]) for row in parsed], list(range(len(parsed))))
        self.assertEqual(sorted({int(row["candidate_k"]) for row in parsed}), [10, 50, 100, 250, 500, 1000])
        self.assertTrue({"cosine", "wildfusion", "vismatch"}.issubset({row["method"] for row in parsed}))
        self.assertEqual(sum(row["matcher"] == "loma" for row in parsed), 12)
        self.assertEqual(sum(row["matcher"] == "rdd-lightglue" for row in parsed), 12)
        self.assertEqual(len({(row["candidate_k"], row["method"], row["matcher"], row["checkpoint"]) for row in parsed}), len(parsed))

    def test_submission_dry_run_uses_array_range_and_cap(self):
        task_count = len(self.task_rows())
        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            result = self.run_script(
                "--dry-run",
                env={**self.checkpoint_env(loma.name, rdd.name), "MAX_CONCURRENT_JOBS": "7"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Submitting {task_count} probe tasks with array throttle 7", result.stdout)
        self.assertRegex(
            result.stdout,
            re.compile(rf"sbatch --array=0-{task_count - 1}%7 .*probe-parallel\.sh"),
        )

    def test_default_and_custom_checkpoint_mapping(self):
        rows = self.task_rows()

        def task_index(method, matcher, checkpoint):
            return next(
                int(row["index"])
                for row in rows
                if row["method"] == method and row["matcher"] == matcher and row["checkpoint"] == checkpoint
            )

        loma_default_index = task_index("vismatch", "loma", "default")
        loma_custom_index = task_index("vismatch", "loma", "custom")
        rdd_custom_index = task_index("vismatch", "rdd-lightglue", "custom")
        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            env = self.checkpoint_env(loma.name, rdd.name)
            default_result = self.run_script(
                "--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": str(loma_default_index), "PROBE_PARALLEL_DRY_RUN": "1"}
            )
            custom_result = self.run_script(
                "--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": str(loma_custom_index), "PROBE_PARALLEL_DRY_RUN": "1"}
            )
            rdd_result = self.run_script(
                "--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": str(rdd_custom_index), "PROBE_PARALLEL_DRY_RUN": "1"}
            )

        self.assertEqual(default_result.returncode, 0, default_result.stderr)
        self.assertIn("matcher=loma checkpoint=default", default_result.stdout)
        self.assertIn("checkpoint_source=default", default_result.stdout)
        self.assertNotIn("checkpoint_components=matcher_only", default_result.stdout)

        self.assertEqual(custom_result.returncode, 0, custom_result.stderr)
        self.assertIn("matcher=loma checkpoint=custom", custom_result.stdout)
        self.assertIn("checkpoint_components=matcher_only", custom_result.stdout)
        self.assertIn(loma.name, custom_result.stdout)

        self.assertEqual(rdd_result.returncode, 0, rdd_result.stderr)
        self.assertIn("matcher=rdd-lightglue checkpoint=custom", rdd_result.stdout)
        self.assertIn(rdd.name, rdd_result.stdout)

    def test_missing_custom_checkpoint_fails_before_probe(self):
        rows = self.task_rows()
        custom_loma_index = next(
            int(row["index"])
            for row in rows
            if row["method"] == "vismatch" and row["matcher"] == "loma" and row["checkpoint"] == "custom"
        )
        result = self.run_script(
            "--dry-run",
            env={
                "SLURM_ARRAY_TASK_ID": str(custom_loma_index),
                "LOMA_CUSTOM_CHECKPOINT_PATH": "/tmp/does-not-exist-loma-checkpoint",
                "RDD_CUSTOM_CHECKPOINT_PATH": "/tmp/does-not-exist-rdd-checkpoint",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LoMa custom checkpoint does not exist", result.stderr)

    def test_array_index_boundaries(self):
        task_count = len(self.task_rows())
        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            env = {**self.checkpoint_env(loma.name, rdd.name), "PROBE_PARALLEL_DRY_RUN": "1"}
            first = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": "0"})
            last = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": str(task_count - 1)})
            outside = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": str(task_count)})
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("candidate_k=10", first.stdout)
        self.assertEqual(last.returncode, 0, last.stderr)
        self.assertIn("candidate_k=1000", last.stdout)
        self.assertNotEqual(outside.returncode, 0)
        self.assertRegex(outside.stderr, re.compile(rf"outside 0\.\.{task_count - 1}"))

    def test_local_lightglue_uses_shared_budget(self):
        text = (ROOT / "conf/probe.yaml").read_text(encoding="utf-8")
        self.assertIn("local_lightglue:", text)
        self.assertNotIn("      B:", text)
        result = self.run_script("--list-tasks")
        local_lines = [line for line in result.stdout.splitlines() if "method=local_lightglue" in line]
        if local_lines:
            self.assertEqual(len(local_lines), 6)


if __name__ == "__main__":
    unittest.main()
