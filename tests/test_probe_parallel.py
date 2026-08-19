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

    def test_shell_syntax_is_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_task_table_has_expected_grid(self):
        result = self.run_script("--list-tasks")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("index=")]
        self.assertEqual(len(lines), 54)

        parsed = []
        for line in lines:
            fields = dict(item.split("=", 1) for item in line.split())
            parsed.append(fields)

        self.assertEqual([int(row["index"]) for row in parsed], list(range(54)))
        self.assertEqual(sorted({int(row["candidate_k"]) for row in parsed}), [10, 50, 100, 250, 500, 1000])
        self.assertEqual(
            {row["method"] for row in parsed},
            {"cosine", "wildfusion", "local_lightglue", "linear_probe", "efficient_probe", "vismatch"},
        )
        self.assertEqual(sum(row["matcher"] == "loma" for row in parsed), 12)
        self.assertEqual(sum(row["matcher"] == "rdd-lightglue" for row in parsed), 12)
        self.assertEqual(len({(row["candidate_k"], row["method"], row["matcher"], row["checkpoint"]) for row in parsed}), 54)

    def test_submission_dry_run_uses_array_range_and_cap(self):
        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            result = self.run_script(
                "--dry-run",
                env={**self.checkpoint_env(loma.name, rdd.name), "MAX_CONCURRENT_JOBS": "7"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Submitting 54 probe tasks with array throttle 7", result.stdout)
        self.assertRegex(
            result.stdout,
            re.compile(r"sbatch --array=0-53%7 .*probe-parallel\.sh"),
        )

    def test_default_and_custom_checkpoint_mapping(self):
        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            env = self.checkpoint_env(loma.name, rdd.name)
            default_result = self.run_script(
                "--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": "5", "PROBE_PARALLEL_DRY_RUN": "1"}
            )
            custom_result = self.run_script(
                "--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": "6", "PROBE_PARALLEL_DRY_RUN": "1"}
            )
            rdd_result = self.run_script(
                "--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": "8", "PROBE_PARALLEL_DRY_RUN": "1"}
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
        result = self.run_script(
            "--dry-run",
            env={
                "SLURM_ARRAY_TASK_ID": "6",
                "LOMA_CUSTOM_CHECKPOINT_PATH": "/tmp/does-not-exist-loma-checkpoint",
                "RDD_CUSTOM_CHECKPOINT_PATH": "/tmp/does-not-exist-rdd-checkpoint",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LoMa custom checkpoint does not exist", result.stderr)

    def test_array_index_boundaries(self):
        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            env = {**self.checkpoint_env(loma.name, rdd.name), "PROBE_PARALLEL_DRY_RUN": "1"}
            first = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": "0"})
            last = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": "53"})
            outside = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": "54"})
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("candidate_k=10", first.stdout)
        self.assertEqual(last.returncode, 0, last.stderr)
        self.assertIn("candidate_k=1000", last.stdout)
        self.assertNotEqual(outside.returncode, 0)
        self.assertRegex(outside.stderr, re.compile(r"outside 0\.\.53"))

    def test_local_lightglue_uses_shared_budget(self):
        text = (ROOT / "conf/probe.yaml").read_text(encoding="utf-8")
        self.assertIn("local_lightglue:", text)
        self.assertNotIn("      B:", text)
        result = self.run_script("--list-tasks")
        local_lines = [line for line in result.stdout.splitlines() if "method=local_lightglue" in line]
        self.assertEqual(len(local_lines), 6)


if __name__ == "__main__":
    unittest.main()
