import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "probe-parallel-wildlife.sh"


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
        self.assertIn('SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"', text)

    def test_new_wildlife_profiles_are_ready_to_activate(self):
        script_text = SCRIPT.read_text(encoding="utf-8")
        expected_profiles = {
            "ATRW": ("metadata_ATRW.csv", "299", "299"),
            "Giraffes": ("metadata_Giraffes.csv", "299", "299"),
            "LeopardID2022": ("metadata_LeopardID2022.csv", "299", "299"),
            "HyenaID2022": ("metadata_HyenaID2022.csv", "299", "299"),
        }
        for animal, (metadata_name, loma_epoch, rdd_epoch) in expected_profiles.items():
            self.assertIn(f"|WildlifeReID-10k|{animal}|", script_text)
            self.assertIn(
                f"metadata_mdsplit_no_background/{metadata_name}",
                script_text,
            )
            self.assertIn(
                f"|100|legacy|legacy|{loma_epoch}|{rdd_epoch}\"",
                script_text,
            )

    def test_slurm_submission_directory_is_used(self):
        with tempfile.TemporaryDirectory() as submit_dir:
            result = self.run_script("--list-tasks", env={"SLURM_SUBMIT_DIR": submit_dir})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(submit_dir) / "logs" / "parallel_run").is_dir())

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
        script_text = SCRIPT.read_text(encoding="utf-8")
        candidate_matches = re.findall(r"(?m)^(?!\s*#)\s*CANDIDATE_K_VALUES=\(([^)]*)\)", script_text)
        self.assertTrue(candidate_matches)
        expected_candidates = sorted(int(value) for value in candidate_matches[-1].split())
        self.assertEqual(sorted({int(row["candidate_k"]) for row in parsed}), expected_candidates)
        methods = {row["method"] for row in parsed}
        self.assertIn("vismatch", methods)
        self.assertTrue(methods.issubset({"cosine", "wildfusion", "local_lightglue", "linear_probe", "efficient_probe", "vismatch"}))
        self.assertTrue(all(row["matcher"] in {"-", "loma", "rdd-lightglue"} for row in parsed))
        self.assertEqual(
            len({(row["candidate_k"], row["method"], row["matcher"], row["checkpoint"]) for row in parsed}),
            len(parsed),
        )

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
            re.compile(rf"sbatch --array=0-{task_count - 1}%7 .*probe-parallel-wildlife\.sh"),
        )

    def test_default_and_custom_checkpoint_mapping(self):
        rows = self.task_rows()
        custom_rows = [
            row for row in rows
            if row["method"] == "vismatch" and row["checkpoint"] == "custom"
        ]
        self.assertTrue(custom_rows)

        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            env = self.checkpoint_env(loma.name, rdd.name)
            for row in custom_rows:
                result = self.run_script(
                    "--dry-run",
                    env={
                        **env,
                        "SLURM_ARRAY_TASK_ID": row["index"],
                        "PROBE_PARALLEL_DRY_RUN": "1",
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("checkpoint_components=matcher_only", result.stdout)
                expected_path = loma.name if row["matcher"] == "loma" else rdd.name
                self.assertIn(expected_path, result.stdout)

            default_rows = [
                row for row in rows
                if row["method"] == "vismatch" and row["checkpoint"] == "default"
            ]
            for row in default_rows:
                result = self.run_script(
                    "--dry-run",
                    env={
                        **env,
                        "SLURM_ARRAY_TASK_ID": row["index"],
                        "PROBE_PARALLEL_DRY_RUN": "1",
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("checkpoint_source=default", result.stdout)
                self.assertNotIn("checkpoint_components=matcher_only", result.stdout)

    def test_missing_custom_checkpoint_fails_before_probe(self):
        rows = self.task_rows()
        custom_index = next(
            int(row["index"])
            for row in rows
            if row["method"] == "vismatch" and row["checkpoint"] == "custom"
        )
        result = self.run_script(
            "--dry-run",
            env={
                "SLURM_ARRAY_TASK_ID": str(custom_index),
                "LOMA_CUSTOM_CHECKPOINT_PATH": "/tmp/does-not-exist-loma-checkpoint",
                "RDD_CUSTOM_CHECKPOINT_PATH": "/tmp/does-not-exist-rdd-checkpoint",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, re.compile(r"(LoMa|RDD-LightGlue) custom checkpoint does not exist"))

    def test_array_index_boundaries(self):
        task_count = len(self.task_rows())
        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            env = {**self.checkpoint_env(loma.name, rdd.name), "PROBE_PARALLEL_DRY_RUN": "1"}
            first = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": "0"})
            last = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": str(task_count - 1)})
            outside = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": str(task_count)})
        self.assertEqual(first.returncode, 0, first.stderr)
        script_text = SCRIPT.read_text(encoding="utf-8")
        candidate_matches = re.findall(r"(?m)^(?!\s*#)\s*CANDIDATE_K_VALUES=\(([^)]*)\)", script_text)
        self.assertTrue(candidate_matches)
        expected_candidates = sorted(int(value) for value in candidate_matches[-1].split())
        self.assertIn(f"candidate_k={expected_candidates[0]}", first.stdout)
        self.assertEqual(last.returncode, 0, last.stderr)
        self.assertIn(f"candidate_k={expected_candidates[-1]}", last.stdout)
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
