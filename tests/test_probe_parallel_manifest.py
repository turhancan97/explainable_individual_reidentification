import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "probe_parallel_manifest.py"


def run_helper(*args):
    return subprocess.run(["python", str(HELPER), *map(str, args)], cwd=ROOT, text=True, capture_output=True)


class ParallelManifestTests(unittest.TestCase):
    def task_line(self, checkpoint, owner="WhaleSharkID", animal="WhaleSharkID"):
        return "|".join([
            "whaleshark", "WildlifeReID-10k", animal, "/data",
            "metadata.csv", "identity", "mask", "false", "no_background",
            "split", "train", "test", "100", "vismatch", "rdd-lightglue",
            "custom", str(checkpoint), owner, "matcher_only", "-", "10",
        ])

    def create(self, root, checkpoint, line=None):
        config = root / "probe.yaml"
        config.write_text("dataset:\n  animal: Original\n")
        tasks = root / "tasks.tsv"
        tasks.write_text(line or self.task_line(checkpoint))
        submission = root / "submission"
        result = run_helper(
            "create", "--submission-dir", submission, "--submission-id", "s1",
            "--config-file", config, "--task-file", tasks,
            "--launcher-path", ROOT / "probe-parallel-wildlife.sh", "--repository-dir", ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return submission, config

    def test_snapshot_and_emit_are_submission_time_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"weights-v1")
            submission, config = self.create(root, checkpoint)
            emitted = run_helper("emit-shell", "--manifest", submission / "manifest.json", "--index", 0)
            self.assertEqual(emitted.returncode, 0, emitted.stderr)
            self.assertIn("DATASET_NAME=WildlifeReID-10k", emitted.stdout)
            self.assertIn("ANIMAL=WhaleSharkID", emitted.stdout)
            self.assertIn(f"CHECKPOINT_PATH={checkpoint}", emitted.stdout)
            config.write_text("dataset:\n  animal: Changed\n")
            self.assertIn("animal: Original", (submission / "probe.yaml").read_text())
            self.assertEqual(run_helper("validate", "--manifest", submission / "manifest.json", "--index", 0).returncode, 0)
            checkpoint.write_bytes(b"weights-v2")
            changed = run_helper("validate", "--manifest", submission / "manifest.json", "--index", 0)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("content changed", changed.stderr)

    def test_owner_and_missing_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"weights")
            mismatch = run_helper(
                "create", "--submission-dir", root / "bad", "--submission-id", "s",
                "--config-file", root / "missing.yaml", "--task-file", root / "tasks.tsv",
                "--launcher-path", ROOT / "probe-parallel-wildlife.sh", "--repository-dir", ROOT,
            )
            self.assertNotEqual(mismatch.returncode, 0)
            (root / "probe.yaml").write_text("x: 1\n")
            (root / "tasks.tsv").write_text(self.task_line(checkpoint, owner="NyalaData"))
            owner = run_helper(
                "create", "--submission-dir", root / "bad-owner", "--submission-id", "s",
                "--config-file", root / "probe.yaml", "--task-file", root / "tasks.tsv",
                "--launcher-path", ROOT / "probe-parallel-wildlife.sh", "--repository-dir", ROOT,
            )
            self.assertNotEqual(owner.returncode, 0)
            self.assertIn("owner mismatch", owner.stderr)

    def test_invalid_index_fails_without_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"weights")
            submission, _ = self.create(root, checkpoint)
            result = run_helper("validate", "--manifest", submission / "manifest.json", "--index", 9)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside manifest", result.stderr)


if __name__ == "__main__":
    unittest.main()
