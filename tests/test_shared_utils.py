import os
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from reid.evaluation.metrics import compute_metrics

from reid.config_defaults import (
    DEFAULT_MODEL_TYPE,
    SUPPORTED_MODEL_TYPES,
    validate_model_type,
)
from reid.training.accumulation import should_step_accumulated_gradients

try:
    from reid.data.dataset_view import BenchmarkDatasetView
    from reid.features.containers import normalize_features
    HAS_DATASET_DEPS = True
except ModuleNotFoundError:
    HAS_DATASET_DEPS = False

try:
    from reid.training.checkpointing import (
        resolve_configured_model_checkpoint,
        resolve_model_checkpoint,
    )
    HAS_CHECKPOINT_DEPS = True
except ModuleNotFoundError:
    HAS_CHECKPOINT_DEPS = False


class DummyDataset:
    def __init__(self):
        self.col_label = "label"
        self.df = pd.DataFrame({"label": ["a", "b"], "mask": [None, None]})
        self.metadata = self.df

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        # HWC image + label
        return np.zeros((4, 4, 3), dtype=np.uint8), self.df.iloc[idx]["label"]


class MetricsTests(unittest.TestCase):
    def test_compute_metrics_top1(self):
        q = DummyDataset()
        d = DummyDataset()
        sim = np.array([[1.0, 0.2], [0.1, 1.0]], dtype=np.float32)
        out = compute_metrics(q, d, sim, [1], compute_map=True)
        self.assertAlmostEqual(out["top_1"], 1.0)
        self.assertGreaterEqual(out["mAP"], 0.99)


class NormalizeTests(unittest.TestCase):
    @unittest.skipUnless(HAS_DATASET_DEPS, "dataset dependencies not available")
    def test_normalize_from_row_pairs(self):
        raw = [
            (np.array([1.0, 2.0], dtype=np.float32), {"meta": 1}),
            (np.array([3.0, 4.0], dtype=np.float32), {"meta": 2}),
        ]
        arr = normalize_features(raw)
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype, np.float32)


class DatasetViewTests(unittest.TestCase):
    @unittest.skipUnless(HAS_DATASET_DEPS, "dataset dependencies not available")
    def test_view_passthrough_without_mask(self):
        base = DummyDataset()
        view = BenchmarkDatasetView(base_dataset=base, label_col="label", no_background=False)
        image, label = view[0]
        self.assertEqual(image.shape, (4, 4, 3))
        self.assertEqual(label, "a")


class ConfigurationTests(unittest.TestCase):
    def test_shipped_yaml_model_defaults_are_supported(self):
        root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "config/finetune_config.yaml",
            "config/probe_config.yaml",
            "config/kaggle_jaguar.yaml",
        ):
            text = (root / relative_path).read_text(encoding="utf-8")
            match = re.search(r'(?m)^  type:\s*"([^"]+)"', text)
            self.assertIsNotNone(match, relative_path)
            self.assertIn(match.group(1), SUPPORTED_MODEL_TYPES)

    def test_default_model_and_legacy_name(self):
        self.assertEqual(DEFAULT_MODEL_TYPE, "megadescriptor-l")
        self.assertIn(DEFAULT_MODEL_TYPE, SUPPORTED_MODEL_TYPES)
        self.assertNotIn("megadescriptor", SUPPORTED_MODEL_TYPES)
        with self.assertRaises(ValueError):
            validate_model_type("megadescriptor")


@unittest.skipUnless(HAS_CHECKPOINT_DEPS, "checkpoint dependencies not available")
class CheckpointResolutionTests(unittest.TestCase):
    def test_canonical_checkpoint_is_preferred_in_newest_run(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            older = root / "older"
            newer = root / "newer"
            older.mkdir()
            newer.mkdir()
            legacy = older / "checkpoint-final_legacy.pth"
            canonical = newer / "checkpoint-final.pth"
            legacy.touch()
            canonical.touch()
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))
            self.assertEqual(resolve_model_checkpoint(root), canonical)

    def test_tagged_checkpoint_fallback(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            tagged = run_dir / "checkpoint-final_CzechLynx.pth"
            tagged.touch()
            self.assertEqual(resolve_model_checkpoint(root), tagged)

    def test_explicit_checkpoint_takes_precedence(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            discovered = run_dir / "checkpoint-final.pth"
            discovered.touch()
            explicit = root / "explicit.pth"
            explicit.touch()
            self.assertEqual(
                resolve_configured_model_checkpoint(explicit, root),
                explicit,
            )

    def test_missing_checkpoint_fails_clearly(self):
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_model_checkpoint(Path(temp_dir))

    def test_full_checkpoint_is_not_selected_as_model_checkpoint(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            full_checkpoint = run_dir / "checkpoint-final-full_legacy.pth"
            full_checkpoint.touch()
            with self.assertRaises(FileNotFoundError):
                resolve_model_checkpoint(root)
            with self.assertRaises(ValueError):
                resolve_model_checkpoint(root, filename="checkpoint-final-full.pth")
            with self.assertRaises(ValueError):
                resolve_configured_model_checkpoint(full_checkpoint, root)


class AccumulationTests(unittest.TestCase):
    def test_divisible_batches_step_only_at_accumulation_boundaries(self):
        flags = [
            should_step_accumulated_gradients(i, total_batches=4, accumulation_steps=2)
            for i in range(4)
        ]
        self.assertEqual(flags, [False, True, False, True])

    def test_partial_final_group_is_flushed(self):
        flags = [
            should_step_accumulated_gradients(i, total_batches=5, accumulation_steps=2)
            for i in range(5)
        ]
        self.assertEqual(flags, [False, True, False, True, True])

    def test_invalid_accumulation_steps_fail(self):
        with self.assertRaises(ValueError):
            should_step_accumulated_gradients(0, total_batches=1, accumulation_steps=0)


if __name__ == "__main__":
    unittest.main()
