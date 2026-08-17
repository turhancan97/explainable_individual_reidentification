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
from reid.utils.io import append_csv_row, update_csv_rows
from reid.methods.vismatch_profiles import FrameFeatures
from reid.methods.vismatch_profiles import (
    FEATURE_SCHEMA_VERSION,
    SUPPORTED_VISMATCH_MATCHERS,
    default_matcher_threshold,
    build_matcher_profile,
    normalize_match_confidences,
    profile_fingerprint,
    validate_matcher_name,
)

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
            "conf/finetune.yaml",
            "conf/probe.yaml",
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

    def test_new_checkpoint_root_falls_back_to_legacy_root(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            modern = root / "experiments" / "finetune"
            legacy = root / "results"
            legacy_run = legacy / "run-legacy"
            legacy_run.mkdir(parents=True)
            checkpoint = legacy_run / "checkpoint-final_legacy.pth"
            checkpoint.touch()
            self.assertEqual(
                resolve_configured_model_checkpoint(
                    explicit_path=None,
                    results_dir=modern,
                    fallback_dirs=[legacy],
                ),
                checkpoint,
            )

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


class CsvRuntimeTests(unittest.TestCase):
    def test_update_csv_rows_adds_final_finetune_runtime(self):
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "train_metrics.csv"
            append_csv_row(csv_path, {"run_id": "run-a", "epoch": 1, "top_1": 0.5})
            append_csv_row(csv_path, {"run_id": "run-a", "epoch": 2, "top_1": 0.6})
            append_csv_row(csv_path, {"run_id": "run-b", "epoch": 1, "top_1": 0.7})

            updated = update_csv_rows(
                csv_path,
                match={"run_id": "run-a"},
                updates={"total_run_sec": 12.5, "total_run_min": 12.5 / 60.0},
            )

            self.assertEqual(updated, 2)
            rows = pd.read_csv(csv_path)
            self.assertEqual(rows.loc[0, "total_run_sec"], 12.5)
            self.assertAlmostEqual(rows.loc[1, "total_run_min"], 12.5 / 60.0)
            self.assertTrue(pd.isna(rows.loc[2, "total_run_sec"]))


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


class VismatchProfileTests(unittest.TestCase):
    def test_frame_features_use_canonical_schema(self):
        features = FrameFeatures(
            keypoints=np.zeros((3, 2), dtype=np.float32),
            descriptors=np.zeros((3, 8), dtype=np.float32),
            scores=np.ones(3, dtype=np.float32),
            image_size=np.asarray([32, 48], dtype=np.int32),
        )
        self.assertEqual(features.schema_version, FEATURE_SCHEMA_VERSION)
        self.assertEqual(features.coordinate_convention, "pixel")
        self.assertEqual(features.image_size_convention, "hw")
        normalized = FrameFeatures(
            keypoints=np.zeros((3, 2), dtype=np.float32),
            descriptors=np.zeros((3, 8), dtype=np.float32),
            scores=np.ones(3, dtype=np.float32),
            image_size=np.asarray([42, 56], dtype=np.int32),
            coordinate_convention="normalized[-1,1]",
            original_image_size=np.asarray([40, 53], dtype=np.int32),
        )
        self.assertEqual(normalized.coordinate_convention, "normalized[-1,1]")
        with self.assertRaises(ValueError):
            FrameFeatures(
                keypoints=np.zeros((3, 3), dtype=np.float32),
                descriptors=np.zeros((3, 8), dtype=np.float32),
                scores=np.ones(3, dtype=np.float32),
                image_size=np.asarray([32, 48], dtype=np.int32),
            )

    def test_supported_matchers_and_legacy_method_rejection(self):
        self.assertEqual(
            SUPPORTED_VISMATCH_MATCHERS,
            ("rdd-lightglue", "aliked-lightglue", "superpoint-lightglue", "loma"),
        )
        self.assertEqual(validate_matcher_name("RDD-LightGlue"), "rdd-lightglue")
        self.assertEqual(validate_matcher_name("LoMa"), "loma")
        self.assertEqual(default_matcher_threshold("loma"), 0.1)
        self.assertEqual(default_matcher_threshold("rdd-lightglue"), 0.01)
        with self.assertRaises(ValueError):
            validate_matcher_name("rdd")
        with self.assertRaises(ValueError):
            validate_matcher_name("unknown-lightglue")

    def test_profiles_are_reproducible_and_matcher_specific(self):
        rdd = build_matcher_profile("rdd-lightglue", 512, 0.01)
        aliked = build_matcher_profile("aliked-lightglue", 512, 0.01)
        loma = build_matcher_profile("loma", 512, 0.1)
        self.assertEqual(rdd.feature_schema_version, FEATURE_SCHEMA_VERSION)
        self.assertNotEqual(profile_fingerprint(rdd), profile_fingerprint(aliked))
        self.assertNotEqual(profile_fingerprint(rdd), profile_fingerprint(loma))
        self.assertIn("multiple of 14", loma.preprocessing)
        self.assertEqual(loma.preprocessing_version, "lynx_loma_finetuning_v1")
        self.assertEqual(loma.score_mode, "mutual_confidence_sum_over_min_keypoints")
        self.assertNotEqual(
            profile_fingerprint(rdd),
            profile_fingerprint(build_matcher_profile("rdd-lightglue", 1024, 0.01)),
        )

    def test_confidence_normalization_preserves_legacy_formula(self):
        score, count, values = normalize_match_confidences([0.01, 0.5, 0.9], 4, 8, 0.01)
        self.assertAlmostEqual(score, (0.01 + 0.5 + 0.9) / 4.0)
        self.assertEqual(count, 3)
        self.assertEqual(values, [0.01, 0.5, 0.9])

    def test_empty_confidences_have_zero_score(self):
        score, count, values = normalize_match_confidences([], 0, 0, 0.01)
        self.assertEqual((score, count, values), (0.0, 0, []))

    def test_shipped_configs_use_vismatch_public_method(self):
        root = Path(__file__).resolve().parents[1]
        probe = (root / "conf/probe.yaml").read_text(encoding="utf-8")
        # Keep the user's current Stage-A default intact while validating that
        # Vismatch remains a shipped, independently selectable public method.
        self.assertRegex(probe, r'(?m)^  method: "(?:vismatch|wildfusion)"')
        self.assertIn('    vismatch:', probe)
        self.assertNotIn('    rdd:', probe)
        jaguar = (root / "config/kaggle_jaguar.yaml").read_text(encoding="utf-8")
        self.assertIn('      local_top_k: 512', probe)
        self.assertIn('    local_top_k: 512', jaguar)
        self.assertIn('vismatch:', jaguar)
        self.assertIn('loma', probe)
        self.assertIn('loma', jaguar)
        self.assertIn('stage_a_plus_vismatch', jaguar)
        self.assertNotIn('stage_a_plus_rdd', jaguar)
    def test_image_variant_cache_identity_is_explicit_and_stable(self):
        from reid.utils.cache_identity import build_dataset_cache_identity, validate_image_variant

        base = type("DatasetConfig", (), {
            "root": "/tmp/dataset",
            "metadata_file": "metadata_with_background/metadata_NyalaData.csv",
            "image_variant": "background",
        })()
        identity = build_dataset_cache_identity(base)
        self.assertEqual(identity["image_variant"], "background")
        self.assertEqual(identity["metadata_file"], "metadata_with_background/metadata_NyalaData.csv")
        self.assertNotIn("/tmp/", identity["metadata_file"])
        self.assertEqual(validate_image_variant("NO_BACKGROUND"), "no_background")

        masked = type("DatasetConfig", (), {
            "root": "/tmp/dataset",
            "metadata_file": "metadata_no_background/metadata_NyalaData.csv",
            "image_variant": "no_background",
        })()
        masked_identity = build_dataset_cache_identity(masked)
        self.assertNotEqual(identity, masked_identity)
        with self.assertRaises(ValueError):
            validate_image_variant("masked-ish")


if __name__ == "__main__":
    unittest.main()
