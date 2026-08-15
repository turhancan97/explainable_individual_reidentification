import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from reid.data.safety_checks import run_split_safety_checks
from reid.evaluation.candidate_scoring import (
    build_shortlist_score_matrix,
    candidate_recall_metrics,
    normalize_shortlist_score,
    shortlist_pair_counts,
)
from reid.evaluation.metrics import _label_retrieval_metrics
from reid.evaluation.ranking import stable_rank_indices
from reid.methods.wildfusion_calibration import fit_pipeline_calibration
from reid.training.accumulation import accumulation_group_size
from reid.training.checkpointing import resolve_model_checkpoint
from reid.training.results import build_final_training_metrics
from reid.utils.fingerprints import hash_state_dict, sha256_file
try:
    from omegaconf import OmegaConf
    from reid.engine.probe_runner import _format_metric_value, extract_deep_features_with_cache
    HAS_PROBE_CACHE_DEPS = True
except ModuleNotFoundError:
    HAS_PROBE_CACHE_DEPS = False



class _FakeCalibration:
    def __init__(self):
        self.scores = None
        self.hits = None

    def fit(self, scores, hits):
        self.scores = np.asarray(scores)
        self.hits = np.asarray(hits)


class _FakeFeatures:
    def __init__(self, labels):
        self.labels_string = np.asarray(labels)


class _FakePipeline:
    def __init__(self):
        self.calibration = _FakeCalibration()
        self.calibration_done = False

    def get_feature_dataset(self, dataset):
        return _FakeFeatures(dataset.df["label"].tolist())

    def matcher(self, left, right):
        return np.arange(len(left.labels_string) * len(right.labels_string), dtype=np.float32).reshape(
            len(left.labels_string), len(right.labels_string)
        )


class ResearchValidityTests(unittest.TestCase):
    def test_stable_ranking_uses_original_index_for_ties(self):
        scores = np.array([[0.5, 0.7, 0.7, 0.1]], dtype=np.float32)
        np.testing.assert_array_equal(stable_rank_indices(scores), [[1, 2, 0, 3]])

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runtime dependencies not available")
    def test_console_metric_format_accepts_numeric_and_string_values(self):
        self.assertEqual(_format_metric_value(0.125), "0.125000")
        self.assertEqual(_format_metric_value(np.float32(0.25)), "0.250000")
        self.assertEqual(_format_metric_value("abc123"), "abc123")

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runtime dependencies not available")
    def test_probe_reporting_imports_file_identity(self):
        from reid.engine import probe_runner
        from reid.reporting.artifacts import file_identity

        self.assertIs(probe_runner.file_identity, file_identity)

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runtime dependencies not available")
    def test_deep_feature_cache_key_is_constructed_and_weight_aware(self):
        class FakeModel:
            def __init__(self, value):
                self.value = value

            def state_dict(self):
                return {"weight": np.asarray([self.value], dtype=np.float32)}

        class CaptureCache:
            def __init__(self):
                self.keys = []

            def get_or_compute(self, key, compute):
                self.keys.append(key)
                return np.zeros((1, 2), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = OmegaConf.create(
                {
                    "dataset": {
                        "root": tmp,
                        "metadata_file": "metadata.csv",
                        "image_variant": "background",
                        "no_background": False,
                        "label_col": "identity",
                    },
                    "model": {"type": "megadescriptor-l", "mode": "pretrained"},
                }
            )
            dataset = type("Dataset", (), {"df": pd.DataFrame({"path": ["missing.jpg"], "identity": ["id"]})})()
            cache = CaptureCache()
            extract_deep_features_with_cache(dataset, "database", FakeModel(1), None, 1, 0, cache, cfg, "cosine", None)
            extract_deep_features_with_cache(dataset, "database", FakeModel(2), None, 1, 0, cache, cfg, "cosine", None)
            self.assertEqual(len(cache.keys), 2)
            self.assertNotEqual(cache.keys[0], cache.keys[1])

    def test_accumulation_group_sizes_include_partial_final_group(self):
        self.assertEqual([accumulation_group_size(i, 5, 2) for i in range(5)], [2, 2, 2, 2, 1])
        self.assertEqual([accumulation_group_size(i, 4, 2) for i in range(4)], [2, 2, 2, 2])

    def test_map_counts_unmatched_queries_as_zero_and_reports_coverage(self):
        metrics = _label_retrieval_metrics(
            np.array(["a", "missing"]),
            np.array(["a", "b"]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            [1, 2],
            True,
        )
        self.assertEqual(metrics["num_queries_without_gallery_match"], 1.0)
        self.assertEqual(metrics["mAP_query_coverage"], 0.5)
        self.assertLess(metrics["mAP"], metrics["mAP_eligible"])

    def test_shortlist_matrix_uses_negative_infinity_and_candidate_recall(self):
        stage = np.array([[0.5, np.nan, 0.2], [np.inf, 0.1, 0.0]], dtype=np.float32)
        candidates = np.array([[0, 1], [1, 2]], dtype=np.int64)
        result = build_shortlist_score_matrix(stage, candidates)
        self.assertTrue(np.isneginf(result).all())
        self.assertNotIn(-1e9, result)
        recall = candidate_recall_metrics(["a", "c"], ["a", "b", "c"], candidates)
        self.assertEqual(recall["candidate_recall_at_k"], 1.0)

    def test_shortlist_pair_counts_are_auditable(self):
        counts = shortlist_pair_counts(2, 5, [[0, 1], [2]])
        self.assertEqual(counts["num_candidate_pairs"], 3.0)
        self.assertEqual(counts["num_unscored_pairs"], 7.0)
        self.assertEqual(counts["candidate_fraction"], 0.3)

    def test_invalid_candidate_scores_are_excluded(self):
        self.assertEqual(normalize_shortlist_score(float("nan")), -np.inf)
        self.assertEqual(normalize_shortlist_score(float("inf")), -np.inf)
        self.assertEqual(normalize_shortlist_score(0.25), 0.25)

    def test_full_matching_matrix_can_be_filled_completely(self):
        matrix = build_shortlist_score_matrix(np.zeros((2, 3), dtype=np.float32), None)
        matrix[:, :] = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.assertTrue(np.isfinite(matrix).all())

    def test_negative_infinity_ties_use_original_database_index(self):
        scores = np.array([[-np.inf, -np.inf, 0.5]], dtype=np.float32)
        np.testing.assert_array_equal(stable_rank_indices(scores), [[2, 0, 1]])

    def test_image_content_hash_changes_when_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.bin"
            path.write_bytes(b"first")
            first = sha256_file(path)
            path.write_bytes(b"second")
            self.assertNotEqual(first, sha256_file(path))

    def test_duplicate_content_is_rejected_even_when_paths_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jpg").write_bytes(b"same")
            (root / "b.jpg").write_bytes(b"same")
            with self.assertRaises(ValueError):
                run_split_safety_checks(
                    df_a=pd.DataFrame({"path": ["a.jpg"], "label": ["one"]}),
                    df_b=pd.DataFrame({"path": ["b.jpg"], "label": ["one"]}),
                    root=root,
                    run_dir=root / "run",
                    split_a_name="train",
                    split_b_name="test",
                    label_col="label",
                )
            summary = json.loads((root / "run" / "safety_checks" / "summary.json").read_text())
            self.assertEqual(summary["num_duplicate_content_hashes"], 1)

    def test_checkpoint_discovery_prefers_completed_nested_canonical_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = root / "dataset" / "failed"
            completed = root / "dataset" / "completed"
            failed.mkdir(parents=True)
            completed.mkdir(parents=True)
            (failed / "checkpoint-final.pth").write_bytes(b"failed")
            (failed / "run_manifest.json").write_text(json.dumps({"status": "failed"}))
            (completed / "checkpoint-final_tagged.pth").write_bytes(b"tagged")
            (completed / "checkpoint-final.pth").write_bytes(b"canonical")
            (completed / "run_manifest.json").write_text(json.dumps({"status": "completed"}))
            self.assertEqual(resolve_model_checkpoint(root).read_bytes(), b"canonical")

    def test_training_metrics_keep_best_and_final_epoch(self):
        metrics = build_final_training_metrics(
            {"top_1": 0.9},
            {"top_1": 0.8},
            best_epoch=3,
            best_metric="top_1",
            selected_checkpoint="checkpoint-best.pth",
        )
        self.assertEqual(metrics["top_1"], 0.9)
        self.assertEqual(metrics["final_epoch_metrics"]["top_1"], 0.8)

    def test_model_weight_hash_changes_when_weights_change(self):
        first = hash_state_dict({"weight": np.array([1.0, 2.0], dtype=np.float32)})
        second = hash_state_dict({"weight": np.array([1.0, 3.0], dtype=np.float32)})
        self.assertNotEqual(first, second)

    def test_official_calibration_mode_keeps_diagonal_pairs(self):
        dataset = type("Dataset", (), {})()
        dataset.df = pd.DataFrame({"path": ["a", "b"], "label": ["x", "y"]})

    def test_calibration_excludes_same_set_diagonal(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = type("Dataset", (), {})()
            dataset.df = pd.DataFrame({"path": ["a", "b"], "label": ["x", "y"]})
            pipeline = _FakePipeline()
            diagnostics = fit_pipeline_calibration(pipeline, dataset, dataset, exclude_self_pairs=True)
            self.assertEqual(diagnostics["excluded_self_pairs"], 2)
            self.assertEqual(len(pipeline.calibration.scores), 2)
            self.assertTrue(pipeline.calibration_done)


if __name__ == "__main__":
    unittest.main()
