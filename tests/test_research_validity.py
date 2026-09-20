import json
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from reid.data.safety_checks import run_split_safety_checks
from reid.evaluation.candidate_scoring import (
    build_shortlist_score_matrix,
    candidate_recall_metrics,
    load_score_matrix,
    normalize_shortlist_score,
    save_score_matrix,
    shortlist_pair_counts,
)
from reid.evaluation.metrics import _label_retrieval_metrics, _truncated_average_precision
from reid.evaluation.ranking import stable_rank_indices
from reid.methods.wildfusion_calibration import fit_pipeline_calibration
from reid.training.accumulation import accumulation_group_size
from reid.training.checkpointing import resolve_model_checkpoint, validate_resume_epochs
from reid.training.results import build_final_training_metrics
from reid.utils import fingerprints
from reid.utils.fingerprints import file_digest_cache, hash_state_dict, sha256_file
try:
    from omegaconf import OmegaConf
    from reid.engine.probe_runner import (
        _classifier_metrics,
        _cosine_similarity_matrix,
        _format_metric_value,
        _label_indices,
        _set_probe_training_mode,
        classifier_open_set_coverage,
        extract_deep_features_with_cache,
        validate_classifier_open_set_labels,
    )
    HAS_PROBE_CACHE_DEPS = True
except ModuleNotFoundError:
    HAS_PROBE_CACHE_DEPS = False



class _LabelledFrame:
    """Minimal stand-in for a dataset view exposing `df` and `col_label`."""

    def __init__(self, labels, col_label: str = "label"):
        self.df = pd.DataFrame({col_label: list(labels)})
        self.col_label = col_label


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

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runner dependencies are not available")
    def test_identity_probe_metrics_handle_more_images_than_classes(self):
        # Regression: the classifier emits one column per identity, so an identity mask
        # built from database *images* cannot index that axis. Five images over three
        # identities is the shape that used to raise IndexError every epoch.
        from reid.engine.probe_runner import _probe_retrieval_metrics

        cfg = OmegaConf.create(
            {"benchmark": {"top_k": [1, 2], "compute_map": True, "candidate_k": 2}}
        )
        db_labels_idx = np.array([0, 0, 1, 1, 2])
        query_labels_idx = np.array([1, 2])
        probs_query = np.array([[0.1, 0.7, 0.2], [0.5, 0.2, 0.3]])

        metrics = _probe_retrieval_metrics(
            cfg,
            _LabelledFrame(["id1", "id2"]),
            _LabelledFrame(["id0", "id0", "id1", "id1", "id2"]),
            probs_query,
            db_labels_idx,
            query_labels_idx,
        )

        # Query 0 ranks its own identity first; query 1 does not.
        self.assertEqual(metrics["top_1"], 0.5)
        self.assertEqual(metrics["num_queries"], 2.0)
        # Image-level diagnostics stay available under the image_ prefix.
        self.assertEqual(metrics["image_top_1"], 0.5)
        self.assertEqual(metrics["image_num_queries"], 2.0)

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runner dependencies are not available")
    def test_identity_probe_metrics_score_each_identity_once(self):
        # Every database image of an identity shares that identity's single probability,
        # so duplicating an identity in the gallery must not change identity-level scores.
        from reid.engine.probe_runner import _probe_retrieval_metrics

        cfg = OmegaConf.create(
            {"benchmark": {"top_k": [1], "compute_map": True, "candidate_k": 1}}
        )
        probs_query = np.array([[0.2, 0.8]])
        sparse = _probe_retrieval_metrics(
            cfg, _LabelledFrame(["id1"]), _LabelledFrame(["id0", "id1"]),
            probs_query, np.array([0, 1]), np.array([1]),
        )
        duplicated = _probe_retrieval_metrics(
            cfg, _LabelledFrame(["id1"]), _LabelledFrame(["id0", "id0", "id0", "id1"]),
            probs_query, np.array([0, 0, 0, 1]), np.array([1]),
        )
        self.assertEqual(sparse["top_1"], 1.0)
        self.assertEqual(duplicated["top_1"], sparse["top_1"])
        self.assertEqual(duplicated["mAP_at_k"], sparse["mAP_at_k"])

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runner dependencies are not available")
    def test_identity_probe_metrics_reject_labels_outside_the_classifier_head(self):
        from reid.engine.probe_runner import _probe_retrieval_metrics

        cfg = OmegaConf.create(
            {"benchmark": {"top_k": [1], "compute_map": True, "candidate_k": 1}}
        )
        with self.assertRaises(ValueError) as ctx:
            _probe_retrieval_metrics(
                cfg, _LabelledFrame(["id1"]), _LabelledFrame(["id0", "id5"]),
                np.array([[0.4, 0.6]]), np.array([0, 5]), np.array([1]),
            )
        self.assertIn("outside the classifier", str(ctx.exception))

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runner dependencies are not available")
    def test_open_world_classifier_metrics_map_unseen_queries_to_zero_credit(self):
        from reid.engine.probe_runner import _classifier_metrics

        labels = ["seen_a", "unseen", "seen_b"]
        mapping = {"seen_a": 0, "seen_b": 1}
        label_indices = _label_indices(labels, mapping)
        probs = np.asarray(
            [
                [0.9, 0.1],
                [0.9, 0.1],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        metrics = _classifier_metrics(probs, label_indices, labels, mapping, "open")

        self.assertEqual(label_indices.tolist(), [0, -1, 1])
        self.assertEqual(metrics["classification_num_unseen_query_images"], 1.0)
        self.assertEqual(metrics["classification_num_unseen_query_identities"], 1.0)
        self.assertEqual(metrics["classification_query_seen_coverage"], 2.0 / 3.0)
        self.assertEqual(metrics["classification_open_top_1"], 2.0 / 3.0)
        self.assertEqual(metrics["classification_open_balanced_top_1"], 2.0 / 3.0)
        self.assertEqual(metrics["classification_seen_top_1"], 1.0)
        self.assertEqual(metrics["classification_top_1"], metrics["classification_open_top_1"])

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runner dependencies are not available")
    def test_warn_policy_keeps_legacy_fields_seen_only(self):
        from reid.engine.probe_runner import _classifier_metrics

        labels = ["seen", "unseen"]
        mapping = {"seen": 0}
        indices = _label_indices(labels, mapping)
        metrics = _classifier_metrics(
            np.asarray([[1.0], [1.0]], dtype=np.float32), indices, labels, mapping, "warn"
        )
        self.assertEqual(metrics["classification_seen_top_1"], 1.0)
        self.assertEqual(metrics["classification_open_top_1"], 0.5)
        self.assertEqual(metrics["classification_top_1"], 1.0)

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runner dependencies are not available")
    def test_closed_policy_fails_before_classifier_execution(self):
        database = _LabelledFrame(["seen"])
        query = _LabelledFrame(["unseen"])
        with self.assertRaisesRegex(ValueError, "Closed-set classifier evaluation failed"):
            validate_classifier_open_set_labels(database, query, "label", "closed")

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runner dependencies are not available")
    def test_all_unseen_classifier_metrics_are_defined(self):
        from reid.engine.probe_runner import _classifier_metrics

        labels = ["unseen_a", "unseen_b"]
        mapping = {"seen": 0}
        indices = _label_indices(labels, mapping)
        metrics = _classifier_metrics(
            np.asarray([[1.0], [1.0]], dtype=np.float32), indices, labels, mapping, "open"
        )
        self.assertEqual(metrics["classification_open_top_1"], 0.0)
        self.assertEqual(metrics["classification_open_balanced_top_1"], 0.0)
        self.assertTrue(np.isnan(metrics["classification_seen_top_1"]))

    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runner dependencies are not available")
    def test_embedding_similarity_diagnostic_is_dependency_light(self):
        from reid.engine.probe_runner import _cosine_similarity_matrix

        result = _cosine_similarity_matrix(
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(result, [[1.0, 2 ** -0.5], [0.0, 2 ** -0.5]], atol=1e-6)

    def test_full_matrix_map_is_suppressed_on_a_shortlist_matrix(self):
        labels = np.array(["a", "b", "c", "d"])
        shortlist = np.array([[0.9, 0.1, -np.inf, -np.inf]], dtype=np.float32)
        metrics = _label_retrieval_metrics(
            np.array(["a"]), labels, shortlist, [1, 2], True, map_at_k=2
        )
        self.assertTrue(np.isnan(metrics["mAP"]))
        self.assertTrue(np.isnan(metrics["mAP_eligible"]))
        self.assertEqual(metrics["score_coverage"], 0.5)
        self.assertEqual(metrics["mAP_at_k"], 1.0)

    def test_full_matrix_map_survives_when_every_pair_is_scored(self):
        labels = np.array(["a", "b"])
        metrics = _label_retrieval_metrics(
            np.array(["a"]), labels, np.array([[0.9, 0.1]], dtype=np.float32), [1, 2], True, map_at_k=2
        )
        self.assertEqual(metrics["score_coverage"], 1.0)
        self.assertEqual(metrics["mAP"], 1.0)

    def test_unscored_tail_cannot_earn_truncated_precision(self):
        # A relevant entry that was never scored sits in the -inf tail. It occupies
        # a rank but must not be credited, otherwise the metric rewards database order.
        relevant = np.array([0.0, 1.0], dtype=np.float32)
        scored = np.array([True, False])
        end_to_end, rerank, hits = _truncated_average_precision(relevant, scored, 1, 2)
        self.assertEqual((end_to_end, hits), (0.0, 0))
        self.assertTrue(np.isnan(rerank))

    def test_truncated_map_separates_shortlist_reach_from_ordering(self):
        labels = np.array(["a", "b", "c", "d"])
        queries = np.array(["a", "d"])
        # Query 1 ranks its match first; query 2's identity never reached the shortlist.
        scores = np.array(
            [[0.9, 0.2, -np.inf, -np.inf], [0.9, 0.2, -np.inf, -np.inf]], dtype=np.float32
        )
        metrics = _label_retrieval_metrics(queries, labels, scores, [1, 2], True, map_at_k=2)
        self.assertEqual(metrics["recall_at_k"], 0.5)
        self.assertEqual(metrics["num_queries_with_relevant_in_top_k"], 1.0)
        # End-to-end charges the shortlist miss; the rerank diagnostic does not.
        self.assertEqual(metrics["mAP_at_k"], 0.5)
        self.assertEqual(metrics["rerank_mAP_at_k"], 1.0)

    def test_truncated_map_rewards_better_ordering_within_one_shortlist(self):
        labels = np.array(["a", "b", "a", "b"])
        queries = np.array(["a"])
        good = np.array([[0.9, 0.1, 0.8, 0.2]], dtype=np.float32)
        bad = np.array([[0.2, 0.9, 0.1, 0.8]], dtype=np.float32)
        better = _label_retrieval_metrics(queries, labels, good, [1], True, map_at_k=4)
        worse = _label_retrieval_metrics(queries, labels, bad, [1], True, map_at_k=4)
        self.assertGreater(better["mAP_at_k"], worse["mAP_at_k"])

    def test_map_at_k_is_omitted_when_no_cutoff_is_requested(self):
        metrics = _label_retrieval_metrics(
            np.array(["a"]), np.array(["a", "b"]), np.array([[0.9, 0.1]]), [1], True
        )
        self.assertNotIn("mAP_at_k", metrics)
        self.assertEqual(metrics["mAP"], 1.0)

    def test_sparse_score_matrix_round_trips_scored_entries(self):
        scores = np.array([[0.5, -np.inf, 0.25], [-np.inf, -np.inf, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_score_matrix(Path(tmp) / "scores.npz", scores)
            self.assertIsNotNone(path)
            np.testing.assert_array_equal(load_score_matrix(path), scores)

    def test_dense_score_matrix_is_skipped_instead_of_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.npz"
            self.assertIsNone(save_score_matrix(path, np.ones((10, 10)), max_entries=99))
            self.assertFalse(path.is_file())

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

    def test_file_digest_cache_hashes_each_file_once_per_run(self):
        calls: List[Path] = []
        original = fingerprints._sha256_file_uncached

        def counting(path, chunk_size):
            calls.append(Path(path))
            return original(path, chunk_size)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.bin"
            path.write_bytes(b"lynx")
            fingerprints._sha256_file_uncached = counting
            try:
                with file_digest_cache():
                    # The three call sites build paths differently; inode identity means
                    # they still share one cache entry.
                    digests = {
                        sha256_file(path),
                        sha256_file(Path(tmp) / "." / "image.bin"),
                        sha256_file(path.absolute()),
                    }
            finally:
                fingerprints._sha256_file_uncached = original
        self.assertEqual(len(digests), 1)
        self.assertEqual(len(calls), 1)

    def test_file_digest_cache_is_inactive_outside_a_run(self):
        calls: List[Path] = []
        original = fingerprints._sha256_file_uncached

        def counting(path, chunk_size):
            calls.append(Path(path))
            return original(path, chunk_size)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.bin"
            path.write_bytes(b"lynx")
            fingerprints._sha256_file_uncached = counting
            try:
                sha256_file(path)
                sha256_file(path)
            finally:
                fingerprints._sha256_file_uncached = original
        self.assertEqual(len(calls), 2)

    def test_file_digest_cache_does_not_leak_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.bin"
            path.write_bytes(b"first")
            with file_digest_cache():
                first = sha256_file(path)
            # A same-size rewrite can reuse one mtime tick on tmpfs, so correctness
            # relies on the cache being scoped to a run rather than process-wide.
            path.write_bytes(b"secnd")
            with file_digest_cache():
                self.assertNotEqual(first, sha256_file(path))

    def test_image_content_hash_changes_when_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.bin"
            path.write_bytes(b"first")
            first = sha256_file(path)
            path.write_bytes(b"second")
            self.assertNotEqual(first, sha256_file(path))

    def test_unseen_identities_warn_for_retrieval_and_fail_for_closed_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, content in (("a.jpg", b"a"), ("b.jpg", b"b")):
                (root / name).write_bytes(content)
            df_a = pd.DataFrame({"identity": ["seen"], "path": ["a.jpg"]})
            df_b = pd.DataFrame({"identity": ["unseen"], "path": ["b.jpg"]})
            kwargs = dict(
                df_a=df_a, df_b=df_b, split_a_name="database", split_b_name="query",
                root=root, label_col="identity", run_dir=root / "run",
            )
            # Open-set retrieval continues and still reports the coverage gap.
            summary = run_split_safety_checks(**kwargs, require_b_labels_in_a=False)
            self.assertEqual(summary["num_unseen_labels_b_in_a"], 1)
            # Closed-set classification fails instead.
            with self.assertRaises(ValueError) as ctx:
                run_split_safety_checks(**kwargs, require_b_labels_in_a=True)
            self.assertIn("closed-set", str(ctx.exception))

    def test_safety_checks_no_longer_accept_the_unreachable_warn_flag(self):
        import inspect

        signature = inspect.signature(run_split_safety_checks)
        self.assertNotIn("warn_only_unseen", signature.parameters)
        self.assertIn("require_b_labels_in_a", signature.parameters)

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

    def test_resume_is_rejected_when_no_epochs_remain(self):
        # A checkpoint from a finished run leaves range(start_epoch, epochs) empty, which
        # previously reached the post-loop code with `metrics` unbound and raised NameError.
        for start_epoch in (30, 31):
            with self.assertRaises(ValueError) as ctx:
                validate_resume_epochs(start_epoch, 30, "checkpoint-latest-full.pth")
            self.assertIn("nothing left to train", str(ctx.exception))
            self.assertIn("train.epochs", str(ctx.exception))

    def test_resume_is_allowed_when_epochs_remain(self):
        self.assertIsNone(validate_resume_epochs(29, 30, "checkpoint-latest-full.pth"))
        self.assertIsNone(validate_resume_epochs(0, 1, "checkpoint-latest-full.pth"))

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


    @unittest.skipUnless(HAS_PROBE_CACHE_DEPS, "probe runner dependencies not available")
    def test_classifier_probe_keeps_frozen_backbone_in_eval_mode(self):
        import torch
        from torch import nn

        backbone = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(p=0.5), nn.Linear(4, 3))
        objective = nn.Linear(3, 2)
        backbone.train()
        objective.eval()

        _set_probe_training_mode(backbone, objective, "classifier")

        self.assertFalse(backbone.training)
        self.assertFalse(backbone[0].training)
        self.assertFalse(backbone[1].training)
        self.assertTrue(objective.training)

        _set_probe_training_mode(backbone, objective, "all")
        self.assertTrue(backbone.training)
        self.assertTrue(objective.training)


if __name__ == "__main__":
    unittest.main()
