import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import torch
    import reid.methods.vismatch as vismatch_module
    from reid.methods.vismatch_profiles import FrameFeatures
    HAS_VISMATCH_RUNTIME = True
except ModuleNotFoundError:
    HAS_VISMATCH_RUNTIME = False

from reid.methods.vismatch_batching import (
    candidate_pair_count,
    grouped_pair_batches,
    run_with_batch_backoff,
)


def _feature(keypoints: int):
    return SimpleNamespace(keypoints=np.zeros((keypoints, 2), dtype=np.float32))


class VismatchBatchingTests(unittest.TestCase):
    @unittest.skipUnless(HAS_VISMATCH_RUNTIME, "Vismatch runtime dependencies are not available")
    def test_batched_extraction_runtime_smoke_evaluates_annotations(self):
        class FakeDataset:
            def __init__(self):
                self.df = __import__("pandas").DataFrame({"path": ["one.jpg", "two.jpg"]})

            def __len__(self):
                return len(self.df)

        class FakeBackend:
            def __init__(self):
                self.batch_diagnostics = {
                    "extract_batches": 0,
                    "effective_extract_batch_size": None,
                    "configured_extract_batch_size": None,
                }

            @staticmethod
            def prepared_shape(image):
                return tuple(image.shape[-2:])

            @staticmethod
            def _feature():
                return FrameFeatures(
                    keypoints=np.zeros((2, 2), dtype=np.float32),
                    descriptors=np.zeros((2, 4), dtype=np.float32),
                    scores=np.ones(2, dtype=np.float32),
                    image_size=np.asarray([16, 16], dtype=np.int32),
                )

            def extract_frames_batch(self, images):
                return [self._feature() for _image in images]

            def extract_frame(self, image):
                return self._feature()

        backend = FakeBackend()
        with TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                vismatch_module,
                "_load_raw_rgb_image",
                return_value=np.zeros((16, 16, 3), dtype=np.uint8),
            ), mock.patch.object(
                vismatch_module,
                "_to_image_tensor",
                return_value=torch.zeros((3, 16, 16), dtype=torch.float32),
            ):
                features = vismatch_module._extract_split_features(
                    FakeDataset(),
                    "query",
                    backend,
                    torch.device("cpu"),
                    top_k=8,
                    resize_max=16,
                    cache_dir=Path(temp_dir),
                    dataset_root=Path(temp_dir),
                    no_background=False,
                    mask_col="mask",
                    path_col="path",
                    cfg_tag="runtime-smoke",
                    batch_mode="batched",
                    extract_batch_size=2,
                    oom_backoff=True,
                )

        self.assertEqual(len(features), 2)
        self.assertEqual(backend.batch_diagnostics["extract_batches"], 1)
        self.assertEqual(backend.batch_diagnostics["effective_extract_batch_size"], 2)

    def test_candidate_pair_count_supports_full_and_shortlisted_matching(self):
        self.assertEqual(candidate_pair_count(3, 5), 15)
        self.assertEqual(candidate_pair_count(3, 5, [[0, 1], [2], []]), 3)
        with self.assertRaises(ValueError):
            candidate_pair_count(2, 5, [[0]])

    def test_shape_bucketing_keeps_loma_shapes_separate_and_flushes_partial_batch(self):
        query = [_feature(3), _feature(4)]
        database = [_feature(5), _feature(5), _feature(6)]
        pairs = [(0, 0), (0, 1), (0, 2), (1, 0)]

        batches = list(grouped_pair_batches(pairs, query, database, batch_size=2))

        self.assertEqual(batches, [[(0, 0), (0, 1)], [(0, 2)], [(1, 0)]])
        for batch in batches:
            shapes = {
                (len(query[q].keypoints), len(database[d].keypoints))
                for q, d in batch
            }
            self.assertEqual(len(shapes), 1)

    def test_candidate_results_are_scattered_to_original_matrix_positions(self):
        query = [_feature(3), _feature(3)]
        database = [_feature(5), _feature(5), _feature(5)]
        pairs = [(0, 2), (0, 0), (1, 1)]
        matrix = np.full((2, 3), -1e9, dtype=np.float32)

        for batch in grouped_pair_batches(pairs, query, database, batch_size=2):
            results = [float(q * 10 + d) for q, d in batch]
            for (q, d), score in zip(batch, results):
                matrix[q, d] = score

        np.testing.assert_array_equal(matrix, np.asarray([[0.0, -1e9, 2.0], [-1e9, 11.0, -1e9]]))

    def test_fake_serial_and_batched_backend_scores_are_identical(self):
        query = [_feature(3), _feature(3)]
        database = [_feature(5), _feature(5), _feature(5)]
        pairs = [(0, 2), (0, 0), (1, 1), (1, 2)]

        def score(pair):
            query_index, database_index = pair
            return float((query_index + 1) * 10 + database_index)

        serial = {pair: score(pair) for pair in pairs}
        batched = {}
        for batch in grouped_pair_batches(pairs, query, database, batch_size=2):
            for pair in batch:
                batched[pair] = score(pair)
        self.assertEqual(serial, batched)

    def test_shipped_config_enables_feature_level_batching(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "conf/probe.yaml").read_text(encoding="utf-8")
        self.assertIn('feature_matching_mode: "feature_level"', text)
        self.assertIn('batch_mode: "batched"', text)
        self.assertIn("match_batch_size: 16", text)
        self.assertIn("extract_batch_size: 8", text)
        self.assertIn("oom_backoff: true", text)

    def test_empty_candidate_groups_do_not_modify_sentinel_scores(self):
        query = [_feature(3)]
        database = [_feature(5)]
        self.assertEqual(list(grouped_pair_batches([], query, database, batch_size=2)), [])

    def test_serial_reference_processes_one_item_at_a_time(self):
        items = list(range(5))
        seen = []

        def process(batch):
            seen.append(list(batch))
            return list(batch)

        results = list(run_with_batch_backoff(items, 1, process, oom_backoff=True))

        self.assertEqual(seen, [[0], [1], [2], [3], [4]])
        self.assertEqual([result for result, _size in results], [[item] for item in items])
        self.assertEqual([size for _result, size in results], [1] * len(items))

    def test_oom_backoff_retries_and_records_effective_size(self):
        items = list(range(7))
        seen = []
        cleared = []

        def process(batch):
            current = list(batch)
            seen.append(current)
            if len(current) > 2:
                raise RuntimeError("CUDA out of memory")
            return current

        results = list(
            run_with_batch_backoff(
                items,
                4,
                process,
                oom_backoff=True,
                clear_memory=lambda: cleared.append(True),
            )
        )

        self.assertEqual(seen[:2], [items[:4], items[:2]])
        self.assertEqual([result for result, _size in results], [[0, 1], [2, 3], [4, 5], [6]])
        self.assertEqual([size for _result, size in results], [2, 2, 2, 2])
        self.assertEqual(len(cleared), 1)

    def test_divisible_batch_does_not_receive_an_extra_empty_step(self):
        seen = []

        def process(batch):
            seen.append(list(batch))
            return len(batch)

        results = list(run_with_batch_backoff(list(range(6)), 3, process, oom_backoff=True))

        self.assertEqual(seen, [[0, 1, 2], [3, 4, 5]])
        self.assertEqual([result for result, _size in results], [3, 3])


if __name__ == "__main__":
    unittest.main()
