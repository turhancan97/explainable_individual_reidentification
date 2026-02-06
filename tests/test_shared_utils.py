import unittest

import numpy as np
import pandas as pd

from reid.evaluation.metrics import compute_metrics

try:
    from reid.data.dataset_view import BenchmarkDatasetView
    from reid.features.containers import normalize_features
    HAS_TORCH_DEPS = True
except ModuleNotFoundError:
    HAS_TORCH_DEPS = False


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
    @unittest.skipUnless(HAS_TORCH_DEPS, "torch dependencies not available")
    def test_normalize_from_row_pairs(self):
        raw = [
            (np.array([1.0, 2.0], dtype=np.float32), {"meta": 1}),
            (np.array([3.0, 4.0], dtype=np.float32), {"meta": 2}),
        ]
        arr = normalize_features(raw)
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype, np.float32)


class DatasetViewTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH_DEPS, "torch dependencies not available")
    def test_view_passthrough_without_mask(self):
        base = DummyDataset()
        view = BenchmarkDatasetView(base_dataset=base, label_col="label", no_background=False)
        image, label = view[0]
        self.assertEqual(image.shape, (4, 4, 3))
        self.assertEqual(label, "a")


if __name__ == "__main__":
    unittest.main()
