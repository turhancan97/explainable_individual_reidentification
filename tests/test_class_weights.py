import unittest

import numpy as np

from reid.training.class_weights import compute_identity_class_weights

try:
    import torch
    from models.objective import SoftmaxLoss
    HAS_TORCH_OBJECTIVE = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH_OBJECTIVE = False


class IdentityClassWeightTests(unittest.TestCase):
    def test_inverse_frequency_is_mean_normalized_in_deterministic_order(self):
        weights, metadata = compute_identity_class_weights(
            ["zebra", "zebra", "antelope", "lynx", "lynx", "lynx"],
            ["antelope", "lynx", "zebra"],
        )
        expected = np.asarray([1.0, 1.0 / 3.0, 0.5], dtype=np.float32)
        expected *= 3.0 / expected.sum()
        np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertEqual(metadata["identity_order"], ["antelope", "lynx", "zebra"])
        self.assertEqual(metadata["counts"], [1, 3, 2])

    def test_cap_is_applied_after_normalization_without_renormalizing(self):
        common_labels = [f"common_{index}" for index in range(10)]
        labels = ["singleton"] + [identity for identity in common_labels for _ in range(1000)]
        weights, metadata = compute_identity_class_weights(
            labels,
            [*common_labels, "singleton"],
            max_weight=5.0,
        )
        self.assertEqual(float(weights[-1]), 5.0)
        self.assertLess(float(weights.mean()), 1.0)
        self.assertEqual(metadata["max"], 5.0)

    def test_none_returns_no_weights_and_is_still_provenanced(self):
        weights, metadata = compute_identity_class_weights(
            ["a", "a", "b"], ["a", "b"], weighting="none"
        )
        self.assertIsNone(weights)
        self.assertEqual(metadata["mode"], "none")
        self.assertEqual(metadata["formula"], "none")

    def test_training_labels_only_and_invalid_configuration(self):
        first, _ = compute_identity_class_weights(["a", "a", "b"], ["a", "b"])
        second, _ = compute_identity_class_weights(["a", "a", "b", "b", "b"], ["a", "b"])
        self.assertFalse(np.array_equal(first, second))
        with self.assertRaisesRegex(ValueError, "class_weighting"):
            compute_identity_class_weights(["a"], ["a"], weighting="bad")
        with self.assertRaisesRegex(ValueError, "positive finite"):
            compute_identity_class_weights(["a"], ["a"], max_weight=0)
        with self.assertRaisesRegex(ValueError, "absent from training"):
            compute_identity_class_weights(["a"], ["a", "b"])

    @unittest.skipUnless(HAS_TORCH_OBJECTIVE, "torch objective dependencies are not available")
    def test_softmax_loss_uses_weights_but_exposes_unweighted_evaluation(self):
        weights = torch.tensor([1.0, 2.0])
        objective = SoftmaxLoss(num_classes=2, embedding_size=2, class_weights=weights)
        self.assertTrue(torch.equal(objective.criterion.weight, weights))
        self.assertTrue(torch.isfinite(objective.unweighted_loss(torch.ones(1, 2), torch.tensor([0]))))


if __name__ == "__main__":
    unittest.main()
