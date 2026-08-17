import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from reid.methods.vismatch_preprocessing import (
    preprocess_vismatch_image,
    resize_long_side_divisible,
    resize_long_side_div32,
    to_rgb_float_tensor,
)
from reid.methods.vismatch_profiles import (
    LOMA_PREPROCESSING_VERSION,
    VISMATCH_PREPROCESSING_VERSION,
    FrameFeatures,
    MatcherProfile,
    build_matcher_profile,
    profile_fingerprint,
)

try:
    from reid.methods.vismatch import VismatchMatcherBackend, _loma_points_to_processed_pixel
    HAS_VISMATCH_BACKEND = True
except ModuleNotFoundError:
    HAS_VISMATCH_BACKEND = False


class VismatchPreprocessingTests(unittest.TestCase):
    def test_uint8_image_conversion_matches_to_tensor_value_semantics(self):
        image = np.asarray(
            [
                [[0, 10, 20], [30, 40, 50]],
                [[60, 70, 80], [90, 100, 110]],
            ],
            dtype=np.uint8,
        )
        tensor = to_rgb_float_tensor(Image.fromarray(image))

        self.assertEqual(tuple(tensor.shape), (3, 2, 2))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.allclose(tensor[:, 0, 0], torch.tensor([0.0, 10.0, 20.0]) / 255.0))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_float_tensor_is_not_quantized_through_uint8(self):
        image = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
                [[0.9, 1.0], [0.0, 0.1]],
            ],
            dtype=torch.float32,
        )

        tensor = to_rgb_float_tensor(image)

        self.assertTrue(torch.allclose(tensor, image))

    def test_resize_matches_lynx_finetuning_formula_and_preserves_aspect_ratio(self):
        image = torch.linspace(0.0, 1.0, 3 * 100 * 50, dtype=torch.float32).reshape(3, 100, 50)
        expected_h = int(100 * (512 / 100)) // 32 * 32
        expected_w = int(50 * (512 / 100)) // 32 * 32
        expected = F.interpolate(
            image.unsqueeze(0),
            size=(expected_h, expected_w),
            mode="bilinear",
            align_corners=False,
        )[0]

        actual, source_size, processed_size = resize_long_side_div32(image, 512)

        self.assertEqual(source_size, (100, 50))
        self.assertEqual(processed_size, (512, 256))
        self.assertEqual(tuple(actual.shape), (3, expected_h, expected_w))
        self.assertTrue(torch.equal(actual, expected))

    def test_smaller_images_are_upscaled(self):
        image = torch.zeros((3, 40, 20), dtype=torch.float32)

        actual, source_size, processed_size = resize_long_side_div32(image, 512)

        self.assertEqual(source_size, (40, 20))
        self.assertEqual(processed_size, (512, 256))
        self.assertEqual(tuple(actual.shape), (3, 512, 256))

    def test_loma_resize_uses_exact_div14_floor_rounding(self):
        image = torch.zeros((3, 100, 50), dtype=torch.float32)

        actual, source_size, processed_size = resize_long_side_divisible(image, 512, divisible_by=14)

        self.assertEqual(source_size, (100, 50))
        self.assertEqual(processed_size, (504, 252))
        self.assertEqual(tuple(actual.shape), (3, 504, 252))
        self.assertEqual(processed_size[0] % 14, 0)
        self.assertEqual(processed_size[1] % 14, 0)

    def test_standard_vismatch_resize_remains_div32(self):
        image = torch.zeros((3, 100, 50), dtype=torch.float32)

        _actual, _source_size, processed_size = resize_long_side_divisible(image, 512, divisible_by=32)

        self.assertEqual(processed_size, (512, 256))
        self.assertEqual(processed_size[0] % 32, 0)
        self.assertEqual(processed_size[1] % 32, 0)

    def test_preprocess_returns_processed_and_source_sizes(self):
        image = np.zeros((80, 40, 3), dtype=np.uint8)

        processed, source_size, processed_size = preprocess_vismatch_image(image, 512)

        self.assertEqual(source_size, (80, 40))
        self.assertEqual(processed_size, (512, 256))
        self.assertEqual(tuple(processed.shape), (3, 512, 256))

    def test_preprocessing_identity_changes_profile_fingerprint(self):
        current = build_matcher_profile("rdd-lightglue", 512, 0.01)
        legacy = MatcherProfile(
            matcher=current.matcher,
            framework=current.framework,
            framework_version=current.framework_version,
            preprocessing=current.preprocessing,
            keypoint_budget=current.keypoint_budget,
            threshold=current.threshold,
            score_mode=current.score_mode,
            feature_matching_mode=current.feature_matching_mode,
            feature_schema_version=2,
            preprocessing_version="legacy_pil_resize_v0",
        )

        self.assertEqual(current.preprocessing_version, VISMATCH_PREPROCESSING_VERSION)
        self.assertNotEqual(profile_fingerprint(current), profile_fingerprint(legacy))

    def test_loma_profile_records_its_div14_preprocessing_identity(self):
        loma = build_matcher_profile("loma", 512, 0.1)

        self.assertIn("multiple of 14", loma.preprocessing)
        self.assertEqual(loma.preprocessing_version, LOMA_PREPROCESSING_VERSION)
        self.assertNotEqual(profile_fingerprint(loma), profile_fingerprint(build_matcher_profile("rdd-lightglue", 512, 0.01)))

    def test_shipped_vismatch_resolution_is_512(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "conf/probe.yaml").read_text(encoding="utf-8")

        self.assertIn("resize_max: 512", text)

    @unittest.skipUnless(HAS_VISMATCH_BACKEND, "Vismatch backend dependencies are not available")
    def test_frame_metadata_uses_processed_coordinates_and_source_size(self):
        class FakeExtractor:
            @staticmethod
            def extract(images):
                batch_size = int(images.shape[0])
                # Native RDD.extract returns one dict per batch item, each with its own
                # threshold-filtered keypoint count, not a dict of stacked tensors.
                return [
                    {
                        "keypoints": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32),
                        "descriptors": torch.ones((2, 4), dtype=torch.float32),
                        "scores": torch.ones(2, dtype=torch.float32),
                    }
                    for _ in range(batch_size)
                ]

        backend = object.__new__(VismatchMatcherBackend)
        backend.matcher_name = "rdd-lightglue"
        backend.resize_max = 512
        backend.preprocessing_divisor = 32
        backend.device = torch.device("cpu")
        backend.extractor = FakeExtractor()

        feature = backend.extract_frame(torch.zeros((3, 40, 20), dtype=torch.float32))

        np.testing.assert_array_equal(feature.image_size, np.asarray([512, 256], dtype=np.int32))
        np.testing.assert_array_equal(feature.original_image_size, np.asarray([40, 20], dtype=np.int32))
        np.testing.assert_array_equal(feature.keypoints, np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))

    @unittest.skipUnless(HAS_VISMATCH_BACKEND, "Vismatch backend dependencies are not available")
    def test_serial_and_batched_preprocessing_have_same_frame_metadata(self):
        class FakeExtractor:
            @staticmethod
            def extract(images):
                # One dict per batch item, matching native RDD.extract. Values are derived
                # from image content so a batched item/feature misalignment is visible
                # instead of being hidden behind identical placeholder tensors.
                return [
                    {
                        "keypoints": torch.full((1, 2), float(images[b].mean()), dtype=torch.float32),
                        "descriptors": torch.full((1, 4), float(images[b].mean()), dtype=torch.float32),
                        "scores": torch.ones(1, dtype=torch.float32),
                    }
                    for b in range(int(images.shape[0]))
                ]

        backend = object.__new__(VismatchMatcherBackend)
        backend.matcher_name = "rdd-lightglue"
        backend.resize_max = 512
        backend.preprocessing_divisor = 32
        backend.device = torch.device("cpu")
        backend.extractor = FakeExtractor()
        images = [torch.zeros((3, 40, 20), dtype=torch.float32), torch.ones((3, 80, 40), dtype=torch.float32)]

        serial = [backend.extract_frame(image) for image in images]
        batched = backend.extract_frames_batch(images)

        for serial_feature, batched_feature in zip(serial, batched):
            np.testing.assert_array_equal(serial_feature.image_size, batched_feature.image_size)
            np.testing.assert_array_equal(serial_feature.original_image_size, batched_feature.original_image_size)
            np.testing.assert_allclose(serial_feature.keypoints, batched_feature.keypoints)
            np.testing.assert_allclose(serial_feature.descriptors, batched_feature.descriptors)

    @unittest.skipUnless(HAS_VISMATCH_BACKEND, "Vismatch backend dependencies are not available")
    def test_each_image_is_preprocessed_exactly_once(self):
        # Bucketing used to call prepared_shape(), which ran the whole resize just to read
        # the output shape, and extraction then repeated it. Preparation must happen once.
        from reid.methods.vismatch import VismatchMatcherBackend

        class FakeExtractor:
            @staticmethod
            def extract(images):
                return [
                    {
                        "keypoints": torch.zeros((1, 2), dtype=torch.float32),
                        "descriptors": torch.ones((1, 4), dtype=torch.float32),
                        "scores": torch.ones(1, dtype=torch.float32),
                    }
                    for _ in range(int(images.shape[0]))
                ]

        backend = object.__new__(VismatchMatcherBackend)
        backend.matcher_name = "rdd-lightglue"
        backend.resize_max = 512
        backend.preprocessing_divisor = 32
        backend.device = torch.device("cpu")
        backend.extractor = FakeExtractor()

        calls = []
        original = VismatchMatcherBackend._prepare_input

        def counting(self, image):
            calls.append(tuple(image.shape[-2:]))
            return original(self, image)

        VismatchMatcherBackend._prepare_input = counting
        try:
            prepared = backend.prepare_image(torch.zeros((3, 40, 20), dtype=torch.float32))
            backend.extract_prepared(prepared)
            backend.extract_prepared_batch([prepared, prepared])
        finally:
            VismatchMatcherBackend._prepare_input = original

        self.assertEqual(len(calls), 1)
        # The prepared tensor carries its own shape, so no second resize is needed.
        self.assertEqual(prepared.processed_size, (512, 256))
        self.assertEqual(tuple(prepared.tensor.shape), (1, 3, 512, 256))

    @unittest.skipUnless(HAS_VISMATCH_BACKEND, "Vismatch backend dependencies are not available")
    def test_prepared_extraction_matches_raw_image_extraction(self):
        from reid.methods.vismatch import VismatchMatcherBackend

        class FakeExtractor:
            @staticmethod
            def extract(images):
                return [
                    {
                        "keypoints": torch.full((2, 2), float(images[b].mean()), dtype=torch.float32),
                        "descriptors": torch.full((2, 4), float(images[b].std()), dtype=torch.float32),
                        "scores": torch.ones(2, dtype=torch.float32),
                    }
                    for b in range(int(images.shape[0]))
                ]

        backend = object.__new__(VismatchMatcherBackend)
        backend.matcher_name = "rdd-lightglue"
        backend.resize_max = 512
        backend.preprocessing_divisor = 32
        backend.device = torch.device("cpu")
        backend.extractor = FakeExtractor()

        image = torch.rand((3, 300, 200), dtype=torch.float32)
        from_raw = backend.extract_frame(image)
        from_prepared = backend.extract_prepared(backend.prepare_image(image))
        np.testing.assert_array_equal(from_raw.keypoints, from_prepared.keypoints)
        np.testing.assert_array_equal(from_raw.descriptors, from_prepared.descriptors)
        np.testing.assert_array_equal(from_raw.image_size, from_prepared.image_size)
        np.testing.assert_array_equal(
            from_raw.original_image_size, from_prepared.original_image_size
        )

    @unittest.skipUnless(HAS_VISMATCH_BACKEND, "Vismatch backend dependencies are not available")
    def test_loma_backend_prepares_div14_spatial_shapes(self):
        backend = object.__new__(VismatchMatcherBackend)
        backend.matcher_name = "loma"
        backend.resize_max = 512
        backend.preprocessing_divisor = 14
        backend.device = torch.device("cpu")

        prepared, source_size, processed_size = backend._prepare_input(torch.zeros((3, 100, 50)))

        self.assertEqual(tuple(prepared.shape), (1, 3, 504, 252))
        self.assertEqual(source_size, (100, 50))
        self.assertEqual(processed_size, (504, 252))

    @unittest.skipUnless(HAS_VISMATCH_BACKEND, "Vismatch backend dependencies are not available")
    def test_loma_visualization_points_stay_in_processed_coordinates(self):
        feature = FrameFeatures(
            keypoints=np.zeros((3, 2), dtype=np.float32),
            descriptors=np.zeros((3, 4), dtype=np.float32),
            scores=np.ones(3, dtype=np.float32),
            image_size=np.asarray([504, 252], dtype=np.int32),
            coordinate_convention="normalized[-1,1]",
            original_image_size=np.asarray([939, 881], dtype=np.int32),
        )
        normalized = np.asarray([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)

        actual = _loma_points_to_processed_pixel(normalized, feature)

        np.testing.assert_allclose(
            actual,
            np.asarray([[-0.5, -0.5], [251.5, 503.5], [125.5, 251.5]], dtype=np.float32),
        )
        self.assertLessEqual(float(actual[:, 0].max()), 251.5)
        self.assertLessEqual(float(actual[:, 1].max()), 503.5)


if __name__ == "__main__":
    unittest.main()
