"""Optional Vismatch smoke tests; enable with RUN_VISMATCH_SMOKE=1."""

import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_VISMATCH_SMOKE") == "1", "Vismatch smoke tests disabled")
class VismatchEnvironmentSmokeTests(unittest.TestCase):
    def test_import_and_supported_factory_names(self):
        import torch
        import vismatch

        self.assertEqual(getattr(vismatch, "__version__", None), "1.3.1")
        for name in ("rdd-lightglue", "aliked-lightglue", "superpoint-lightglue", "loma"):
            self.assertIn(name, vismatch.available_models)
        self.assertEqual(torch.__version__.split("+")[0], "2.8.0")

    def test_loma_factory_constructs(self):
        from vismatch import get_matcher

        matcher = get_matcher("loma", device="cuda", max_num_keypoints=32)
        self.assertTrue(hasattr(matcher, "matcher"))
        self.assertTrue(hasattr(matcher, "preprocess"))
        self.assertAlmostEqual(float(matcher.matcher.cfg.filter_threshold), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
