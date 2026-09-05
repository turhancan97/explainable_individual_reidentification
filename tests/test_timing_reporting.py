import unittest

from reid.reporting.timing import primary_runtime_key, set_primary_compute_runtime


class TimingReportingTests(unittest.TestCase):
    def test_matcher_methods_use_matcher_runtime(self):
        timings = {
            "matcher_runtime_sec": 3.5,
            "method_compute_runtime_sec": 20.0,
            "vismatch_stage_a_sec": 8.0,
            "feature_extraction_compute_sec": 12.0,
        }
        set_primary_compute_runtime("vismatch", timings)
        self.assertEqual(primary_runtime_key("wildfusion"), "matcher_runtime_sec")
        self.assertEqual(timings["primary_compute_runtime_sec"], 3.5)

    def test_primary_runtime_does_not_change_total_runtime(self):
        timings = {"matcher_runtime_sec": 3.5, "total_run_sec": 99.0}
        set_primary_compute_runtime("local_lightglue", timings)
        self.assertEqual(timings["total_run_sec"], 99.0)

    def test_non_matcher_methods_use_method_runtime(self):
        timings = {"method_compute_runtime_sec": 7.25}
        set_primary_compute_runtime("linear_probe", timings)
        self.assertEqual(primary_runtime_key("efficient_probe"), "method_compute_runtime_sec")
        self.assertEqual(timings["primary_compute_runtime_sec"], 7.25)

    def test_missing_primary_source_does_not_invent_runtime(self):
        timings = {"total_run_sec": 100.0}
        self.assertIsNone(set_primary_compute_runtime("cosine", timings))
        self.assertNotIn("primary_compute_runtime_sec", timings)


if __name__ == "__main__":
    unittest.main()
