import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "conf"


def compose_config(name, overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name=name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


class HydraConfigurationTests(unittest.TestCase):
    def test_probe_and_finetune_defaults_compose(self):
        probe = compose_config("probe")
        finetune = compose_config("finetune")

        self.assertEqual(probe.model.type, "megadescriptor-l")
        self.assertEqual(probe.benchmark.method, "vismatch")
        self.assertEqual(probe.benchmark.methods.vismatch.matcher, "rdd-lightglue")
        self.assertEqual(probe.benchmark.methods.wildfusion.B, 100)
        self.assertEqual(probe.benchmark.methods.vismatch.candidate_k, 100)
        # The shipped evaluation cutoff must fit inside the shortlist, otherwise
        # metrics would rank unscored database entries by original index.
        self.assertEqual(probe.benchmark.map_at_k, 100)
        self.assertLessEqual(probe.benchmark.map_at_k, probe.benchmark.methods.vismatch.candidate_k)
        self.assertLessEqual(max(probe.benchmark.top_k), probe.benchmark.methods.vismatch.candidate_k)
        self.assertEqual(probe.benchmark.methods.vismatch.checkpoint_source, "default")
        self.assertEqual(probe.benchmark.methods.vismatch.checkpoint_components, "auto")
        self.assertEqual(probe.dataset.image_variant, "no_background")
        self.assertEqual(finetune.model.type, "megadescriptor-l")
        self.assertEqual(finetune.dataset.image_variant, "background")
        self.assertEqual(finetune.train.epochs, 30)

    def test_nested_dotlist_overrides_convert_types(self):
        cfg = compose_config(
            "probe",
            [
                "benchmark.method=vismatch",
                "benchmark.methods.vismatch.matcher=loma",
                "benchmark.methods.vismatch.top_k=1024",
                "benchmark.methods.vismatch.oom_backoff=false",
                "benchmark.top_k=[1,10]",
                "dataset.metadata_file=custom.csv",
            ],
        )

        self.assertEqual(cfg.benchmark.method, "vismatch")
        self.assertEqual(cfg.benchmark.methods.vismatch.matcher, "loma")
        self.assertEqual(cfg.benchmark.methods.vismatch.top_k, 1024)
        self.assertFalse(cfg.benchmark.methods.vismatch.oom_backoff)
        self.assertEqual(list(cfg.benchmark.top_k), [1, 10])
        self.assertEqual(cfg.dataset.metadata_file, "custom.csv")

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(ConfigCompositionException):
            compose_config("probe", ["benchmark.methd=vismatch"])

    def test_output_paths_and_interpolations_remain_project_managed(self):
        probe = compose_config("probe")
        finetune = compose_config("finetune")

        self.assertEqual(probe.output.experiment_root, "experiments")
        self.assertEqual(probe.output.run_dir, "benchmark_runs")
        self.assertEqual(probe.output.csv_path, "benchmark_runs/benchmark_results.csv")
        self.assertEqual(finetune.output.experiment_root, "experiments")
        self.assertEqual(finetune.output.run_dir, "results/CzechLynx_v2/CzechLynx/mask_False")
        self.assertTrue(finetune.reporting.enabled)
        self.assertNotIn("${", str(probe.dataset.root))
        self.assertNotIn("${", str(finetune.output.csv_path))

    def test_resolved_snapshot_contains_overrides(self):
        cfg = compose_config(
            "probe",
            ["benchmark.method=vismatch", "benchmark.methods.vismatch.matcher=loma"],
        )
        with TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "config.snapshot.yaml"
            OmegaConf.save(cfg, snapshot, resolve=True)
            saved = OmegaConf.load(snapshot)
            snapshot_text = snapshot.read_text(encoding="utf-8")

        self.assertEqual(saved.benchmark.method, "vismatch")
        self.assertEqual(saved.benchmark.methods.vismatch.matcher, "loma")
        self.assertNotIn("${", snapshot_text)

    def test_jaguar_keeps_its_argparse_config_workflow(self):
        jaguar = OmegaConf.load(ROOT / "config/kaggle_jaguar.yaml")
        submitter = (ROOT / "scripts/kaggle_jaguar_submit.py").read_text(encoding="utf-8")

        self.assertEqual(jaguar.finetune.base_config, "conf/finetune.yaml")
        self.assertIn("--config", submitter)
        self.assertIn("argparse", submitter)


if __name__ == "__main__":
    unittest.main()
