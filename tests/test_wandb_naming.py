import unittest

from reid.reporting.wandb_naming import finetune_wandb_name, probe_wandb_name


class WandbNamingTests(unittest.TestCase):
    def test_probe_name_contains_experiment_identity(self):
        cfg = {
            "dataset": {"name": "WildlifeReID-10k", "animal": "NyalaData", "split_col": "split", "no_background": True},
            "model": {"type": "megadescriptor-l", "mode": "pretrained"},
            "benchmark": {
                "method": "vismatch",
                "candidate_k": 100,
                "methods": {"vismatch": {"matcher": "loma", "checkpoint_source": "custom"}},
            },
        }
        name = probe_wandb_name(cfg, "20260916T120000Z_abcdef12")
        self.assertIn("probe-wildlifereid-10k-nyaladata-split-megadescriptor-l-pretrained-vismatch-loma-finetuned-k100-masked", name)
        self.assertTrue(name.endswith("-abcdef12"))

    def test_linear_weighting_and_finetune_name(self):
        probe_cfg = {
            "dataset": {"name": "CzechLynx_v2", "animal": "CzechLynx", "split_col": "split-time_closed"},
            "model": {"type": "megadescriptor-l", "mode": "pretrained"},
            "benchmark": {
                "method": "linear_probe",
                "methods": {"linear_probe": {"train_mode": "all", "class_weighting": "none"}},
            },
        }
        self.assertIn("linear-probe-all-unweighted-background", probe_wandb_name(probe_cfg, "run_1"))
        finetune_cfg = {
            "dataset": {"name": "CzechLynx_v2", "animal": "CzechLynx", "split_col": "split-time_closed"},
            "model": {"type": "megadescriptor-l"},
            "train": {"epochs": 30, "lr": 0.0001},
        }
        name = finetune_wandb_name(finetune_cfg, "20260916T120000Z_12345678")
        self.assertIn("finetune-czechlynx-v2-czechlynx-split-time-closed-megadescriptor-l-30ep-lr0.0001", name)


if __name__ == "__main__":
    unittest.main()
