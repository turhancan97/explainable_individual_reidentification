import os
import unittest
from pathlib import Path


CHECKPOINT_DIR = Path(
    os.environ.get(
        "VISMATCH_CUSTOM_CHECKPOINT",
        "/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/matches-lg-wandb/epoch_15",
    )
)
RUN_SMOKE = os.environ.get("RUN_VISMATCH_CHECKPOINT_SMOKE") == "1"


@unittest.skipUnless(
    RUN_SMOKE and CHECKPOINT_DIR.exists(),
    "set RUN_VISMATCH_CHECKPOINT_SMOKE=1 with an available custom checkpoint",
)
class VismatchCheckpointEnvironmentTests(unittest.TestCase):
    def test_custom_rdd_lightglue_checkpoint_constructs(self):
        import torch

        from reid.methods.vismatch import VismatchMatcherBackend

        backend = VismatchMatcherBackend(
            "rdd-lightglue",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            32,
            0.01,
            checkpoint_source="custom",
            checkpoint_path=CHECKPOINT_DIR,
            checkpoint_components="auto",
        )
        components = {item["component"] for item in backend.checkpoint_resolution.as_dict()["components"]}
        self.assertIn("lightglue", components)


if __name__ == "__main__":
    unittest.main()
