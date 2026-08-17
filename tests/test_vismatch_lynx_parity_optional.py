import os
import sys
import unittest
from pathlib import Path

import torch

from reid.methods.vismatch_preprocessing import resize_long_side_div32


LYNX_ROOT = Path(
    os.environ.get(
        "LYNX_FINETUNING_ROOT",
        "/home/kargin/Projects/repositories/lynx-finetuning",
    )
)
RUN_PARITY = os.environ.get("RUN_VISMATCH_LYNX_PARITY") == "1"


@unittest.skipUnless(
    RUN_PARITY and (LYNX_ROOT / "contrastive_finetuning" / "train_common.py").is_file(),
    "set RUN_VISMATCH_LYNX_PARITY=1 with an available lynx-finetuning checkout",
)
class LynxPreprocessingParityTests(unittest.TestCase):
    def test_resize_matches_external_finetuning_helper(self):
        sys.path.insert(0, str(LYNX_ROOT))
        try:
            try:
                from contrastive_finetuning.train_common import resize_long_side
            except ModuleNotFoundError as exc:
                self.skipTest(f"lynx-finetuning dependencies are unavailable: {exc}")

            image = torch.linspace(0.0, 1.0, 3 * 100 * 50, dtype=torch.float32).reshape(1, 3, 100, 50)
            expected = resize_long_side(image, 512)[0]
            actual, _source_size, _processed_size = resize_long_side_div32(image[0], 512)

            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
