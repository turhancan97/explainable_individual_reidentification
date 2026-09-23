import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from reid.config_defaults import SUPPORTED_MODEL_TYPES
from reid.engine.probe_runner import _set_trainable_params

ROOT = Path(__file__).resolve().parents[1]
DINO_TYPES = [name for name in SUPPORTED_MODEL_TYPES if name.startswith("dino")]


def _dino_names(model_type: str):
    """Parameter names as exposed through ViTCLSAdapter (``backbone.`` prefix).

    Mirrors the Hugging Face naming verified on the real checkpoints: DINOv2 uses
    ``encoder.layer.N`` and a final ``layernorm``; DINOv3 uses ``layer.N`` and ``norm``.
    """
    depth = 24 if model_type.endswith("-l") else 12
    v2 = model_type.startswith("dinov2")
    block = "backbone.encoder.layer" if v2 else "backbone.layer"
    names = ["backbone.embeddings.cls_token", "backbone.embeddings.register_tokens"]
    for index in range(depth):
        for leaf in ("norm1.weight", "norm1.bias", "mlp.fc1.weight", "norm2.weight", "layer_scale2.lambda1"):
            names.append(f"{block}.{index}.{leaf}")
    final = "backbone.layernorm" if v2 else "backbone.norm"
    names += [f"{final}.weight", f"{final}.bias"]
    return names, depth, block, final


class _FakeModel:
    def __init__(self, names):
        self.params = {name: torch.nn.Parameter(torch.zeros(1)) for name in names}

    def named_parameters(self):
        return iter(self.params.items())


def _cfg(model_type: str, method_key: str):
    probe = OmegaConf.load(ROOT / "conf" / "probe.yaml")
    probe.model.type = model_type
    probe.benchmark.methods[method_key].train_mode = "partial"
    return probe


class PartialRuleTests(unittest.TestCase):
    def test_every_dino_type_has_explicit_rules(self):
        probe = OmegaConf.load(ROOT / "conf" / "probe.yaml")
        self.assertEqual(sorted(DINO_TYPES), ["dinov2", "dinov2-l", "dinov3", "dinov3-l"])
        for method_key in ("linear_probe", "efficient_probe"):
            rules = probe.benchmark.methods[method_key].partial_rules
            for model_type in DINO_TYPES:
                # The Swin default's bare "norm" would unfreeze every DINO LayerNorm.
                self.assertIn(model_type, rules, f"{method_key} lacks {model_type}")

    def test_dino_rules_unfreeze_only_last_block_and_final_norm(self):
        for method_key in ("linear_probe", "efficient_probe"):
            for model_type in DINO_TYPES:
                names, depth, block, final = _dino_names(model_type)
                model = _FakeModel(names)
                _set_trainable_params(model, _cfg(model_type, method_key), method_key)
                trainable = {name for name, p in model.named_parameters() if p.requires_grad}
                expected = {n for n in names if n.startswith(f"{block}.{depth - 1}.") or n.startswith(f"{final}.")}
                self.assertEqual(trainable, expected, f"{method_key}/{model_type}")

    def test_partial_rules_matching_nothing_fail_closed(self):
        cfg = _cfg("dinov3", "efficient_probe")
        cfg.benchmark.methods.efficient_probe.partial_rules.dinov3 = ["encoder.layer.11", "layernorm"]
        names, *_ = _dino_names("dinov3")
        with self.assertRaisesRegex(ValueError, "match no backbone parameters"):
            _set_trainable_params(_FakeModel(names), cfg, "efficient_probe")


if __name__ == "__main__":
    unittest.main()
