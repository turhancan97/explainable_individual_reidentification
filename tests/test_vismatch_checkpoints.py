import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

from reid.methods.vismatch_checkpoints import (
    apply_vismatch_checkpoint,
    resolve_vismatch_checkpoint,
)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for checkpoint fixture tests")
class VismatchCheckpointTests(unittest.TestCase):
    @staticmethod
    def _save(path: Path, state):
        torch.save(state, path)

    def test_default_resolution_has_no_custom_files(self):
        resolution = resolve_vismatch_checkpoint("rdd-lightglue")
        self.assertEqual(resolution.source, "default")
        self.assertEqual(resolution.default_components, ("rdd_extractor", "lightglue"))
        self.assertEqual(resolution.files, ())

    def test_single_lightglue_file_is_detected_without_filename_convention(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arbitrary-name.pth"
            self._save(
                path,
                {
                    "transformers.0.weight": torch.zeros(2, 2),
                    "log_assignment.0.weight": torch.zeros(2, 2),
                    "token_confidence.0.weight": torch.zeros(1, 2),
                },
            )
            resolution = resolve_vismatch_checkpoint("rdd-lightglue", "custom", path)
            self.assertEqual([item.component for item in resolution.files], ["lightglue"])
            self.assertEqual(resolution.default_components, ("rdd_extractor",))

    def test_directory_detects_rdd_and_lightglue_independent_of_file_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._save(root / "model_1.pth", {"detector.weight": torch.zeros(2, 2)})
            self._save(
                root / "model.pth",
                {
                    "transformers.0.weight": torch.zeros(2, 2),
                    "log_assignment.0.weight": torch.zeros(2, 2),
                    "token_confidence.0.weight": torch.zeros(1, 2),
                },
            )
            resolution = resolve_vismatch_checkpoint("rdd-lightglue", "custom", root)
            self.assertEqual({item.component for item in resolution.files}, {"rdd_extractor", "lightglue"})
            self.assertEqual(resolution.default_components, ())

    def test_manifest_mapping_is_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.safetensors.pth"
            self._save(model, {"detector.weight": torch.zeros(2, 2)})
            (root / "checkpoint_manifest.json").write_text(
                json.dumps({"components": {"rdd_extractor": model.name}}), encoding="utf-8"
            )
            resolution = resolve_vismatch_checkpoint("rdd-lightglue", "custom", root, "extractor_only")
            self.assertEqual(resolution.files[0].component, "rdd_extractor")
            self.assertIsNotNone(resolution.manifest_path)

    def test_missing_explicit_component_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lightglue.pth"
            self._save(path, {"detector.weight": torch.zeros(2, 2)})
            with self.assertRaisesRegex(ValueError, "requires a lightglue"):
                resolve_vismatch_checkpoint("rdd-lightglue", "custom", path, "matcher_only")

    def test_loma_rejects_lightglue_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lightglue.pth"
            self._save(
                path,
                {
                    "transformers.0.weight": torch.zeros(2, 2),
                    "log_assignment.0.weight": torch.zeros(2, 2),
                    "token_confidence.0.weight": torch.zeros(1, 2),
                },
            )
            with self.assertRaisesRegex(ValueError, "LoMa"):
                resolve_vismatch_checkpoint("loma", "custom", path)

    def test_checkpoint_content_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lightglue.pth"
            state = {
                "transformers.0.weight": torch.zeros(2, 2),
                "log_assignment.0.weight": torch.zeros(2, 2),
                "token_confidence.0.weight": torch.zeros(1, 2),
            }
            self._save(path, state)
            first = resolve_vismatch_checkpoint("rdd-lightglue", "custom", path).fingerprint
            state["transformers.0.weight"][0, 0] = 1.0
            self._save(path, state)
            second = resolve_vismatch_checkpoint("rdd-lightglue", "custom", path).fingerprint
            self.assertNotEqual(first, second)

    def test_lightglue_component_loads_strictly(self):
        class FakeLightGlue(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
                self.log_assignment = nn.ModuleList([nn.Linear(2, 2, bias=False)])
                self.token_confidence = nn.ModuleList([nn.Linear(2, 1, bias=False)])

        class FakeMatcher(nn.Module):
            def __init__(self):
                super().__init__()
                self.RDD = nn.Module()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lightglue = FakeLightGlue()
                self.matcher = FakeMatcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pth"
            source = FakeLightGlue()
            self._save(path, source.state_dict())
            resolution = resolve_vismatch_checkpoint("rdd-lightglue", "custom", path, "matcher_only")
            target = FakeModel()
            apply_vismatch_checkpoint(target, resolution)
            for name, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.lightglue.state_dict()[name]))

    def test_rdd_descriptor_checkpoint_applies_only_descriptor(self):
        class RDD(nn.Module):
            def __init__(self):
                super().__init__()
                self.detector = nn.Linear(2, 2, bias=False)
                self.descriptor = nn.Linear(2, 2, bias=False)

        class Matcher(nn.Module):
            def __init__(self):
                super().__init__()
                self.RDD = RDD()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.matcher = Matcher()
                self.lightglue = nn.Identity()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = RDD()
            source.descriptor.weight.data.fill_(7.0)
            source.detector.weight.data.fill_(9.0)
            path = root / "model.pth"
            self._save(path, source.state_dict())
            (root / "czechlynx_protocol.json").write_text(
                json.dumps({"rdd_train_component": "descriptor"}), encoding="utf-8"
            )
            resolution = resolve_vismatch_checkpoint("rdd-lightglue", "custom", path, "auto")
            self.assertEqual(resolution.resolved_component_mode, "descriptor_only")
            self.assertEqual(resolution.ignored_prefixes, ("detector.",))
            target = Model()
            target.matcher.RDD.detector.weight.data.fill_(3.0)
            apply_vismatch_checkpoint(target, resolution)
            self.assertTrue(torch.all(target.matcher.RDD.descriptor.weight == 7.0))
            self.assertTrue(torch.all(target.matcher.RDD.detector.weight == 3.0))

    def test_rdd_descriptor_detector_tensors_are_shape_validated_but_not_applied(self):
        class RDD(nn.Module):
            def __init__(self):
                super().__init__()
                self.detector = nn.Linear(2, 2, bias=False)
                self.descriptor = nn.Linear(2, 2, bias=False)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.matcher = nn.Module()
                self.matcher.RDD = RDD()
                self.lightglue = nn.Identity()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "model.pth"
            self._save(
                path,
                {
                    "descriptor.weight": torch.zeros(2, 2),
                    "detector.weight": torch.zeros(3, 3),
                },
            )
            (root / "czechlynx_protocol.json").write_text(
                json.dumps({"rdd_train_component": "descriptor"}), encoding="utf-8"
            )
            resolution = resolve_vismatch_checkpoint("rdd-lightglue", "custom", path, "auto")
            with self.assertRaisesRegex(RuntimeError, "ignored tensor validation"):
                apply_vismatch_checkpoint(Model(), resolution)

    def test_descriptor_loader_keeps_missing_batchnorm_buffers(self):
        class LoMa(nn.Module):
            def __init__(self):
                super().__init__()
                self._descriptor = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.matcher = LoMa()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = Model().matcher._descriptor
            state = {
                key: value
                for key, value in source.state_dict().items()
                if not key.endswith(("running_mean", "running_var", "num_batches_tracked"))
            }
            path = root / "model.pth"
            self._save(path, {f"_descriptor.{key}": value for key, value in state.items()})
            (root / "czechlynx_protocol.json").write_text(
                json.dumps({"loma_train_component": "descriptor"}), encoding="utf-8"
            )
            resolution = resolve_vismatch_checkpoint("loma", "custom", path, "auto")
            apply_vismatch_checkpoint(Model(), resolution)

    def test_loma_descriptor_checkpoint_applies_only_descriptor(self):
        class LoMa(nn.Module):
            def __init__(self):
                super().__init__()
                self._detector = nn.Linear(2, 2, bias=False)
                self._descriptor = nn.Linear(2, 2, bias=False)
                self.transformers = nn.ModuleList()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.matcher = LoMa()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = Model().matcher
            source._descriptor.weight.data.fill_(5.0)
            path = root / "model.pth"
            self._save(path, {f"_descriptor.{key}": value for key, value in source._descriptor.state_dict().items()})
            (root / "czechlynx_protocol.json").write_text(
                json.dumps({"loma_train_component": "descriptor"}), encoding="utf-8"
            )
            resolution = resolve_vismatch_checkpoint("loma", "custom", path, "auto")
            self.assertEqual(resolution.resolved_component_mode, "descriptor_only")
            target = Model()
            target.matcher._detector.weight.data.fill_(2.0)
            apply_vismatch_checkpoint(target, resolution)
            self.assertTrue(torch.all(target.matcher._descriptor.weight == 5.0))
            self.assertTrue(torch.all(target.matcher._detector.weight == 2.0))

    def test_rng_state_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._save(root / "rng_state.pt", {"state": torch.zeros(1)})
            self._save(root / "model.pth", {"detector.weight": torch.zeros(2, 2)})
            resolution = resolve_vismatch_checkpoint("rdd-lightglue", "custom", root, "extractor_only")
            self.assertEqual(resolution.files[0].path.name, "model.pth")


if __name__ == "__main__":
    unittest.main()
