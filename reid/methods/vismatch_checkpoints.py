"""Discovery and loading helpers for custom Vismatch checkpoints.

This module intentionally keeps checkpoint inspection independent from Vismatch,
CUDA, and model construction so configuration and discovery can be tested with
small local state dictionaries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from reid.utils.fingerprints import sha256_file


SUPPORTED_CHECKPOINT_SOURCES = {"default", "custom"}
SUPPORTED_COMPONENT_MODES = {"auto", "matcher_only", "extractor_only", "full"}
SUPPORTED_LOMA_ARCHITECTURES = {"LoMa-B", "LoMa-L", "LoMa-G", "LoMa-B128", "LoMa-R"}
SUPPORTED_CHECKPOINT_SUFFIXES = {".safetensors", ".pth", ".pt", ".bin"}
_IGNORED_NAME_TOKENS = ("optimizer", "scheduler", "random_state", "random-states", "scaler")


@dataclass(frozen=True)
class CheckpointFile:
    component: str
    path: Path
    file_format: str
    sha256: str
    keys: Tuple[str, ...]
    shapes: Tuple[Tuple[str, Tuple[int, ...]], ...]

    def manifest_entry(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "path": self.path.resolve().as_posix(),
            "format": self.file_format,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class VismatchCheckpointResolution:
    source: str
    requested_path: Optional[str]
    matcher: str
    component_mode: str
    loma_arch: str
    files: Tuple[CheckpointFile, ...]
    default_components: Tuple[str, ...]
    manifest_path: Optional[str] = None
    validation: str = "default"

    @property
    def file_map(self) -> Dict[str, CheckpointFile]:
        return {item.component: item for item in self.files}

    @property
    def fingerprint(self) -> str:
        payload = {
            "source": self.source,
            "requested_path": self.requested_path,
            "matcher": self.matcher,
            "component_mode": self.component_mode,
            "loma_arch": self.loma_arch,
            "files": [item.manifest_entry() for item in self.files],
            "default_components": list(self.default_components),
            "manifest_path": self.manifest_path,
            "validation": self.validation,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "requested_path": self.requested_path,
            "matcher": self.matcher,
            "component_mode": self.component_mode,
            "loma_arch": self.loma_arch,
            "components": [item.manifest_entry() for item in self.files],
            "default_components": list(self.default_components),
            "manifest_path": self.manifest_path,
            "validation": self.validation,
            "fingerprint": self.fingerprint,
        }


def _unwrap_state_dict(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint does not contain a mapping state dictionary")
    for key in ("state_dict", "model_state_dict", "model"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and nested and all(isinstance(name, str) for name in nested):
            return nested
    if value and all(isinstance(name, str) for name in value):
        return value
    raise ValueError("checkpoint mapping does not contain a recognizable model state dictionary")


def _load_torch_state(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"PyTorch is required to inspect checkpoint {path}") from exc
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    return _unwrap_state_dict(value)


def _inspect_state(path: Path) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, Tuple[int, ...]], ...]]:
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as exc:
            raise RuntimeError("safetensors is required to inspect .safetensors checkpoints") from exc
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = tuple(sorted(handle.keys()))
            shapes = tuple((key, tuple(int(value) for value in handle.get_slice(key).get_shape())) for key in keys)
        return keys, shapes
    state = _load_torch_state(path)
    keys = tuple(sorted(str(key) for key in state))
    shapes = tuple((str(key), tuple(int(value) for value in getattr(state[key], "shape", ()))) for key in keys)
    return keys, shapes


def _candidate_files(directory: Path) -> Sequence[Path]:
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_CHECKPOINT_SUFFIXES
            and not any(token in path.name.lower() for token in _IGNORED_NAME_TOKENS)
        )
    )


def _classify_keys(keys: Sequence[str]) -> Optional[str]:
    names = tuple(str(key).removeprefix("module.") for key in keys)
    has_transformer = any(key.startswith("transformers.") for key in names)
    has_assignment = any(key.startswith("log_assignment.") for key in names)
    has_token_confidence = any(key.startswith("token_confidence.") for key in names)
    has_loma_parts = any(key.startswith(("_detector.", "_descriptor.")) for key in names)
    has_rdd_parts = any(
        key.startswith(("detector.", "descriptor.", "interpolator.", "softdetect.", "backbone."))
        for key in names
    )
    if has_transformer and has_assignment and has_token_confidence:
        return "lightglue"
    if has_transformer and has_assignment and (has_loma_parts or not has_token_confidence):
        return "loma_model"
    if has_rdd_parts:
        return "rdd_extractor"
    return None


def _manifest_mapping(path: Path) -> Tuple[Optional[Path], Dict[str, Path], Dict[str, str]]:
    manifest = path / "checkpoint_manifest.json"
    if not manifest.is_file():
        return None, {}, {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint manifest: {manifest}") from exc
    components = payload.get("components") if isinstance(payload, Mapping) else None
    if not isinstance(components, Mapping):
        raise ValueError(f"Checkpoint manifest must contain a components mapping: {manifest}")
    paths: Dict[str, Path] = {}
    hashes: Dict[str, str] = {}
    for component, value in components.items():
        if isinstance(value, str):
            filename = value
            declared_hash = ""
        elif isinstance(value, Mapping):
            filename = value.get("path") or value.get("file")
            declared_hash = str(value.get("sha256") or "")
        else:
            raise ValueError(f"Invalid manifest entry for component {component}: {manifest}")
        if not isinstance(filename, str):
            raise ValueError(f"Manifest entry for component {component} has no file path: {manifest}")
        candidate = (path / filename).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint manifest references missing file: {candidate}")
        paths[str(component)] = candidate
        hashes[str(component)] = declared_hash
    return manifest, paths, hashes


def _inspect_file(path: Path, component: Optional[str] = None) -> CheckpointFile:
    keys, shapes = _inspect_state(path)
    schema_component = _classify_keys(keys)
    if component is not None and schema_component != component:
        raise ValueError(
            f"Checkpoint manifest component {component!r} does not match tensor schema {schema_component!r} for {path}"
        )
    detected = component or schema_component
    if detected is None:
        raise ValueError(f"Could not identify checkpoint component from tensor schema: {path}")
    return CheckpointFile(
        component=detected,
        path=path.resolve(),
        file_format=path.suffix.lower().lstrip("."),
        sha256=sha256_file(path),
        keys=keys,
        shapes=shapes,
    )


def _select_components(
    matcher: str,
    mode: str,
    detected: Mapping[str, CheckpointFile],
    loma_arch: str,
) -> Tuple[Tuple[CheckpointFile, ...], Tuple[str, ...]]:
    if matcher == "rdd-lightglue":
        incompatible = sorted(name for name in detected if name not in {"rdd_extractor", "lightglue"})
        if incompatible:
            raise ValueError(f"RDD-LightGlue cannot use incompatible checkpoint components: {incompatible}")
        if mode == "auto":
            selected = tuple(detected[name] for name in ("rdd_extractor", "lightglue") if name in detected)
            if not selected:
                raise ValueError("Custom RDD-LightGlue checkpoint contains no RDD or LightGlue model component")
            defaults = tuple(name for name in ("rdd_extractor", "lightglue") if name not in detected)
            return selected, defaults
        required = "lightglue" if mode == "matcher_only" else "rdd_extractor" if mode == "extractor_only" else None
        if required is not None:
            if required not in detected:
                raise ValueError(f"checkpoint_components={mode} requires a {required} checkpoint")
            other = "rdd_extractor" if required == "lightglue" else "lightglue"
            return (detected[required],), (other,)
        missing = [name for name in ("rdd_extractor", "lightglue") if name not in detected]
        if missing:
            raise ValueError(f"checkpoint_components=full requires both RDD and LightGlue files; missing {missing}")
        return (detected["rdd_extractor"], detected["lightglue"]), ()

    if matcher == "loma":
        if mode == "extractor_only":
            raise ValueError("checkpoint_components=extractor_only is not supported for the current LoMa wrapper")
        incompatible = sorted(name for name in detected if name != "loma_model")
        if incompatible:
            raise ValueError(
                "LoMa cannot use generic RDD or LightGlue checkpoint components: "
                f"{incompatible}"
            )
        if "loma_model" not in detected:
            incompatible = sorted(detected)
            raise ValueError(
                "LoMa requires a LoMa-compatible checkpoint; detected incompatible components: "
                f"{incompatible or ['none']}"
            )
        if mode == "full":
            item = detected["loma_model"]
            names = set(item.keys)
            if not any(name.startswith("_detector.") for name in names) or not any(name.startswith("_descriptor.") for name in names):
                raise ValueError(f"checkpoint_components=full requires complete {loma_arch} LoMa weights")
        return (detected["loma_model"],), ()

    raise ValueError(f"Unsupported Vismatch matcher for checkpoint loading: {matcher}")


def resolve_vismatch_checkpoint(
    matcher: str,
    source: str = "default",
    path: Optional[str | Path] = None,
    component_mode: str = "auto",
    loma_arch: str = "LoMa-B",
) -> VismatchCheckpointResolution:
    matcher = str(matcher).strip().lower()
    source = str(source).strip().lower()
    component_mode = str(component_mode).strip().lower()
    loma_arch = str(loma_arch)
    if source not in SUPPORTED_CHECKPOINT_SOURCES:
        raise ValueError(f"checkpoint_source must be one of {sorted(SUPPORTED_CHECKPOINT_SOURCES)}")
    if component_mode not in SUPPORTED_COMPONENT_MODES:
        raise ValueError(f"checkpoint_components must be one of {sorted(SUPPORTED_COMPONENT_MODES)}")
    if loma_arch not in SUPPORTED_LOMA_ARCHITECTURES:
        raise ValueError(f"Unsupported LoMa architecture: {loma_arch}")
    if matcher == "loma" and component_mode == "extractor_only":
        raise ValueError("checkpoint_components=extractor_only is not supported for the current LoMa wrapper")
    requested = None if path is None else str(Path(path).expanduser())
    if source == "default":
        if path is not None:
            raise ValueError("checkpoint_source=default cannot be combined with checkpoint_path")
        defaults = ("rdd_extractor", "lightglue") if matcher == "rdd-lightglue" else ("loma_model",)
        return VismatchCheckpointResolution(source, None, matcher, component_mode, loma_arch, (), defaults)
    if path is None:
        raise ValueError("checkpoint_source=custom requires checkpoint_path")
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Custom Vismatch checkpoint path does not exist: {root}")
    manifest_path: Optional[Path] = None
    detected: Dict[str, CheckpointFile] = {}
    if root.is_file():
        item = _inspect_file(root)
        detected[item.component] = item
    elif root.is_dir():
        manifest_path, manifest_files, manifest_hashes = _manifest_mapping(root)
        if manifest_files:
            for component, file_path in manifest_files.items():
                if component not in {"rdd_extractor", "lightglue", "loma_model"}:
                    raise ValueError(f"Unsupported checkpoint manifest component: {component}")
                item = _inspect_file(file_path, component=component)
                declared_hash = manifest_hashes.get(component, "")
                if declared_hash and declared_hash != item.sha256:
                    raise ValueError(f"Checkpoint manifest SHA-256 mismatch for {file_path}")
                detected[component] = item
        else:
            for file_path in _candidate_files(root):
                item = _inspect_file(file_path)
                if item.component in detected:
                    raise ValueError(
                        f"Multiple custom checkpoint files detected for component {item.component}: "
                        f"{detected[item.component].path} and {item.path}; add checkpoint_manifest.json or use one file"
                    )
                detected[item.component] = item
    else:
        raise ValueError(f"Custom Vismatch checkpoint path is neither a file nor directory: {root}")
    selected, defaults = _select_components(matcher, component_mode, detected, loma_arch)
    return VismatchCheckpointResolution(
        source="custom",
        requested_path=requested,
        matcher=matcher,
        component_mode=component_mode,
        loma_arch=loma_arch,
        files=selected,
        default_components=defaults,
        manifest_path=manifest_path.resolve().as_posix() if manifest_path else None,
        validation="schema_validated",
    )


def _load_state(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise RuntimeError("safetensors is required to load custom .safetensors checkpoints") from exc
        return load_file(str(path), device="cpu")
    return _load_torch_state(path)


def _normalize_state_keys(state: Mapping[str, Any], component: str) -> Dict[str, Any]:
    result = {str(key): value for key, value in state.items()}
    prefixes = ["module."]
    if component == "lightglue":
        prefixes += ["lightglue.", "model.lightglue.", "matcher.lightglue."]
    elif component == "rdd_extractor":
        prefixes += ["RDD.", "model.RDD.", "matcher.RDD.", "rdd.", "model.rdd."]
    else:
        prefixes += ["model.", "matcher.", "model.matcher.", "loma."]
    changed = True
    while changed and result:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in result):
                result = {key[len(prefix):]: value for key, value in result.items()}
                changed = True
                break
    return result


def _load_into(module: Any, item: CheckpointFile, *, allow_loma_partial: bool) -> None:
    state = _normalize_state_keys(_load_state(item.path), item.component)
    if item.component == "loma_model" and allow_loma_partial:
        allowed_prefixes = ("input_proj.", "posenc.", "transformers.", "log_assignment.")
        ignored_prefixes = ("_detector.", "_descriptor.")
        unknown = [key for key in state if not key.startswith(allowed_prefixes + ignored_prefixes)]
        if unknown:
            raise RuntimeError(f"LoMa partial checkpoint contains unknown keys: {unknown}")
        state = {key: value for key, value in state.items() if key.startswith(allowed_prefixes)}
        result = module.load_state_dict(state, strict=False)
        unexpected = list(result.unexpected_keys)
        missing = list(result.missing_keys)
        allowed_missing = tuple(name for name in missing if name.startswith(("_detector.", "_descriptor.")))
        if unexpected or len(allowed_missing) != len(missing):
            raise RuntimeError(
                f"LoMa partial checkpoint validation failed for {item.path}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"{item.component} checkpoint is incompatible with the active Vismatch model: {item.path}") from exc


def apply_vismatch_checkpoint(model: Any, resolution: VismatchCheckpointResolution) -> None:
    """Apply a validated resolution to a constructed Vismatch model."""

    if resolution.source == "default":
        model.eval()
        return
    for item in resolution.files:
        if item.component == "lightglue":
            target = getattr(model, "lightglue", None)
            if target is None:
                raise RuntimeError("Vismatch model does not expose a LightGlue component")
            _load_into(target, item, allow_loma_partial=False)
        elif item.component == "rdd_extractor":
            target = getattr(getattr(model, "matcher", None), "RDD", None)
            if target is None:
                raise RuntimeError("Vismatch model does not expose an RDD extractor component")
            _load_into(target, item, allow_loma_partial=False)
        elif item.component == "loma_model":
            target = getattr(model, "matcher", None)
            if target is None:
                raise RuntimeError("Vismatch model does not expose a LoMa model component")
            has_backbone = any(name.startswith(("_detector.", "_descriptor.")) for name in item.keys)
            _load_into(target, item, allow_loma_partial=resolution.component_mode == "matcher_only" or (resolution.component_mode == "auto" and not has_backbone))
        else:
            raise RuntimeError(f"Unsupported resolved Vismatch component: {item.component}")
    model.eval()

