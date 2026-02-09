import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from reid.utils.io import ensure_dir, ensure_file


@dataclass
class FrameFeat:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    image_size: np.ndarray


def _choose_rdd_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("RDD device is set to cuda but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_rdd_modules(repo_dir: Path):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    try:
        from RDD.RDD import build as build_rdd  # type: ignore
        # Import LightGlue directly to avoid RDD.matchers.__init__ side imports
        # (e.g., dense_matcher -> poselib) that are not required for this benchmark.
        from RDD.matchers.lightglue import LightGlue  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import RDD modules from repo_dir={repo_dir}. "
            "Expected to import RDD.RDD and RDD.matchers.lightglue.LightGlue."
        ) from exc
    return build_rdd, LightGlue


def _build_rdd_models(repo_dir: Path, config_path: Path, weights_path: Path, device: torch.device, top_k: int):
    build_rdd, LightGlue = _resolve_rdd_modules(repo_dir)
    ensure_file(config_path, "RDD config file")
    ensure_file(weights_path, "RDD weights file")

    # Keep matcher config hardcoded to mirror the previous implementation.
    lg_conf = {
        "name": "lightglue",
        "input_dim": 256,
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,
        "mp": False,
        "filter_threshold": 0.01,
        "depth_confidence": -1,
        "width_confidence": -1,
        "weights": str(repo_dir / "weights" / "RDD_lg-v2.pth"),
    }
    ensure_file(Path(lg_conf["weights"]), "RDD LightGlue weights file")

    rdd_conf = None
    model = build_rdd(rdd_conf, weights=str(weights_path))
    model.to(device).eval()
    model.top_k = int(top_k)
    model.set_softdetect(top_k=int(top_k))

    matcher = LightGlue("rdd", **lg_conf).to(device).eval()
    return model, matcher


def _to_image_tensor(image: Any, resize_max: int) -> torch.Tensor:
    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        arr = np.asarray(image)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

    if resize_max and resize_max > 0:
        w, h = img.size
        scale = float(resize_max) / float(max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _to_uint8_rgb(
    image: Any,
    resize_max: int,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            if arr.min() >= 0.0 and arr.max() <= 1.0:
                arr = arr * 255.0
            elif mean is not None and std is not None and arr.ndim == 3 and arr.shape[2] >= 3:
                mean_np = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
                std_np = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
                arr = arr[..., :3] * std_np + mean_np
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            elif arr.max() <= 5.0 and arr.min() >= -5.0:
                # Fallback for normalized tensors when explicit stats are unavailable.
                arr_min = float(arr.min())
                arr_max = float(arr.max())
                if arr_max > arr_min:
                    arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
                else:
                    arr = np.zeros_like(arr)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        arr = np.asarray(image)
        if np.issubdtype(arr.dtype, np.floating) and arr.min() >= 0.0 and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

    if resize_max and resize_max > 0:
        w, h = img.size
        scale = float(resize_max) / float(max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return np.asarray(img)


def _cache_key(
    image_path: str,
    split_name: str,
    resize_max: int,
    top_k: int,
    cfg_tag: str,
) -> str:
    payload = f"{split_name}|{image_path}|resize_max={resize_max}|top_k={top_k}|{cfg_tag}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.npz"


def _save_cached_feat(path: Path, feat: FrameFeat) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=feat.keypoints,
        descriptors=feat.descriptors,
        scores=feat.scores,
        image_size=feat.image_size,
    )


def _load_cached_feat(path: Path) -> FrameFeat:
    data = np.load(path)
    return FrameFeat(
        keypoints=data["keypoints"],
        descriptors=data["descriptors"],
        scores=data["scores"],
        image_size=data["image_size"],
    )


def _resolve_image_path(image_path: str, dataset_root: Path) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return dataset_root / path


def _decode_mask_from_row(row: Any, mask_col: str, idx: int) -> np.ndarray:
    raw_mask = row.get(mask_col)
    if raw_mask is None:
        raise ValueError(f"Missing mask at row index {idx}")
    if isinstance(raw_mask, float) and np.isnan(raw_mask):
        raise ValueError(f"Missing mask at row index {idx}")

    if isinstance(raw_mask, str):
        try:
            mask_data = json.loads(raw_mask)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON mask at row index {idx}") from exc
    elif isinstance(raw_mask, dict):
        mask_data = raw_mask
    else:
        raise ValueError(
            f"Unsupported mask type at row index {idx}: {type(raw_mask)}. "
            "Expected JSON string or COCO-RLE dict."
        )

    try:
        mask = mask_utils.decode(mask_data).astype(np.uint8)
    except Exception as exc:
        raise ValueError(f"Failed to decode mask at row index {idx}") from exc

    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask.max(axis=-1)
    if mask.ndim != 2:
        raise ValueError(f"Decoded mask must be 2D at row index {idx}, got shape={mask.shape}")
    return mask


def _load_raw_rgb_image(
    row: Any,
    idx: int,
    dataset_root: Path,
    path_col: str,
    no_background: bool,
    mask_col: str,
) -> Image.Image:
    if path_col not in row.index:
        raise KeyError(f"RDD method requires path_col='{path_col}' in dataset dataframe")
    image_path = _resolve_image_path(str(row[path_col]), dataset_root=dataset_root)
    ensure_file(image_path, "RDD image file")
    image = Image.open(image_path).convert("RGB")
    if not no_background:
        return image

    image_np = np.asarray(image, dtype=np.uint8)
    mask = _decode_mask_from_row(row=row, mask_col=mask_col, idx=idx)
    if image_np.shape[0] != mask.shape[0] or image_np.shape[1] != mask.shape[1]:
        raise ValueError(
            f"Mask/Image size mismatch at row index {idx}: mask={mask.shape}, image={image_np.shape[:2]}"
        )
    image_np = image_np * np.expand_dims(np.asfortranarray(mask), axis=-1)
    return Image.fromarray(image_np)


@torch.no_grad()
def _extract_frame(model: Any, image_tensor: torch.Tensor, device: torch.device, top_k: int) -> FrameFeat:
    model.top_k = int(top_k)
    model.set_softdetect(top_k=int(top_k))
    image_tensor = image_tensor.to(device)
    out = model.extract(image_tensor)[0]
    return FrameFeat(
        keypoints=out["keypoints"].detach().cpu().numpy(),
        descriptors=out["descriptors"].detach().cpu().numpy(),
        scores=out["scores"].detach().cpu().numpy(),
        image_size=np.array(image_tensor.shape[-2:], dtype=np.int32),
    )


def _score_pair(matcher: Any, fa: FrameFeat, fb: FrameFeat, device: torch.device) -> Tuple[float, int]:
    k0 = torch.from_numpy(fa.keypoints).to(device).unsqueeze(0)
    k1 = torch.from_numpy(fb.keypoints).to(device).unsqueeze(0)
    d0 = torch.from_numpy(fa.descriptors).to(device).unsqueeze(0)
    d1 = torch.from_numpy(fb.descriptors).to(device).unsqueeze(0)
    size0 = torch.tensor(fa.image_size[::-1].copy(), device=device).unsqueeze(0)
    size1 = torch.tensor(fb.image_size[::-1].copy(), device=device).unsqueeze(0)
    pred = matcher(
        {
            "image0": {"keypoints": k0, "descriptors": d0, "image_size": size0},
            "image1": {"keypoints": k1, "descriptors": d1, "image_size": size1},
        }
    )
    if pred["scores"][0].numel() == 0:
        return 0.0, 0
    conf = pred["scores"][0]
    sum_conf = float(conf.sum().item())
    norm = min(max(1, int(fa.keypoints.shape[0])), max(1, int(fb.keypoints.shape[0])))
    score = sum_conf / float(norm)
    n_matches = int((conf > 0).sum().item())
    return score, n_matches


@torch.no_grad()
def _match_frames(matcher: Any, fa: FrameFeat, fb: FrameFeat, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k0 = torch.from_numpy(fa.keypoints).to(device).unsqueeze(0)
    k1 = torch.from_numpy(fb.keypoints).to(device).unsqueeze(0)
    d0 = torch.from_numpy(fa.descriptors).to(device).unsqueeze(0)
    d1 = torch.from_numpy(fb.descriptors).to(device).unsqueeze(0)
    size0 = torch.tensor(fa.image_size[::-1].copy(), device=device).unsqueeze(0)
    size1 = torch.tensor(fb.image_size[::-1].copy(), device=device).unsqueeze(0)
    pred = matcher(
        {
            "image0": {"keypoints": k0, "descriptors": d0, "image_size": size0},
            "image1": {"keypoints": k1, "descriptors": d1, "image_size": size1},
        }
    )
    matches = pred["matches"][0]
    conf = pred["scores"][0]
    if conf.numel() == 0 or matches.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    mkpts0 = fa.keypoints[matches[:, 0].detach().cpu().numpy()]
    mkpts1 = fb.keypoints[matches[:, 1].detach().cpu().numpy()]
    return mkpts0, mkpts1, conf.detach().cpu().numpy()


def _draw_matches_save(
    out_path: Path,
    img0: np.ndarray,
    img1: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    conf: np.ndarray,
    max_matches: int,
    title: str,
) -> str:
    if len(conf) > int(max_matches):
        order = np.argsort(-conf)[: int(max_matches)]
        mkpts0 = mkpts0[order]
        mkpts1 = mkpts1[order]
        conf = conf[order]

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    target_h = max(h0, h1)
    target_w = max(w0, w1)

    img0, mkpts0 = _letterbox_image_and_keypoints(img0, mkpts0, target_h=target_h, target_w=target_w)
    img1, mkpts1 = _letterbox_image_and_keypoints(img1, mkpts1, target_h=target_h, target_w=target_w)

    out = np.zeros((target_h, target_w * 2, 3), dtype=np.uint8)
    out[:, :target_w] = img0
    out[:, target_w:] = img1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(out)
    ax.axis("off")
    if title:
        ax.set_title(title)

    if len(conf) > 0:
        conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        colors = cm.viridis(conf_norm)
        for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, colors):
            ax.plot([x0, x1 + target_w], [y0, y1], color=c, linewidth=1)
        ax.scatter(mkpts0[:, 0], mkpts0[:, 1], s=6, c=colors, marker="o")
        ax.scatter(mkpts1[:, 0] + target_w, mkpts1[:, 1], s=6, c=colors, marker="o")
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=float(conf.min()), vmax=float(conf.max())))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("match confidence")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _letterbox_image_and_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: int = 114,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image, keypoints

    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))

    canvas = np.full((target_h, target_w, 3), fill_value=pad_value, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    if keypoints.size == 0:
        return canvas, keypoints
    remapped = keypoints.astype(np.float32, copy=True)
    remapped[:, 0] = remapped[:, 0] * scale + float(pad_x)
    remapped[:, 1] = remapped[:, 1] * scale + float(pad_y)
    return canvas, remapped


def _extract_split_features(
    dataset: Any,
    split_name: str,
    model: Any,
    device: torch.device,
    top_k: int,
    resize_max: int,
    cache_dir: Path,
    dataset_root: Path,
    no_background: bool,
    mask_col: str,
    path_col: str,
    cfg_tag: str,
) -> List[FrameFeat]:
    if path_col not in dataset.df.columns:
        raise KeyError(f"RDD method requires path_col='{path_col}' in dataset dataframe")

    feats: List[FrameFeat] = []
    iterator = tqdm(range(len(dataset)), desc=f"[rdd][extract:{split_name}]", mininterval=1, ncols=120)
    for idx in iterator:
        image_path = str(dataset.df.iloc[idx][path_col])
        key = _cache_key(image_path=image_path, split_name=split_name, resize_max=resize_max, top_k=top_k, cfg_tag=cfg_tag)
        cp = _cache_path(cache_dir, key)
        if cp.is_file():
            feats.append(_load_cached_feat(cp))
            continue
        row = dataset.df.iloc[idx]
        image = _load_raw_rgb_image(
            row=row,
            idx=idx,
            dataset_root=dataset_root,
            path_col=path_col,
            no_background=no_background,
            mask_col=mask_col,
        )
        image_tensor = _to_image_tensor(image=image, resize_max=resize_max)
        feat = _extract_frame(model=model, image_tensor=image_tensor, device=device, top_k=top_k)
        _save_cached_feat(cp, feat)
        feats.append(feat)
    return feats


def run_rdd_benchmark(
    cfg: Any,
    dataset_query: Any,
    dataset_database: Any,
    run_dir: Path,
    checkpoint_path: Optional[Path] = None,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    candidate_indices: Optional[np.ndarray] = None,
    method_artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    settings = cfg.benchmark.methods.rdd
    repo_dir = Path(str(settings.repo_dir))
    config_path = Path(str(settings.config_path))
    weights_path = Path(str(settings.weights))
    ensure_dir(repo_dir, "RDD repository directory")

    device = _choose_rdd_device(str(settings.device))
    top_k = int(settings.top_k)
    resize_max = int(settings.resize_max)
    path_col = str(settings.path_col)
    mask_col = str(cfg.dataset.mask_col)
    dataset_root = Path(str(cfg.dataset.root))
    no_background = bool(cfg.dataset.no_background)
    cache_dir = Path(str(settings.cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_tag = "none"
    if checkpoint_path is not None:
        if checkpoint_path.is_file():
            stat = checkpoint_path.stat()
            checkpoint_tag = f"{checkpoint_path}|{stat.st_size}|{int(stat.st_mtime)}"
        else:
            checkpoint_tag = str(checkpoint_path)
    stage_a_method = str(settings.stage_a_method) if "stage_a_method" in settings else "none"
    cfg_tag = hashlib.sha256(
        (
            f"repo={repo_dir}|cfg={config_path}|weights={weights_path}|"
            f"stage_a={stage_a_method}|model_type={cfg.model.type}|model_mode={cfg.model.mode}|"
            f"checkpoint={checkpoint_tag}|top_k={top_k}|resize_max={resize_max}|"
            f"no_bg={no_background}|path_col={path_col}|mask_col={mask_col}"
        ).encode("utf-8")
    ).hexdigest()
    print(
        "[rdd] cache tag components: "
        f"stage_a={stage_a_method} model_type={cfg.model.type} model_mode={cfg.model.mode} "
        f"checkpoint={checkpoint_tag} no_background={no_background} top_k={top_k} resize_max={resize_max}"
    )

    t_build = time.perf_counter()
    rdd_model, matcher = _build_rdd_models(
        repo_dir=repo_dir,
        config_path=config_path,
        weights_path=weights_path,
        device=device,
        top_k=top_k,
    )
    model_build_sec = time.perf_counter() - t_build

    t_extract = time.perf_counter()
    query_feats = _extract_split_features(
        dataset=dataset_query,
        split_name="query",
        model=rdd_model,
        device=device,
        top_k=top_k,
        resize_max=resize_max,
        cache_dir=cache_dir,
        dataset_root=dataset_root,
        no_background=no_background,
        mask_col=mask_col,
        path_col=path_col,
        cfg_tag=cfg_tag,
    )
    db_feats = _extract_split_features(
        dataset=dataset_database,
        split_name="database",
        model=rdd_model,
        device=device,
        top_k=top_k,
        resize_max=resize_max,
        cache_dir=cache_dir,
        dataset_root=dataset_root,
        no_background=no_background,
        mask_col=mask_col,
        path_col=path_col,
        cfg_tag=cfg_tag,
    )
    extract_sec = time.perf_counter() - t_extract

    t_sim = time.perf_counter()
    similarity = np.full((len(query_feats), len(db_feats)), fill_value=-1e9, dtype=np.float32)
    match_counts: List[int] = []
    for qi in tqdm(range(len(query_feats)), desc="[rdd][match]", mininterval=1, ncols=120):
        qf = query_feats[qi]
        if candidate_indices is None:
            db_candidates = range(len(db_feats))
        else:
            db_candidates = candidate_indices[qi].tolist()
        for di in db_candidates:
            score, nm = _score_pair(matcher, qf, db_feats[di], device=device)
            similarity[qi, di] = float(score)
            match_counts.append(int(nm))
    rerank_sec = time.perf_counter() - t_sim

    if bool(cfg.visualization.enabled):
        rng = np.random.default_rng(int(cfg.benchmark.seed))
        num_examples = min(int(cfg.visualization.num_examples), len(dataset_query))
        sampled_indices = rng.choice(np.arange(len(dataset_query)), size=num_examples, replace=False) if num_examples > 0 else []
        max_matches = int(getattr(cfg.visualization, "rdd_max_matches", 200))
        out_dir = Path(cfg.visualization.dir) / run_dir.name
        match_paths: List[str] = []
        for q_idx in sampled_indices:
            q_idx_int = int(q_idx)
            db_idx = int(np.argmax(similarity[q_idx_int]))
            q_row = dataset_query.df.iloc[q_idx_int]
            db_row = dataset_database.df.iloc[db_idx]
            q_img = _load_raw_rgb_image(
                row=q_row,
                idx=q_idx_int,
                dataset_root=dataset_root,
                path_col=path_col,
                no_background=no_background,
                mask_col=mask_col,
            )
            db_img = _load_raw_rgb_image(
                row=db_row,
                idx=db_idx,
                dataset_root=dataset_root,
                path_col=path_col,
                no_background=no_background,
                mask_col=mask_col,
            )
            q_vis = _to_uint8_rgb(q_img, resize_max=resize_max, mean=mean, std=std)
            db_vis = _to_uint8_rgb(db_img, resize_max=resize_max, mean=mean, std=std)
            mkpts0, mkpts1, conf = _match_frames(matcher, query_feats[q_idx_int], db_feats[db_idx], device=device)
            q_label = str(dataset_query.df.iloc[q_idx_int][cfg.dataset.label_col])
            db_label = str(dataset_database.df.iloc[db_idx][cfg.dataset.label_col])
            title = f"q{q_idx_int}:{q_label} -> db{db_idx}:{db_label} | n_matches={len(conf)}"
            out_path = out_dir / f"rdd_matches_q{q_idx_int}_db{db_idx}.png"
            match_paths.append(
                _draw_matches_save(
                    out_path=out_path,
                    img0=q_vis,
                    img1=db_vis,
                    mkpts0=mkpts0,
                    mkpts1=mkpts1,
                    conf=conf,
                    max_matches=max_matches,
                    title=title,
                )
            )
        if method_artifacts is not None:
            method_artifacts["rdd_match_paths"] = match_paths

    timings = {
        "rdd_model_build_sec": float(model_build_sec),
        "rdd_feature_extraction_sec": float(extract_sec),
        "rdd_rerank_sec": float(rerank_sec),
        "total_method_sec": float(model_build_sec + extract_sec + rerank_sec),
    }
    method_metrics = {
        "rdd_avg_matches": float(np.mean(match_counts)) if match_counts else 0.0,
    }
    return similarity, timings, method_metrics
