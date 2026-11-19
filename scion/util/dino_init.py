"""DINO image preprocessing for codebook init.

Pure helpers for the codebook DINO-init path: enumerate a scene's training
images, run a DINOv2/v3 backbone over them with on-disk caching, project
3D positions into the cached feature maps, and optionally PCA-reduce the
aggregated features.  No model state — everything operates on plain
tensors and paths.

The cache key hashes the sorted train image names, model name, resolution,
PCA dim and downscale factor so reruns skip the per-image forward passes
(the dominant cost; ~5–15 min/scene).  Canonical cache location is
``data/{scene}/dino/``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Training views
# ---------------------------------------------------------------------------


def train_views(
    scene,
    device: str = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], int, int]:
    """``(viewmats, Ks, image_paths, W, H)`` of a :class:`~scion.data.Scene`'s
    train split only, so eval images never reach DINO initialization."""
    cams = scene.train_cameras()
    if not cams:
        raise RuntimeError("scene has no training views")
    image_paths = [str(scene.image_paths[i]) for i in scene.train_indices]
    viewmats = [c.viewmat().to(device=device, dtype=torch.float32) for c in cams]
    Ks = [c.K.to(device=device, dtype=torch.float32) for c in cams]
    return viewmats, Ks, image_paths, int(cams[0].width), int(cams[0].height)


def train_feature_maps(
    scene,
    scene_dir: Path | str,
    *,
    model_name: str,
    resolution: int,
    pca_dim: int,
    device: str = "cuda",
    no_cache: bool = False,
) -> List[torch.Tensor]:
    """Cached DINO maps of ``scene``'s train views: the ``<scene_dir>/dino``
    cache training reads (``scripts/preprocess_dino.py`` fills the same one).
    Features come from the full-resolution images (``images/`` for COLMAP)."""
    scene_dir = Path(scene_dir)
    blender = (scene_dir / "transforms_train.json").exists()
    return extract_dino_features_cached(
        [str(scene.image_paths[i]) for i in scene.train_indices],
        cache_dir=scene_dir / "dino",
        model_name=model_name,
        resolution=resolution,
        images_dir=str(scene_dir if blender else scene_dir / "images"),
        device=device,
        no_cache=no_cache,
        pca_dim=pca_dim,
        downscale_factor=scene.downscale_factor,
    )


# ---------------------------------------------------------------------------
# DINOv2/v3 feature extraction (cached)
# ---------------------------------------------------------------------------


def _extract_dino_features(
    image_paths: List[str],
    *,
    images_dir: Optional[str],
    resolution: int,
    model_name: str,
    device: str,
    bf16_autocast: bool = True,
) -> List[torch.Tensor]:
    """Forward each image through the chosen DINO backbone.

    Resizes so the larger dimension is ``resolution`` and both axes are
    multiples of the model's patch size.  Returns ``[D, h, w]`` per image.
    ``bf16_autocast=False`` runs the backbone in fp32.
    """
    from torchvision import transforms as T

    print(f"  Loading DINO model: {model_name}")

    try:
        from scion.util.dino_models import (
            DINO_MODELS, load_dino_model,
        )
        registry_name = model_name
        if registry_name in DINO_MODELS:
            model, patch_size, feat_dim = load_dino_model(registry_name)
            model = model.to(device)
            num_prefix = getattr(model, "num_prefix_tokens", 1)
            print(f"  Loaded from registry: {registry_name}")
        else:
            raise KeyError(f"Not in registry: {model_name}")
    except (ImportError, KeyError):
        import timm
        model = timm.create_model(
            model_name, pretrained=True, dynamic_img_size=True,
        ).to(device).eval()
        feat_dim = model.embed_dim
        patch_size = 14
        num_prefix = getattr(model, "num_prefix_tokens", 1)

    print(f"  Feature dimension: {feat_dim}, patch size: {patch_size}")
    print(f"  Prefix tokens to skip: {num_prefix}")

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_list: List[torch.Tensor] = []
    for i, img_path in enumerate(image_paths):
        if images_dir:
            p = Path(images_dir) / Path(img_path).name
            if not p.exists():
                p = Path(img_path)
        else:
            p = Path(img_path)

        img = Image.open(p).convert("RGB")
        orig_w, orig_h = img.size

        scale = resolution / max(orig_w, orig_h)
        new_w = int(round(orig_w * scale / patch_size)) * patch_size
        new_h = int(round(orig_h * scale / patch_size)) * patch_size
        img = img.resize((new_w, new_h), Image.BILINEAR)

        x = normalize(img).unsqueeze(0).to(device)

        # bf16 autocast: DINOv3 ViT-L attention is O(N²) in tokens; at
        # resolution > ~1200 the fp32 path OOMs on 32 GB GPUs.  Under
        # autocast SDPA dispatches to FlashAttention/mem-efficient kernels,
        # which fits ~2500-pixel inputs comfortably.  Output is recast to
        # float32 below before storage.
        autocast_enabled = bf16_autocast and (
            (isinstance(device, str) and device.startswith("cuda"))
            or (hasattr(device, "type") and device.type == "cuda")
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled,
        ):
            feats = model.forward_features(x)
            if isinstance(feats, dict):
                patch_feats = feats["x_norm_patchtokens"]
            else:
                patch_feats = feats[:, num_prefix:, :]
            patch_feats = patch_feats.float()

            h_feat = new_h // patch_size
            w_feat = new_w // patch_size
            expected = h_feat * w_feat
            actual = patch_feats.shape[1]
            if actual != expected:
                raise RuntimeError(
                    f"Patch count mismatch: got {actual}, expected {expected} "
                    f"({w_feat}x{h_feat}) from image {new_w}x{new_h}. "
                    f"Total tokens before prefix removal: {feats.shape[1]}, "
                    f"num_prefix={num_prefix}"
                )

            patch_feats = patch_feats.reshape(1, h_feat, w_feat, feat_dim)
            patch_feats = patch_feats.permute(0, 3, 1, 2).squeeze(0)

        features_list.append(patch_feats)

        if (i + 1) % 10 == 0 or i == 0 or i == len(image_paths) - 1:
            print(
                f"    [{i+1}/{len(image_paths)}] {p.name}: "
                f"input {new_w}x{new_h} -> features {w_feat}x{h_feat}"
            )

    return features_list


def extract_dino_features_cached(
    image_paths: List[str],
    *,
    cache_dir: Path | str,
    model_name: str = "dinov3_vitl16",
    resolution: int = 832,
    images_dir: Optional[str] = None,
    device: str = "cuda",
    no_cache: bool = False,
    bf16_autocast: bool = True,
    pca_dim: int = 0,
    downscale_factor: int = 1,
) -> List[torch.Tensor]:
    """Extract per-image 2D DINO feature maps with on-disk caching.

    Cache key: sha256 of the sorted image file names (not paths, so a cache
    moves with the data) + ``model_name`` + ``resolution`` + ``pca_dim`` +
    ``downscale_factor``; a different train split is a different cache. fp32
    maps (``bf16_autocast=False``) are cached under a separate ``_fp32`` name.

    Returned tensors live on CPU; move them to ``device`` at the call site.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = "\n".join(
        sorted(Path(p).name for p in image_paths)
        + [model_name, str(resolution), str(pca_dim), str(downscale_factor)]
    )
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
    cache_path = cache_dir / (
        f"dino_feature_maps_{model_name}_res{resolution}_{key_hash}"
        f"{'' if bf16_autocast else '_fp32'}.pt"
    )

    if cache_path.exists() and not no_cache:
        print(f"  Reusing cached DINO feature maps: {cache_path}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        return [t for t in blob["features"]]

    features = _extract_dino_features(
        image_paths,
        images_dir=images_dir,
        resolution=resolution,
        model_name=model_name,
        device=device,
        bf16_autocast=bf16_autocast,
    )
    # Temp file + rename: a killed or concurrent writer never leaves a partial .pt.
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save({"features": [f.cpu() for f in features]}, tmp)
        os.replace(tmp, cache_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"  Cached DINO feature maps: {cache_path}")
    return features


# ---------------------------------------------------------------------------
# Direct per-position projection (no per-Gaussian intermediate)
# ---------------------------------------------------------------------------


def _project_points_pinhole(
    positions: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
    near_plane: float,
    far_plane: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project ``positions [N, 3]`` into one camera.

    Returns ``(uv [N, 2], depth [N], in_frustum [N])`` — pixel-space
    coordinates, camera-space depth, and a boolean mask combining the
    near/far plane and on-image-plane checks.
    """
    N = positions.shape[0]
    ones = torch.ones(N, 1, device=positions.device, dtype=positions.dtype)
    pts_h = torch.cat([positions, ones], dim=-1)            # [N, 4]
    pts_cam = (viewmat @ pts_h.T).T                          # [N, 4]
    xyz_cam = pts_cam[:, :3]
    z = xyz_cam[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z_safe = z.clamp(min=1e-6)
    u = fx * xyz_cam[:, 0] / z_safe + cx
    v = fy * xyz_cam[:, 1] / z_safe + cy

    in_z = (z > near_plane) & (z < far_plane)
    in_uv = (u >= 0) & (u <= (W - 1)) & (v >= 0) & (v <= (H - 1))
    uv = torch.stack([u, v], dim=-1)
    return uv, z, in_z & in_uv


def project_features_at_positions(
    positions: torch.Tensor,
    viewmats: List[torch.Tensor],
    Ks: List[torch.Tensor],
    features_2d: List[torch.Tensor],
    W: int,
    H: int,
    *,
    distance_alpha: float = 0.0,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    device: str = "cuda",
    align_corners: bool = True,
    fill_unseen: bool = True,
) -> torch.Tensor:
    """Sample 2D DINO features at ``positions`` across all views.

    For each view, projects every position with a pinhole camera, samples
    the per-image feature map bilinearly, and accumulates with a
    distance-weighted falloff:

    ``w_v(i) = 1[in_frustum, in_image] * exp(-distance_alpha * ||p_i - cam_origin_v||)``

    Setting ``distance_alpha=0`` recovers uniform per-view averaging.

    Args:
        positions: ``[N, 3]`` world-space points.
        viewmats: List of ``[4, 4]`` view matrices (world → camera).
        Ks: List of ``[3, 3]`` intrinsics.
        features_2d: List of ``[D, H_feat, W_feat]`` feature tensors.
        W, H: Original image width / height.
        distance_alpha: Distance falloff (0 = uniform).
        near_plane, far_plane: Frustum z-clip.
        device: Torch device.
        align_corners: Passed to ``grid_sample`` (pixel coords are always
            normalised by ``W - 1`` / ``H - 1``).
        fill_unseen: Copy features from the nearest seen position into
            positions no view sees; ``False`` leaves them zero.

    Returns:
        ``[N, D]`` per-position aggregated features.
    """
    pos = positions.detach().to(device).float()
    N = pos.shape[0]
    D = features_2d[0].shape[0]

    feat_accum = torch.zeros(N, D, device=device)
    weight_accum = torch.zeros(N, device=device)

    n_views = len(viewmats)
    for v_idx in range(n_views):
        viewmat = viewmats[v_idx].to(device)
        K = Ks[v_idx].to(device)
        feat_map = features_2d[v_idx].to(device)
        D_feat, H_feat, W_feat = feat_map.shape

        uv, _depth, in_frustum = _project_points_pinhole(
            pos, viewmat, K, W, H, near_plane, far_plane,
        )
        if not in_frustum.any():
            continue

        valid_idx = in_frustum.nonzero(as_tuple=True)[0]
        proj_xy = uv[valid_idx]

        # grid_sample uses [-1, 1] normalised pixel coords.
        grid_x = (proj_xy[:, 0] / (W - 1)) * 2 - 1
        grid_y = (proj_xy[:, 1] / (H - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).clamp(-1, 1)

        grid_4d = grid.unsqueeze(0).unsqueeze(0)
        feat_map_4d = feat_map.unsqueeze(0)
        sampled = F.grid_sample(
            feat_map_4d, grid_4d, mode="bilinear",
            padding_mode="border", align_corners=align_corners,
        ).squeeze(0).squeeze(1).t()  # [M, D]

        if distance_alpha > 0:
            # Camera origin in world space: c = -R^T t for viewmat = [R | t].
            R = viewmat[:3, :3]
            t = viewmat[:3, 3]
            cam_origin = -(R.T @ t)
            dists = (pos[valid_idx] - cam_origin.unsqueeze(0)).norm(dim=-1)
            w = torch.exp(-distance_alpha * dists)
        else:
            w = torch.ones(valid_idx.shape[0], device=device)

        feat_accum[valid_idx] += sampled * w.unsqueeze(-1)
        weight_accum[valid_idx] += w

        if (v_idx + 1) % 10 == 0 or v_idx == 0 or v_idx == n_views - 1:
            print(
                f"    [{v_idx+1}/{n_views}] {int(in_frustum.sum().item()):,} valid positions"
            )

    mask = weight_accum > 0
    feat_accum[mask] /= weight_accum[mask].unsqueeze(-1)
    n_unseen = int((~mask).sum().item())
    if n_unseen > 0 and not fill_unseen:
        print(f"  Warning: {n_unseen:,} positions unseen in any view; features left zero")
    elif n_unseen > 0:
        n_seen = int(mask.sum().item())
        if n_seen == 0:
            print(
                f"  Warning: ALL {n_unseen:,} positions unseen in any view; "
                "features remain zero"
            )
        else:
            seen_pos = pos[mask]
            seen_feat = feat_accum[mask]
            unseen_idx = (~mask).nonzero(as_tuple=True)[0]
            # Chunk over the (typically small) unseen set so the
            # cdist intermediate stays bounded for large N_query.
            chunk = 2048
            for start in range(0, unseen_idx.shape[0], chunk):
                end = min(start + chunk, unseen_idx.shape[0])
                rows = unseen_idx[start:end]
                d = torch.cdist(pos[rows], seen_pos)
                nearest = d.argmin(dim=-1)
                feat_accum[rows] = seen_feat[nearest]
            print(
                f"  Filled {n_unseen:,} unseen positions from "
                f"nearest-seen-neighbour features"
            )
    return feat_accum


# ---------------------------------------------------------------------------
# PCA on aggregated features (used by direct path)
# ---------------------------------------------------------------------------


def pca_reduce(
    features: torch.Tensor,
    n_components: int,
) -> torch.Tensor:
    """Fit PCA on ``features [N, D]`` and project to ``n_components``.

    Returns ``[N, n_components]`` (float32).  ``n_components <= 0`` returns
    the input unchanged.
    """
    if n_components <= 0 or n_components >= features.shape[-1]:
        return features
    X = features.detach().cpu().numpy().astype(np.float32)
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(X)
    print(
        f"  PCA explained variance ({n_components}/{features.shape[-1]} comps): "
        f"{pca.explained_variance_ratio_.sum():.1%}"
    )
    return torch.from_numpy(proj.astype(np.float32)).to(features.device)
