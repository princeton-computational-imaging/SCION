"""A real (tiny) CodebookModel + one-camera Scene that trains on CPU.

gsplat is CUDA-only, so :func:`fake_rasterize` stands in for
``scion.rasterizer.rasterize``: a differentiable soft splat of every
Gaussian parameter onto a small image. Used by the trainer tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scion.camera import Camera
from scion.data.scene import Scene
from scion.losses.discriminator import DiscriminatorConfig
from scion.models.codebook import CodebookModel, CodebookModelConfig

W, H = 64, 48
LRS = {"splat_means": 1e-3, "splat_features_dc": 1e-2, "splat_opacities": 1e-2,
       "splat_scales": 1e-2, "splat_quats": 1e-3, "instance_positions": 1e-3,
       "instance_rotations": 1e-3, "instance_scales": 1e-3, "instance_opacities": 1e-2}


def fake_rasterize(*, means, quats, scales_log, opacities_logit, colors, viewmat, K,
                   width, height, background, **_):
    cam = means @ viewmat[0, :3, :3].T + viewmat[0, :3, 3]
    z = cam[:, 2:3].clamp(min=0.1)
    uv = (cam[:, :2] / z) @ K[0, :2, :2].T + K[0, :2, 2]
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 1, 2).to(means)
    sigma = torch.exp(scales_log).mean(-1) * K[0, 0, 0] / z[:, 0] + 1.0
    w = torch.sigmoid(opacities_logit[:, 0]) * torch.exp(-((pix - uv) ** 2).sum(-1) / (2 * sigma ** 2))
    w = w * (quats.norm(dim=-1) / quats.norm(dim=-1).detach())  # unit value, live gradient
    alpha = 1 - torch.exp(-w.sum(-1, keepdim=True))
    rgb = (w @ colors) / (w.sum(-1, keepdim=True) + 1e-6) * alpha + (1 - alpha) * background
    return (rgb.clamp(0, 1).reshape(height, width, 3), alpha.reshape(height, width, 1), None,
            background, {"means2d": uv[None]})


def tiny_config() -> CodebookModelConfig:
    return CodebookModelConfig(
        num_candidates=4, max_candidates=4, initial_splats_per_candidate=2,
        max_splats_per_candidate=4, num_instances=-1, max_num_instances=1000, sh_degree=0,
        background_color="white", init_assignment_mode="random_uniform", num_downscales=0,
        lpips_lambda=0.05, lpips_every=1,
        discriminator=DiscriminatorConfig(enabled=True, warmup_step=0, update_freq=1,
                                          patch_size=32, num_patches=2, loss_weight=0.01),
    )


def tiny_model(seed: int = 0) -> CodebookModel:
    g = torch.Generator().manual_seed(seed)
    xyz = torch.cat([torch.rand(48, 2, generator=g) * 2 - 1,
                     -1.5 - torch.rand(48, 1, generator=g)], dim=1)
    rgb = (torch.rand(48, 3, generator=g) * 255).to(torch.uint8)
    torch.manual_seed(seed)
    return CodebookModel(tiny_config(), seed_points=(xyz, rgb))


def tiny_scene(tmp_path: Path, n: int = 1) -> Scene:
    """``n`` copies of one camera + image (distinct indices, so camera order is observable)."""
    arr = (np.random.default_rng(0).random((H, W, 3)) * 255).astype(np.uint8)
    path = Path(tmp_path) / "img_000.png"
    Image.fromarray(arr).save(path)
    K = torch.tensor([[60.0, 0.0, W / 2], [0.0, 60.0, H / 2], [0.0, 0.0, 1.0]])
    cam = Camera(cam_to_world=torch.eye(4)[:3], K=K, width=W, height=H)
    return Scene(cameras=[cam] * n, image_paths=[path] * n, train_indices=list(range(n)),
                 eval_indices=[0])
