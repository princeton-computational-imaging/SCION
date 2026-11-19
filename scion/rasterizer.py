"""Public rasterization entry point for the SCION core.

A thin wrapper over :func:`gsplat.rasterization` that knows how to:
  * accept either SH coefficients (``[N, K, 3]``) or per-Gaussian RGB
    (``[N, 3]``) colors;
  * exponentiate scales and sigmoid-activate opacities (the
    rasterizer expects positive scales + ``[0, 1]`` opacities);
  * splat-composite onto a configurable background;
  * apply (optionally) a bilateral-grid post-process to the image plane.

This is the **only** call into ``gsplat`` from the core layer. Strategies
and visualisation helpers must funnel through this function (or its
private cousin in ``util.vis`` for the visualisation-only single-shot
path that doesn't need bilateral grid / SH).

See :meth:`CodebookModel.get_outputs` for the caller side.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from gsplat.rendering import rasterization


def rasterize(
    *,
    means: Tensor,                      # [N, 3] world-space positions
    quats: Tensor,                      # [N, 4] (unnormalized OK; gsplat normalizes)
    scales_log: Tensor,                 # [N, 3] LOG-space scales
    opacities_logit: Tensor,            # [N, 1] LOGIT-space opacities
    colors: Tensor,                     # [N, 3] sigmoid-RGB OR [N, K, 3] SH coeffs
    viewmat: Tensor,                    # [1, 4, 4] world-to-camera
    K: Tensor,                          # [1, 3, 3] intrinsics
    width: int,
    height: int,
    background: Tensor,                 # [3] color to composite under alpha
    sh_degree: Optional[int] = None,    # None = colors are RGB; else SH
    rasterize_mode: str = "classic",    # "classic" | "antialiased"
    render_mode: str = "RGB+ED",        # gsplat render_mode passthrough
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    absgrad: bool = True,
    packed: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Dict]:
    """Rasterize Gaussians and composite onto a background.

    Returns
    -------
    rgb : Tensor of shape ``(H, W, 3)``
        Composited image, clamped to ``[0, 1]``.
    alpha : Tensor of shape ``(H, W, 1)``
        Per-pixel accumulation / coverage from gsplat.
    depth : Optional[Tensor] of shape ``(H, W, 1)``
        Expected depth when ``render_mode`` includes ``"ED"``; else ``None``.
    background_full : Tensor of shape ``(3,)`` or ``(H, W, 3)``
        The background tensor used for compositing (broadcast-ready).
    info : Dict
        Raw gsplat info dict (means2d, radii, etc.) — strategies hook
        into ``info["means2d"]`` for screen-space gradient bookkeeping.
    """
    render, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=torch.exp(scales_log),
        opacities=torch.sigmoid(opacities_logit).squeeze(-1),
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode=render_mode,
        sh_degree=sh_degree,
        sparse_grad=False,
        absgrad=absgrad,
        rasterize_mode=rasterize_mode,
    )

    # `render` is [B, H, W, C]; with B=1 we squeeze later.  alpha is [B, H, W, 1].
    rgb = render[..., :3] + (1.0 - alpha) * background  # broadcast [1,H,W,3]
    rgb = torch.clamp(rgb, 0.0, 1.0)

    if render_mode in ("RGB+ED", "RGB+D", "D", "ED"):
        depth_im = render[..., 3:4]
        depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
    else:
        depth_im = None

    rgb = rgb.squeeze(0)
    alpha_out = alpha.squeeze(0)

    return rgb, alpha_out, depth_im, background, info
