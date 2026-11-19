import math
from typing import List, Tuple

import torch

from gsplat.rendering import rasterization


def setup_side_view_camera(
    min_bounds: torch.Tensor,
    max_bounds: torch.Tensor,
    resolution: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Set up orthographic-like camera looking along Y-axis (side view showing XZ plane).

    Returns:
        viewmat: [4, 4] view matrix
        K: [3, 3] intrinsic matrix
        width: image width
        height: image height
    """
    # Scene center and extent
    center = (min_bounds + max_bounds) / 2
    extent = max_bounds - min_bounds

    # For side view (XZ plane), we care about X and Z extents
    # Make square by using max of X and Z
    scene_size = max(extent[0].item(), extent[2].item())

    # Add some padding
    scene_size *= 1.1

    # Camera positioned along negative Y axis, looking at center
    # Distance should be far enough for near-orthographic projection
    cam_distance = scene_size * 5.0

    cam_pos = torch.tensor([center[0].item(), center[1].item() - cam_distance, center[2].item()], device=device)

    # View matrix: camera looks along +Y
    # Right = +X, Up = +Z, Forward = +Y
    R = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],  # Y -> -Z in camera space
        [0.0, 1.0, 0.0],   # Z -> Y in camera space
    ], device=device)

    t = -R @ cam_pos

    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t

    # Intrinsic matrix for near-orthographic projection
    # Use a narrow FOV to approximate orthographic
    fov = 2 * math.atan(scene_size / (2 * cam_distance))
    focal = resolution / (2 * math.tan(fov / 2))

    K = torch.tensor([
        [focal, 0.0, resolution / 2],
        [0.0, focal, resolution / 2],
        [0.0, 0.0, 1.0],
    ], device=device)

    return viewmat, K, resolution, resolution


def render_single_candidate(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    Render a single candidate using gsplat.

    Args:
        means: [N, 3] splat positions
        scales: [N, 3] splat scales (log-space)
        quats: [N, 4] splat quaternions
        colors: [N, 3] splat colors
        opacities: [N, 1] splat opacities (logit-space)
        viewmat: [4, 4] view matrix
        K: [3, 3] intrinsic matrix
        width: image width
        height: image height
        background_color: RGB tuple for background

    Returns:
        [H, W, 3] rendered image
    """
    device = means.device
    # Convert from log/logit space
    scales_exp = torch.exp(scales)
    opacities_sig = torch.sigmoid(opacities).squeeze(-1)

    # Prepare for gsplat (expects batch dimension)
    means = means.contiguous()
    scales_exp = scales_exp.contiguous()
    quats = quats.contiguous()

    # Normalize quaternions
    quats = quats / quats.norm(dim=-1, keepdim=True)

    # gsplat takes colors as [N, 3] in RGB mode
    colors = colors.contiguous()

    # Background
    bg = torch.as_tensor(background_color, device=device, dtype=torch.float32)

    # Render using gsplat
    render, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales_exp,
        opacities=opacities_sig,
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        width=width,
        height=height,
        packed=False,
        render_mode="RGB",
    )

    alpha = alpha[:, ...]

    rgb = render[:, ..., :3] + (1 - alpha) * bg
    rgb = torch.clamp(rgb, 0.0, 1.0)

    # renders: [1, H, W, 3]
    return rgb.squeeze(0)


# ---------------------------------------------------------------------------
# Codebook visualization helpers
# ---------------------------------------------------------------------------


def index_to_color(indices: torch.Tensor, max_val: int) -> torch.Tensor:
    """Map integer indices [N] to RGB [N, 3] via HSV colormap."""
    hue = (indices.float() / max(max_val, 1)) % 1.0
    h6 = hue * 6.0
    c = torch.ones_like(hue)
    x = c * (1 - (h6 % 2 - 1).abs())
    z = torch.zeros_like(hue)
    rgb = torch.zeros(indices.shape[0], 3, device=indices.device)
    m0 = h6 < 1
    m1 = (h6 >= 1) & (h6 < 2)
    m2 = (h6 >= 2) & (h6 < 3)
    m3 = (h6 >= 3) & (h6 < 4)
    m4 = (h6 >= 4) & (h6 < 5)
    m5 = h6 >= 5
    rgb[m0] = torch.stack([c[m0], x[m0], z[m0]], -1)
    rgb[m1] = torch.stack([x[m1], c[m1], z[m1]], -1)
    rgb[m2] = torch.stack([z[m2], c[m2], x[m2]], -1)
    rgb[m3] = torch.stack([z[m3], x[m3], c[m3]], -1)
    rgb[m4] = torch.stack([x[m4], z[m4], c[m4]], -1)
    rgb[m5] = torch.stack([c[m5], z[m5], x[m5]], -1)
    return rgb


def tile_images_into_grid(
    images: List[torch.Tensor],
    cell_h: int,
    cell_w: int,
    bg_value: float = 1.0,
) -> torch.Tensor:
    """Tile a list of [H, W, 3] images into a square-ish grid.

    Args:
        images: List of [cell_h, cell_w, 3] tensors (all same device).
        cell_h: Height of each cell.
        cell_w: Width of each cell.
        bg_value: Fill value for empty cells (default white).

    Returns:
        [nrows * cell_h, ncols * cell_w, 3] tensor.
    """
    n = len(images)
    if n == 0:
        return torch.full((cell_h, cell_w, 3), bg_value)
    ncols = max(1, int(math.ceil(math.sqrt(n))))
    nrows = max(1, int(math.ceil(n / ncols)))
    device = images[0].device
    grid = torch.full((nrows * cell_h, ncols * cell_w, 3), bg_value, device=device)
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = img
    return grid


def extract_camera_params(
    camera,
    downscale_factor: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """``(viewmat [1,4,4], K [1,3,3], W, H)`` of a :class:`scion.Camera`
    downscaled by ``downscale_factor`` (floored sizes, as in
    ``CodebookModel.get_outputs``)."""
    viewmat = camera.viewmat().to(device).unsqueeze(0)
    scale = 1.0 / float(downscale_factor)
    W = int(camera.width * scale)
    H = int(camera.height * scale)
    K_mat = camera.K.to(device).clone()
    K_mat[0, 0] *= scale
    K_mat[1, 1] *= scale
    K_mat[0, 2] *= scale
    K_mat[1, 2] *= scale
    return viewmat, K_mat.unsqueeze(0), W, H


def rasterize_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales_log: torch.Tensor,
    opacities_logit: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    W: int,
    H: int,
    rasterize_mode: str = "classic",
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Rasterize Gaussian splats with standard default parameters.

    Convenience wrapper around gsplat.rasterization with the default
    near/far planes, packed=False, RGB mode, and no SH/sparse/abs grads.

    Converts scales from log-space and opacities from logit-space internally.

    Args:
        means: [N, 3] world-space positions.
        quats: [N, 4] quaternions.
        scales_log: [N, 3] log-space scales.
        opacities_logit: [N, 1] logit-space opacities.
        colors: [N, 3] RGB colors.
        viewmats: [B, 4, 4] view matrices.
        Ks: [B, 3, 3] intrinsic matrices.
        W: Image width.
        H: Image height.
        rasterize_mode: "classic" or "antialiased".

    Returns:
        (render, alpha, info) from gsplat.rasterization.
    """
    return rasterization(
        means=means,
        quats=quats,
        scales=torch.exp(scales_log),
        opacities=torch.sigmoid(opacities_logit).squeeze(-1),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        sh_degree=None,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode=rasterize_mode,
    )


