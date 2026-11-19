"""Per-instance 2D flow maps for wind-animated CodebookModel renders.

Opacity-weighted instance centroids are tracked per frame, projected into a
fixed camera with the same ``K·(x, y, z)/z`` pinhole convention as ``gsplat``
and each instance is coloured by its image-plane displacement in an
optical-flow-style field: **white = no motion**, hue = 2D direction,
saturation grows with speed. Instances behind the camera are drawn black.
These are the flow-map panels of the paper's animation figure.

Outputs written by :func:`save_instance_motion_diagnostics` under ``out_dir``:

* ``instance_flow.mp4`` — flow ``t → t + stride`` for every ``step``-th frame.
* ``reference_instance_flow/from_first_frame/flow_ref_0000_to_XXXX.png`` —
  flow from frame 0 to frames ``k, 2k, …`` and the last frame, plus
  ``from_first_frame.mp4``.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image

Z_NEAR = 0.02
# Hue = atan2(dv, du) + 60° in image coordinates (u right, v down): +u → 60°
# (yellow), +v → 150° (green), −u → 240° (blue), −v → 330° (pink).
FLOW_WHEEL_H0_FRAC = 60.0 / 360.0
# Treat sub-pixel centroid drift as no visible flow.
FLOW_ZERO_EPS = 1e-3


def project_world_to_pixels(
    points_world: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D world → 2D pixel coordinates (``gsplat``-compatible pinhole).

    ``viewmat`` is world-to-camera ``[4, 4]`` (or ``[1, 4, 4]``), ``K`` is
    ``[3, 3]`` (or ``[1, 3, 3]``).

    Returns:
        pixels: ``[..., 2]``; NaN where the point is behind the camera.
        valid:  ``[...]`` bool, True iff in front of the camera.
    """
    lead = points_world.shape[:-1]
    pw = points_world.reshape(-1, 3)
    vm = viewmat[0] if viewmat.dim() == 3 else viewmat
    k = K[0] if K.dim() == 3 else K
    pc = pw @ vm[:3, :3].T + vm[:3, 3]
    z = pc[:, 2]
    valid = z > Z_NEAR
    uv = (pc @ k[:2, :3].T) / z.clamp(min=1e-6).unsqueeze(-1)
    uv = torch.where(valid.unsqueeze(-1), uv, torch.full_like(uv, float("nan")))
    return uv.view(*lead, 2), valid.view(*lead)


def instance_flow2d_to_rgb(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Optical-flow-style ``[N, 3]`` RGB in [0, 1] for per-instance flow.

    White = ``|v| ≈ 0``, hue = ``atan2(dy, dx) + 60°`` in image coordinates
    (+x right, +y down; see :data:`FLOW_WHEEL_H0_FRAC`), saturation grows
    with ``|v|`` against a robust percentile scale. Non-finite flow → black.
    """
    N = dx.shape[0]
    mag = np.hypot(dx, dy)
    fin = np.isfinite(mag)
    mags = mag[fin]
    if mags.size == 0:
        return np.zeros((N, 3), dtype=np.float32)

    p50 = float(np.percentile(mags, 50))
    p90 = float(np.percentile(mags, 90))
    p99 = float(np.percentile(mags, 99))
    if p99 < FLOW_ZERO_EPS:
        out = np.ones((N, 3), dtype=np.float32)
        out[~fin] = 0.0
        return out

    # Robust saturation scale: ignore the top 1% but never let a few fast
    # instances wash out a mostly-slow field.
    hi = min(p99, max(6.0 * max(p50, 1e-5), 2.0 * p90, 1e-3))

    dx_s = np.where(fin, dx, 0.0).astype(np.float64)
    dy_s = np.where(fin, dy, 0.0).astype(np.float64)
    h = np.mod(np.arctan2(dy_s, dx_s) / (2.0 * np.pi) + FLOW_WHEEL_H0_FRAC, 1.0)
    s = np.clip(np.where(fin, mag, 0.0) / hi, 0.0, 1.0) ** 0.42
    s[mag <= FLOW_ZERO_EPS] = 0.0
    out = mcolors.hsv_to_rgb(np.stack([h, s, np.ones(N)], axis=1)).astype(np.float32)
    out[~fin] = 0.0
    return out


@torch.no_grad()
def _rasterize_instance_colors(
    model, camera, cached_inst_idx: torch.Tensor, inst_colors: torch.Tensor,
) -> np.ndarray:
    """Render the (un-animated) codebook with every splat in its instance's colour."""
    from scion.util.vis import extract_camera_params, rasterize_gaussians

    bundle = model._get_assembled()
    device = bundle.means.device
    splat_colors = inst_colors.to(device)[cached_inst_idx]
    viewmat, K_mat, W, H = extract_camera_params(
        camera, model._get_downscale_factor(), device,
    )
    bg = model._get_background_color().to(device)
    render, alpha, _ = rasterize_gaussians(
        bundle.means, bundle.quats, bundle.scales, bundle.opacities,
        splat_colors, viewmat, K_mat, W, H,
        rasterize_mode=model.config.rasterize_mode,
    )
    out = (render.squeeze(0) + (1 - alpha.squeeze(0)) * bg).clamp(0, 1)
    return (out.cpu().numpy() * 255).astype(np.uint8)


def _flow_image(model, camera, cached_inst_idx, dxy: torch.Tensor) -> np.ndarray:
    dxy = dxy.cpu().numpy()
    rgb = instance_flow2d_to_rgb(dxy[:, 0], dxy[:, 1])
    return _rasterize_instance_colors(
        model, camera, cached_inst_idx, torch.from_numpy(rgb),
    )


def save_instance_motion_diagnostics(
    model,
    camera,
    cached_inst_idx: torch.Tensor,
    inst_traj_3d: torch.Tensor,
    out_dir: Path,
    fps: float,
    flow_frame_stride: int = 1,
    flow_frame_step: int = 1,
    reference_flow_every: int = 10,
) -> None:
    """Write per-frame and first-frame-reference instance flow maps.

    Args:
        model: CodebookModel (rendered un-animated, coloured by flow).
        camera: the fixed ``scion.Camera`` the animation was rendered from.
        cached_inst_idx: ``[T]`` splat → instance index.
        inst_traj_3d: ``[N, F, 3]`` opacity-weighted instance centroids per frame.
        out_dir: output directory.
        fps: frame rate of the written flow videos.
        flow_frame_stride / flow_frame_step: ``instance_flow.mp4`` shows flow
            ``t → t + stride`` for ``t = 0, step, 2·step, …``.
        reference_flow_every: reference maps ``0 → k, 2k, …`` (+ last frame).
    """
    from scion.util.animate_codebook import write_video
    from scion.util.vis import extract_camera_params

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = inst_traj_3d.device
    viewmat, K, _, _ = extract_camera_params(
        camera, model._get_downscale_factor(), device,
    )
    N, T, _ = inst_traj_3d.shape
    uv = project_world_to_pixels(inst_traj_3d.reshape(-1, 3), viewmat, K)[0]
    uv = uv.view(N, T, 2)

    stride = max(1, int(flow_frame_stride))
    step = max(1, int(flow_frame_step))
    frames: List[np.ndarray] = [
        _flow_image(model, camera, cached_inst_idx, uv[:, t + stride] - uv[:, t])
        for t in range(0, T - stride, step)
    ]
    if frames:
        path = out_dir / "instance_flow.mp4"
        write_video(frames, path, fps=int(round(fps)))
        print(f"  motion: {len(frames)} instance-flow frames (stride={stride}) → {path}")
    else:
        print(f"  motion: instance flow skipped (T={T} <= stride={stride})")

    if T < 2:
        print(f"  motion: reference flow skipped (T={T} < 2)")
        return
    ref_dir = out_dir / "reference_instance_flow" / "from_first_frame"
    if ref_dir.exists():
        shutil.rmtree(ref_dir)  # drop maps from an earlier run with other frame indices
    ref_dir.mkdir(parents=True)
    k = max(1, int(reference_flow_every))
    targets = sorted(set(range(k, T, k)) | {T - 1})
    ref_frames: List[np.ndarray] = []
    for target in targets:
        img = _flow_image(model, camera, cached_inst_idx, uv[:, target] - uv[:, 0])
        Image.fromarray(img).save(ref_dir / f"flow_ref_0000_to_{target:04d}.png")
        ref_frames.append(img)
    write_video(ref_frames, ref_dir / "from_first_frame.mp4", fps=int(round(fps)))
    print(f"  motion: {len(targets)} reference flow map(s) (every {k} frames) → {ref_dir}/")
