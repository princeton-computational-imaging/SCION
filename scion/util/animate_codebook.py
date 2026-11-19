"""Codebook-specific wind-animation helpers.

These helpers apply ``scion.util.animate`` to :class:`CodebookModel`, where:

- Instances are free 3D points with *free* SO(3) rotations (no mesh).
- Each instance has its own up-axis given by ``R(instance_rotations) @ (0,0,1)``.
- Splats live in a flat ``[T_active, ...]`` buffer and are mapped to their
  instance via ``model._cached_inst_idx``.

All functions here are pure tensor ops — they never touch model state — so
they can be composed into a post-assembly mutation of a
:class:`scion.models.codebook_base.GaussianBundle`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from scion.util.animate import (
    AnimationConfig,
    WindAnimationState,
    WindDisplacement,
    compute_wind_displacement,
)
from scion.util.transformation import quat_mul, quat_rotate


STATIC_LABEL = -1  # reserved label for instances that should not animate


# ---------------------------------------------------------------------------
# Basic instance-level geometry helpers
# ---------------------------------------------------------------------------


def instance_up_axes(inst_rot: torch.Tensor) -> torch.Tensor:
    """Compute the per-instance "up" axis in world space.

    Args:
        inst_rot: ``[N, 4]`` unit quaternions (w, x, y, z).

    Returns:
        ``[N, 3]`` unit vectors ``R_i @ (0, 0, 1)`` — each instance's local
        +Z axis expressed in world coordinates.
    """
    N = inst_rot.shape[0]
    z_local = torch.zeros(N, 3, device=inst_rot.device, dtype=inst_rot.dtype)
    z_local[:, 2] = 1.0
    up = quat_rotate(inst_rot, z_local)
    return up / up.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def instance_heights_along_up(
    means: torch.Tensor,
    cached_inst_idx: torch.Tensor,
    inst_pos: torch.Tensor,
    up_axes: torch.Tensor,
) -> torch.Tensor:
    """Per-instance blade length along each instance's up-axis.

    Projects each splat's offset ``(means_s - inst_pos_i)`` onto ``up_i``,
    clamps to non-negative, and takes the max per instance.

    Args:
        means: ``[T, 3]`` world-space splat means.
        cached_inst_idx: ``[T]`` per-splat instance index.
        inst_pos: ``[N, 3]`` instance centers.
        up_axes: ``[N, 3]`` per-instance up axes.

    Returns:
        ``[N]`` tensor of heights. Instances with no splats get 0.
    """
    N = inst_pos.shape[0]
    if means.shape[0] == 0:
        return torch.zeros(N, device=means.device, dtype=means.dtype)

    offsets = means - inst_pos[cached_inst_idx]           # [T, 3]
    up_per_splat = up_axes[cached_inst_idx]               # [T, 3]
    z_local = (offsets * up_per_splat).sum(dim=-1).clamp_min(0.0)  # [T]

    heights = torch.zeros(N, device=means.device, dtype=means.dtype)
    heights.scatter_reduce_(
        0, cached_inst_idx, z_local, reduce="amax", include_self=True,
    )
    return heights


# ---------------------------------------------------------------------------
# Feature aggregation for DINOtxt classification
# ---------------------------------------------------------------------------


def aggregate_splat_features_to_instances(
    splat_feats: torch.Tensor,
    cached_inst_idx: torch.Tensor,
    splat_opacities_sigmoid: torch.Tensor,
    num_instances: int,
) -> torch.Tensor:
    """Opacity-weighted mean of per-splat features, grouped by instance.

    Args:
        splat_feats: ``[T, D]`` per-splat features.
        cached_inst_idx: ``[T]`` per-splat instance index.
        splat_opacities_sigmoid: ``[T]`` or ``[T, 1]`` per-splat opacity in
            ``[0, 1]``. Must be non-negative and sum to >0 per instance for
            a meaningful mean; instances with zero aggregate weight get all
            zeros.
        num_instances: ``N``, size of the output tensor.

    Returns:
        ``[N, D]`` per-instance feature tensor.
    """
    if splat_opacities_sigmoid.dim() > 1:
        splat_opacities_sigmoid = splat_opacities_sigmoid.squeeze(-1)
    T, D = splat_feats.shape
    device = splat_feats.device
    dtype = splat_feats.dtype

    weighted = splat_feats * splat_opacities_sigmoid[:, None]

    sum_feats = torch.zeros(num_instances, D, device=device, dtype=dtype)
    sum_feats.index_add_(0, cached_inst_idx, weighted)

    sum_w = torch.zeros(num_instances, device=device, dtype=dtype)
    sum_w.index_add_(0, cached_inst_idx, splat_opacities_sigmoid)

    mask = sum_w > 1e-8
    out = torch.zeros_like(sum_feats)
    out[mask] = sum_feats[mask] / sum_w[mask, None]
    return out


# ---------------------------------------------------------------------------
# Per-group delta quaternions
# ---------------------------------------------------------------------------


def _axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Axis-angle → quaternion (w, x, y, z). Accepts arbitrary leading dims."""
    half = angle * 0.5
    s = torch.sin(half)
    return torch.stack(
        [torch.cos(half), axis[..., 0] * s, axis[..., 1] * s, axis[..., 2] * s],
        dim=-1,
    )


def _bend_axes(
    up_axes: torch.Tensor,
    wind_dir_world: torch.Tensor,
    bend_axis_mode: str,
) -> torch.Tensor:
    """Per-instance unit rotation axis = normalize(wind × up_ref).

    Used by the per-splat rotation in :func:`apply_wind_to_bundle`.
    """
    device = up_axes.device
    dtype = up_axes.dtype
    N = up_axes.shape[0]

    wind_xy = wind_dir_world.to(device=device, dtype=dtype).flatten()
    wind3 = torch.stack(
        [wind_xy[0], wind_xy[1], torch.zeros((), device=device, dtype=dtype)]
    )
    wind3 = (wind3 / wind3.norm().clamp(min=1e-8)).unsqueeze(0).expand(N, 3)

    if bend_axis_mode == "world_z":
        up_ref = torch.zeros(N, 3, device=device, dtype=dtype)
        up_ref[:, 2] = 1.0
    elif bend_axis_mode == "instance_local":
        up_ref = up_axes
    else:
        raise ValueError(
            f"bend_axis_mode must be instance_local|world_z, got {bend_axis_mode!r}"
        )

    axis = torch.cross(wind3, up_ref, dim=-1)
    axis_n = axis.norm(dim=-1, keepdim=True)
    fb = torch.zeros_like(axis); fb[:, 0] = 1.0
    return torch.where(axis_n > 1e-6, axis / axis_n.clamp(min=1e-8), fb)


# ---------------------------------------------------------------------------
# Post-assembly mutation
# ---------------------------------------------------------------------------


@dataclass
class GroupAnimationSpec:
    """Per-group wind configuration resolved from YAML.

    ``size_impact_alpha`` scales the per-splat wind angle with world-space
    splat size: ``weight = clamp(splat_size / median_size, size_impact_min,
    size_impact_max) ** alpha``. The clamp happens before the power, so the
    weight spans ``[size_impact_min ** alpha, size_impact_max ** alpha]``
    (``leaves.yaml``: up to ``4 ** 1.5 = 8``). ``0.0`` is a no-op (every splat
    in an instance moves together); tune per scene.

    ``pivot_mode``:
      - ``"base"`` (default): pivot at ``(inst_xy, min_splat_z)`` — rotation
        axis stays in the horizontal plane at the cluster base, so bending
        produces mostly lateral motion, mimicking a rooted blade/branch.
      - ``"center"``: pivot at ``instance_positions[i]`` (cluster centroid).
        Produces the "scissoring" look where splats above/below the centroid
        move in opposite directions on each cycle — more vertical bob.

    ``pivot_down_offset`` subtracts a fixed world-Z value from the resolved
    pivot for every animated instance.
    """
    label: str
    animation_config: AnimationConfig
    bend_axis_mode: str = "instance_local"   # "instance_local" | "world_z"
    per_splat_flutter: bool = False
    size_impact_alpha: float = 0.0           # 0 = per-splat scaling off
    size_impact_min: float = 0.2
    size_impact_max: float = 5.0
    pivot_mode: str = "base"                 # "base" | "center"
    pivot_down_offset: float = 0.0


def apply_wind_to_bundle(
    bundle,                                  # GaussianBundle — mutated in place
    cached_inst_idx: torch.Tensor,
    inst_pos: torch.Tensor,                  # [N, 3]
    inst_rot: torch.Tensor,                  # [N, 4]
    inst_heights: torch.Tensor,              # [N]
    inst_log_scales: torch.Tensor,           # [N] or [N, 1]
    labels: torch.Tensor,                    # [N] long, -1 = STATIC
    specs_by_label: Dict[int, GroupAnimationSpec],
    states_by_label: Dict[int, Optional[WindAnimationState]],
    t: float,
    dt: float,
) -> None:
    """Mutate ``bundle.means`` / ``bundle.quats`` in place with per-group wind.

    For each non-STATIC group ``g``:
      1. Compute a :class:`WindDisplacement` for that group's instances.
      2. Build ``Δq`` from ``primary_angle`` + bend axis.
      3. Apply rigid rotation around each instance's pivot to its splats.
      4. Optionally apply per-splat height-falloff flutter on top.

    The caller owns the ``states_by_label`` dict; this function writes back
    the new state (identity → ``None`` stays ``None``).

    Bundle tensors must already be cloned by the caller if the originals
    need to be restored.
    """
    if labels.numel() == 0:
        return

    device = bundle.means.device
    N = inst_pos.shape[0]

    up_axes_all = instance_up_axes(inst_rot)     # [N, 3]

    means_w = bundle.means
    quats_w = bundle.quats

    for group_id, spec in specs_by_label.items():
        if group_id == STATIC_LABEL:
            continue
        member_mask = labels == group_id
        if not member_mask.any():
            continue
        member_idx = member_mask.nonzero(as_tuple=True)[0]   # [M]

        xy_g = inst_pos[member_idx, :2]
        heights_g = inst_heights[member_idx]
        scales_g = inst_log_scales[member_idx]
        if scales_g.dim() > 1:
            scales_g = scales_g.squeeze(-1)

        cfg = spec.animation_config
        state = states_by_label.get(group_id)
        wind, new_state = compute_wind_displacement(
            xy_g, t, cfg,
            heights=heights_g,
            scales=scales_g,
            state=state,
            dt=dt,
        )
        states_by_label[group_id] = new_state

        up_g = up_axes_all[member_idx]
        wind_dir_world = torch.tensor(
            [cfg.wind_dir_normalized[0], cfg.wind_dir_normalized[1], 0.0],
            device=device, dtype=means_w.dtype,
        )
        axis_per_inst = _bend_axes(up_g, wind_dir_world, spec.bend_axis_mode)
        angle_per_inst = wind.primary_angle                     # [M]

        # Which splats belong to this group?
        splat_member = member_mask[cached_inst_idx]     # [T]
        if not splat_member.any():
            continue
        splat_idx = splat_member.nonzero(as_tuple=True)[0]

        inst_for_splat = cached_inst_idx[splat_idx]     # [Tg] global instance ids
        # Map global instance id -> local row in the group's [M]-slab.
        # member_idx is sorted (comes from .nonzero); searchsorted is safe.
        local_of_splat = torch.searchsorted(member_idx, inst_for_splat)

        # --- Pivot: "base" = (inst_xy, min splat z per instance); "center" = inst_pos
        old_means = means_w[splat_idx]
        if spec.pivot_mode == "base":
            min_z_per_inst = torch.full(
                (member_idx.shape[0],), float("inf"),
                device=device, dtype=means_w.dtype,
            )
            min_z_per_inst.scatter_reduce_(
                0, local_of_splat, old_means[:, 2],
                reduce="amin", include_self=True,
            )
            # Fall back to centroid z if no splats (shouldn't happen here).
            min_z_per_inst = torch.where(
                torch.isfinite(min_z_per_inst),
                min_z_per_inst,
                inst_pos[member_idx, 2],
            )
            pivot_per_inst = inst_pos[member_idx].clone()
            pivot_per_inst[:, 2] = min_z_per_inst
        else:  # "center"
            pivot_per_inst = inst_pos[member_idx]
        if spec.pivot_down_offset != 0.0:
            pivot_per_inst = pivot_per_inst.clone()
            pivot_per_inst[:, 2] = pivot_per_inst[:, 2] - float(spec.pivot_down_offset)
        pivot = pivot_per_inst[local_of_splat]           # [Tg, 3]

        # --- Per-splat angle = angle_per_inst * size_weight(splat)
        # World-space splat size: exp(log_scale).mean over the 3 axes.
        # Weight = clamp(size / median, min, max) ** alpha, i.e. in
        # [min ** alpha, max ** alpha]: larger splats sway more.
        world_scale = bundle.scales[splat_idx].detach().exp().mean(dim=-1).clamp(min=1e-8)
        if spec.size_impact_alpha > 0.0:
            ref_scale = world_scale.median().clamp(min=1e-8)
            size_w = (world_scale / ref_scale).clamp(
                min=spec.size_impact_min, max=spec.size_impact_max,
            ) ** spec.size_impact_alpha
        else:
            size_w = torch.ones_like(world_scale)
        angle_per_splat = angle_per_inst[local_of_splat] * size_w     # [Tg]

        # --- Build per-splat Δq around the per-instance axis
        axis_per_splat = axis_per_inst[local_of_splat]                # [Tg, 3]
        dq = _axis_angle_to_quat(axis_per_splat, angle_per_splat)
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        new_means = pivot + quat_rotate(dq, old_means - pivot)
        means_w[splat_idx] = new_means

        old_quats = quats_w[splat_idx]
        new_quats = quat_mul(dq, old_quats)
        new_quats = new_quats / new_quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        quats_w[splat_idx] = new_quats

        # Optional per-splat tip flutter around the same per-instance axis,
        # with z^1.5 height falloff so tips flutter more than bases. Primary
        # bend was already applied above; flutter composes on top.
        if spec.per_splat_flutter:
            offsets_g = means_w[splat_idx] - pivot
            up_for_splat = up_axes_all[inst_for_splat]
            local_z = (offsets_g * up_for_splat).sum(dim=-1).clamp_min(0.0)

            h_max = torch.zeros(
                member_idx.shape[0], device=device, dtype=means_w.dtype,
            )
            h_max.scatter_reduce_(
                0, local_of_splat, local_z,
                reduce="amax", include_self=True,
            )
            z_norm = local_z / h_max[local_of_splat].clamp(min=1e-6)
            flutter_weight = z_norm ** 1.5
            flutter_per_splat = wind.flutter_angle[local_of_splat] * flutter_weight

            q_flutter = _axis_angle_to_quat(axis_per_splat, flutter_per_splat)
            q_flutter = q_flutter / q_flutter.norm(
                dim=-1, keepdim=True,
            ).clamp(min=1e-8)

            means_w[splat_idx] = pivot + quat_rotate(
                q_flutter, means_w[splat_idx] - pivot,
            )
            q_combined = quat_mul(q_flutter, quats_w[splat_idx])
            quats_w[splat_idx] = q_combined / q_combined.norm(
                dim=-1, keepdim=True,
            ).clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Shared I/O helpers
# ---------------------------------------------------------------------------


def write_video(
    frames: List["np.ndarray"],
    output_path: "Path",
    fps: int,
) -> None:
    """Write frames to ``output_path`` with a fallback chain.

    Order: ``mediapy`` -> ``imageio.v3`` -> PNG sequence. Errors only bubble
    up from the final fallback.
    """
    import numpy as np

    try:
        import mediapy as media
        media.write_video(str(output_path), frames, fps=fps)
        return
    except Exception as e_mp:  # noqa: BLE001 — any import/runtime issue falls through
        print(f"  [video] mediapy failed ({e_mp}); trying imageio.")

    try:
        import imageio.v3 as iio
        iio.imwrite(
            str(output_path),
            np.stack(frames),
            fps=fps,
            codec="libx264",
        )
        return
    except Exception as e_iio:  # noqa: BLE001
        print(f"  [video] imageio failed ({e_iio}); dumping PNG sequence.")

    from PIL import Image
    frames_dir = output_path.with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frames_dir / f"frame_{i:05d}.png")
    print(f"  [video] saved {len(frames)} frames to {frames_dir}/")
