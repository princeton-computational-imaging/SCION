"""Pure-PyTorch reference implementation of assemble_fused.

This mirrors CodebookMixin.assemble_gaussians but as a stand-alone function
whose signature matches the CUDA kernel's. Used as the fallback tier and
as the reference in the CUDA / Triton parity tests.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

from scion.util.transformation import (
    quat_mul_normalized,
    quat_rotate,
)


# Match _MIN_WORLD_SCALE_LOG in scion/models/codebook_base.py.
_DEFAULT_MIN_WORLD_SCALE_LOG = math.log(1e-4)


def assemble_fused(
    splat_means:       Tensor,
    splat_scales:      Tensor,
    splat_quats:       Tensor,
    splat_opacities:   Tensor,
    splat_features_dc: Tensor,
    inst_positions:    Tensor,
    inst_rotations:    Tensor,
    inst_scales:       Tensor,
    inst_opacities:    Tensor,
    cand_splat_rows:   Tensor,       # [total] int32
    inst_offsets:      Tensor,       # [N+1]  int32 (prefix sum)
    min_world_scale_log: float = _DEFAULT_MIN_WORLD_SCALE_LOG,
    max_world_scale_log: float = 5.0,
    opac_clamp:        float = 20.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pure-PyTorch reference of the ``assemble_fused`` kernel.

    Returns ``(world_means, world_scales, world_rot, world_opac, world_feat)``
    each shape ``[total, ...]`` contiguous fp32. The body mirrors
    ``CodebookMixin.assemble_gaussians`` from ``codebook_base.py`` -- in
    particular it uses ``util.transformation.quat_rotate`` (xyz part is
    ``q[..., 1:]``, scalar is ``q[..., 0]``) and ``quat_mul_normalized``
    so that this reference is bit-identical to the in-tree consumer.

    Args:
        splat_means: ``[T_max, 3]``.
        splat_scales: ``[T_max, 3]`` log-space.
        splat_quats: ``[T_max, 4]`` quaternions (wxyz).
        splat_opacities: ``[T_max, 1]`` logit-space.
        splat_features_dc: ``[T_max, 3]``.
        inst_positions: ``[N, 3]``.
        inst_rotations: ``[N, 4]`` quaternions (wxyz).
        inst_scales: ``[N, 1]`` log-space global scale per instance.
        inst_opacities: ``[N, 1]`` logit-space opacity bias.
        cand_splat_rows: ``[total]`` int (32 or 64) — row of the flat splat
            buffer used by each world-space Gaussian.
        inst_offsets: ``[N+1]`` int (32 or 64) prefix-sum into
            ``cand_splat_rows`` giving each instance's slice.
        min_world_scale_log / max_world_scale_log: clamp bounds for
            ``world_scales`` (log-space sum).
        opac_clamp: symmetric clamp for ``world_opac`` (logit space).
    """
    # ---- 1. Derive (N, total) and the (g -> inst_idx) map. ----
    N = inst_offsets.numel() - 1
    # cand_splat_rows has exactly inst_offsets[-1] entries by construction
    # (_rebuild_layout builds it from arange(total)), so its static length
    # stands in for the offset sum — no host sync, torch.compile-safe.
    total = cand_splat_rows.numel()
    device = cand_splat_rows.device

    if total == 0:
        # Empty bundle — match assemble_gaussians' empty path.
        return (
            torch.zeros((0, 3), device=device),
            torch.zeros((0, 3), device=device),
            torch.zeros((0, 4), device=device),
            torch.zeros((0, 1), device=device),
            torch.zeros((0, 3), device=device),
        )

    # sizes_per_inst = inst_offsets[1:] - inst_offsets[:-1]  (matches the
    # codebook_base.py path exactly). output_size pins the result shape to
    # the statically-known total so torch.compile need not guess it.
    sizes_per_inst = (inst_offsets[1:] - inst_offsets[:-1]).long()  # [N]
    inst_idx = torch.repeat_interleave(
        torch.arange(N, device=device, dtype=torch.long),
        sizes_per_inst,
        output_size=total,
    )                                                                # [total]

    splat_row = cand_splat_rows.long()                               # [total]

    # ---- 2. Gather splat fields. ----
    local_means   = splat_means[splat_row]      # [total, 3]
    local_scales  = splat_scales[splat_row]     # [total, 3]
    local_rot     = splat_quats[splat_row]      # [total, 4]
    local_opac    = splat_opacities[splat_row]  # [total, 1]
    local_feat_dc = splat_features_dc[splat_row]  # [total, 3]

    # ---- 3. Gather instance fields. ----
    pos = inst_positions[inst_idx]      # [total, 3]
    rot = inst_rotations[inst_idx]      # [total, 4]
    scl = inst_scales[inst_idx]         # [total, 1]
    opac = inst_opacities[inst_idx]     # [total, 1]

    # ---- 4. Compose. Must mirror codebook_base.py:assemble_gaussians. ----
    # world_means = pos + rotate(rot, local_means * exp(scl))
    world_means = pos + quat_rotate(rot, local_means * scl.exp())
    # world_scales = clamp(scl + local_scales, min, max)
    world_scales = (scl + local_scales).clamp(
        min=min_world_scale_log, max=max_world_scale_log
    )
    # world_rot = normalize(quat_mul(rot, local_rot))
    world_rot = quat_mul_normalized(rot, local_rot)
    # Match codebook_base.py's defensive renorm (clamp(min=1e-6) on the norm).
    world_rot = world_rot / world_rot.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # world_opac = clamp(local_opac + opac, -opac_clamp, opac_clamp)
    world_opac = (local_opac + opac).clamp(min=-opac_clamp, max=opac_clamp)

    # Instances share the candidate's colour.
    return world_means, world_scales, world_rot, world_opac, local_feat_dc
