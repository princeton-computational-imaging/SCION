"""Triton-tier ``assemble_fused``.

Forward only. The backward goes through the CUDA tier (or python fallback)
inside ``scion.ops.assemble.AssembleFusedFunction.backward``.

Convention (must match ``scion/ops/_ref/assemble.py`` and
``AssembleFused.cu``):

- Quaternions are stored as ``(w, x, y, z)``; Hamilton product.
- ``world_scales = clamp(iscl + ls, min_world_scale_log, max_world_scale_log)``
  (log-space sum).
- ``world_rot = normalize(quat_multiply(ir, lq))`` (Hamilton product, renorm
  with ``clamp(norm, min=1e-6)`` to match the reference's defensive renorm).
- ``world_means = ip + R(ir) * (lm * exp(iscl))`` where ``R(ir)`` is the
  standard 3×3 rotation matrix from quaternion ``ir``. This expression is
  algebraically identical (for any quaternion, unit or not) to the
  cross-product form ``v + 2(w * (qvec×v) + qvec×(qvec×v))`` that
  ``util.transformation.quat_rotate`` uses.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def _assemble_fused_kernel(
    # ---- Splat (local) buffers, all rank-1 over flat fp32 storage. ----
    splat_means_ptr, splat_scales_ptr, splat_quats_ptr,
    splat_opa_ptr, splat_feat_dc_ptr,
    # ---- Instance buffers. ----
    inst_pos_ptr, inst_rot_ptr, inst_scl_ptr, inst_opa_ptr,
    # ---- Layout. ----
    cand_splat_rows_ptr,                         # int32  [total]
    inst_idx_per_g_ptr,                          # int32  [total] -- precomputed host-side
    # ---- Outputs. ----
    world_means_ptr, world_scales_ptr,
    world_rot_ptr, world_opa_ptr, world_feat_ptr,
    # ---- Scalars / configuration. ----
    total,
    BLOCK: tl.constexpr,
    min_scale_log, max_scale_log, opac_clamp,
):
    """One thread per world Gaussian; block-tile size = BLOCK.

    The kernel reads ``inst_idx_per_g`` directly instead of doing an
    in-kernel binary search over ``inst_offsets``. The Python wrapper
    materialises ``inst_idx_per_g`` via ``torch.repeat_interleave`` --
    this is what gsplat does and keeps the kernel body simple.
    """
    pid = tl.program_id(0)
    g_offsets = pid * BLOCK + tl.arange(0, BLOCK)
    g_mask = g_offsets < total

    # Load indices. tl.load with mask returns 0 for masked-out lanes, which is
    # a safe index into the row-0 of each splat / instance buffer (those
    # values will be discarded by the masked stores below). int32 -> int64
    # for use as offsets.
    splat_row = tl.load(cand_splat_rows_ptr + g_offsets, mask=g_mask, other=0).to(tl.int64)
    inst_idx  = tl.load(inst_idx_per_g_ptr  + g_offsets, mask=g_mask, other=0).to(tl.int64)

    # ---- 2. Load splat fields. ----
    lm0 = tl.load(splat_means_ptr  + splat_row * 3 + 0, mask=g_mask, other=0.0)
    lm1 = tl.load(splat_means_ptr  + splat_row * 3 + 1, mask=g_mask, other=0.0)
    lm2 = tl.load(splat_means_ptr  + splat_row * 3 + 2, mask=g_mask, other=0.0)
    ls0 = tl.load(splat_scales_ptr + splat_row * 3 + 0, mask=g_mask, other=0.0)
    ls1 = tl.load(splat_scales_ptr + splat_row * 3 + 1, mask=g_mask, other=0.0)
    ls2 = tl.load(splat_scales_ptr + splat_row * 3 + 2, mask=g_mask, other=0.0)
    lq0 = tl.load(splat_quats_ptr  + splat_row * 4 + 0, mask=g_mask, other=0.0)
    lq1 = tl.load(splat_quats_ptr  + splat_row * 4 + 1, mask=g_mask, other=0.0)
    lq2 = tl.load(splat_quats_ptr  + splat_row * 4 + 2, mask=g_mask, other=0.0)
    lq3 = tl.load(splat_quats_ptr  + splat_row * 4 + 3, mask=g_mask, other=0.0)
    lo  = tl.load(splat_opa_ptr    + splat_row,         mask=g_mask, other=0.0)
    lf0 = tl.load(splat_feat_dc_ptr + splat_row * 3 + 0, mask=g_mask, other=0.0)
    lf1 = tl.load(splat_feat_dc_ptr + splat_row * 3 + 1, mask=g_mask, other=0.0)
    lf2 = tl.load(splat_feat_dc_ptr + splat_row * 3 + 2, mask=g_mask, other=0.0)

    # ---- 3. Load instance fields. ----
    ip0 = tl.load(inst_pos_ptr + inst_idx * 3 + 0, mask=g_mask, other=0.0)
    ip1 = tl.load(inst_pos_ptr + inst_idx * 3 + 1, mask=g_mask, other=0.0)
    ip2 = tl.load(inst_pos_ptr + inst_idx * 3 + 2, mask=g_mask, other=0.0)
    ir0 = tl.load(inst_rot_ptr + inst_idx * 4 + 0, mask=g_mask, other=0.0)
    ir1 = tl.load(inst_rot_ptr + inst_idx * 4 + 1, mask=g_mask, other=0.0)
    ir2 = tl.load(inst_rot_ptr + inst_idx * 4 + 2, mask=g_mask, other=0.0)
    ir3 = tl.load(inst_rot_ptr + inst_idx * 4 + 3, mask=g_mask, other=0.0)
    iscl = tl.load(inst_scl_ptr + inst_idx, mask=g_mask, other=0.0)
    iopa = tl.load(inst_opa_ptr + inst_idx, mask=g_mask, other=0.0)

    # ---- 4. world_means = ip + R(ir) * (lm * exp(iscl)) ----
    es = tl.exp(iscl)
    sx = lm0 * es
    sy = lm1 * es
    sz = lm2 * es
    # Standard quaternion -> rotation-matrix expansion (works for non-unit
    # quaternions too, matching util.transformation.quat_rotate exactly).
    tx = 2.0 * ir1
    ty = 2.0 * ir2
    tz = 2.0 * ir3
    twx = ir0 * tx
    twy = ir0 * ty
    twz = ir0 * tz
    txx = ir1 * tx
    txy = ir1 * ty
    txz = ir1 * tz
    tyy = ir2 * ty
    tyz = ir2 * tz
    tzz = ir3 * tz
    R00 = 1.0 - tyy - tzz; R01 = txy - twz;       R02 = txz + twy
    R10 = txy + twz;       R11 = 1.0 - txx - tzz; R12 = tyz - twx
    R20 = txz - twy;       R21 = tyz + twx;       R22 = 1.0 - txx - tyy
    rx = R00 * sx + R01 * sy + R02 * sz
    ry = R10 * sx + R11 * sy + R12 * sz
    rz = R20 * sx + R21 * sy + R22 * sz
    tl.store(world_means_ptr + g_offsets * 3 + 0, ip0 + rx, mask=g_mask)
    tl.store(world_means_ptr + g_offsets * 3 + 1, ip1 + ry, mask=g_mask)
    tl.store(world_means_ptr + g_offsets * 3 + 2, ip2 + rz, mask=g_mask)

    # ---- 5. world_scales = clamp(iscl + ls, min, max) ----
    ws0 = tl.minimum(tl.maximum(iscl + ls0, min_scale_log), max_scale_log)
    ws1 = tl.minimum(tl.maximum(iscl + ls1, min_scale_log), max_scale_log)
    ws2 = tl.minimum(tl.maximum(iscl + ls2, min_scale_log), max_scale_log)
    tl.store(world_scales_ptr + g_offsets * 3 + 0, ws0, mask=g_mask)
    tl.store(world_scales_ptr + g_offsets * 3 + 1, ws1, mask=g_mask)
    tl.store(world_scales_ptr + g_offsets * 3 + 2, ws2, mask=g_mask)

    # ---- 6. world_rot = normalize(quat_mul(ir, lq)) ----
    # Hamilton product (w, x, y, z) ordering.
    w = ir0 * lq0 - ir1 * lq1 - ir2 * lq2 - ir3 * lq3
    x = ir0 * lq1 + ir1 * lq0 + ir2 * lq3 - ir3 * lq2
    y = ir0 * lq2 - ir1 * lq3 + ir2 * lq0 + ir3 * lq1
    z = ir0 * lq3 + ir1 * lq2 - ir2 * lq1 + ir3 * lq0
    # Reference does `q / clamp(norm(q), min=1e-6)` -- not 1/sqrt(... + eps).
    # We mirror that exactly so atol=1e-6 holds on inputs where the norm is
    # near zero.
    nrm = tl.sqrt(w * w + x * x + y * y + z * z)
    nrm = tl.maximum(nrm, 1.0e-6)
    inv = 1.0 / nrm
    tl.store(world_rot_ptr + g_offsets * 4 + 0, w * inv, mask=g_mask)
    tl.store(world_rot_ptr + g_offsets * 4 + 1, x * inv, mask=g_mask)
    tl.store(world_rot_ptr + g_offsets * 4 + 2, y * inv, mask=g_mask)
    tl.store(world_rot_ptr + g_offsets * 4 + 3, z * inv, mask=g_mask)

    # ---- 7. world_opa = clamp(iopa + lo, -opac_clamp, opac_clamp) ----
    op_ = tl.minimum(tl.maximum(iopa + lo, -opac_clamp), opac_clamp)
    tl.store(world_opa_ptr + g_offsets, op_, mask=g_mask)

    # ---- 8. world_feat = lf (instances share the candidate's colour) ----
    tl.store(world_feat_ptr + g_offsets * 3 + 0, lf0, mask=g_mask)
    tl.store(world_feat_ptr + g_offsets * 3 + 1, lf1, mask=g_mask)
    tl.store(world_feat_ptr + g_offsets * 3 + 2, lf2, mask=g_mask)


def assemble_fused(
    splat_means:       Tensor, splat_scales:      Tensor,
    splat_quats:       Tensor, splat_opacities:   Tensor,
    splat_features_dc: Tensor,
    inst_positions:    Tensor, inst_rotations:    Tensor,
    inst_scales:       Tensor, inst_opacities:    Tensor,
    cand_splat_rows:   Tensor,
    inst_offsets:      Tensor,
    min_world_scale_log: float = -9.21034,
    max_world_scale_log: float = 5.0,
    opac_clamp:        float = 20.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Triton-tier ``assemble_fused``. **Forward only** -- no gradient support.

    The caller should either hold a ``torch.no_grad()`` context, or use this
    only as the forward fast-path inside an autograd ``Function`` whose
    backward goes through the CUDA or Python tier (see
    ``scion.ops.assemble``).
    """
    assert splat_means.is_cuda, "Triton kernels require CUDA tensors"
    assert cand_splat_rows.is_cuda and inst_offsets.is_cuda

    total = int(cand_splat_rows.numel())
    N = int(inst_offsets.numel()) - 1
    device = splat_means.device
    dtype = splat_means.dtype  # all splat / inst params are fp32 in 0.2

    # Allocate outputs (must allocate even for total == 0 so callers can
    # unpack the five-tuple unconditionally).
    world_means  = torch.empty((total, 3), device=device, dtype=dtype)
    world_scales = torch.empty((total, 3), device=device, dtype=dtype)
    world_rot    = torch.empty((total, 4), device=device, dtype=dtype)
    world_opa    = torch.empty((total, 1), device=device, dtype=dtype)
    world_feat   = torch.empty((total, 3), device=device, dtype=dtype)

    if total == 0:
        return world_means, world_scales, world_rot, world_opa, world_feat

    # Build inst_idx_per_g host-side via repeat_interleave (single int32
    # kernel; cheap relative to the kernel body). This avoids an
    # in-kernel binary search and keeps the kernel a clean rank-1 SIMT.
    cand_splat_rows_i32 = cand_splat_rows
    if cand_splat_rows_i32.dtype != torch.int32:
        cand_splat_rows_i32 = cand_splat_rows_i32.to(torch.int32)

    sizes = (inst_offsets[1:] - inst_offsets[:-1]).to(torch.long)
    inst_idx_per_g = torch.repeat_interleave(
        torch.arange(N, device=device, dtype=torch.int32),
        sizes,
    )                                                       # [total] int32

    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _assemble_fused_kernel[grid](
        splat_means, splat_scales, splat_quats, splat_opacities, splat_features_dc,
        inst_positions, inst_rotations, inst_scales, inst_opacities,
        cand_splat_rows_i32, inst_idx_per_g,
        world_means, world_scales, world_rot, world_opa, world_feat,
        total,
        BLOCK=BLOCK,
        min_scale_log=min_world_scale_log,
        max_scale_log=max_world_scale_log,
        opac_clamp=opac_clamp,
    )
    return world_means, world_scales, world_rot, world_opa, world_feat
