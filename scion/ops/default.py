"""Gradient-accumulation, relocation and instance clone/cull/subdivide ops.

Used by MHSplatStrategy and DefaultInstanceStrategy on the index-based
codebook layout with free-list allocation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .common import compute_relocation, rows_for_candidate


# ===========================================================================
# Gradient accumulation + instance clone/cull
# ===========================================================================

def accumulate_splat_grad2d(
    absgrad: Tensor,
    global_idx: Tensor,
    accum: Tensor,
    count: Tensor,
    T_active: Optional[int] = None,
) -> None:
    """Reverse-map rasteriser absgrad to flat-buffer splat indices via scatter_add.

    Each world Gaussian maps back to a candidate-local splat via global_idx.
    Multiple instances share the same splat row, so gradients accumulate.

    Args:
        absgrad: [total_world_gaussians] or [total_world_gaussians, 2] absgrad
            from the gsplat rasteriser (attached to means2d).
        global_idx: [total_world_gaussians] int32 OR int64 — maps each world
            Gaussian to a row in the flat ``[T_max, ...]`` parameter buffer.
            Rows where ``global_idx == -1`` are sentinels and are skipped.
        accum: [T_max] running accumulator (modified in-place).
        count: [T_max] running count (modified in-place).
        T_active: Optional watermark — entries with ``global_idx >= T_active``
            are skipped. ``None`` (default) means no upper-bound guard.
    """
    # PyTorch's scatter_add_ requires int64 indices; cast once.
    if global_idx.dtype != torch.int64:
        global_idx_long = global_idx.long()
    else:
        global_idx_long = global_idx
    valid = global_idx_long != -1
    if T_active is not None:
        valid = valid & (global_idx_long < T_active)
    idx = global_idx_long[valid]
    sub = absgrad[valid]
    if sub.dim() == 1:
        grad = sub
    else:
        grad = torch.linalg.vector_norm(sub, dim=-1)
    accum.scatter_add_(0, idx, grad)
    count.scatter_add_(0, idx, torch.ones_like(grad))


def accumulate_instance_grad2d(
    absgrad: Tensor,       # [total_world_gaussians] or [total_world_gaussians, 2]
    inst_idx: Tensor,      # [total_world_gaussians] int32 or int64 -> instance index
    accum: Tensor,         # [N_max] running accumulator (modified in-place)
    count: Tensor,         # [N_max] running count (modified in-place)
    N_active: Optional[int] = None,
) -> None:
    """Reverse-map rasteriser absgrad to per-instance accumulators via scatter_add.

    Each world Gaussian maps back to an instance via ``inst_idx``.  Both the
    accumulator and the count are incremented *per splat-per-visit*: an
    instance with ``S`` visible splats contributes ``S`` entries per frame,
    so ``avg_grad = accum/count`` is the per-splat-per-visit average.
    The signal therefore scales down with the number of splats per
    candidate, so large candidates clone less. On garden this gave higher
    PSNR than a per-frame-visit average, which over-fires on detail-rich
    foreground regions (the centre table) and destabilises reconstruction.

    Args:
        inst_idx: int32 or int64. Entries with ``inst_idx == -1`` are
            sentinels and are skipped.
        N_active: Optional watermark — entries with ``inst_idx >= N_active``
            are skipped. Defaults to ``None`` (no upper-bound guard).
    """
    if inst_idx.dtype != torch.int64:
        inst_idx_long = inst_idx.long()
    else:
        inst_idx_long = inst_idx
    valid = inst_idx_long != -1
    if N_active is not None:
        valid = valid & (inst_idx_long < N_active)
    idx = inst_idx_long[valid]
    grad = absgrad[valid] if absgrad.dim() == 1 else absgrad[valid].norm(dim=-1)
    accum.scatter_add_(0, idx, grad)
    count.scatter_add_(0, idx, torch.ones_like(grad))


def clone_instances_by_mask(
    positions: Tensor,          # [N, 3]
    rotations: Tensor,          # [N, 4]
    scales: Tensor,             # [N, 1]
    opacities: Tensor,          # [N, 1]
    assignments: Tensor,        # [N] long
    clone_mask: Tensor,         # [N] bool — True = clone this instance
    cap: int,                   # max total instances
    opa_offset: float = 0.0,    # Subtracted from each clone's opacity logit.
    min_opa_logit: float = -6.0,  # Floor so clones stay above the prune threshold.
    noise_scale_factor: float = 0.1,  # Multiplier on per-instance scale for clone displacement noise.
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Clone selected instances with small position perturbation.

    Duplicates instances marked by clone_mask, adding noise proportional to
    scale. Respects cap.

    Args:
        opa_offset: Non-negative offset subtracted from each clone's opacity
            logit before concatenation.  ``0`` (default) copies the parent's
            opacity verbatim — causes two co-located clones to compound to
            ``1 - (1-σ)²`` so the region over-saturates relative to the parent
            alone.  ``log(2) ≈ 0.693`` halves σ so two overlapping clones
            approximately recover the parent's α for σ in [0.3, 0.7].  Match
            the splat-split path's 2-child α correction when clones are
            expected to overlap.
        min_opa_logit: Lower clamp so clones don't get pushed below the
            ``instance_min_opacity`` / ``default_inst_prune_opa`` floor and
            immediately culled on the next refine tick (σ(-6) ≈ 2.5e-3).

    Returns:
        Concatenated (positions, rotations, scales, opacities, assignments).
        Plain tensors — caller must replace Parameters.
    """
    N = positions.shape[0]
    n_clone = int(clone_mask.sum().item())
    n_clone = min(n_clone, cap - N)
    if n_clone <= 0:
        return (positions, rotations, scales,
                opacities, assignments)

    clone_indices = torch.where(clone_mask)[0][:n_clone]

    # Clone params
    new_pos = positions[clone_indices].clone()
    new_rot = rotations[clone_indices].clone()
    new_scl = scales[clone_indices].clone()
    new_opa = opacities[clone_indices].clone()
    new_asgn = assignments[clone_indices].clone()

    # Add noise proportional to scale
    noise_scale = new_scl.exp()  # [n_clone, 1]
    new_pos += noise_scale * noise_scale_factor * torch.randn_like(new_pos)

    # Damp the clone's opacity in logit space so two co-located clones
    # don't α-compose to a higher joint α than the parent alone.  The
    # offset is in logit space; floor the result at ``min_opa_logit`` so
    # the clone stays above the prune threshold.
    if opa_offset != 0.0:
        new_opa = (new_opa - opa_offset).clamp(min=min_opa_logit)

    # Concatenate
    return (
        torch.cat([positions, new_pos], dim=0),
        torch.cat([rotations, new_rot], dim=0),
        torch.cat([scales, new_scl], dim=0),
        torch.cat([opacities, new_opa], dim=0),
        torch.cat([assignments, new_asgn], dim=0),
    )


def cull_instances_by_mask(
    positions: Tensor,          # [N, 3]
    rotations: Tensor,          # [N, 4]
    scales: Tensor,             # [N, 1]
    opacities: Tensor,          # [N, 1]
    assignments: Tensor,        # [N] long
    cull_mask: Tensor,          # [N] bool — True = remove this instance
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Remove instances marked for culling.

    Returns compacted tensors (positions, rotations, scales, opacities, assignments).
    """
    keep = ~cull_mask
    return (
        positions[keep],
        rotations[keep],
        scales[keep],
        opacities[keep],
        assignments[keep],
    )


# ===========================================================================
# Splat relocation (mass-preserving within candidate)
# ===========================================================================

def relocate_dead_splats(
    splat_params: Dict[str, Tensor],
    splat_candidate_id: Tensor,
    K: int,
    threshold: float,
    binoms: Tensor,
    volume_exp: float = 1.0 / 3.0,
    fw_offsets: "Tensor | None" = None,
    fw_sorted_rows: "Tensor | None" = None,
) -> int:
    """Recycle dead Gaussians by copying alive ones within each candidate.

    Dead capacity is reused in-place; splat count per candidate does not
    change. Copied Gaussians have their opacity and scale adjusted via
    ``compute_relocation`` to preserve total probability mass.

    Caller should pass ``param.data`` for nn.Parameter inputs.
    """
    device = splat_params["opacities"].device
    use_fw = fw_offsets is not None and fw_sorted_rows is not None

    all_opacities = splat_params["opacities"]
    all_scales = splat_params["scales"]

    n_relocated_total = 0

    for k in range(K):
        if use_fw:
            s, e = int(fw_offsets[k].item()), int(fw_offsets[k + 1].item())
            if s == e:
                continue
            rows = fw_sorted_rows[s:e]
        else:
            rows = rows_for_candidate(splat_candidate_id, k)
            if rows.numel() == 0:
                continue

        seg_opacities_sig = torch.sigmoid(all_opacities[rows, 0])

        dead_mask = seg_opacities_sig < threshold
        alive_mask = ~dead_mask

        if not dead_mask.any() or not alive_mask.any():
            continue

        n_dead = int(dead_mask.sum().item())
        n_relocated_total += n_dead

        alive_weights = seg_opacities_sig[alive_mask]
        sampled_local = torch.multinomial(
            alive_weights, num_samples=n_dead, replacement=True
        )

        alive_indices = alive_mask.nonzero(as_tuple=False).squeeze(-1)
        src_local = alive_indices[sampled_local]
        dst_local = dead_mask.nonzero(as_tuple=False).squeeze(-1)

        src_global = rows[src_local]
        dst_global = rows[dst_local]

        n_alive = int(alive_mask.sum().item())
        copy_counts = torch.zeros(n_alive, device=device, dtype=torch.long)
        copy_counts.scatter_add_(
            0, sampled_local,
            torch.ones_like(sampled_local, dtype=torch.long),
        )
        donor_alive_idx = (copy_counts > 0).nonzero(as_tuple=False).squeeze(-1)
        donor_global = rows[alive_indices[donor_alive_idx]]
        N_per_donor = copy_counts[donor_alive_idx] + 1

        new_op_sig, new_sc_exp = compute_relocation(
            torch.sigmoid(all_opacities[donor_global, 0]),
            torch.exp(all_scales[donor_global]),
            N_per_donor,
            binoms,
            volume_exp,
        )
        splat_params["opacities"][donor_global, 0] = torch.logit(
            new_op_sig.clamp(1e-6, 1 - 1e-6)
        )
        splat_params["scales"][donor_global] = torch.log(
            new_sc_exp.clamp(min=1e-8)
        )

        for key, param in splat_params.items():
            param[dst_global] = param[src_global]

        splat_params["means"][dst_global] += 1e-5 * torch.randn_like(
            splat_params["means"][dst_global]
        )

    return n_relocated_total


# ===========================================================================
# Instance subdivision (mass-preserving children at disjoint positions)
# ===========================================================================

def subdivide_instance(
    positions: Tensor,
    rotations: Tensor,
    scales: Tensor,
    opacities: Tensor,
    assignments: Tensor,
    src_indices: Tensor,
    M: int,
    *,
    volume_exp: float = 1.0 / 3.0,
    keep_parent: bool = True,
    position_noise_scale: float = 1.0,
    min_child_scale: float = 1e-4,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Replace each selected parent with ``M`` mass-preserving children.

    Children tile the parent's footprint at disjoint world positions, so
    instance-scope mass preservation is opacity verbatim + scale shrunk by
    ``M^(1/3)`` (not the splat-scope α-compositing formula, which only
    applies when children co-render at the same pixel).
    """
    device = positions.device
    P = src_indices.shape[0]
    if P == 0 or M < 2:
        return positions, rotations, scales, opacities, assignments

    parent_scale_exp = scales[src_indices].exp()
    shrink = float(M) ** float(volume_exp)
    predicted_min = parent_scale_exp.min(dim=-1).values / shrink
    keep_mask = predicted_min >= min_child_scale
    if not bool(keep_mask.all().item()):
        src_indices = src_indices[keep_mask]
        P = src_indices.shape[0]
        if P == 0:
            return positions, rotations, scales, opacities, assignments

    src = src_indices

    src_scl_exp = scales[src].exp()
    new_opa_logit = opacities[src].clone()
    new_scl_exp = src_scl_exp / shrink
    new_scl_log = new_scl_exp.log()

    D_pos = positions.shape[-1]
    noise = torch.randn(P, M, D_pos, device=device)
    sigma = src_scl_exp
    if sigma.dim() == 2 and sigma.shape[-1] == 1 and D_pos > 1:
        sigma = sigma.expand(P, D_pos)
    disp = noise * sigma.unsqueeze(1) * position_noise_scale

    parent_pos = positions[src].unsqueeze(1)
    child_pos = parent_pos + disp

    child_rot = rotations[src].unsqueeze(1).expand(P, M, rotations.shape[-1])
    child_asgn = assignments[src].unsqueeze(1).expand(P, M)
    child_scl = new_scl_log.unsqueeze(1).expand(P, M, -1)
    child_opa = new_opa_logit.unsqueeze(1).expand(P, M, -1)

    if keep_parent:
        positions = positions.clone()
        rotations = rotations.clone()
        scales = scales.clone()
        opacities = opacities.clone()
        assignments = assignments.clone()
        positions[src] = child_pos[:, 0]
        rotations[src] = child_rot[:, 0]
        scales[src] = child_scl[:, 0]
        opacities[src] = child_opa[:, 0]
        assignments[src] = child_asgn[:, 0]
        extra_pos = child_pos[:, 1:].reshape(-1, D_pos)
        extra_rot = child_rot[:, 1:].reshape(-1, rotations.shape[-1])
        extra_scl = child_scl[:, 1:].reshape(-1, scales.shape[-1])
        extra_opa = child_opa[:, 1:].reshape(-1, opacities.shape[-1])
        extra_asgn = child_asgn[:, 1:].reshape(-1)
    else:
        extra_pos = child_pos.reshape(-1, D_pos)
        extra_rot = child_rot.reshape(-1, rotations.shape[-1])
        extra_scl = child_scl.reshape(-1, scales.shape[-1])
        extra_opa = child_opa.reshape(-1, opacities.shape[-1])
        extra_asgn = child_asgn.reshape(-1)

    return (
        torch.cat([positions, extra_pos], dim=0),
        torch.cat([rotations, extra_rot], dim=0),
        torch.cat([scales, extra_scl], dim=0),
        torch.cat([opacities, extra_opa], dim=0),
        torch.cat([assignments, extra_asgn], dim=0),
    )
