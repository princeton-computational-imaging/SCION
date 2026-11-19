"""MH-3DGS importance-guided splat birth proposals for the flat-buffer codebook.

Uses index-based storage via ``splat_candidate_id`` / ``free_list`` rather
than contiguous offset-based segments.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from .common import (
    rows_for_candidate,
    build_binoms,
    compute_relocation,
)


# Local voxel-hash helpers.  Only ``mh_birth_splats`` uses them, so they live where they're
# used.

def _voxel_key(grid: Tensor) -> Tensor:
    """Pack ``[N, 3]`` integer voxel coords into a single key per row.

    Uses a large prime-multiplied XOR hash so two voxels collide only when
    their integer coords match.
    """
    return (
        grid[:, 0] * 73856093
        ^ grid[:, 1] * 19349663
        ^ grid[:, 2] * 83492791
    )


def voxel_count_at_positions(
    instance_positions: Tensor,
    query_positions: Tensor,
    voxel_size: float,
    *,
    aabb_min: "Tensor | None" = None,
) -> Tensor:
    """Count existing instances sharing each query's voxel.

    Returns ``[M]`` long tensor — count of existing instances in each
    query's voxel (0 when the voxel is empty, or when
    ``instance_positions`` is empty).
    """
    device = query_positions.device
    M = query_positions.shape[0]
    if M == 0:
        return torch.zeros(0, device=device, dtype=torch.long)
    N = instance_positions.shape[0]
    if N == 0:
        return torch.zeros(M, device=device, dtype=torch.long)

    if aabb_min is None:
        combined = torch.cat([instance_positions, query_positions], dim=0)
        aabb_min = combined.min(dim=0).values - 1e-6
    aabb_min = aabb_min.to(device=device, dtype=instance_positions.dtype)

    inst_grid = ((instance_positions - aabb_min) / voxel_size).floor().long()
    q_grid = ((query_positions - aabb_min) / voxel_size).floor().long()
    inst_key = _voxel_key(inst_grid)
    q_key = _voxel_key(q_grid)

    uniq, counts = torch.unique(inst_key, return_counts=True)
    pos = torch.searchsorted(uniq, q_key)
    pos_clamped = pos.clamp(max=uniq.numel() - 1)
    hit = (pos < uniq.numel()) & (uniq[pos_clamped] == q_key)
    zero = torch.zeros_like(q_key)
    return torch.where(hit, counts[pos_clamped], zero).long()


# ===========================================================================
# 5b. MH-3DGS importance-guided birth proposals for codebook splats
# ===========================================================================

def mh_birth_splats(
    splat_params: Dict[str, Tensor],
    splat_candidate_id: Tensor,
    free_list: Tensor,
    free_count: int,
    K: int,
    cap_max: int,
    *,
    accept_threshold: float = 0.5,
    binoms: Tensor | None = None,
    volume_exp: float = 1.0 / 3.0,
    fw_offsets: Tensor | None = None,
    fw_sorted_rows: Tensor | None = None,
    # New paper-aligned signals + knobs
    splat_grad_accum: Tensor | None = None,
    splat_grad_count: Tensor | None = None,
    splat_l1_accum: Tensor | None = None,
    splat_l1_count: Tensor | None = None,
    splat_denom: Tensor | None = None,
    w_opa: float = 0.8,
    w_grad: float = 0.5,
    w_l1: float = 0.5,
    coarse_scale: float = 10.0,
    fine_scale: float = 2.0,
    coarse_batch_per_cand: int = 2,
    fine_batch_per_cand: int = 6,
    voxel_size: float = 0.05,
    voxel_alpha: float = 5.0,
    use_voxel_penalty: bool = True,
    use_density_factor: bool = True,
) -> Tuple[int, Dict[str, float]]:
    """MH-3DGS importance-guided birth proposals for codebook splats.

    Paper-faithful (Kim, Jung & Park, MH-3DGS) version adapted to the
    flat-buffer codebook:

      1. Per-candidate importance ``sigmoid(w_opa·z(opa) + w_grad·z(grad) + w_l1·z(L1))``,
         z-scored per candidate (births are per-candidate).
      2. Two proposal batches per candidate per tick — coarse (large noise)
         + fine (small noise) — each with its own batch size and proposal
         std.  Caller anneals the std/batch knobs externally to match the
         paper's ramp-down over the densification window.
      3. Per-parent proposal std = ``prop_scale · parent_scale · density_factor``,
         where ``density_factor = 1 / (1 + log(1 + denom))`` damps the
         std for parents that have already been visited many times
         (paper's ``densification_postfix`` ``denom`` counter).
      4. MH acceptance ``sigmoid(imp − accept_threshold) · voxel_factor``,
         where ``voxel_factor = 1 / (1 + voxel_alpha · V)`` and ``V`` is
         the number of existing same-candidate splats sharing the
         proposal's voxel.

    Args:
        splat_params: Dict of Tensor [T_max, dims].  Caller should pass
            ``param.data`` for nn.Parameter inputs.
        splat_candidate_id: [T_max] long tensor mapping rows to candidates
            (-1 = free).
        free_list: [T_max] free-list stack buffer.
        free_count: Current number of free slots.
        K: Number of active candidates.
        cap_max: Maximum splats per candidate.
        accept_threshold: Importance centre for sigmoid acceptance (paper: 0.5).
        binoms: Binomial coefficient table from ``build_binoms()``.
        volume_exp: Exponent for volume scaling in ``compute_relocation``.
        fw_offsets: Optional [K+1] forward-map offsets (``model._fw_offsets``).
        fw_sorted_rows: Optional [T_active] sorted row indices.
        splat_grad_accum, splat_grad_count: Per-row 2D absgrad accumulator
            (caller maintains; reset after this call fires).  Both required
            to use the gradient component of importance; pass None to fall
            back to opacity-only.
        splat_l1_accum, splat_l1_count: Per-row L1 error accumulator
            sampled at ``means2d`` from the prior step's cached L1 map.
        splat_denom: [T_max] per-row visited counter (paper's ``denom``);
            damps proposal std for already-densified parents.
        w_opa, w_grad, w_l1: Importance fusion weights (paper: 0.8 / 0.5 / 0.5).
        coarse_scale, fine_scale: Per-batch proposal-std multipliers in
            units of parent scale (paper anneals 10→5 / 2→1 over window).
        coarse_batch_per_cand, fine_batch_per_cand: Per-candidate batch sizes
            for the coarse / fine proposals.  Caller is responsible for
            annealing these to 0 at the end of the window.
        voxel_size: Voxel side length (in candidate-local coords) for the
            density penalty.  Caller anneals from large→small.
        voxel_alpha: Voxel-density penalty strength.  0 disables.
        use_voxel_penalty: Apply ``1/(1 + α·V)`` to acceptance prob.
        use_density_factor: Apply ``1/(1 + log(1 + denom))`` to proposal std.

    Returns:
        Tuple ``(free_count, metrics)`` where ``metrics`` carries
        per-tick aggregates (n_births, n_proposals, accept_rate, parent_imp_mean).
    """
    device = splat_candidate_id.device

    if binoms is None:
        binoms = build_binoms().to(device)
    else:
        binoms = binoms.to(device)

    use_fw = fw_offsets is not None and fw_sorted_rows is not None

    n_births_total = 0
    n_proposals_total = 0
    parent_imp_sum = 0.0
    parent_imp_count = 0

    have_grad = splat_grad_accum is not None and splat_grad_count is not None
    have_l1 = splat_l1_accum is not None and splat_l1_count is not None
    have_denom = splat_denom is not None

    for k in range(K):
        if use_fw:
            s, e = int(fw_offsets[k].item()), int(fw_offsets[k + 1].item())
            S_k = e - s
            if S_k >= cap_max or S_k == 0:
                continue
            rows = fw_sorted_rows[s:e]
        else:
            rows = rows_for_candidate(splat_candidate_id, k)
            S_k = rows.shape[0]
            if S_k >= cap_max or S_k == 0:
                continue

        if free_count <= 0:
            break

        # --- 1. Per-row signals (candidate-scoped) ---
        opa = torch.sigmoid(splat_params["opacities"][rows, 0])  # [S_k]
        alive_mask = opa >= 1e-4
        if not alive_mask.any():
            continue

        # z-score helper, candidate-scoped
        def _z(x: Tensor) -> Tensor:
            return (x - x.mean()) / (x.std().clamp(min=1e-6))

        log_imp = w_opa * _z(opa)
        if have_grad:
            grad = (
                splat_grad_accum[rows]
                / splat_grad_count[rows].clamp(min=1)
            )
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            log_imp = log_imp + w_grad * _z(grad)
        if have_l1:
            l1 = (
                splat_l1_accum[rows]
                / splat_l1_count[rows].clamp(min=1)
            )
            l1 = torch.nan_to_num(l1, nan=0.0, posinf=0.0, neginf=0.0)
            log_imp = log_imp + w_l1 * _z(l1)
        importance = torch.sigmoid(log_imp)
        # Mask out dead splats so multinomial ignores them.
        importance = importance * alive_mask.float()

        # --- 2. Coarse + fine proposals ---
        existing_means = splat_params["means"][rows]  # [S_k, D]

        for batch_size, prop_scale in (
            (coarse_batch_per_cand, coarse_scale),
            (fine_batch_per_cand, fine_scale),
        ):
            if batch_size <= 0:
                continue
            if free_count <= 0:
                break
            if importance.sum() <= 0:
                continue

            n_request = min(batch_size, free_count, cap_max - S_k)
            if n_request <= 0:
                continue

            parents_local = torch.multinomial(
                importance, num_samples=n_request, replacement=True,
            )
            # Paper dedupes proposals; matches its "unique" step.
            parents_local = torch.unique(parents_local)
            n_request = parents_local.shape[0]
            if n_request <= 0:
                continue

            parent_global = rows[parents_local]
            parent_scale_3d = torch.exp(
                splat_params["scales"][parent_global].clamp(max=5.0)
            )  # [B, D]

            if use_density_factor and have_denom:
                d = splat_denom[parent_global].float()
                density_factor = 1.0 / (1.0 + torch.log1p(d))
            else:
                density_factor = torch.ones(
                    n_request, device=device, dtype=parent_scale_3d.dtype,
                )

            std = (
                prop_scale * parent_scale_3d * density_factor.unsqueeze(-1)
            ).clamp(min=1e-6)

            proposed_xyz = (
                splat_params["means"][parent_global]
                + torch.randn_like(parent_scale_3d) * std
            )

            # --- 3. MH acceptance ---
            parent_imp = importance[parents_local]
            if use_voxel_penalty and voxel_alpha > 0.0:
                V = voxel_count_at_positions(
                    existing_means, proposed_xyz, voxel_size,
                )
                voxel_factor = 1.0 / (1.0 + voxel_alpha * V.float())
            else:
                voxel_factor = torch.ones_like(parent_imp)

            p_accept = (
                torch.sigmoid(parent_imp - accept_threshold) * voxel_factor
            )
            accepted = torch.rand(n_request, device=device) < p_accept
            if not accepted.any():
                n_proposals_total += n_request
                parent_imp_sum += float(parent_imp.sum().item())
                parent_imp_count += n_request
                continue

            parents_local_acc = parents_local[accepted]
            parent_global_acc = parent_global[accepted]
            parent_scale_acc = parent_scale_3d[accepted]
            proposed_xyz_acc = proposed_xyz[accepted]
            n_accepted = int(parents_local_acc.shape[0])

            n_accepted = min(n_accepted, free_count, cap_max - S_k)
            if n_accepted <= 0:
                n_proposals_total += n_request
                parent_imp_sum += float(parent_imp.sum().item())
                parent_imp_count += n_request
                continue

            parents_local_acc = parents_local_acc[:n_accepted]
            parent_global_acc = parent_global_acc[:n_accepted]
            parent_scale_acc = parent_scale_acc[:n_accepted]
            proposed_xyz_acc = proposed_xyz_acc[:n_accepted]

            # --- 4. Mass-preserving relocation adjustments ---
            src_op = opa[parents_local_acc]
            ratios = torch.full(
                (n_accepted,), 2, dtype=torch.long, device=device,
            )
            new_op_sig, new_sc_exp = compute_relocation(
                src_op, parent_scale_acc, ratios, binoms, volume_exp,
            )
            new_op_logit = torch.logit(new_op_sig.clamp(1e-6, 1 - 1e-6))
            new_sc_log = torch.log(new_sc_exp.clamp(min=1e-8))

            # --- 5. Allocate + write ---
            new_rows = free_list[
                free_count - n_accepted : free_count
            ].clone()
            free_count -= n_accepted

            for key, param in splat_params.items():
                if key == "means":
                    param[new_rows] = proposed_xyz_acc
                elif key == "opacities":
                    cloned = param[parent_global_acc].clone()
                    cloned[:, 0] = new_op_logit
                    param[new_rows] = cloned
                elif key == "scales":
                    param[new_rows] = new_sc_log
                else:
                    param[new_rows] = param[parent_global_acc].clone()

            splat_candidate_id[new_rows] = k

            # Source-side mass adjustment (each parent sampled N times in this
            # batch accounts for N+1 total copies after the birth fires).
            unique_parents, parent_counts = torch.unique(
                parents_local_acc, return_counts=True,
            )
            if unique_parents.numel() > 0:
                dup_global = rows[unique_parents]
                adj_op, adj_sc = compute_relocation(
                    torch.sigmoid(
                        splat_params["opacities"][dup_global, 0]
                    ),
                    torch.exp(splat_params["scales"][dup_global]),
                    parent_counts + 1,
                    binoms,
                    volume_exp,
                )
                splat_params["opacities"][dup_global, 0] = torch.logit(
                    adj_op.clamp(1e-6, 1 - 1e-6)
                )
                splat_params["scales"][dup_global] = torch.log(
                    adj_sc.clamp(min=1e-8)
                )

            # Update local count + existing_means so the next batch sees
            # the just-added rows in its voxel-density check.
            S_k += n_accepted
            existing_means = torch.cat(
                [existing_means, proposed_xyz_acc], dim=0,
            )

            n_births_total += n_accepted
            n_proposals_total += n_request
            parent_imp_sum += float(parent_imp.sum().item())
            parent_imp_count += n_request

            if S_k >= cap_max:
                break

    accept_rate = (
        float(n_births_total) / max(1, n_proposals_total)
        if n_proposals_total > 0
        else 0.0
    )
    parent_imp_mean = (
        parent_imp_sum / parent_imp_count
        if parent_imp_count > 0
        else 0.0
    )

    metrics: Dict[str, float] = {
        "n_births": float(n_births_total),
        "n_proposals_total": float(n_proposals_total),
        "accept_rate": float(accept_rate),
        "parent_imp_mean": float(parent_imp_mean),
    }
    return free_count, metrics
