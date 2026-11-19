"""Shared helpers and scoring utilities for the flat-buffer codebook model.

Contains math primitives and metric helpers used by the MH / default
density-control sub-modules and the model's training metrics.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor

# Sentinel value for free (unowned) splat rows in splat_candidate_id.
# Defined here (rather than codebook_base.py) to avoid circular imports.
SPLAT_FREE: int = -1


# ===========================================================================
# Helpers
# ===========================================================================

def rows_for_candidate(splat_candidate_id: Tensor, k: Union[int, Tensor]) -> Tensor:
    """Return buffer row indices owned by candidate *k*."""
    return torch.where(splat_candidate_id == k)[0]


# ===========================================================================
# 1. Binomial coefficient table
# ===========================================================================

def build_binoms(n_max: int = 51) -> Tensor:
    """Build a table of binomial coefficients C(n, k).

    Uses Pascal's recurrence: C(n, k) = C(n-1, k-1) + C(n-1, k).

    Args:
        n_max: Size of the square table. Entry [n, k] = C(n, k) for
            0 <= k <= n < n_max; entries where k > n are 0.

    Returns:
        Float32 tensor of shape [n_max, n_max].
    """
    binoms = torch.zeros(n_max, n_max, dtype=torch.float32)
    for n in range(n_max):
        binoms[n, 0] = 1.0
        for k in range(1, n + 1):
            binoms[n, k] = binoms[n - 1, k - 1] + binoms[n - 1, k]
    return binoms


# ===========================================================================
# 2. Opacity / scale relocation for splits
# ===========================================================================

def compute_relocation(
    opacities: Tensor,
    scales: Tensor,
    ratios: Tensor,
    binoms: Tensor,
    volume_exp: float = 1.0 / 3.0,
) -> Tuple[Tensor, Tensor]:
    """Compute new opacity and scale when splitting each Gaussian into n copies.

    Preserves total probability mass so that n independent copies with the
    returned opacity collectively explain the same probability as the original.

    Math (from gsplat MCMC):
        new_opacity = 1 - (1 - opacity)^(1/n)
        new_scale   = scale / n^volume_exp   (preserves total volume/area)

    Args:
        opacities: [N] sigmoid-space opacities in [0, 1].
        scales: [N, D] exp-space scales (positive), D=3 for 3D, D=2 for 2D.
        ratios: [N] long tensor — number of copies each Gaussian becomes.
        binoms: Binomial coefficient table [n_max, n_max] from build_binoms().
            (Accepted for API consistency with gsplat; not used in the
            simplified closed-form derivation.)
        volume_exp: Exponent for scale shrinkage. Use 1/3 for 3D (volume
            preserving) or 1/2 for 2D (area preserving). Default: 1/3.

    Returns:
        Tuple (new_opacities [N], new_scales [N, D]) in sigmoid / exp space.
    """
    ratios_clamped = ratios.long().clamp(min=1, max=binoms.shape[0] - 1)
    ratios_f = ratios_clamped.float()

    # Each of the n copies has this opacity; n independent copies together
    # cover the same probability mass as the original single Gaussian.
    new_opacities = 1.0 - (1.0 - opacities) ** (1.0 / ratios_f)

    # Shrink each linear scale dimension so that total volume/area is preserved.
    new_scales = scales / (ratios_f ** volume_exp).unsqueeze(-1)

    return new_opacities, new_scales


# ===========================================================================
# Gradient-informed metrics
# ===========================================================================


def compute_instance_urgency(
    pos_grad_ema: Tensor,
    scale_grad_ema: Tensor | None = None,
    scale_weight: float = 1.0,
) -> Tensor:
    """Per-instance urgency from gradient EMA magnitudes.

    High urgency means the optimizer is fighting to compensate for a poor
    candidate assignment via instance transforms.

    Args:
        pos_grad_ema: [N, 3] EMA of position gradients (Adam exp_avg).
        scale_grad_ema: Optional [N, 1] EMA of scale gradients.
        scale_weight: Relative weight of scale gradient contribution.

    Returns:
        [N] float tensor of urgency scores (non-negative).
    """
    urgency = pos_grad_ema.norm(dim=-1)  # [N]
    if scale_grad_ema is not None:
        urgency = urgency + scale_weight * scale_grad_ema.abs().squeeze(-1)
    return urgency


def compute_gradient_disagreement(
    pos_grad_ema: Tensor,
    assignments: Tensor,
    K: int,
) -> Tensor:
    """Within-candidate gradient direction disagreement.

    For each candidate, measures how much its instances' position gradient
    directions diverge.  High disagreement means instances want contradictory
    changes — a stronger split signal than spatial spread alone.

    Candidates with fewer than 2 instances get score 0.

    Args:
        pos_grad_ema: [N, 3] EMA of position gradients (Adam exp_avg).
        assignments: [N] long tensor of candidate indices.
        K: Number of active candidates.

    Returns:
        [K] float tensor of disagreement scores in [0, 1].
            0 = all gradients aligned, 1 = maximally opposing.
    """
    device = pos_grad_ema.device
    D = pos_grad_ema.shape[-1]

    # Normalize directions for instances with non-trivial gradients
    norms = pos_grad_ema.norm(dim=-1)  # [N]
    active = norms > 1e-6
    directions = torch.zeros_like(pos_grad_ema)
    directions[active] = pos_grad_ema[active] / norms[active].unsqueeze(-1)

    # Per-candidate sum of unit directions
    dir_sum = torch.zeros(K, D, device=device)
    dir_sum.scatter_add_(0, assignments.unsqueeze(1).expand(-1, D), directions)

    # Per-candidate count of active instances
    active_counts = torch.zeros(K, device=device)
    active_counts.scatter_add_(0, assignments, active.float())

    # Mean direction magnitude squared = ||sum/count||^2
    mean_dir = dir_sum / active_counts.clamp(min=1).unsqueeze(1)
    scores = 1.0 - mean_dir.norm(dim=-1).pow(2).clamp(max=1.0)
    scores[active_counts < 2] = 0.0
    return scores


# Parameter counts per element.
# Splat: 3 (means) + 3 (scales) + 4 (quats) + 1 (opacity) + 3 (color_dc) = 14
# Instance: 3 (position) + 4 (rotation) + 1 (scale) + 1 (opacity) = 9
_P_SPLAT = 14
_P_INST = 9


def codebook_description_length(
    splats_per_candidate: Tensor,
    num_instances: int,
    params_per_splat: int = _P_SPLAT,
    params_per_instance: int = _P_INST,
) -> Tensor:
    """Storage cost of the codebook representation.

    DL = P_inst * N + P_splat * sum_c(s_c)

    Each candidate's splats are stored once; instances are lightweight
    references.

    Args:
        splats_per_candidate: [K] splat counts per candidate.
        num_instances: Total number of instances N.
        params_per_splat: Scalar parameters per Gaussian splat.
        params_per_instance: Scalar parameters per instance.

    Returns:
        Scalar tensor with the description length.
    """
    dl = params_per_instance * num_instances + params_per_splat * splats_per_candidate.float().sum()
    return torch.tensor(dl.item(), device=splats_per_candidate.device)


def codebook_compression_ratio(
    splats_per_candidate: Tensor,
    assignments: Tensor,
    num_instances: int,
    params_per_splat: int = _P_SPLAT,
    params_per_instance: int = _P_INST,
) -> Tensor:
    """Compression ratio: codebook storage / naive per-instance storage.

    ratio = DL / (P_splat * sum_i s_{a_i})

    where a_i is instance i's assigned candidate.  Values < 1 mean the
    codebook is more compact than storing each instance independently.

    Args:
        splats_per_candidate: [K] splat counts per candidate.
        assignments: [N] integer candidate assignments per instance.
        num_instances: Total number of instances N.
        params_per_splat: Scalar parameters per Gaussian splat.
        params_per_instance: Scalar parameters per instance.

    Returns:
        Scalar tensor with the compression ratio.
    """
    dl = codebook_description_length(
        splats_per_candidate, num_instances,
        params_per_splat, params_per_instance,
    )
    splats_per_instance = splats_per_candidate[assignments[:num_instances]]
    naive_cost = params_per_splat * splats_per_instance.float().sum()
    return dl / naive_cost.clamp(min=1.0)
