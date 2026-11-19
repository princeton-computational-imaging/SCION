"""Loss functions specific to the flat-buffer codebook model.

Operates on pre-allocated [T_max, dims] flat buffers with candidate_offsets
to demarcate per-candidate segments.
"""

import torch
from torch import Tensor


def compactness_loss(
    splat_means: Tensor,
    offsets: Tensor,
    K: int,
    T_active: int = -1,
) -> Tensor:
    """Within-candidate spatial variance loss.

    Penalizes spread of splat positions within each candidate, encouraging
    tight spatial clusters (e.g. grass bushels). Candidates with fewer than
    2 splats contribute zero via masked mean.

    Args:
        splat_means: [T, 3] flat splat position buffer (gradients flow through).
        offsets: [K+1] long tensor of candidate segment boundaries.
        K: Number of active candidates.
        T_active: Pre-cached active splat count. If -1, computed from offsets
            (fallback for callers outside the compiled region).

    Returns:
        Scalar loss tensor (mean across multi-splat candidates).
    """
    device = splat_means.device
    sizes = offsets[1:K + 1] - offsets[:K]  # [K]
    if T_active < 0:
        T_active = int(offsets[K].item())

    if T_active == 0:
        return torch.zeros(1, device=device)

    means = splat_means[:T_active]  # [T_active, 3]

    # Build splat-to-candidate mapping
    cand_idx = torch.repeat_interleave(
        torch.arange(K, device=device), sizes
    )  # [T_active]
    counts = sizes.float().clamp(min=1)  # [K]

    # Per-candidate centroid via scatter
    centroid_sum = torch.zeros(K, 3, device=device)
    centroid_sum.scatter_add_(0, cand_idx.unsqueeze(1).expand(-1, 3), means)
    centroids = centroid_sum / counts.unsqueeze(1)

    # Per-splat squared distance from its candidate's centroid
    sq_dist = (means - centroids[cand_idx]).pow(2).sum(-1)  # [T_active]

    # Per-candidate mean squared distance
    dist_sum = torch.zeros(K, device=device)
    dist_sum.scatter_add_(0, cand_idx, sq_dist)
    per_cand = dist_sum / counts

    # Branchless masked mean over multi-splat candidates
    multi = (sizes > 1).float()
    n_multi = multi.sum().clamp(min=1.0)
    return (per_cand * multi).sum() / n_multi
