"""Clustering utilities for codebook initialization."""

import numpy as np
import torch
from torch import Tensor

import faiss


@torch.no_grad()
def kmeans(
    features: Tensor,
    K: int,
    n_iter: int = 20,
    spherical: bool = False,
) -> Tensor:
    """GPU-accelerated k-means via faiss.

    Args:
        features: [N, D] feature vectors.  Caller handles preparation
            (concatenation, weighting, normalization).
        K: Number of clusters.
        n_iter: Lloyd iterations.
        spherical: Use spherical k-means (cosine distance).  Features
            should be L2-normalized when this is ``True``.

    Returns:
        [N] long tensor of cluster indices in [0, K), on the same device
        as *features*.
    """
    orig_device = features.device
    features_np = features.cpu().numpy().astype(np.float32)
    D = features_np.shape[1]

    use_gpu = torch.cuda.is_available() and faiss.get_num_gpus() > 0
    km = faiss.Kmeans(
        d=D, k=K, niter=n_iter, gpu=use_gpu, verbose=False,
        spherical=spherical,
    )
    km.train(features_np)
    _, assignments_np = km.index.search(features_np, 1)

    return torch.from_numpy(assignments_np[:, 0].astype(np.int64)).to(orig_device)


@torch.no_grad()
def kmeans_farthest_point(
    positions: Tensor,
    colors: Tensor,
    K: int,
    n_iter: int = 20,
    color_weight: float = 0.5,
) -> Tensor:
    """K-means on concatenated position + weighted color features.

    Convenience wrapper around :func:`kmeans` that builds a ``[N, 6]``
    feature vector from positions and colors.

    Args:
        positions: [N, 3] point positions.
        colors: [N, 3] point colors (same device as positions).
        K: Number of clusters.
        n_iter: Lloyd iterations.
        color_weight: Scaling factor for color features.

    Returns:
        [N] long tensor of cluster indices in [0, K), on the same device
        as the input positions.
    """
    features = torch.cat(
        [positions, colors.to(positions.device) * color_weight], dim=-1
    )
    return kmeans(features, K, n_iter=n_iter)


