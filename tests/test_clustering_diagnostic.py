"""Tests for clustering utilities used in codebook initialization."""

import pytest
import torch
from collections import Counter

from scion.util.clustering import kmeans_farthest_point


def cluster_stats(assignments: torch.Tensor, K: int) -> dict:
    """Compute statistics about cluster assignments."""
    counts = Counter(assignments.cpu().tolist())
    non_empty = sum(1 for c in range(K) if counts.get(c, 0) > 0)
    sizes = [counts.get(c, 0) for c in range(K)]
    sizes_nonzero = [s for s in sizes if s > 0]
    return {
        "K": K,
        "non_empty": non_empty,
        "min_size": min(sizes_nonzero) if sizes_nonzero else 0,
        "max_size": max(sizes_nonzero) if sizes_nonzero else 0,
    }


class TestClusteringBasic:
    """Test that clustering works on well-separated data."""

    def test_well_separated_blobs(self):
        """K blobs that are clearly separable should get K clusters."""
        K = 16
        N_per = 500
        torch.manual_seed(42)

        positions = []
        colors = []
        for _ in range(K):
            center = torch.randn(3) * 10
            blob = center + torch.randn(N_per, 3) * 0.1
            color = torch.rand(3).unsqueeze(0).expand(N_per, -1)
            positions.append(blob)
            colors.append(color)

        positions = torch.cat(positions)
        colors = torch.cat(colors)

        assignments = kmeans_farthest_point(positions, colors, K, color_weight=2.0)
        stats = cluster_stats(assignments, K)
        assert stats["non_empty"] == K, f"Expected {K} clusters, got {stats['non_empty']}"

    def test_uniform_random(self):
        """Uniform random points should distribute across K clusters."""
        K = 64
        N = 30000
        torch.manual_seed(42)

        positions = torch.rand(N, 3) * 4 - 2  # [-2, 2]
        colors = torch.rand(N, 3)

        assignments = kmeans_farthest_point(positions, colors, K, color_weight=2.0)
        stats = cluster_stats(assignments, K)
        assert stats["non_empty"] == K


class TestLogitColorCollapse:
    """Verify that logit-space colors with ±inf values break clustering,
    and that RGB-space colors work correctly. This was the root cause of
    codebook assignments collapsing to ~2 clusters."""

    def test_logit_inf_causes_failure(self):
        """Logit colors from 0/255 pixel values produce ±inf, breaking k-means."""
        K = 64
        N = 10000
        torch.manual_seed(42)

        positions = torch.rand(N, 3) * 4 - 2
        # Simulate SfM colors with some 0 and 255 values
        colors_uint8 = torch.randint(0, 256, (N, 3)).float()
        colors_logit = torch.logit(colors_uint8 / 255.0, eps=1e-10)

        # Inf values cause faiss to raise an error (or collapse to fewer clusters)
        with pytest.raises(RuntimeError, match="NaN.*Inf"):
            kmeans_farthest_point(positions, colors_logit, K, color_weight=2.0)

    def test_rgb_colors_no_collapse(self):
        """RGB [0,1] colors (sigmoid of logit) produce healthy K clusters."""
        K = 64
        N = 10000
        torch.manual_seed(42)

        positions = torch.rand(N, 3) * 4 - 2
        colors_uint8 = torch.randint(0, 256, (N, 3)).float()
        colors_rgb = colors_uint8 / 255.0  # the fix: use RGB, not logit

        assignments = kmeans_farthest_point(positions, colors_rgb, K, color_weight=2.0)
        stats = cluster_stats(assignments, K)
        assert stats["non_empty"] == K

    def test_bimodal_colors_all_clusters_populated(self):
        """Two dominant color groups should still produce K spatial clusters."""
        K = 64
        N = 30000
        torch.manual_seed(42)

        positions = torch.zeros(N, 3)
        positions[:, 0] = torch.rand(N) * 4 - 2
        positions[:, 1] = torch.rand(N) * 4 - 2
        positions[:, 2] = torch.randn(N) * 0.05

        # Two color groups in RGB space
        colors = torch.empty(N, 3)
        half = N // 2
        colors[:half] = torch.tensor([0.2, 0.5, 0.15]) + torch.randn(half, 3) * 0.02
        colors[half:] = torch.tensor([0.4, 0.3, 0.1]) + torch.randn(N - half, 3) * 0.02

        assignments = kmeans_farthest_point(positions, colors, K, color_weight=2.0)
        stats = cluster_stats(assignments, K)
        assert stats["non_empty"] == K


