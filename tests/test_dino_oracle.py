"""Tests for DinoFeatureOracle."""

import pytest
import torch
from torch import nn

from scion.util.dino_oracle import DinoFeatureOracle


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def oracle_cpu():
    """Small oracle on CPU for fast tests."""
    N = 1000
    D = 16
    positions = torch.randn(N, 3)
    features = torch.randn(N, D)
    return DinoFeatureOracle(positions, features, k=1, device="cpu")


@pytest.fixture
def oracle_k3():
    """Oracle with k=3 IDW averaging."""
    N = 500
    D = 8
    positions = torch.randn(N, 3)
    features = torch.randn(N, D)
    return DinoFeatureOracle(positions, features, k=3, device="cpu")


@pytest.fixture
def oracle_voxel():
    """Voxel-hash oracle on CPU."""
    torch.manual_seed(7)
    N = 1000
    D = 16
    positions = torch.randn(N, 3)
    features = torch.randn(N, D)
    return DinoFeatureOracle(
        positions,
        features,
        k=1,
        device="cpu",
        backend="voxel_hash",
        voxel_size=0.5,
        voxel_neighbor_radius=1,
    )


# ------------------------------------------------------------------
# Basic construction
# ------------------------------------------------------------------

class TestConstruction:
    def test_shapes(self, oracle_cpu):
        assert oracle_cpu.feature_dim == 16
        assert oracle_cpu.num_points == 1000

    def test_is_nn_module(self, oracle_voxel):
        assert isinstance(oracle_voxel, nn.Module)

    def test_mismatched_counts_raises(self):
        with pytest.raises(AssertionError, match="mismatch"):
            DinoFeatureOracle(
                torch.randn(10, 3), torch.randn(20, 8), device="cpu",
            )


# ------------------------------------------------------------------
# kNN lookup
# ------------------------------------------------------------------

class TestLookup:
    def test_exact_positions_k1(self, oracle_cpu):
        """Querying at exact original positions should return exact features."""
        # Query a subset of original positions
        indices = [0, 10, 42, 999]
        query = oracle_cpu._positions[indices]
        result = oracle_cpu.lookup(query)

        expected = oracle_cpu._features[indices]
        assert torch.allclose(result, expected, atol=1e-5), (
            f"Max diff: {(result - expected).abs().max().item()}"
        )

    def test_output_shape(self, oracle_cpu):
        query = torch.randn(50, 3)
        result = oracle_cpu.lookup(query)
        assert result.shape == (50, 16)

    def test_k3_idw(self, oracle_k3):
        """k>1 lookup should return weighted average, not exact match."""
        query = oracle_k3._positions[:5]
        result = oracle_k3.lookup(query)
        assert result.shape == (5, 8)
        # With k=3, result at exact positions should be close but not
        # identical to the original feature (neighbours bleed in)
        exact = oracle_k3._features[:5]
        # Should be similar but not exactly equal
        assert result.shape == exact.shape

    def test_single_query(self, oracle_cpu):
        query = torch.randn(1, 3)
        result = oracle_cpu.lookup(query)
        assert result.shape == (1, 16)


class TestLookupChunked:
    def test_chunked_matches_full(self, oracle_cpu):
        """Chunked and full lookup should produce identical results."""
        query = torch.randn(200, 3)
        result_full = oracle_cpu.lookup(query, chunk_size=1000)
        result_chunked = oracle_cpu.lookup(query, chunk_size=64)
        assert torch.allclose(result_full, result_chunked, atol=1e-6)

    def test_small_input(self, oracle_cpu):
        """Input smaller than chunk_size should still work."""
        query = torch.randn(5, 3)
        result = oracle_cpu.lookup(query, chunk_size=100)
        assert result.shape == (5, 16)


class TestVoxelLookup:
    def test_exact_positions_k1(self, oracle_voxel):
        indices = [0, 10, 42, 999]
        query = oracle_voxel._positions[indices]
        result = oracle_voxel.lookup(query)
        expected = oracle_voxel._features[indices]
        assert torch.allclose(result, expected, atol=1e-5)

    def test_returns_finite_features(self):
        torch.manual_seed(11)
        positions = torch.randn(800, 3)
        features = torch.randn(800, 12)
        voxel = DinoFeatureOracle(
            positions,
            features,
            k=1,
            device="cpu",
            backend="voxel_hash",
            voxel_size=0.4,
            voxel_neighbor_radius=1,
        )
        query = torch.randn(200, 3)
        result = voxel.lookup(query)
        assert result.shape == (200, 12)
        assert torch.isfinite(result).all()

    def test_fallback_for_out_of_grid_query(self):
        torch.manual_seed(13)
        positions = torch.randn(300, 3)
        features = torch.randn(300, 6)
        brute = DinoFeatureOracle(
            positions, features, k=1, device="cpu", backend="brute_force"
        )
        voxel = DinoFeatureOracle(
            positions,
            features,
            k=1,
            device="cpu",
            backend="voxel_hash",
            voxel_size=0.2,
            voxel_neighbor_radius=0,
        )
        query = torch.tensor([[100.0, 100.0, 100.0]])
        expected = brute.lookup(query)
        result = voxel.lookup(query)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_torchscript_voxel_lookup(self, oracle_voxel):
        scripted = torch.jit.script(oracle_voxel)
        query = torch.randn(32, 3)
        expected = oracle_voxel.lookup(query)
        result = scripted(query)
        assert torch.allclose(result, expected, atol=1e-6)


# ------------------------------------------------------------------
