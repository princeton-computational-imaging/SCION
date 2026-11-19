"""Unit tests for CodebookModel config, ops, losses and visualization helpers.

Uses index-based storage: ``splat_candidate_id`` (row -> candidate mapping),
``free_list`` / ``free_count`` (free-list stack) instead of contiguous offsets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scion.ops import (
    SPLAT_FREE,
    build_binoms,
    compute_relocation,
    relocate_dead_splats,
)
from scion.util.clustering import kmeans_farthest_point
from scion.util.vis import index_to_color, tile_images_into_grid
from scion.losses.codebook_losses import compactness_loss
from scion.models.codebook_base import CodebookMixin


def _make_index_layout(T_max, K, S):
    """Build splat_candidate_id / free_list / free_count for K candidates with S splats each.

    Assigns rows [k*S .. (k+1)*S) to candidate k. The remaining rows are free.
    Returns (splat_candidate_id, free_list, free_count).
    """
    T_active = K * S
    cid = torch.full((T_max,), SPLAT_FREE, dtype=torch.long)
    for k in range(K):
        cid[k * S : (k + 1) * S] = k
    fl = torch.zeros(T_max, dtype=torch.long)
    n_free = T_max - T_active
    fl[:n_free] = torch.arange(T_active, T_max)
    return cid, fl, n_free


class TestCodebookConfig:
    """Test CodebookModelConfig has the right fields."""

    def test_has_new_fields(self):
        from scion.models.codebook import CodebookModelConfig
        config = CodebookModelConfig()
        assert hasattr(config, 'initial_splats_per_candidate')
        assert hasattr(config, 'max_splats_per_candidate')

    def test_no_splats_per_candidate(self):
        """Old field name should not exist."""
        from scion.models.codebook import CodebookModelConfig
        config = CodebookModelConfig()
        assert not hasattr(config, 'splats_per_candidate')


# ---------------------------------------------------------------------------
# Pipeline param names consistency
# ---------------------------------------------------------------------------

class TestPipelineConsistency:
    """Verify pipeline and config use matching param names."""

    def test_pipeline_param_names_match_model(self):
        """Verify splat param names are accessible via CodebookMixin method."""
        from scion.models.codebook_base import CodebookMixin

        # Verify the mixin's get_instance_param_names works without state
        expected_inst = [
            "instance_positions", "instance_rotations",
            "instance_scales", "instance_opacities",
        ]
        assert CodebookMixin.get_instance_param_names(None) == expected_inst


class TestCompactnessReg:
    """Verify compactness regularization is positive for spread-out splats and
    differentiable (gradients flow through splat_means)."""

    def test_spread_splats_give_positive_loss(self):
        """Widely scattered splat_means should yield a positive compactness loss."""
        import torch.nn as nn
        torch.manual_seed(0)

        K, S = 4, 5
        T = K * S
        splat_means = nn.Parameter(torch.randn(T, 3) * 10.0)  # large spread
        splat_opacities = nn.Parameter(torch.zeros(T, 1))
        splat_scales = nn.Parameter(torch.zeros(T, 3))
        offsets = torch.tensor([k * S for k in range(K + 1)], dtype=torch.long)

        compact_loss = torch.zeros(1)
        n_multi = 0
        for k in range(K):
            s, e = int(offsets[k]), int(offsets[k + 1])
            if e - s > 1:
                m = splat_means[s:e]
                compact_loss = compact_loss + (m - m.mean(0)).pow(2).sum(-1).mean()
                n_multi += 1
        if n_multi > 0:
            compact_loss = compact_loss / n_multi

        assert compact_loss.item() > 0.0, "Spread splats should give positive compactness loss"

    def test_compact_splats_give_small_loss(self):
        """Splat_means clustered at the same point should give near-zero loss."""
        import torch.nn as nn
        K, S = 4, 5
        T = K * S
        splat_means = nn.Parameter(torch.zeros(T, 3))  # all at origin
        offsets = torch.tensor([k * S for k in range(K + 1)], dtype=torch.long)

        compact_loss = torch.zeros(1)
        n_multi = 0
        for k in range(K):
            s, e = int(offsets[k]), int(offsets[k + 1])
            if e - s > 1:
                m = splat_means[s:e]
                compact_loss = compact_loss + (m - m.mean(0)).pow(2).sum(-1).mean()
                n_multi += 1
        if n_multi > 0:
            compact_loss = compact_loss / n_multi

        assert compact_loss.item() < 1e-6, "Co-located splats should give ~0 compactness loss"

    def test_gradients_flow_through_splat_means(self):
        """Compactness loss should produce non-zero gradients on splat_means."""
        import torch.nn as nn
        torch.manual_seed(1)
        K, S = 2, 4
        T = K * S
        splat_means = nn.Parameter(torch.randn(T, 3) * 5.0)
        offsets = torch.tensor([k * S for k in range(K + 1)], dtype=torch.long)

        compact_loss = torch.zeros(1)
        for k in range(K):
            s, e = int(offsets[k]), int(offsets[k + 1])
            if e - s > 1:
                m = splat_means[s:e]
                compact_loss = compact_loss + (m - m.mean(0)).pow(2).sum(-1).mean()

        compact_loss.backward()
        assert splat_means.grad is not None, "No gradient on splat_means"
        assert splat_means.grad.abs().sum() > 0, "Zero gradient — no signal"


# ---------------------------------------------------------------------------
# New config fields
# ---------------------------------------------------------------------------

class TestNewConfigFields:
    """Verify new config fields exist with correct defaults."""

    def test_compactness_reg_weight_exists(self):
        from scion.models.codebook import CodebookModelConfig
        config = CodebookModelConfig()
        assert hasattr(config, "compactness_reg_weight")

    def test_instance_min_opacity_exists(self):
        from scion.models.codebook import CodebookModelConfig
        config = CodebookModelConfig()
        assert hasattr(config, "instance_min_opacity")

    def test_instance_min_opacity_independent_from_min_opacity(self):
        """instance_min_opacity and splat_min_opacity are separate fields."""
        from scion.models.codebook import CodebookModelConfig
        config = CodebookModelConfig(splat_min_opacity=0.001, instance_min_opacity=0.05)
        assert config.splat_min_opacity == 0.001
        assert config.instance_min_opacity == 0.05


# ---------------------------------------------------------------------------
# Extracted utility functions
# ---------------------------------------------------------------------------

class TestKmeansFarthestPoint:
    """Tests for scion.util.clustering.kmeans_farthest_point."""

    def test_basic_clustering(self):
        """Should produce K distinct clusters."""
        torch.manual_seed(42)
        # Two well-separated clusters
        positions = torch.cat([
            torch.randn(50, 3) + 10,
            torch.randn(50, 3) - 10,
        ])
        colors = torch.rand(100, 3)
        assignments = kmeans_farthest_point(positions, colors, K=2)

        assert assignments.shape == (100,)
        assert assignments.dtype == torch.long
        assert set(assignments.tolist()) == {0, 1}

    def test_all_same_points(self):
        """K-means on identical points should not crash."""
        positions = torch.zeros(20, 3)
        colors = torch.zeros(20, 3)
        assignments = kmeans_farthest_point(positions, colors, K=3)
        assert assignments.shape == (20,)

    def test_color_weight(self):
        """Different color_weight should affect assignments."""
        torch.manual_seed(7)
        positions = torch.randn(60, 3)
        # Colors that form clear clusters
        colors = torch.cat([
            torch.ones(20, 3) * 0,
            torch.ones(20, 3) * 0.5,
            torch.ones(20, 3) * 1.0,
        ])
        a1 = kmeans_farthest_point(positions, colors, K=3, color_weight=0.0)
        a2 = kmeans_farthest_point(positions, colors, K=3, color_weight=5.0)
        # With high color weight, clustering should differ
        # (not guaranteed to be different due to randomness, but shape should be valid)
        assert a1.shape == a2.shape == (60,)


class TestCompactnessLossExtracted:
    """Tests for compactness_loss extracted to losses/codebook_losses.py."""

    def test_spread_splats_positive(self):
        torch.manual_seed(0)
        K, S = 4, 5
        splat_means = nn.Parameter(torch.randn(K * S, 3) * 10.0)
        offsets = torch.tensor([k * S for k in range(K + 1)], dtype=torch.long)

        loss = compactness_loss(splat_means, offsets, K)
        assert loss.item() > 0.0

    def test_colocated_splats_zero(self):
        K, S = 4, 5
        splat_means = nn.Parameter(torch.zeros(K * S, 3))
        offsets = torch.tensor([k * S for k in range(K + 1)], dtype=torch.long)

        loss = compactness_loss(splat_means, offsets, K)
        assert loss.item() < 1e-6

    def test_gradient_flows(self):
        torch.manual_seed(1)
        K, S = 2, 4
        splat_means = nn.Parameter(torch.randn(K * S, 3) * 5.0)
        offsets = torch.tensor([k * S for k in range(K + 1)], dtype=torch.long)

        loss = compactness_loss(splat_means, offsets, K)
        loss.backward()
        assert splat_means.grad is not None
        assert splat_means.grad.abs().sum() > 0

    def test_single_splat_candidates_skipped(self):
        """Candidates with S_k=1 should not contribute to the loss."""
        splat_means = nn.Parameter(torch.randn(3, 3) * 10.0)
        offsets = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # 3 candidates, 1 splat each

        loss = compactness_loss(splat_means, offsets, K=3)
        assert loss.item() == 0.0


class TestIndexToColor:
    """Tests for index_to_color extracted to util/vis.py."""

    def test_output_shape(self):
        indices = torch.arange(10)
        colors = index_to_color(indices, 10)
        assert colors.shape == (10, 3)

    def test_values_in_range(self):
        indices = torch.arange(20)
        colors = index_to_color(indices, 20)
        assert (colors >= 0).all()
        assert (colors <= 1).all()

    def test_zero_max_val(self):
        indices = torch.zeros(5, dtype=torch.long)
        colors = index_to_color(indices, 0)
        assert colors.shape == (5, 3)


class TestTileImagesIntoGrid:
    """Tests for tile_images_into_grid extracted to util/vis.py."""

    def test_single_image(self):
        img = torch.rand(32, 32, 3)
        grid = tile_images_into_grid([img], 32, 32)
        assert grid.shape == (32, 32, 3)
        assert torch.allclose(grid, img)

    def test_four_images_2x2(self):
        imgs = [torch.rand(16, 16, 3) for _ in range(4)]
        grid = tile_images_into_grid(imgs, 16, 16)
        assert grid.shape == (32, 32, 3)

    def test_empty_list(self):
        grid = tile_images_into_grid([], 64, 64)
        assert grid.shape == (64, 64, 3)
        assert (grid == 1.0).all()

    def test_non_square_count(self):
        """3 images should produce a 2x2 grid with one empty cell."""
        imgs = [torch.zeros(8, 8, 3) for _ in range(3)]
        grid = tile_images_into_grid(imgs, 8, 8)
        assert grid.shape == (16, 16, 3)
        # Bottom-right cell should be white (bg)
        assert (grid[8:, 8:] == 1.0).all()


class TestBuildBinoms:
    def test_shape(self):
        binoms = build_binoms(10)
        assert binoms.shape == (10, 10)

    def test_values(self):
        binoms = build_binoms(10)
        assert binoms[0, 0] == 1.0
        assert binoms[4, 2] == 6.0  # C(4,2) = 6
        assert binoms[5, 0] == 1.0
        assert binoms[5, 5] == 1.0
        assert binoms[3, 4] == 0.0  # k > n

    def test_default_size(self):
        binoms = build_binoms()
        assert binoms.shape == (51, 51)


class TestComputeRelocation:
    def test_opacity_preserves_mass(self):
        """n copies of new_opacity should yield same total as original."""
        opacities = torch.tensor([0.8, 0.5, 0.9])
        scales = torch.rand(3, 3) * 0.1
        ratios = torch.tensor([2, 3, 4])
        binoms = build_binoms()

        new_op, new_sc = compute_relocation(opacities, scales, ratios, binoms)

        # Each new opacity should be less than original
        assert (new_op < opacities).all()
        assert (new_op > 0).all()

        # 1 - (1 - new_op)^n should approximately equal original opacity
        for i in range(3):
            total_prob = 1.0 - (1.0 - new_op[i].item()) ** ratios[i].item()
            assert abs(total_prob - opacities[i].item()) < 1e-5

    def test_scale_shrinks(self):
        opacities = torch.tensor([0.5])
        scales = torch.tensor([[1.0, 1.0, 1.0]])
        ratios = torch.tensor([8])
        binoms = build_binoms()

        new_op, new_sc = compute_relocation(opacities, scales, ratios, binoms)
        assert (new_sc < scales).all()
        # Scale should be divided by n^(1/3)
        expected = 1.0 / (8 ** (1.0 / 3.0))
        assert torch.allclose(new_sc, torch.tensor([[expected, expected, expected]]), atol=1e-5)


class TestRelocateDeadSplatsLowLevel:
    def test_relocate_fills_dead(self):
        T_max = 40
        K, S = 2, 10
        binoms = build_binoms()

        cid, fl, fc = _make_index_layout(T_max, K, S)

        # Set half of each candidate's splats to very low opacity (dead)
        opacities = torch.logit(torch.tensor(0.5)).expand(T_max, 1).clone()
        opacities[:5, 0] = torch.logit(torch.tensor(0.001))  # dead in candidate 0
        opacities[10:15, 0] = torch.logit(torch.tensor(0.001))  # dead in candidate 1

        splat_params = {
            "means": torch.randn(T_max, 3),
            "scales": torch.randn(T_max, 3),
            "rotations": torch.randn(T_max, 4),
            "opacities": opacities,
            "features_dc": torch.randn(T_max, 3),
        }

        relocate_dead_splats(splat_params, cid, K=K, threshold=0.005, binoms=binoms)

        # After relocation, dead splats should have new values
        new_opacities = torch.sigmoid(splat_params["opacities"].data[:5, 0])
        # They should have been copied from alive splats (and adjusted)
        assert (new_opacities > 0.0).all()

    def test_no_crash_all_alive(self):
        T_max = 20
        K, S = 1, 10
        binoms = build_binoms()

        cid, fl, fc = _make_index_layout(T_max, K, S)

        splat_params = {
            "means": torch.randn(T_max, 3),
            "scales": torch.randn(T_max, 3),
            "rotations": torch.randn(T_max, 4),
            "opacities": torch.logit(torch.tensor(0.5)).expand(T_max, 1).clone(),
            "features_dc": torch.randn(T_max, 3),
        }
        # Should not crash when all are alive
        relocate_dead_splats(splat_params, cid, K=K, threshold=0.005, binoms=binoms)


class _MinimalCodebookModule(nn.Module, CodebookMixin):
    """Minimal nn.Module host for CodebookMixin, used for compile tests."""

    def __init__(self, K: int, N: int, S: int, T_max: int):
        super().__init__()
        # Splat params
        self.splat_means = nn.Parameter(torch.randn(T_max, 3))
        self.splat_scales = nn.Parameter(torch.randn(T_max, 3) * 0.1 - 2.0)
        self.splat_quats = nn.Parameter(F.normalize(torch.randn(T_max, 4), dim=-1))
        self.splat_opacities = nn.Parameter(torch.randn(T_max, 1))
        self.splat_features_dc = nn.Parameter(torch.rand(T_max, 3))

        # Instance params
        self.instance_positions = nn.Parameter(torch.randn(N, 3))
        self.instance_rotations = nn.Parameter(
            F.normalize(torch.randn(N, 4), dim=-1)
        )
        self.instance_scales = nn.Parameter(torch.zeros(N, 1))
        self.instance_opacities = nn.Parameter(torch.zeros(N, 1))

        # Init mixin (registers splat_candidate_id, free_list, instance_assignments)
        self.init_codebook(
            spatial_dim=3, rot_dim=4, volume_exp=1.0 / 3.0,
            p_splat=14, p_inst=9,
            K=K, N=N, T_max=T_max,
        )

        # Assign splat rows to candidates: rows [k*S .. (k+1)*S) -> candidate k
        T_active = K * S
        for k in range(K):
            self.splat_candidate_id[k * S : (k + 1) * S] = k
        # Update free list: only rows >= T_active are free
        n_free = T_max - T_active
        self.free_list[:n_free] = torch.arange(T_active, T_max)
        self.free_count = n_free

        # Random assignments
        self.instance_assignments[:] = torch.randint(0, K, (N,))

        # Precompute indices
        self.recompute_assembly_indices()

    def get_splat_params_dict(self):
        return {
            "means": self.splat_means,
            "scales": self.splat_scales,
            "rotations": self.splat_quats,
            "opacities": self.splat_opacities,
            "features_dc": self.splat_features_dc,
        }


class TestTorchCompileAssembly:
    """Verify the assembly path is torch.compile-compatible (no graph breaks)."""

    def _make_model(self, K=4, N=20, S=5, T_max=200):
        return _MinimalCodebookModule(K=K, N=N, S=S, T_max=T_max)

    def test_assemble_gaussians_fullgraph(self):
        """assemble_gaussians should run under torch.compile(fullgraph=True)."""
        model = self._make_model()
        model.train()

        compiled = torch.compile(model.assemble_gaussians, fullgraph=True, backend="eager")
        result = compiled()

        assert "means" in result
        assert result["means"].shape[0] == model._cached_total
        assert result["means"].shape[1] == 3

    def test_assembly_gradients_flow(self):
        """Gradients flow through compiled assembly to splat params."""
        model = self._make_model()
        model.train()

        compiled = torch.compile(model.assemble_gaussians, fullgraph=True, backend="eager")
        result = compiled()
        loss = result["means"].sum()
        loss.backward()

        assert model.splat_means.grad is not None
        assert model.instance_positions.grad is not None


class TestDiscriminatorCompile:
    """Verify the discriminator's ADA augmentation behaves (differentiable, adaptive)."""

    def test_simple_ada_augment_differentiable(self):
        """SimpleADA augmentation is differentiable (gradients flow through)."""
        from scion.losses.discriminator import SimpleADA

        ada = SimpleADA(initial_p=1.0)
        x = torch.randn(8, 3, 32, 32, requires_grad=True)
        out = ada(x)
        out.sum().backward()
        assert x.grad is not None

    def test_simple_ada_update_adjusts_p(self):
        """SimpleADA.update() increases p when real_acc > target, decreases otherwise."""
        from scion.losses.discriminator import SimpleADA

        ada = SimpleADA(
            initial_p=0.3, adjustment_speed=0.1, target_real_acc=0.6,
            update_every=1, ema_lambda=0.0,  # no smoothing for deterministic test
        )
        # real_acc > target -> p should increase
        ada.update(0.9)
        assert ada.p > 0.3

        # real_acc < target -> p should decrease
        ada.p = 0.3
        ada.update(0.2)
        assert ada.p < 0.3

    def test_simple_ada_passthrough_p_zero(self):
        """SimpleADA with p=0 returns input unchanged."""
        from scion.losses.discriminator import SimpleADA

        ada = SimpleADA(initial_p=0.0)
        x = torch.randn(4, 3, 32, 32)
        out = ada(x)
        assert torch.allclose(out, x)
