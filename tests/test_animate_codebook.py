"""Tests for scion.util.animate_codebook helpers."""

import math
from dataclasses import dataclass, field

import pytest
import torch

from scion.util.animate import AnimationConfig
from scion.util.animate_codebook import (
    GroupAnimationSpec,
    STATIC_LABEL,
    aggregate_splat_features_to_instances,
    apply_wind_to_bundle,
    instance_heights_along_up,
    instance_up_axes,
)
from scion.util.transformation import quat_mul, quat_rotate


# ---------------------------------------------------------------------------
# GaussianBundle stand-in (mirrors scion.models.codebook_base.GaussianBundle
# surface we actually touch: means + quats — kept local to avoid importing the
# full model tree just for tests)
# ---------------------------------------------------------------------------

@dataclass
class _FakeBundle:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor = field(default=None)

    def __post_init__(self):
        # Default: log(1) = 0 (unit world scale) for every splat — keeps the
        # size-weight path a no-op unless a test supplies its own scales.
        if self.scales is None:
            self.scales = torch.zeros_like(self.means)


def _identity_quats(N: int) -> torch.Tensor:
    q = torch.zeros(N, 4)
    q[:, 0] = 1.0
    return q


# ---------------------------------------------------------------------------
# instance_up_axes
# ---------------------------------------------------------------------------


class TestInstanceUpAxes:
    def test_identity_is_z(self):
        q = _identity_quats(3)
        up = instance_up_axes(q)
        expected = torch.tensor([[0.0, 0.0, 1.0]] * 3)
        assert torch.allclose(up, expected, atol=1e-5)

    def test_90_about_x_sends_z_to_negative_y(self):
        # 90° around X axis: (w, x, y, z) = (cos 45°, sin 45°, 0, 0)
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        q = torch.tensor([[c, s, 0.0, 0.0]])
        up = instance_up_axes(q)
        # R_x(90°) · (0,0,1) = (0, -1, 0)
        assert torch.allclose(up, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5)


# ---------------------------------------------------------------------------
# instance_heights_along_up
# ---------------------------------------------------------------------------


class TestHeights:
    def test_height_matches_max_projection(self):
        inst_pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        up = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        means = torch.tensor([
            [0.0, 0.0, 0.5],   # inst 0
            [0.0, 0.0, 1.5],   # inst 0 → max
            [10.0, 0.0, 0.2],  # inst 1
            [10.0, 0.0, 0.8],  # inst 1 → max
        ])
        idx = torch.tensor([0, 0, 1, 1])
        heights = instance_heights_along_up(means, idx, inst_pos, up)
        assert torch.allclose(heights, torch.tensor([1.5, 0.8]), atol=1e-6)


# ---------------------------------------------------------------------------
# aggregate_splat_features_to_instances
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_opacity_weighted_mean(self):
        # 3 instances, 5 splats
        feats = torch.tensor([
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 4.0],
            [2.0, 2.0],
        ])
        idx = torch.tensor([0, 0, 1, 1, 2])
        opa = torch.tensor([1.0, 3.0, 2.0, 2.0, 1.0])
        out = aggregate_splat_features_to_instances(feats, idx, opa, 3)

        # inst 0: (1*1 + 3*3)/(1+3) = 10/4 = 2.5
        assert torch.allclose(out[0], torch.tensor([2.5, 0.0]), atol=1e-6)
        # inst 1: ((0*2+0*2)/4, (2*2+4*2)/4) = (0, 3.0)
        assert torch.allclose(out[1], torch.tensor([0.0, 3.0]), atol=1e-6)
        # inst 2: single splat, weight 1 → passes through
        assert torch.allclose(out[2], torch.tensor([2.0, 2.0]), atol=1e-6)

    def test_empty_instance_yields_zero(self):
        feats = torch.tensor([[1.0, 2.0]])
        idx = torch.tensor([0])
        opa = torch.tensor([1.0])
        out = aggregate_splat_features_to_instances(feats, idx, opa, 2)
        assert torch.allclose(out[1], torch.zeros(2))


# ---------------------------------------------------------------------------
# apply_wind_to_bundle — STATIC stays put, pivot invariance holds
# ---------------------------------------------------------------------------


def _make_group_spec(
    strength_deg: float,
    label: str = "g",
    pivot_mode: str = "center",
    size_impact_alpha: float = 0.0,
) -> GroupAnimationSpec:
    """Defaults match the pre-size/base-pivot rigid-rotation behaviour so
    the rigid-body invariants in these tests still hold.  Individual tests
    opt into base-pivot / size-weighted scaling via kwargs."""
    cfg = AnimationConfig(
        wind_strength=strength_deg,
        axis_perturbation=0.0,
        flutter_amplitude=0.0,
    )
    return GroupAnimationSpec(
        label=label,
        animation_config=cfg,
        bend_axis_mode="world_z",
        per_splat_flutter=False,
        pivot_mode=pivot_mode,
        size_impact_alpha=size_impact_alpha,
    )


class TestApplyWindToBundle:
    def test_static_instance_untouched(self):
        torch.manual_seed(0)
        inst_pos = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        inst_rot = _identity_quats(2)
        means = torch.tensor([
            [0.0, 0.0, 0.5],   # inst 0 — animated
            [0.1, 0.0, 0.8],   # inst 0
            [5.0, 0.0, 0.3],   # inst 1 — STATIC
            [5.2, 0.0, 0.7],   # inst 1
        ]).clone()
        quats = _identity_quats(4).clone()
        bundle = _FakeBundle(means=means, quats=quats)

        cached_inst_idx = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0, STATIC_LABEL])

        orig_means = means.clone()
        orig_quats = quats.clone()

        specs = {0: _make_group_spec(30.0)}
        apply_wind_to_bundle(
            bundle,
            cached_inst_idx=cached_inst_idx,
            inst_pos=inst_pos,
            inst_rot=inst_rot,
            inst_heights=torch.tensor([0.8, 0.7]),
            inst_log_scales=torch.zeros(2),
            labels=labels,
            specs_by_label=specs,
            states_by_label={0: None},
            t=1.0,
            dt=1 / 30.0,
        )

        # STATIC instance's splats unchanged
        assert torch.allclose(bundle.means[2:], orig_means[2:], atol=1e-7)
        assert torch.allclose(bundle.quats[2:], orig_quats[2:], atol=1e-7)

    def test_zero_strength_is_noop(self):
        inst_pos = torch.tensor([[0.0, 0.0, 0.0]])
        inst_rot = _identity_quats(1)
        means = torch.tensor([[0.2, 0.1, 0.7], [-0.1, 0.3, 0.9]]).clone()
        quats = _identity_quats(2).clone()
        bundle = _FakeBundle(means=means, quats=quats)

        orig_means = means.clone()
        orig_quats = quats.clone()

        apply_wind_to_bundle(
            bundle,
            cached_inst_idx=torch.tensor([0, 0]),
            inst_pos=inst_pos,
            inst_rot=inst_rot,
            inst_heights=torch.tensor([0.9]),
            inst_log_scales=torch.zeros(1),
            labels=torch.tensor([0]),
            specs_by_label={0: _make_group_spec(0.0)},
            states_by_label={0: None},
            t=0.5,
            dt=1 / 30.0,
        )
        assert torch.allclose(bundle.means, orig_means, atol=1e-5)
        assert torch.allclose(bundle.quats, orig_quats, atol=1e-5)

    def test_size_weight_scales_per_splat_angle(self):
        """Larger splats should move more than smaller ones when
        ``size_impact_alpha > 0`` (per-splat angle is linear in size)."""
        pivot = torch.tensor([[0.0, 0.0, 0.0]])
        inst_rot = _identity_quats(1)
        # Two co-located splats 1m above pivot along +z; one tiny (exp=0.1),
        # one large (exp=2.0).
        orig_means = torch.tensor([
            [0.0, 0.0, 1.0],   # small splat
            [0.0, 0.0, 1.0],   # large splat
        ])
        means = orig_means.clone()
        quats = _identity_quats(2).clone()
        log_scales = torch.log(torch.tensor([
            [0.1, 0.1, 0.1],
            [2.0, 2.0, 2.0],
        ]))
        bundle = _FakeBundle(means=means, quats=quats, scales=log_scales)

        apply_wind_to_bundle(
            bundle,
            cached_inst_idx=torch.tensor([0, 0]),
            inst_pos=pivot,
            inst_rot=inst_rot,
            inst_heights=torch.tensor([1.0]),
            inst_log_scales=torch.zeros(1),
            labels=torch.tensor([0]),
            specs_by_label={0: _make_group_spec(
                30.0, pivot_mode="center", size_impact_alpha=1.0,
            )},
            states_by_label={0: None},
            t=0.25,
            dt=1 / 30.0,
        )
        # Displacement magnitude: large splat > small splat.
        disp_small = (bundle.means[0] - orig_means[0]).norm()
        disp_large = (bundle.means[1] - orig_means[1]).norm()
        assert disp_large > disp_small, (
            f"expected large-splat disp ({disp_large}) > small ({disp_small})"
        )

    def test_pivot_invariance(self):
        """Mean-offset-from-pivot should be preserved under rigid rotation.

        For a single instance, the mean of (means - pivot) after mutation
        should equal the rotated mean of (means - pivot) before.
        """
        torch.manual_seed(1)
        pivot = torch.tensor([[1.0, 2.0, 0.0]])
        inst_rot = _identity_quats(1)
        local = torch.tensor([
            [0.1, 0.0, 0.5],
            [-0.1, 0.2, 0.8],
            [0.0, -0.1, 1.0],
        ])
        means = (pivot + local).clone()
        quats = _identity_quats(3).clone()
        bundle = _FakeBundle(means=means, quats=quats)

        before_mean_offset = (means - pivot).mean(dim=0)

        apply_wind_to_bundle(
            bundle,
            cached_inst_idx=torch.tensor([0, 0, 0]),
            inst_pos=pivot,
            inst_rot=inst_rot,
            inst_heights=torch.tensor([1.0]),
            inst_log_scales=torch.zeros(1),
            labels=torch.tensor([0]),
            specs_by_label={0: _make_group_spec(20.0)},
            states_by_label={0: None},
            t=0.3,
            dt=1 / 30.0,
        )

        after_mean_offset = (bundle.means - pivot).mean(dim=0)
        # Both before and after vectors should have the same norm
        # (rigid rotation preserves length).
        assert torch.isclose(
            before_mean_offset.norm(), after_mean_offset.norm(), atol=1e-5,
        )

    def test_pivot_down_offset_moves_splat_at_pivot(self):
        """A splat sitting on the centre pivot only moves once the pivot drops."""
        inst_pos = torch.tensor([[0.0, 0.0, 0.0]])

        def run(offset: float) -> torch.Tensor:
            spec = _make_group_spec(30.0, pivot_mode="center")
            spec.pivot_down_offset = offset
            bundle = _FakeBundle(
                means=torch.zeros(1, 3), quats=_identity_quats(1).clone(),
            )
            apply_wind_to_bundle(
                bundle,
                cached_inst_idx=torch.tensor([0]),
                inst_pos=inst_pos,
                inst_rot=_identity_quats(1),
                inst_heights=torch.tensor([1.0]),
                inst_log_scales=torch.zeros(1),
                labels=torch.tensor([0]),
                specs_by_label={0: spec},
                states_by_label={0: None},
                t=0.25,
                dt=1 / 30.0,
            )
            return bundle.means[0]

        assert torch.allclose(run(0.0), torch.zeros(3), atol=1e-6)
        assert run(0.1).norm() > 1e-4


def test_animation_yamls_parse():
    """Every shipped animation YAML maps onto AnimationConfig / GroupAnimationSpec."""
    from pathlib import Path

    import yaml

    anim_dir = Path(__file__).parents[1] / "config" / "animation_codebook"
    yamls = sorted(anim_dir.glob("*.yaml"))
    assert yamls
    spec_fields = set(GroupAnimationSpec.__dataclass_fields__)
    for path in yamls:
        data = yaml.safe_load(path.read_text())
        AnimationConfig(**data["animation_config"])
        for key in data:
            if key != "animation_config":
                # the script maps YAML ``bend_axis`` onto ``bend_axis_mode``
                field_name = "bend_axis_mode" if key == "bend_axis" else key
                assert field_name in spec_fields, (path.name, key)


# ---------------------------------------------------------------------------
# motion_diagnostics — projection + flow colouring
# ---------------------------------------------------------------------------


class TestMotionDiagnostics:
    def test_projection_matches_pinhole_and_masks_behind(self):
        from scion.util.motion_diagnostics import project_world_to_pixels

        K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        pts = torch.tensor([[0.2, -0.1, 2.0], [0.0, 0.0, -1.0]])
        uv, valid = project_world_to_pixels(pts, torch.eye(4), K)
        assert valid.tolist() == [True, False]
        assert torch.allclose(uv[0], torch.tensor([60.0, 35.0]))
        assert torch.isnan(uv[1]).all()

    def test_flow_colours(self):
        import numpy as np

        from scion.util.motion_diagnostics import instance_flow2d_to_rgb

        import matplotlib.colors as mcolors

        # +u, +v, -u, -v (image coords, v down), still, behind the camera.
        dx = np.array([5.0, 0.0, -5.0, 0.0, 0.0, np.nan])
        dy = np.array([0.0, 5.0, 0.0, -5.0, 0.0, 0.0])
        rgb = instance_flow2d_to_rgb(dx, dy)
        hsv = mcolors.rgb_to_hsv(rgb[:4])
        assert np.allclose(hsv[:, 0] * 360.0, [60.0, 150.0, 240.0, 330.0], atol=1e-3)
        assert np.allclose(hsv[:, 1], 1.0)        # equal speeds → full saturation
        assert np.allclose(rgb[4], 1.0)           # still → white
        assert np.allclose(rgb[5], 0.0)           # behind camera → black
        assert np.allclose(instance_flow2d_to_rgb(np.zeros(3), np.zeros(3)), 1.0)
