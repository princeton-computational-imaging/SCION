"""Scene-size-dependent LR scaling (scion.engine.run).

Mirrors INRIA 3DGS's ``getNerfppNorm``: ``scene_lr_scale`` =
``1.1 * max(||c_i - mean(c)||)`` over training cameras, applied *only* to
the world-space "means" LRs (``splat_means``, ``instance_positions``).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scion.camera import Camera
from scion.engine.run import camera_radius_scale, learning_rates

BASE_LRS = {
    "splat_means": 1.6e-4, "splat_features_dc": 0.0025, "splat_opacities": 0.02,
    "splat_scales": 0.002, "splat_quats": 0.001, "instance_positions": 7.0e-5,
    "instance_rotations": 7.0e-4, "instance_scales": 5.0e-4, "instance_opacities": 0.02,
}
MEANS = ("splat_means", "instance_positions")


def _scene(centers):
    cams = []
    for c in centers:
        c2w = torch.eye(4)[:3].clone()
        c2w[:, 3] = torch.tensor(c)
        cams.append(Camera(cam_to_world=c2w, K=torch.eye(3), width=1, height=1))
    return SimpleNamespace(train_cameras=lambda: cams)


def _cfg(scene_lr_scale):
    return OmegaConf.create({"method": {"optimizer": {**BASE_LRS, "scene_lr_scale": scene_lr_scale}}})


def test_camera_radius_three_collinear_centers():
    """Centers 0, 1, 2 on x -> mean 1, max distance 1, scale 1.1."""
    assert camera_radius_scale(_scene([[0, 0, 0], [1, 0, 0], [2, 0, 0]])) == pytest.approx(1.1)


def test_degenerate_cameras_raise():
    with pytest.raises(ValueError, match="non-positive scene radius"):
        camera_radius_scale(_scene([[1, 2, 3], [1, 2, 3]]))


@pytest.mark.parametrize("sls,scale", [("auto", 2.2), (4.0, 4.0), (1.0, 1.0)])
def test_only_means_lrs_scale(sls, scale):
    lrs = learning_rates(_cfg(sls), _scene([[0, 0, 0], [4, 0, 0]]))  # radius 1.1 * 2
    for key, lr in BASE_LRS.items():
        assert lrs[key] == pytest.approx(lr * (scale if key in MEANS else 1.0)), key


def test_bad_scene_lr_scale_string_raises():
    with pytest.raises(ValueError, match="'auto' or a number"):
        learning_rates(_cfg("big"), _scene([[0, 0, 0], [1, 0, 0]]))
