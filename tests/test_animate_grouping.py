"""Which instances animate: DINOtxt thresholds, CLI parsing, absorber freeze."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scion.camera import Camera
from scion.util import dinotxt
from scion.util.dinotxt import DINOtxtHead, classify_instances

_SCRIPT = Path(__file__).parents[1] / "scripts" / "animate_codebook.py"
_spec = importlib.util.spec_from_file_location("animate_codebook_script", _SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

LABELS = ["grass", "leaves", "flower", "sky", "dirt", "wood"]


def test_classify_uses_best_labels_own_threshold(monkeypatch):
    monkeypatch.setattr(dinotxt, "encode_text", lambda head, texts: torch.eye(3))
    head = DINOtxtHead(
        model=nn.Identity(), tokenizer=None, vision_dim=3,
        device=torch.device("cpu"),
    )
    feats = torch.tensor([
        [1.0, 0.0, 0.0],       # best 0 at 1.00 > 0.8 → 0
        [0.75, 0.6614, 0.0],   # best 0 at 0.75 < 0.8 → STATIC, though label 1 > 0.1
        [0.55, 0.60, 0.58],    # best 1 at ~0.60 > 0.1 → 1, though < label 0's 0.8
    ])
    labels, _ = classify_instances(
        head, feats, ["a", "b", "c"], match_threshold=[0.8, 0.1, 0.1],
    )
    assert labels.tolist() == [0, -1, 1]


def test_parse_label_thresholds():
    thr = script._parse_label_thresholds(
        "grass=0.08,leaves=0.10,flower=0.08", LABELS, 0.05,
    )
    assert thr == [0.08, 0.10, 0.08, 0.05, 0.05, 0.05]
    with pytest.raises(ValueError):
        script._parse_label_thresholds("tree=0.1", LABELS, 0.05)


def test_static_groups_freeze():
    labels = torch.tensor([0, 1, 2, 3, 4, 5, -1])
    script._freeze_static_groups(labels, LABELS, ["sky", "dirt", "wood"])
    assert labels.tolist() == [0, 1, 2, -1, -1, -1, -1]


def test_instance_features_average_splats_over_all_views(monkeypatch):
    """Train + eval views both contribute; unseen splats stay zero (cz parity)."""
    from types import SimpleNamespace

    from scion.data.scene import Scene
    from scion.util import dino_init

    calls = {}

    def fake_extract(paths, **kw):
        calls.update(paths=paths, **kw)
        # view 0 → features (1, 2), view 1 → (3, 4), constant over the map
        return [torch.tensor([1.0, 2.0])[:, None, None].expand(2, 4, 4),
                torch.tensor([3.0, 4.0])[:, None, None].expand(2, 4, 4)]

    monkeypatch.setattr(dino_init, "extract_dino_features_cached", fake_extract)
    K = torch.tensor([[50.0, 0, 32], [0, 50.0, 18], [0, 0, 1]])
    cam = Camera(cam_to_world=torch.eye(4), K=K, width=64, height=36)
    scene = Scene(
        cameras=[cam, cam], image_paths=[Path("a.png"), Path("b.png")],
        train_indices=[0], eval_indices=[1],
    )
    # OpenGL camera looks down -z: inst 0's splats are visible, inst 1's is behind.
    means = torch.tensor([[0.0, 0.0, -2.0], [0.1, 0.0, -2.0], [0.0, 0.0, 2.0]])
    model = SimpleNamespace(
        _get_assembled=lambda: SimpleNamespace(means=means, opacities=torch.zeros(3, 1)),
        _cached_inst_idx=torch.tensor([0, 0, 1]),
        instance_positions=torch.zeros(2, 3),
    )
    feats = script.compute_instance_dino_features(
        model, scene, "dinov3_vitl16", 1024, Path("unused"), "cpu",
    )
    assert calls["paths"] == ["a.png", "b.png"]
    assert calls["bf16_autocast"] is False
    assert torch.allclose(feats, torch.tensor([[2.0, 3.0], [0.0, 0.0]]))


def test_static_camera_path_detection():
    K = torch.tensor([[100.0, 0, 64], [0, 100.0, 36], [0, 0, 1]])
    cam = Camera(cam_to_world=torch.eye(4), K=K, width=128, height=72)
    moved = torch.eye(4)
    moved[0, 3] = 0.5
    assert script._is_static_path([cam, cam, cam])
    assert not script._is_static_path(
        [cam, Camera(cam_to_world=moved, K=K, width=128, height=72)],
    )
