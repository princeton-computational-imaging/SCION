"""DINO feature cache: scripts/preprocess_dino.py fills exactly the cache that
training reads (same train-split key), the key tracks names / model /
resolution / PCA dim / downscale, and writes are atomic."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

import scion.data.synthetic as synthetic
import scion.util.dino_init as dino_init
from scion.data import load_scene
from scion.engine.run import scene_kwargs
from scion.util.provenance import compose

from test_scene_loading import _colmap_scene

REPO = Path(__file__).resolve().parents[1]


def _fake_extract(image_paths, **_):
    return [torch.full((2, 3, 4), float(i)) for i in range(len(image_paths))]


def _no_extract(*_, **__):
    raise AssertionError("cache miss")


PLY = ("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
       "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
       "end_header\n0 0 0 1 2 3\n")


def test_preprocess_seeds_a_synthetic_scene_first(tmp_path, monkeypatch):
    """The runner's per-scene step triangulates the seed points before any job runs."""
    import numpy as np
    from PIL import Image

    scene_dir = tmp_path / "toylego"
    for split in ("train", "test"):
        (scene_dir / split).mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 4), np.uint8)).save(scene_dir / split / "r_0.png")
        (scene_dir / f"transforms_{split}.json").write_text(json.dumps(
            {"camera_angle_x": 0.7, "frames": [{"file_path": f"./{split}/r_0",
                                                "transform_matrix": np.eye(4).tolist()}]}))
    pre = _preprocess_module()
    seeded = []

    def fake_seed(path):
        seeded.append(path)
        (path / synthetic.SEED_NAME).write_text(PLY)

    monkeypatch.setattr(synthetic, "write_seed", fake_seed)
    monkeypatch.setattr(dino_init, "_extract_dino_features", _fake_extract)
    monkeypatch.setattr(sys, "argv", ["preprocess_dino.py", "--scene", "toylego",
                                      "--data-root", str(tmp_path), "--device", "cpu"])
    pre.main()
    assert seeded == [scene_dir] and list((scene_dir / "dino").glob("*.pt"))
    pre.main()                                     # idempotent: seed exists, not re-run
    assert len(seeded) == 1


def _preprocess_module():
    spec = importlib.util.spec_from_file_location("preprocess_dino", REPO / "scripts" / "preprocess_dino.py")
    pre = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre)
    return pre


def test_preprocess_and_training_share_the_cache(tmp_path, monkeypatch):
    names = [f"frame_{i:03d}.png" for i in range(10)]
    scene_dir = _colmap_scene(tmp_path / "toyscene", names, w=640, h=400)
    pre = _preprocess_module()

    monkeypatch.setattr(dino_init, "_extract_dino_features", _fake_extract)
    monkeypatch.setattr(sys, "argv", ["preprocess_dino.py", "--scene", "toyscene",
                                      "--data-root", str(tmp_path), "--device", "cpu"])
    pre.main()
    cached = sorted((scene_dir / "dino").glob("*.pt"))
    assert len(cached) == 1 and not list((scene_dir / "dino").glob("*.tmp"))

    # Training's view of the same job (default split: no dataset group).
    cfg = compose(["method=codebook", "scene=toyscene"])
    scene = load_scene(scene_dir, **scene_kwargs(cfg))
    monkeypatch.setattr(dino_init, "_extract_dino_features", _no_extract)
    m = cfg.method.model
    maps = dino_init.train_feature_maps(scene, scene_dir, model_name=m.dino_direct_model,
                                        resolution=m.dino_direct_resolution,
                                        pca_dim=m.dino_direct_pca_dim, device="cpu")
    assert len(maps) == len(scene.train_indices) < len(names)  # train split only

    # A different split / PCA dim / downscale is a different cache.
    with pytest.raises(AssertionError, match="cache miss"):
        dino_init.train_feature_maps(scene, scene_dir, model_name=m.dino_direct_model,
                                     resolution=m.dino_direct_resolution, pca_dim=32, device="cpu")
    scene.train_indices = scene.train_indices[:-1]
    with pytest.raises(AssertionError, match="cache miss"):
        dino_init.train_feature_maps(scene, scene_dir, model_name=m.dino_direct_model,
                                     resolution=m.dino_direct_resolution,
                                     pca_dim=m.dino_direct_pca_dim, device="cpu")
