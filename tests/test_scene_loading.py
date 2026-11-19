"""scion.data loaders: the behaviours the trainer relies on (filename split,
auto downscale, Blender test split + white alpha compositing), and the camera
models the COLMAP loader refuses."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scion.data import load_blender, load_scene

_MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "OPENCV": 4, "OPENCV_FISHEYE": 5}


def _colmap_scene(root: Path, names, w=3200, h=2000, cameras=(("PINHOLE", (2000.0, 2000.0, 1600.0, 1000.0)),)):
    """A tiny COLMAP scene: images*/ plus a binary model in sparse/0/."""
    for sub, f in (("images", 1), ("images_2", 2), ("images_4", 4)):
        (root / sub).mkdir(parents=True)
        for n in names:
            Image.fromarray(np.zeros((h // f, w // f, 3), np.uint8)).save(root / sub / n)
    model = root / "sparse" / "0"
    model.mkdir(parents=True)
    cams = struct.pack("<Q", len(cameras))
    for cam_id, (name, params) in enumerate(cameras, 1):
        cams += struct.pack("<iiQQ", cam_id, _MODEL_IDS[name], w, h) + struct.pack(f"<{len(params)}d", *params)
    (model / "cameras.bin").write_bytes(cams)
    ims = struct.pack("<Q", len(names))
    for i, n in enumerate(reversed(names)):  # file order is not name order
        ims += struct.pack("<i7di", i + 1, 1, 0, 0, 0, i, 0, 2 * i, 1) + n.encode() + b"\0" + struct.pack("<Q", 0)
    (model / "images.bin").write_bytes(ims)
    pts = struct.pack("<Q", 2)
    for i in range(2):
        pts += struct.pack("<Q3d3BdQ", i, 0.1 * i, 0.2, 0.3, 10, 20, 30, 0.5, 1) + struct.pack("<ii", 1, 0)
    (model / "points3D.bin").write_bytes(pts)
    return root


def test_filename_split_auto_downscale_and_seed_points(tmp_path):
    names = ["eval_a.png", "train_b.png", "train_c.png", "eval_d.png"]
    scene = load_scene(_colmap_scene(tmp_path, names), eval_mode="filename")
    assert [scene.image_paths[i].name for i in scene.train_indices] == ["train_b.png", "train_c.png"]
    assert [scene.image_paths[i].name for i in scene.eval_indices] == ["eval_a.png", "eval_d.png"]
    assert scene.downscale_factor == 2  # 3200 px -> images_2 (1600 px) is the first <= 1600
    cam = scene.cameras[0]
    assert cam.width == 1600 and scene.image_paths[0].parent.name == "images_2"
    assert cam.K[0, 0] == 1000.0 and cam.K[0, 2] == 800.0  # intrinsics follow the images on disk
    assert scene.sparse_points_3d.shape == (2, 3) and scene.sparse_points_rgb.tolist() == [[10, 20, 30]] * 2


@pytest.mark.parametrize("cameras, match", [
    ((("OPENCV", (2000.0, 2000.0, 1600.0, 1000.0, -0.05, 0.0, 0.001, 0.0)),), "camera model OPENCV"),
    ((("OPENCV_FISHEYE", (2000.0, 2000.0, 1600.0, 1000.0, 0.0, 0.0, 0.0, 0.0)),), "OPENCV_FISHEYE"),
    ((("PINHOLE", (2000.0, 2000.0, 1600.0, 1000.0)), ("PINHOLE", (2000.0, 2000.0, 1600.0, 1000.0))),
     "2 cameras; one shared pinhole camera only"),
])
def test_distorted_or_multi_camera_models_are_refused(tmp_path, cameras, match):
    root = _colmap_scene(tmp_path, ["train_a.png", "eval_b.png"], cameras=cameras)
    with pytest.raises(ValueError, match=match):
        load_scene(root, eval_mode="filename", downscale_factor=4)


def test_split_file_mode_drops_unlisted_and_refuses_bad_lists(tmp_path):
    root = _colmap_scene(tmp_path, ["a.png", "b.png", "c.png", "d.png", "e.png"])
    (root / "split.json").write_text(json.dumps(
        {"train": ["a.png", "c.png"], "test": {"id": ["b.png"], "ood": ["e.png"]}}))
    scene = load_scene(root, eval_mode="split", downscale_factor=4)
    assert [scene.image_paths[i].name for i in scene.train_indices] == ["a.png", "c.png"]
    assert [scene.image_paths[i].name for i in scene.eval_indices] == ["b.png", "e.png"]  # d.png listed nowhere
    assert scene.eval_subsets == {"id": [0], "ood": [1]}  # positions in eval order
    for bad in ({"train": ["a.png"], "test": {"id": ["a.png"]}}, {"train": ["x.png"], "test": {}}):
        (root / "split.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="split.json"):
            load_scene(root, eval_mode="split", downscale_factor=4)


def test_missing_colmap_model_names_the_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="sparse/0"):
        load_scene(tmp_path)


def test_blender_test_split_seed_and_white_alpha(tmp_path):
    for split, n in (("train", 2), ("val", 1), ("test", 3)):
        (tmp_path / split).mkdir()
        frames = []
        for i in range(n):
            rgba = np.zeros((8, 8, 4), np.uint8)
            rgba[..., 0], rgba[..., 3] = 200, 128  # red at alpha 0.5
            Image.fromarray(rgba).save(tmp_path / split / f"r_{i}.png")
            frames.append({"file_path": f"./{split}/r_{i}", "transform_matrix": np.eye(4).tolist()})
        (tmp_path / f"transforms_{split}.json").write_text(
            json.dumps({"camera_angle_x": 0.69, "frames": frames}))
    (tmp_path / "colmap_points.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
        "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n0.5 0 0 10 20 30\n")
    scene = load_blender(tmp_path)
    assert len(scene.train_indices) == 2 and len(scene.eval_indices) == 3  # val unused
    assert scene.sparse_points_3d.shape == (1, 3) and scene.sparse_points_rgb.tolist() == [[10, 20, 30]]
    px = scene.load_image_uint8(0)[0, 0].float()
    a = 128 / 255.0
    expect = torch.tensor([200 * a + 255 * (1 - a), 255 * (1 - a), 255 * (1 - a)])
    assert (px - expect).abs().max() <= 1.0  # truncated to uint8
