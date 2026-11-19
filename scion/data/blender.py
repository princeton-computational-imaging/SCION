# Copyright 2022 the nerfstudio team. Copyright 2026 the SCION
# authors. Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may
# obtain a copy at http://www.apache.org/licenses/LICENSE-2.0.
#
# This file contains code modified from nerfstudio
# (https://github.com/nerfstudio-project/nerfstudio); see the module
# docstring and THIRD_PARTY_NOTICES.md for what was changed.
"""Blender-format (Synthetic-NeRF) scene loader.

Reads ``transforms_{train,val,test}.json`` (the NeRF synthetic format) and
returns a :class:`scion.data.Scene`.

Math is byte-for-byte from
``nerfstudio.data.dataparsers.blender_dataparser.Blender``: no
orient/center, ``scale_factor=1.0`` by default applied directly to the
translation column. Camera poses are kept in nerfstudio (OpenGL)
c2w convention — no axis flip. The Camera's ``viewmat()`` handles the
GL→CV flip downstream at the rasterizer boundary.

Images carry an alpha channel; the foreground/background composite
happens at the loader / model boundary (see ``loader.py`` and the
``background_color`` knob in the training config) — not here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import math
import numpy as np
import torch
from PIL import Image

from scion.camera import Camera
from scion.data.scene import Scene
from scion.data.synthetic import SEED_NAME


def load_blender(
    scene_dir: Path,
    split: Literal["train", "val", "test", "all"] = "all",
    scale_factor: float = 1.0,
) -> Scene:
    """Load a NeRF synthetic (Blender) scene.

    Parameters
    ----------
    scene_dir
        Directory containing ``transforms_{train,val,test}.json`` and the
        ``train/`` / ``test/`` / ``val/`` image folders.
    split
        Which JSON to read. ``"all"`` reads train + test (train/eval indices
        = the train-JSON / test-JSON frames: nerfstudio's eval split for
        ``ns-eval`` and the paper protocol; val frames are not used).
    scale_factor
        Scalar multiplied into the translation column of every pose.
        Default 1.0 matches nerfstudio.

    A ``colmap_points.ply`` in ``scene_dir`` (written by
    :mod:`scion.data.synthetic`) becomes the seed point cloud. RGBA
    images are composited onto white at load (nerfstudio's ``alpha_color``).
    """
    scene_dir = Path(scene_dir)

    if split == "all":
        splits_to_read = ["train", "test"]
    else:
        splits_to_read = [split]

    all_frames: list[dict] = []
    split_membership: list[str] = []
    camera_angle_x: Optional[float] = None
    for s in splits_to_read:
        p = scene_dir / f"transforms_{s}.json"
        if not p.exists():
            raise FileNotFoundError(f"transforms_{s}.json not found at {p}")
        with open(p) as f:
            meta = json.load(f)
        if camera_angle_x is None:
            camera_angle_x = float(meta["camera_angle_x"])
        for fr in meta["frames"]:
            all_frames.append(fr)
            split_membership.append(s)

    if camera_angle_x is None:
        raise FileNotFoundError(f"No transforms_*.json files found in {scene_dir}")

    # Read first image to discover (H, W); matches BlenderDataParser exactly.
    first_path = _blender_image_path(scene_dir, all_frames[0]["file_path"])
    img0 = np.asarray(Image.open(first_path), dtype=np.uint8)
    image_height, image_width = img0.shape[:2]

    focal = 0.5 * image_width / math.tan(0.5 * camera_angle_x)
    cx = image_width / 2.0
    cy = image_height / 2.0
    K = torch.tensor(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    cameras: list[Camera] = []
    image_paths: list[Path] = []
    for fr in all_frames:
        pose = np.array(fr["transform_matrix"], dtype=np.float32)  # (4, 4)
        c2w_full = torch.from_numpy(pose)
        # Match BlenderDataParser: camera_to_world = poses[:, :3] (3-row slice),
        # then translation column scaled. Note that nerfstudio's
        # ``camera_to_world[..., 3] *= scale_factor`` scales the 3-row 4th
        # column — i.e. only the translation; the bottom homogeneous row is
        # already dropped.
        c2w = c2w_full[:3, :4].clone()
        c2w[:, 3] = c2w[:, 3] * scale_factor
        cameras.append(
            Camera(
                cam_to_world=c2w,
                K=K.clone(),
                width=image_width,
                height=image_height,
                metadata={"file_path": fr["file_path"]},
            )
        )
        image_paths.append(_blender_image_path(scene_dir, fr["file_path"]))

    # Train / eval indices follow the JSON they came from: train.json -> train,
    # test.json -> eval (the standard Blender protocol; val is not read).
    train_indices = [i for i, s in enumerate(split_membership) if s == "train"]
    eval_indices = [i for i, s in enumerate(split_membership) if s in ("val", "test")]

    xyz = rgb = None
    ply = scene_dir / SEED_NAME
    if ply.exists():
        from scion.data.colmap import _load_ply

        xyz, rgb = _load_ply(ply)
        xyz = xyz * scale_factor

    return Scene(
        cameras=cameras,
        image_paths=image_paths,
        train_indices=train_indices,
        eval_indices=eval_indices,
        sparse_points_3d=xyz,
        sparse_points_rgb=rgb,
        scene_scale=float(scale_factor),
        scene_translate=None,
        coordinate_convention="opengl",
        dataparser_transform=None,
        downscale_factor=1,
        alpha_color=(1.0, 1.0, 1.0),
    )


def _blender_image_path(scene_dir: Path, file_path: str) -> Path:
    """Resolve a Blender JSON ``file_path`` (e.g. ``./train/r_0``) to an
    on-disk ``.png``. Mirrors BlenderDataParser exactly."""
    return scene_dir / (file_path.replace("./", "") + ".png")
