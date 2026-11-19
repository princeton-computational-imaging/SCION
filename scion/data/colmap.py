# Copyright 2022 the nerfstudio team. Copyright 2026 the SCION
# authors. Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may
# obtain a copy at http://www.apache.org/licenses/LICENSE-2.0.
#
# This file contains code modified from nerfstudio
# (https://github.com/nerfstudio-project/nerfstudio); see the module
# docstring and THIRD_PARTY_NOTICES.md for what was changed.
"""COLMAP scene loader.

Reads a scene in the layout the Mip-NeRF 360 and 3DGS Tanks & Temples /
Deep Blending archives ship: ``images/`` (plus optional downscaled
``images_N/``) and a COLMAP model in ``sparse/0/`` (``cameras.bin``,
``images.bin``, ``points3D.bin``). Returns a :class:`scion.data.Scene`.

Poses go through the same conversion as nerfstudio's ``ns-process-data``
(``colmap_to_json``: OpenCV -> OpenGL camera axes, world z-up swap) and
the same orient / center / scale as its dataparser
(``auto_orient_and_center_poses``, ported below), so poses and seed points
are bit-identical to nerfstudio's. Intrinsics are scaled to the size of the images
actually read, so ``images_N/`` folders and archives whose model is at a
higher resolution than its images (the 3DGS T&T / DB bundle) both work.

Defaults match nerfstudio: ``orientation_method='up'``,
``center_method='poses'``, ``auto_scale_poses=True``, ``eval_mode='fraction'``
with ``train_split_fraction=0.9``.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Literal, Optional

import math
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from scion.camera import Camera
from scion.data.scene import Scene


# Public API --------------------------------------------------------------


def load_colmap(
    scene_dir: Path,
    downscale_factor: Optional[int] = 1,
    orientation_method: Literal["up", "pca", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
    auto_scale_poses: bool = True,
    scale_factor: float = 1.0,
    eval_mode: Literal["fraction", "interval", "all", "filename", "split"] = "fraction",
    train_split_fraction: float = 0.9,
    eval_interval: int = 8,
    load_sparse_points: bool = True,
) -> Scene:
    """Load a COLMAP scene (``images*/`` + ``sparse/0/``).

    Parameters
    ----------
    scene_dir
        Directory containing ``images/`` (or ``images_N/``) and ``sparse/0/``.
    downscale_factor
        ``1`` reads ``images/``; ``N`` reads ``images_N/``. ``None`` picks the
        smallest factor whose image max side is <= 1600 px among the
        ``images_N/`` folders on disk (nerfstudio's auto rule).
    orientation_method, center_method, auto_scale_poses, scale_factor
        Match nerfstudio's dataparser semantics.
    eval_mode, train_split_fraction, eval_interval
        Match nerfstudio's split semantics (``filename``: basenames contain
        ``train`` / ``eval``). ``split`` reads ``scene_dir/split.json``
        (``{"train": [names], "test": {bucket: [names]}}``): eval is every
        test bucket (each also scored on its own, ``Scene.eval_subsets``);
        images listed nowhere are dropped.
    load_sparse_points
        Load ``sparse/0/points3D.bin`` as the seed cloud, with the same
        orient / scale transform as the poses.

    Returns
    -------
    A :class:`Scene` whose cameras are sorted by image name and whose
    train/eval indices index into the parallel cameras / image_paths lists.
    """
    scene_dir = Path(scene_dir)
    model = scene_dir / "sparse" / "0"
    if not (model / "cameras.bin").is_file():
        raise FileNotFoundError(f"no COLMAP model at {model}/ (expected cameras.bin, images.bin, points3D.bin)")

    cameras_bin = _read_cameras(model / "cameras.bin")
    if len(cameras_bin) != 1:
        raise ValueError(f"{model}: {len(cameras_bin)} cameras; one shared pinhole camera only")
    cam_model, cam_w, cam_h, params = next(iter(cameras_bin.values()))
    if cam_model == "SIMPLE_PINHOLE":
        fl_x = fl_y = params[0]
        cx, cy = params[1], params[2]
    elif cam_model == "PINHOLE" or (cam_model == "OPENCV" and not any(params[4:])):
        fl_x, fl_y, cx, cy = params[:4]
    else:
        raise ValueError(f"{model}: camera model {cam_model} (params {params}) is not supported; only "
                         "undistorted pinhole cameras (undistort first, e.g. colmap image_undistorter)")

    images_bin = sorted(_read_images(model / "images.bin"), key=lambda im: im[4])
    if downscale_factor is None:
        downscale_factor = _auto_downscale(scene_dir, images_bin[0][4])
    image_dir = scene_dir / ("images" if downscale_factor == 1 else f"images_{downscale_factor}")
    fnames = [image_dir / im[4] for im in images_bin]

    # nerfstudio's colmap_to_json, in float64 like the original.
    poses = []
    for _, qvec, tvec, _, _ in images_bin:
        w2c = np.concatenate([_qvec2rotmat(qvec), tvec.reshape(3, 1)], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1  # OpenCV -> OpenGL camera axes
        c2w = c2w[np.array([0, 2, 1, 3]), :]  # world z-up
        c2w[2, :] *= -1
        poses.append(c2w)
    poses_t = torch.from_numpy(np.stack(poses).astype(np.float32))  # (N, 4, 4)

    # Auto orient + center.
    oriented, transform_matrix = auto_orient_and_center_poses(
        poses_t, method=orientation_method, center_method=center_method
    )

    # Auto scale.
    scale = 1.0
    if auto_scale_poses:
        scale = scale / float(torch.max(torch.abs(oriented[:, :3, 3])))
    scale = scale * scale_factor
    oriented[:, :3, 3] = oriented[:, :3, 3] * scale

    # Scale intrinsics from the model's resolution to the images on disk.
    width, height = Image.open(fnames[0]).size
    sx, sy = width / cam_w, height / cam_h
    K = torch.tensor(
        [[fl_x * sx, 0.0, cx * sx], [0.0, fl_y * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    cameras: list[Camera] = []
    for i, (im_id, _, _, _, name) in enumerate(images_bin):
        cameras.append(
            Camera(
                cam_to_world=oriented[i, :3, :4].clone(),  # (3, 4), OpenGL
                K=K.clone(),
                width=width,
                height=height,
                metadata={"image_index": i, "file_path": name, "colmap_im_id": im_id},
            )
        )

    # Train / eval split (in sorted-name order, like nerfstudio).
    num_images = len(cameras)
    eval_subsets = None
    if eval_mode == "fraction":
        i_train, i_eval = _split_fraction(num_images, train_split_fraction)
    elif eval_mode == "interval":
        i_train, i_eval = _split_interval(num_images, eval_interval)
    elif eval_mode == "all":
        i_train, i_eval = np.arange(num_images), np.arange(num_images)
    elif eval_mode == "filename":
        i_train, i_eval = _split_filename(fnames)
    elif eval_mode == "split":
        i_train, i_eval, eval_subsets = _split_file(scene_dir / "split.json", [f.name for f in fnames])
    else:
        raise ValueError(f"Unknown eval_mode={eval_mode!r}")

    # Sparse points (optional).
    sparse_xyz: Optional[Tensor] = None
    sparse_rgb: Optional[Tensor] = None
    if load_sparse_points and (model / "points3D.bin").is_file():
        xyz, rgb = _read_points3d(model / "points3D.bin")
        if len(xyz):
            # ns-process-data's world z-up swap, then the orient/center
            # transform (3, 4) and the scale: (homogenized_xyz @ T.T) * scale.
            xyz = xyz.astype(np.float32)[:, [0, 2, 1]] * np.float32([1, 1, -1])
            # 6 decimals, like the ASCII sparse_pc.ply ns-process-data wrote, so the
            # seed points match the ones behind evals/runs/ bit for bit.
            xyz = torch.from_numpy(np.char.mod("%f", xyz.astype(np.float64)).astype(np.float64).astype(np.float32))
            xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)
            sparse_xyz = (xyz_h @ transform_matrix.T) * scale
            sparse_rgb = torch.from_numpy(rgb)

    return Scene(
        cameras=cameras,
        image_paths=fnames,
        train_indices=list(map(int, i_train)),
        eval_indices=list(map(int, i_eval)),
        sparse_points_3d=sparse_xyz,
        sparse_points_rgb=sparse_rgb,
        scene_scale=float(scale),
        scene_translate=None,
        coordinate_convention="opengl",
        dataparser_transform=transform_matrix,
        downscale_factor=int(downscale_factor),
        eval_subsets=eval_subsets,
    )


def _auto_downscale(scene_dir: Path, first_name: str, max_res: int = 1600) -> int:
    """nerfstudio's ``downscale_factor=None`` rule: halve until the full-res
    image's max side is <= ``max_res`` or the next ``images_N`` is missing."""
    w, h = Image.open(scene_dir / "images" / first_name).size
    df = 0
    while max(w, h) / 2 ** df > max_res and (scene_dir / f"images_{2 ** (df + 1)}" / first_name).exists():
        df += 1
    return 2 ** df


# COLMAP binary model (https://colmap.github.io/format.html#binary-file-format)

_CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                  4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
                  8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}


def _read_cameras(path: Path) -> dict:
    """camera_id -> (model name, width, height, params)."""
    buf = path.read_bytes()
    (n,), off, out = struct.unpack_from("<Q", buf), 8, {}
    for _ in range(n):
        cam_id, model_id, w, h = struct.unpack_from("<iiQQ", buf, off)
        name, k = _CAMERA_MODELS[model_id]
        out[cam_id] = (name, w, h, struct.unpack_from(f"<{k}d", buf, off + 24))
        off += 24 + 8 * k
    return out


def _read_images(path: Path) -> list:
    """[(image_id, qvec (4,), tvec (3,), camera_id, name)] in file order."""
    buf = path.read_bytes()
    (n,), off, out = struct.unpack_from("<Q", buf), 8, []
    for _ in range(n):
        im_id, *qt, cam_id = struct.unpack_from("<i7di", buf, off)
        end = buf.index(b"\0", off + 64)
        name = buf[off + 64:end].decode("utf-8")
        (n2d,) = struct.unpack_from("<Q", buf, end + 1)
        off = end + 9 + 24 * n2d  # skip the 2D keypoints
        out.append((im_id, np.array(qt[:4]), np.array(qt[4:]), cam_id, name))
    return out


def _read_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """xyz (N, 3) float64 and rgb (N, 3) uint8, in file order."""
    buf = path.read_bytes()
    (n,), off = struct.unpack_from("<Q", buf), 8
    xyz, rgb = np.empty((n, 3)), np.empty((n, 3), np.uint8)
    for i in range(n):
        _, *p, _, track_len = struct.unpack_from("<Q3d3BdQ", buf, off)
        xyz[i], rgb[i] = p[:3], p[3:]
        off += 51 + 8 * track_len  # skip the track
    return xyz, rgb


def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP's scalar-first quaternion -> rotation (same expression as nerfstudio's)."""
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


# Orient + center port ----------------------------------------------------


def _rotation_matrix_between(a: Tensor, b: Tensor) -> Tensor:
    """Rotation that maps unit vector ``a`` to unit vector ``b``.

    Direct port of ``nerfstudio.cameras.camera_utils.rotation_matrix_between``.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.linalg.cross(a, b)

    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0.0, 0.0]) if abs(a[0]) < eps else torch.tensor([0.0, 1.0, 0.0])
        v = torch.linalg.cross(a, x)
    v = v / torch.linalg.norm(v)
    skew = torch.tensor(
        [
            [0.0, -v[2].item(), v[1].item()],
            [v[2].item(), 0.0, -v[0].item()],
            [-v[1].item(), v[0].item(), 0.0],
        ]
    )
    theta = torch.acos(torch.clip(torch.dot(a, b), -1.0, 1.0))
    return torch.eye(3) + torch.sin(theta) * skew + (1 - torch.cos(theta)) * (skew @ skew)


def _focus_of_attention(poses: Tensor, initial_focus: Tensor) -> Tensor:
    """Port of ``nerfstudio.cameras.camera_utils.focus_of_attention``."""
    active_directions = -poses[:, :3, 2:3]
    active_origins = poses[:, :3, 3:4]
    focus_pt = initial_focus
    active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
    done = False
    while torch.sum(active.int()) > 1 and not done:
        active_directions = active_directions[active]
        active_origins = active_origins[active]
        m = torch.eye(3) - active_directions * torch.transpose(active_directions, -2, -1)
        mt_m = torch.transpose(m, -2, -1) @ m
        focus_pt = torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ active_origins).mean(0)[:, 0]
        active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
        if active.all():
            done = True
    return focus_pt


def auto_orient_and_center_poses(
    poses: Tensor,
    method: Literal["up", "pca", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
) -> tuple[Tensor, Tensor]:
    """Port of ``nerfstudio.cameras.camera_utils.auto_orient_and_center_poses``.

    Returns
    -------
    oriented_poses : ``(N, 3, 4)``
    transform : ``(3, 4)`` — the world-to-oriented transform.
    """
    origins = poses[..., :3, 3]
    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = _focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown center_method: {center_method}")

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))
        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]
        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[1:3, :] = -1 * oriented_poses[1:3, :]
            transform[1:3, :] = -1 * transform[1:3, :]
    elif method in ("up", "vertical"):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == "vertical":
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                up_vertical = Vh[2, :]
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                up = up / torch.linalg.norm(up)
        rotation = _rotation_matrix_between(up, torch.tensor([0.0, 0.0, 1.0]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown method: {method}")

    return oriented_poses, transform


# Splits ------------------------------------------------------------------


def _split_fraction(num_images: int, train_split_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    num_train = math.ceil(num_images * train_split_fraction)
    i_all = np.arange(num_images)
    i_train = np.linspace(0, num_images - 1, num_train, dtype=int)
    i_eval = np.setdiff1d(i_all, i_train)
    return i_train, i_eval


def _split_filename(fnames: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """nerfstudio's ``eval_mode="filename"``: basename contains train / eval."""
    i_train, i_eval = [], []
    for i, f in enumerate(fnames):
        if "train" in f.name:
            i_train.append(i)
        elif "eval" in f.name:
            i_eval.append(i)
        else:
            raise ValueError(f"eval_mode=filename: {f.name} contains neither 'train' nor 'eval'")
    return np.array(i_train, dtype=int), np.array(i_eval, dtype=int)


def _split_file(path: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """``eval_mode="split"``: train / test names from ``split.json``, plus each
    test group as positions in the eval indices."""
    split = json.loads(path.read_text())
    train, test = set(split["train"]), set().union(*split["test"].values())
    missing = (train | test) - set(names)
    if missing or train & test:
        raise ValueError(f"{path}: {len(missing)} names not in the COLMAP model, "
                         f"{len(train & test)} in both train and test")
    i_eval = [i for i, n in enumerate(names) if n in test]
    groups = {g: [p for p, i in enumerate(i_eval) if names[i] in v] for g, v in split["test"].items()}
    return (np.array([i for i, n in enumerate(names) if n in train], dtype=int),
            np.array(i_eval, dtype=int), groups)


def _split_interval(num_images: int, eval_interval: int) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(num_images)
    i_train = all_indices[all_indices % eval_interval != 0]
    i_eval = all_indices[all_indices % eval_interval == 0]
    return i_train, i_eval


# PLY loading -------------------------------------------------------------


def _load_ply(path: Path) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Load XYZ + RGB from a PLY file (the Synthetic-NeRF seed cloud). Tries
    ``plyfile`` first, else parses ASCII PLY with a tiny inline parser.
    """
    try:
        from plyfile import PlyData
        ply = PlyData.read(str(path))
        v = ply["vertex"].data
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        if all(k in v.dtype.names for k in ("red", "green", "blue")):
            rgb = np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.uint8)
        else:
            rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
        return torch.from_numpy(xyz), torch.from_numpy(rgb)
    except ImportError:
        pass
    # Fallback: parse ASCII PLY manually.
    with open(path, "rb") as f:
        head = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            head.append(line)
            if line == "end_header":
                break
        n_vertex = 0
        is_ascii = False
        props: list[tuple[str, str]] = []
        for line in head:
            if line.startswith("format ascii"):
                is_ascii = True
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            if line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))
        if not is_ascii:
            raise RuntimeError(f"Binary PLY unsupported without plyfile: {path}")
        name_to_idx = {name: i for i, (_, name) in enumerate(props)}
        xs, ys, zs, rs, gs, bs = [], [], [], [], [], []
        for _ in range(n_vertex):
            toks = f.readline().decode("ascii").split()
            xs.append(float(toks[name_to_idx["x"]]))
            ys.append(float(toks[name_to_idx["y"]]))
            zs.append(float(toks[name_to_idx["z"]]))
            if "red" in name_to_idx:
                rs.append(int(toks[name_to_idx["red"]]))
                gs.append(int(toks[name_to_idx["green"]]))
                bs.append(int(toks[name_to_idx["blue"]]))
        xyz = torch.tensor(np.stack([xs, ys, zs], axis=-1), dtype=torch.float32)
        if rs:
            rgb = torch.tensor(np.stack([rs, gs, bs], axis=-1), dtype=torch.uint8)
        else:
            rgb = torch.zeros((xyz.shape[0], 3), dtype=torch.uint8)
        return xyz, rgb
