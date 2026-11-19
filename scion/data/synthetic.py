"""Seed points for a Synthetic-NeRF (Blender) scene, triangulated with COLMAP.

Synthetic-NeRF ships no SfM points, and SCION starts each instance at a seed
point. We match SIFT features across the training views and triangulate them
with the dataset's camera poses held fixed (COLMAP ``point_triangulator``), so
the seed uses only the training images and their given poses. The points are
written to ``colmap_points.ply`` in 3DGS's PLY layout (x, y, z, nx, ny, nz, red,
green, blue): copied to ``points3d.ply``, 3DGS-family methods start from the
same points.

    python -m scion.data.synthetic data/lego     # needs `colmap` (in the pixi env)

Training runs this on the first run of a scene without the file
(:func:`scion.engine.run.ensure_synthetic_seed`).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SEED_NAME = "colmap_points.ply"
# Synthetic-NeRF's scene box, where 3DGS draws its random start points. Dropping
# points outside it removes far-off low-parallax matches (under 1% of them).
BOUND = 1.3


def colmap_pose(c2w) -> tuple[np.ndarray, np.ndarray]:
    """Blender (OpenGL) camera-to-world (3x4 or 4x4) -> COLMAP world-to-camera
    ``(qvec, tvec)``, quaternion scalar first."""
    from scipy.spatial.transform import Rotation

    c2w = np.vstack([np.asarray(c2w, np.float64)[:3, :4], [0.0, 0.0, 0.0, 1.0]])
    w2c = np.linalg.inv(c2w @ np.diag([1.0, -1.0, -1.0, 1.0]))  # OpenGL -> OpenCV camera axes
    x, y, z, w = Rotation.from_matrix(w2c[:3, :3]).as_quat()
    return np.array([w, x, y, z]), w2c[:3, 3]


def _cpu_flag(colmap: str, command: str) -> list[str]:
    """CPU SIFT for ``command``: works headless with any COLMAP build, ~1 min per
    100 views. COLMAP 3.12 renamed the option group (Sift* -> Feature*)."""
    new, old = {"feature_extractor": ("FeatureExtraction", "SiftExtraction"),
                "exhaustive_matcher": ("FeatureMatching", "SiftMatching")}[command]
    usage = subprocess.run([colmap, command, "-h"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True).stdout
    return [f"--{new if f'{new}.use_gpu' in usage else old}.use_gpu", "0"]


def write_seed(scene_dir: Path, colmap: str = "colmap") -> Path:
    """Triangulate ``scene_dir``'s training views and write ``scene_dir/colmap_points.ply``."""
    from PIL import Image
    from plyfile import PlyData, PlyElement

    from scion.data.blender import load_blender
    from scion.data.colmap import _read_points3d

    scene_dir = Path(scene_dir)
    if shutil.which(colmap) is None:
        raise RuntimeError(f"{scene_dir.name}: a Synthetic-NeRF scene needs COLMAP to triangulate its seed "
                           f"points, and `{colmap}` is not on PATH. It comes with the pixi env.")
    scene = load_blender(scene_dir, split="train")
    cam = scene.cameras[0]
    fx, fy, cx, cy = (float(cam.K[0, 0]), float(cam.K[1, 1]), float(cam.K[0, 2]), float(cam.K[1, 2]))

    def run(*args) -> None:
        res = subprocess.run([colmap, *map(str, args)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode:
            raise RuntimeError(f"colmap {args[0]} failed (exit {res.returncode}) on {scene_dir.name}:\n"
                               + "\n".join(res.stderr.splitlines()[-20:]))

    with tempfile.TemporaryDirectory(dir=scene_dir, prefix=".colmap_") as tmp:
        tmp = Path(tmp)
        images, known, tri, db = tmp / "images", tmp / "known", tmp / "tri", tmp / "database.db"
        for d in (images, known, tri):
            d.mkdir()
        poses = {}
        for c, path in zip(scene.cameras, scene.image_paths):
            rgba = Image.open(path).convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))  # the background SCION trains on
            Image.alpha_composite(white, rgba).convert("RGB").save(images / path.name)
            poses[path.name] = colmap_pose(c.cam_to_world.numpy())

        run("feature_extractor", "--database_path", db, "--image_path", images,
            "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}", *_cpu_flag(colmap, "feature_extractor"))
        run("exhaustive_matcher", "--database_path", db, *_cpu_flag(colmap, "exhaustive_matcher"))

        # The known model: the database's camera and image ids with the dataset's poses, no points.
        with sqlite3.connect(db) as con:
            (cam_id,), = con.execute("SELECT camera_id FROM cameras").fetchall()
            ids = con.execute("SELECT image_id, name FROM images").fetchall()
        (known / "cameras.txt").write_text(f"{cam_id} PINHOLE {cam.width} {cam.height} {fx} {fy} {cx} {cy}\n")
        lines = []
        for image_id, name in ids:
            q, t = poses[name]
            lines += [f"{image_id} " + " ".join(f"{v:.17g}" for v in (*q, *t)) + f" {cam_id} {name}", ""]
        (known / "images.txt").write_text("\n".join(lines) + "\n")
        (known / "points3D.txt").write_text("")
        run("point_triangulator", "--database_path", db, "--image_path", images,
            "--input_path", known, "--output_path", tri)

        xyz, rgb = _read_points3d(tri / "points3D.bin")
        keep = (np.abs(xyz) <= BOUND).all(1)
        if not len(xyz) or keep.mean() < 0.9:
            raise ValueError(f"{scene_dir.name}: {int(keep.sum())} of {len(xyz)} triangulated points lie in "
                             f"Synthetic-NeRF's scene box (±{BOUND}); a Blender scene at another scale "
                             f"needs its own box (scion.data.synthetic.BOUND).")
        fields = [(k, "f4") for k in ("x", "y", "z", "nx", "ny", "nz")] + [(k, "u1") for k in ("red", "green", "blue")]
        v = np.zeros(int(keep.sum()), fields)
        for i, k in enumerate(("x", "y", "z")):
            v[k] = xyz[keep, i]
        for i, k in enumerate(("red", "green", "blue")):
            v[k] = rgb[keep, i]
        PlyData([PlyElement.describe(v, "vertex")]).write(str(tmp / SEED_NAME))
        os.replace(tmp / SEED_NAME, scene_dir / SEED_NAME)  # whole-file swap: concurrent first runs don't clash
    logger.info("[seed:%s] %d COLMAP points (%d outside ±%s dropped) -> %s",
                scene_dir.name, len(v), len(xyz) - len(v), BOUND, scene_dir / SEED_NAME)
    return scene_dir / SEED_NAME


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(prog="python -m scion.data.synthetic",
                                description="Triangulate Synthetic-NeRF seed points with COLMAP.")
    p.add_argument("scene_dirs", nargs="+", type=Path)
    p.add_argument("--colmap", default="colmap", help="COLMAP executable (default: `colmap` on PATH)")
    args = p.parse_args(argv)
    for d in args.scene_dirs:
        write_seed(d, args.colmap)


if __name__ == "__main__":
    main()
