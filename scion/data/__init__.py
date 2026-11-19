"""Data layer for ``scion``.

Public re-exports — these are the symbols a downstream trainer / renderer
should import.
"""
from pathlib import Path
from typing import Optional

from scion.data.scene import Scene
from scion.data.colmap import load_colmap
from scion.data.blender import load_blender
from scion.data.loader import FullImageLoader


def load_scene(
    scene_dir,
    eval_mode: str = "fraction",
    train_split_fraction: float = 0.9,
    eval_interval: int = 8,
    downscale_factor: Optional[int] = None,
) -> Scene:
    """Blender scene if ``transforms_train.json`` exists, else COLMAP
    (``images*/`` + ``sparse/0/``). The keys are the Hydra ``dataparser`` group;
    Blender scenes have a fixed train/test split and ignore them."""
    scene_dir = Path(scene_dir)
    if (scene_dir / "transforms_train.json").exists():
        return load_blender(scene_dir)
    return load_colmap(
        scene_dir,
        downscale_factor=downscale_factor,
        eval_mode=eval_mode,
        train_split_fraction=train_split_fraction,
        eval_interval=eval_interval,
    )


__all__ = [
    "Scene",
    "load_colmap",
    "load_blender",
    "load_scene",
    "FullImageLoader",
]
