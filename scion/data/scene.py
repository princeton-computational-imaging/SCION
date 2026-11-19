"""Scene type for the SCION core package.

A Scene is what a trainer / renderer consumes: list of cameras paired
with image paths (lazy-loaded), train/val split indices, optional sparse
COLMAP points for codebook initialization, and optional per-image masks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from scion.camera import Camera


@dataclass
class Scene:
    """Scene container.

    Cameras are stored in the OpenGL ``cam_to_world`` convention to
    match :class:`scion.camera.Camera`. ``train_indices`` /
    ``eval_indices`` index into ``cameras`` and ``image_paths`` (parallel
    lists, same length).
    """

    cameras: list[Camera]
    image_paths: list[Path]
    train_indices: list[int]
    eval_indices: list[int]
    sparse_points_3d: Optional[torch.Tensor] = None
    sparse_points_rgb: Optional[torch.Tensor] = None
    scene_scale: float = 1.0
    scene_translate: Optional[torch.Tensor] = None
    # Informational: what frame convention the cameras are stored in.
    # We always store OpenGL c2w; use Camera.viewmat() for gsplat.
    coordinate_convention: str = "opengl"
    # Optional 3x4 transform from the loader's input world coordinates to the
    # current oriented/centered coordinates. ``Scene.cameras`` poses are
    # already this transform applied to the raw on-disk poses; this field is
    # informational.
    dataparser_transform: Optional[torch.Tensor] = None
    # The dataset-level downscale factor that was applied to intrinsics /
    # width / height (so callers can locate the matching image folder).
    downscale_factor: int = 1
    # RGB in [0, 1] that RGBA images are composited onto at load time
    # (Blender: white); None keeps the alpha channel.
    alpha_color: Optional[tuple] = None
    # Named groups of eval views (``split.json`` test groups): group ->
    # positions in ``eval_indices``. None when the split has no groups.
    eval_subsets: Optional[dict] = None

    def __len__(self) -> int:
        return len(self.cameras)

    def train_cameras(self) -> list[Camera]:
        return [self.cameras[i] for i in self.train_indices]

    def eval_cameras(self) -> list[Camera]:
        return [self.cameras[i] for i in self.eval_indices]

    def load_image_uint8(self, idx: int) -> torch.Tensor:
        """One image as a uint8 ``(H, W, C)`` tensor, RGBA composited onto
        :attr:`alpha_color` (when set)."""
        arr = np.array(Image.open(self.image_paths[idx]), dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        img = torch.from_numpy(arr)
        if self.alpha_color is not None and img.shape[-1] == 4:
            bg = torch.tensor(self.alpha_color, dtype=torch.float32)
            a = img[:, :, -1:] / 255.0
            img = (img[:, :, :3] * a + 255.0 * bg * (1.0 - a)).clamp(0, 255).to(torch.uint8)
        return img

    def load_image(self, idx: int) -> torch.Tensor:
        """:meth:`load_image_uint8` as float32 in ``[0, 1]``."""
        return self.load_image_uint8(idx).float() / 255.0
