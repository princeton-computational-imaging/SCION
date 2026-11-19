"""Minimal camera type for the SCION core.

One pinhole view (pose in the OpenGL convention, intrinsics, size):
what the data loaders produce and every render / projection consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class Camera:
    """Single pinhole camera.

    Attributes
    ----------
    cam_to_world : Tensor of shape ``(4, 4)`` or ``(3, 4)`` — OpenGL
        convention: x right, y up, z back. Float32. The
        :meth:`viewmat` method converts this to the gsplat / OpenCV
        world-to-camera convention (x right, y down, z forward).
    K : Tensor of shape ``(3, 3)`` — pinhole intrinsics.
    width, height : image resolution.
    near, far : optional near/far planes for the rasterizer.
    metadata : per-image extras a loader wants to carry through
        (e.g. image index, distortion params); core code MUST NOT depend
        on this dict.
    """

    cam_to_world: Tensor
    K: Tensor
    width: int
    height: int
    near: float = 0.01
    far: float = 100.0
    metadata: Optional[dict] = None

    @property
    def device(self) -> torch.device:
        return self.cam_to_world.device

    def to(self, device) -> "Camera":
        return Camera(
            cam_to_world=self.cam_to_world.to(device),
            K=self.K.to(device),
            width=self.width,
            height=self.height,
            near=self.near,
            far=self.far,
            metadata=self.metadata,
        )

    def viewmat(self) -> Tensor:
        """gsplat-compatible ``(4, 4)`` world-to-camera matrix.

        Flips the y/z axes of the rotation block to map the OpenGL frame
        to gsplat's OpenCV frame, then takes the analytic inverse.
        """
        c2w = self.cam_to_world
        if c2w.dim() == 2 and c2w.shape == (3, 4):
            bottom = torch.tensor(
                [[0.0, 0.0, 0.0, 1.0]], device=c2w.device, dtype=c2w.dtype
            )
            c2w = torch.cat([c2w, bottom], dim=0)
        return _viewmat_from_c2w(c2w.unsqueeze(0)).squeeze(0)


def viewmat_from_c2w(c2w: Tensor) -> Tensor:
    """Batched gsplat world-to-camera viewmat from camera-to-world poses.

    Accepts ``(B, 3, 4)`` or ``(B, 4, 4)`` and returns ``(B, 4, 4)``.
    y/z flip on the rotation block followed by analytic inverse.
    """
    if c2w.dim() != 3 or c2w.shape[-2] not in (3, 4) or c2w.shape[-1] != 4:
        raise ValueError(
            f"viewmat_from_c2w expects (B, 3, 4) or (B, 4, 4), got {tuple(c2w.shape)}"
        )
    if c2w.shape[-2] == 3:
        B = c2w.shape[0]
        bottom = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=c2w.device, dtype=c2w.dtype)
            .view(1, 1, 4)
            .expand(B, 1, 4)
        )
        c2w = torch.cat([c2w, bottom], dim=1)
    return _viewmat_from_c2w(c2w)


def _viewmat_from_c2w(c2w: Tensor) -> Tensor:
    """Core (B, 4, 4) → (B, 4, 4) gsplat viewmat conversion."""
    R = c2w[:, :3, :3]
    T = c2w[:, :3, 3:4]
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat
