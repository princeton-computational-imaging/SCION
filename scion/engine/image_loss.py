"""SSIM + L1 helpers for the image-loss path.

Wraps ``pytorch_msssim.SSIM``.

LPIPS is intentionally NOT wrapped here — :class:`CodebookModel` instances
``lpips.LPIPS(net=...)`` directly so the VGG weights ship as a registered
submodule and round-trip through the state_dict unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pytorch_msssim import SSIM as _SSIM


class SSIMLoss(nn.Module):
    """SSIM with ``data_range=1.0``, averaged over channels."""

    def __init__(self, data_range: float = 1.0, channel: int = 3):
        super().__init__()
        self.ssim = _SSIM(data_range=data_range, size_average=True, channel=channel)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Both ``pred`` / ``gt`` are ``(B, C, H, W)`` in ``[0, 1]``."""
        return self.ssim(pred, gt)


def composite_with_background(image: Tensor, background: Tensor) -> Tensor:
    """Alpha-composite an RGBA ground-truth image onto a solid background.

    ``image`` is ``(H, W, 3)`` or ``(H, W, 4)``; ``background`` broadcasts.
    No-op when the image has no alpha channel.
    """
    if image.shape[2] == 4:
        alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
        return alpha * image[..., :3] + (1 - alpha) * background
    return image


def gt_image_to_float(image: Tensor, device: torch.device) -> Tensor:
    """uint8 ``[0,255]`` → float ``[0,1]``, with a device cast.

    No downscaling: the caller knows whether the rendered output is
    downscaled.
    """
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    return image.to(device)
