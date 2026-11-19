"""Base class for frozen feature extractors with ImageNet normalization."""

import torch
import torch.nn as nn

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FrozenFeatureExtractor(nn.Module):
    """Base class for frozen pre-trained feature extractors.

    Registers ImageNet mean/std as buffers and overrides ``train()``
    to always stay in eval mode.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input images from [0, 1] to ImageNet statistics."""
        return (x - self.mean) / self.std

    def train(self, mode: bool = True):
        """Always keep in eval mode — weights are frozen."""
        return super().train(False)
