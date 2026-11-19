"""Ordered VGG-16 feature maps for the projected discriminator."""

import torch
import torch.nn as nn


class VGGBackboneExtractor(nn.Module):
    """Adapter around VGGFeatureExtractor that returns an ordered list of feature maps.

    ``VGGFeatureExtractor`` (from :mod:`scion.losses.gram_loss`) returns
    a *dict* keyed by layer name.  This thin wrapper converts that dict into a
    plain ``list[Tensor]`` in a fixed order (consumed by
    :class:`scion.losses.discriminator.ProjectedDiscriminator`).
    """

    _LAYER_ORDER = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]

    def __init__(self) -> None:
        super().__init__()
        from scion.losses.gram_loss import VGGFeatureExtractor

        self.vgg = VGGFeatureExtractor()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract ordered feature maps from VGG-16.

        Args:
            x: Input image tensor of shape [B, 3, H, W] with values in [0, 1].

        Returns:
            List of feature tensors in layer order
            (relu1_2, relu2_2, relu3_3, relu4_3).
        """
        features = self.vgg(x)
        return [features[k] for k in self._LAYER_ORDER]

    def train(self, mode: bool = True) -> "VGGBackboneExtractor":
        """Always keep in eval mode — delegates to frozen VGGFeatureExtractor."""
        return super().train(False)

