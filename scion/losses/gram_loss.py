"""Frozen VGG-16 feature extractor (relu1_2 .. relu4_3).

Backbone of the projected discriminator (see :mod:`scion.losses.vgg`).
"""

from typing import Dict

import torch

from scion.losses._feature_extractor_base import FrozenFeatureExtractor

# VGG-16 layer indices (0-based within features module) for the four target layers:
#   relu1_2 -> index  3  (after conv1_1, relu, conv1_2, relu)
#   relu2_2 -> index  8  (after pool1, conv2_1, relu, conv2_2, relu)
#   relu3_3 -> index 15  (after pool2, conv3_1, relu, conv3_2, relu, conv3_3, relu)
#   relu4_3 -> index 22  (after pool3, conv4_1, relu, conv4_2, relu, conv4_3, relu)
_LAYER_INDICES = {
    "relu1_2": 3,
    "relu2_2": 8,
    "relu3_3": 15,
    "relu4_3": 22,
}


class VGGFeatureExtractor(FrozenFeatureExtractor):
    """Extracts intermediate feature maps from a frozen VGG-16 network.

    Accepts images in [B, 3, H, W] format with pixel values in [0, 1].
    Internally normalizes using ImageNet mean/std before forward pass.

    Target layers:
        relu1_2, relu2_2, relu3_3, relu4_3
    """

    def __init__(self) -> None:
        super().__init__()

        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            import torchvision.models as tv_models
            vgg = tv_models.vgg16(pretrained=True)

        self.features = vgg.features
        self.features.requires_grad_(False)

        self._max_index = max(_LAYER_INDICES.values())
        self._index_to_name = {v: k for k, v in _LAYER_INDICES.items()}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from the four target VGG layers.

        Args:
            x: Input image tensor of shape [B, 3, H, W] with values in [0, 1].

        Returns:
            Dict mapping layer names to feature tensors.
        """
        x = self._normalize(x)

        features: Dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self._index_to_name:
                features[self._index_to_name[idx]] = x
            if idx == self._max_index:
                break

        return features

