"""Shared DINO model registry and loader.

Used by ``scion.util.dino_init`` (DINO-feature codebook init) to load
DINOv2 and DINOv3 models with a single, consistent API.

Each entry:  ``(hub_repo, hub_model, hf_name, patch_size, embed_dim)``
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

# (hub_repo, hub_model, hf_name, patch_size, embed_dim)
DINO_MODELS = {
    # DINOv2 ViT models (patch size 14)
    "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14", "facebook/dinov2-small", 14, 384),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14", "facebook/dinov2-base", 14, 768),
    "dinov2_vitl14": ("facebookresearch/dinov2", "dinov2_vitl14", "facebook/dinov2-large", 14, 1024),
    "dinov2_vitg14": ("facebookresearch/dinov2", "dinov2_vitg14", "facebook/dinov2-giant", 14, 1536),
    # DINOv2 ViT with registers (patch size 14)
    "dinov2_vits14_reg4": ("facebookresearch/dinov2", "dinov2_vits14_reg4", "facebook/dinov2-small", 14, 384),
    "dinov2_vitb14_reg4": ("facebookresearch/dinov2", "dinov2_vitb14_reg4", "facebook/dinov2-base", 14, 768),
    "dinov2_vitl14_reg4": ("facebookresearch/dinov2", "dinov2_vitl14_reg4", "facebook/dinov2-large", 14, 1024),
    "dinov2_vitg14_reg4": ("facebookresearch/dinov2", "dinov2_vitg14_reg4", "facebook/dinov2-giant", 14, 1536),
    # DINOv3 ViT models (patch size 16)
    "dinov3_vits16": ("facebookresearch/dinov3", "dinov3_vits16", "facebook/dinov3-vits16-pretrain-lvd1689m", 16, 384),
    "dinov3_vitb16": ("facebookresearch/dinov3", "dinov3_vitb16", "facebook/dinov3-vitb16-pretrain-lvd1689m", 16, 768),
    "dinov3_vitl16": ("facebookresearch/dinov3", "dinov3_vitl16", "facebook/dinov3-vitl16-pretrain-lvd1689m", 16, 1024),
}


class _TransformersWrapper(nn.Module):
    """Thin wrapper around HuggingFace transformers DINO models to mimic hub API."""

    def __init__(self, model: nn.Module, num_register_tokens: int = 0):
        super().__init__()
        self.model = model
        self._skip = 1 + num_register_tokens  # CLS + registers

    def forward_features(self, x: torch.Tensor) -> dict:
        outputs = self.model(pixel_values=x, output_hidden_states=False)
        last_hidden = outputs.last_hidden_state
        patch_tokens = last_hidden[:, self._skip:, :]
        return {"x_norm_patchtokens": patch_tokens}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).last_hidden_state[:, self._skip:, :]

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)


def load_dino_model(
    name: str,
    local_path: Optional[str] = None,
) -> Tuple[nn.Module, int, int]:
    """Load a DINO model by registry name.

    Args:
        name: Key from :data:`DINO_MODELS` (e.g. ``"dinov3_vitl16"``).
        local_path: Optional local checkpoint path.

    Returns:
        ``(model, patch_size, embed_dim)`` — model is in eval mode with
        frozen parameters.
    """
    import os

    if name not in DINO_MODELS:
        raise ValueError(
            f"Unknown DINO model '{name}'. "
            f"Available: {sorted(DINO_MODELS.keys())}"
        )

    hub_repo, hub_model, hf_name, patch_size, embed_dim = DINO_MODELS[name]

    # Try local checkpoint first
    if local_path is not None:
        path = os.path.expanduser(local_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[DINO] Local checkpoint not found: {path}")
        repo_dir = os.path.join(
            torch.hub.get_dir(), hub_repo.replace("/", "_") + "_main",
        )
        print(f"[DINO] Loading {hub_model} from {path} (repo: {repo_dir})")
        model = torch.hub.load(repo_dir, hub_model, source="local", weights=path)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, patch_size, embed_dim

    # Try torch.hub
    try:
        model = torch.hub.load(hub_repo, hub_model, verbose=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, patch_size, embed_dim
    except Exception as e:
        print(f"[DINO] torch.hub load of {hub_model} failed ({e}), "
              f"falling back to transformers.")

    # Fallback to HuggingFace transformers
    from transformers import AutoModel
    model = AutoModel.from_pretrained(hf_name)
    # DINOv3 ViTs emit 4 register tokens after CLS.
    num_reg = 4 if name.startswith("dinov3") else 0
    model = _TransformersWrapper(model, num_register_tokens=num_reg)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, patch_size, embed_dim
