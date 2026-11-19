"""Thin wrapper around the DINOv3 DINOtxt text-aligned model.

Exposes a minimal interface used by ``scripts/animate_codebook.py``:
``load_dinotxt()``, ``encode_text()``, ``project_vision()`` and
``classify_instances()``.

Follows the upstream `DINOtxt segmentation inference notebook
<https://github.com/facebookresearch/dinov3/blob/main/notebooks/dinotxt_segmentation_inference.ipynb>`_.
The model is loaded via ``torch.hub`` against the ``facebookresearch/dinov3``
registry, mirroring how ``scion/util/dino_models.py`` loads the
vision-only ViTs.

The wrapper compares *pre-extracted* DINOv3 ViT-L/16 backbone patch features
(pooled per instance) against the patch-aligned half of the DINOtxt text
embedding, so the vision tower never runs inside the animator.

DINOv3 weights are gated: put the backbone checkpoint
(``dinov3_vitl16_pretrain_lvd1689m-*.pth``) and the DINOtxt head
(``dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth``) in
``<torch.hub.get_dir()>/checkpoints/``, or pass the head via ``weights_path``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default entry point name in the DINOv3 torch.hub registry for the
# ViT-L/16 + text-encoder DINOtxt head used in the upstream notebook.
DEFAULT_DINOTXT_HUB_MODEL = "dinov3_vitl16_dinotxt_tet1280d20h24l"

PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of a {} in a scene.",
    "a rendering of a {}.",
    "a {} in a 3D scene.",
    "a close-up photo of a {}.",
    "a photo of many {}.",
]


@dataclass
class DINOtxtHead:
    """Container for a loaded DINOTxt model.

    Wraps the full ``DINOTxt`` module plus its tokenizer. Downstream
    functions (:func:`encode_text`, :func:`project_vision`) use the
    model's native methods so the wrapper stays thin.
    """

    model: nn.Module                       # full DINOTxt model
    tokenizer: object                      # BPE tokenizer (callable or .tokenize)
    vision_dim: int                        # backbone embed_dim (1024 for ViT-L)
    device: torch.device

    def to(self, device) -> "DINOtxtHead":
        self.model.to(device)
        self.device = torch.device(device)
        return self


def load_dinotxt(
    hub_model: str = DEFAULT_DINOTXT_HUB_MODEL,
    weights_path: Optional[str] = None,
    hub_repo: str = "facebookresearch/dinov3",
    device: str = "cuda",
) -> DINOtxtHead:
    """Load the DINOtxt model and wrap it in a :class:`DINOtxtHead`.

    ``weights_path`` optionally points at a local DINOtxt head checkpoint;
    otherwise the hub default is resolved through the torch.hub cache.
    """
    kwargs = {}
    if weights_path is not None:
        wpath = Path(weights_path).expanduser()
        if not wpath.is_file():
            raise FileNotFoundError(f"[DINOtxt] local weights not found: {wpath}")
        kwargs["weights"] = str(wpath)

    # Prefer an already-cloned hub repo so offline machines still work.
    repo_dir = Path(torch.hub.get_dir()) / (hub_repo.replace("/", "_") + "_main")
    if repo_dir.is_dir():
        print(f"[DINOtxt] loading {hub_model} from local repo {repo_dir}")
        model, tokenizer = torch.hub.load(str(repo_dir), hub_model, source="local", **kwargs)
    else:
        print(f"[DINOtxt] torch.hub.load('{hub_repo}', '{hub_model}')")
        model, tokenizer = torch.hub.load(hub_repo, hub_model, verbose=False, **kwargs)
    if tokenizer is None:
        raise AttributeError("[DINOtxt] no tokenizer returned from loader.")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Text features are [class-aligned | patch-aligned] halves of embed_dim;
    # instance features are pooled backbone patch tokens.
    embed_dim = model.model_config.embed_dim
    vision_dim = model.visual_model.backbone.embed_dim
    head = DINOtxtHead(
        model=model,
        tokenizer=tokenizer,
        vision_dim=vision_dim,
        device=torch.device(device),
    ).to(device)
    print(f"[DINOtxt] loaded: embed_dim={embed_dim}, vision_dim={vision_dim}")
    return head


@torch.no_grad()
def encode_text(head: DINOtxtHead, texts: List[str]) -> torch.Tensor:
    """Encode text labels into the patch-aligned ``[L, D_text]`` space.

    For each label, encodes :data:`PROMPT_TEMPLATES` and averages the
    resulting features (following the official DINOtxt notebook). Returns
    only the patch-aligned second half since instance features are pooled
    backbone patch features. Returned features are L2-normalised.
    """
    tokenize_fn = (head.tokenizer.tokenize
                   if hasattr(head.tokenizer, "tokenize")
                   else head.tokenizer)
    all_feats = []
    for label in texts:
        prompts = [t.format(label) for t in PROMPT_TEMPLATES]
        tok = tokenize_fn(prompts)
        if isinstance(tok, torch.Tensor):
            tok = tok.to(head.device)
        feats = head.model.encode_text(tok, normalize=False)  # [P, 2*D]
        feats = feats.to(torch.float32)
        feats = feats[:, feats.shape[1] // 2:]  # [P, D] patch-aligned half
        feats = F.normalize(feats, dim=-1)
        feats = F.normalize(feats.mean(dim=0), dim=-1)  # average over prompts
        all_feats.append(feats)
    return torch.stack(all_feats)  # [L, D]


@torch.no_grad()
def project_vision(head: DINOtxtHead, feats: torch.Tensor) -> torch.Tensor:
    """L2-normalise backbone vision features ``[N, D]`` for cosine similarity."""
    if feats.shape[-1] != head.vision_dim:
        raise ValueError(
            f"Vision feature dim {feats.shape[-1]} does not match DINOtxt "
            f"head's expected {head.vision_dim}."
        )
    x = feats.to(head.device, dtype=torch.float32)
    return F.normalize(x, dim=-1)


def classify_instances(
    head: DINOtxtHead,
    inst_feats: torch.Tensor,
    labels: List[str],
    match_threshold: Union[float, Sequence[float]] = 0.25,
    static_index: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each instance to its best-matching text label.

    ``match_threshold`` is a single float or one value per label. Returns
    ``(labels [N] long, sim [N, L])`` where entry ``i`` is the index into
    ``labels`` if its best-label cosine sim exceeds that label's threshold,
    else ``static_index``.
    """
    txt = encode_text(head, labels)             # [L, D_text]
    vis = project_vision(head, inst_feats)       # [N, D_text]
    sim = vis @ txt.T                             # [N, L]
    thresholds = torch.as_tensor(match_threshold, device=sim.device, dtype=sim.dtype)
    if thresholds.dim() == 0:
        thresholds = thresholds.expand(len(labels))
    if thresholds.numel() != len(labels):
        raise ValueError(
            f"match_threshold has {thresholds.numel()} values but "
            f"there are {len(labels)} labels."
        )
    best_val, best_idx = sim.max(dim=-1)
    result = torch.where(
        best_val > thresholds[best_idx],
        best_idx,
        torch.full_like(best_idx, static_index),
    )
    return result, sim
