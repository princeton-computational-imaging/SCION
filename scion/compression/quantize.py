"""SCION native quantization: the ``raw`` and ``quantized`` compression levels.

* :func:`raw_state_dict` keeps only the learned splat / instance tensors and
  the ``__codebook_meta__`` header (no optimizer state, no frozen DINO /
  LPIPS / discriminator weights). Stored fp32: the ``raw`` level.
* :func:`quantize_state_dict` turns that into the ``quantized`` level:
  fp16 geometry, uint8 colour / opacity (per-tensor min-max affine), with
  the affine parameters stored under ``__quantized_meta__.<key>.*``.
* :func:`dequantize_state_dict` is the inverse (float tensors out), so the
  result loads straight into ``CodebookModel.load_state_dict``.

Keys may carry any prefix (the training checkpoint uses ``_model.``);
suffix matching is prefix-agnostic. ``__quantized_meta__`` is the on-disk key name.
"""

from __future__ import annotations

from typing import Dict

import torch

_RAW_DROP_SUBSTRS = (
    "dino_oracle", "lpips", "loss_fn_vgg",
    "discriminator", "perceptual", "vgg",
)
_RAW_KEEP_SUBSTRS = (
    "splat_", "instance_", "__codebook_meta__",
    "splat_candidate_id", "free_list",
)

_FP16_SUFFIXES = (
    "splat_means", "splat_scales", "splat_quats",
    "instance_positions", "instance_rotations", "instance_scales",
)
_UINT8_SUFFIXES = (
    "splat_features_dc", "splat_opacities", "instance_opacities",
)

META_PREFIX = "__quantized_meta__."


def raw_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract the learned splat/instance/metadata slice of ``state_dict``."""
    return {
        k: v for k, v in state_dict.items()
        if any(s in k for s in _RAW_KEEP_SUBSTRS)
        and not any(d in k for d in _RAW_DROP_SUBSTRS)
    }


def _quant_uint8(t: torch.Tensor) -> tuple:
    tf = t.detach().float().contiguous()
    finite_mask = torch.isfinite(tf)
    if finite_mask.any():
        finite_vals = tf[finite_mask]
        lo = float(finite_vals.min().item())
        hi = float(finite_vals.max().item())
    else:
        lo = 0.0
        hi = 0.0
    rng = max(hi - lo, 1e-12)
    scale = rng / 255.0
    pos_inf = torch.isposinf(tf)
    neg_inf_or_nan = ~(finite_mask | pos_inf)
    tf = torch.where(pos_inf, torch.full_like(tf, hi), tf)
    tf = torch.where(neg_inf_or_nan, torch.full_like(tf, lo), tf)
    q = ((tf - lo) / scale).round().clamp(0, 255).to(torch.uint8)
    return q, scale, lo


def quantize_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """``raw`` level -> ``quantized`` level (see module docstring)."""
    out: Dict[str, torch.Tensor] = {}
    meta: Dict[str, torch.Tensor] = {}

    for key, val in raw.items():
        if not isinstance(val, torch.Tensor):
            out[key] = val
            continue
        if "__codebook_meta__" in key:
            out[key] = val
            continue
        if any(key.endswith(suf) for suf in _UINT8_SUFFIXES):
            q, scale, zero = _quant_uint8(val)
            out[key] = q
            meta[f"{META_PREFIX}{key}.scale"] = torch.tensor(scale)
            meta[f"{META_PREFIX}{key}.zero"] = torch.tensor(zero)
            continue
        if any(key.endswith(suf) for suf in _FP16_SUFFIXES):
            out[key] = val.detach().to(torch.float16).contiguous()
            continue
        out[key] = val

    out.update(meta)
    return out


def is_quantized_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith(META_PREFIX) for k in state_dict)


def dequantize_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`quantize_state_dict`: float tensors, meta keys dropped."""
    per_key_meta: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in state_dict.items():
        if not k.startswith(META_PREFIX):
            continue
        rest = k[len(META_PREFIX):]
        parent, field_ = rest.rsplit(".", 1)
        per_key_meta.setdefault(parent, {})[field_] = v

    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(META_PREFIX):
            continue
        meta = per_key_meta.get(k)
        if meta is None:
            if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                out[k] = v.float()
            else:
                out[k] = v
            continue
        out[k] = v.float() * float(meta["scale"].item()) + float(meta["zero"].item())

    return out
