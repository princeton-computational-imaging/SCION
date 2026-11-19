"""Compression levels ``raw`` < ``quantized`` < ``codec``: encode, decode, evaluate.

* ``raw`` -> ``checkpoints/raw.ckpt``: fp32 learned params
  (:func:`~scion.compression.quantize.raw_state_dict`)
* ``quantized`` -> ``checkpoints/quantized.ckpt``: fp16 geometry, uint8
  colour/opacity
* ``codec`` -> ``codec.omg``: quantized + OMG-style SVQ / G-PCC / range coding

Artifact paths are relative to the run directory.

Selected with Hydra ``compression=<level>`` (``config/compression/<level>.yaml``);
the default is ``codec``. Each level's ``size_bytes`` is the byte length of
its artifact file, and its quality is measured on the model decoded from
that file.
"""

from __future__ import annotations

import importlib.metadata
import io
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch

from scion.compression import omg_posthoc as omg
from scion.compression import tmc13
from scion.compression.omg_posthoc import EncodeOptions
from scion.compression.quantize import (
    dequantize_state_dict,
    quantize_state_dict,
    raw_state_dict,
)
from scion.util.units import bits_per_gaussian

LEVELS = ("raw", "quantized", "codec")
ARTIFACTS = {
    "raw": "checkpoints/raw.ckpt",
    "quantized": "checkpoints/quantized.ckpt",
    "codec": "codec.omg",
}


def _check_level(level: str) -> None:
    if level not in LEVELS:
        raise ValueError(f"compression level must be one of {LEVELS}, got {level!r}")


def from_config(cfg: Mapping[str, Any]) -> Tuple[str, EncodeOptions]:
    """``(level, options)`` from a ``compression`` config group node."""
    cfg = dict(cfg)
    level = cfg.pop("level")
    _check_level(level)
    return level, EncodeOptions(**cfg)


def preflight(
    level: str,
    options: Optional[EncodeOptions] = None,
    skip_hint: str = "compression=quantized",
) -> None:
    """Fail now, not after training, if ``level`` cannot run on this machine."""
    _check_level(level)
    if level == "codec":
        omg.require_dependencies(options or EncodeOptions(), skip_hint)


def encode(
    state_dict: Mapping[str, torch.Tensor],
    level: str,
    options: Optional[EncodeOptions] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Artifact bytes of ``level`` for a model ``state_dict``.

    ``info`` is the codec's report (``position_format``, byte budget) at
    level ``codec`` and ``{}`` below it.
    """
    _check_level(level)
    sd = raw_state_dict(state_dict)
    if level != "raw":
        sd = quantize_state_dict(sd)
    if level == "codec":
        return omg.encode_quantized_to_omg(sd, options or EncodeOptions())
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in sd.items()}, buf)
    return buf.getvalue(), {}


def decode(blob: bytes, level: str) -> Dict[str, torch.Tensor]:
    """Float state_dict for ``CodebookModel.load_state_dict`` from artifact bytes."""
    _check_level(level)
    if level == "codec":
        sd = omg.decode_omg_to_state_dict(blob)
    else:
        sd = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    return dequantize_state_dict(sd)


def library_versions() -> Dict[str, Optional[str]]:
    """Versions of the codec's Python libraries (None if not installed)."""
    out: Dict[str, Optional[str]] = {}
    for key, dist in (("constriction", "constriction"), ("scikit_learn", "scikit-learn")):
        try:
            out[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[key] = None
    return out


_QUALITY = ("psnr", "ssim", "lpips")


def _mean_quality(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    return {k: sum(float(r[k]) for r in rows) / len(rows) for k in _QUALITY}


def evaluate_levels(
    model,
    *,
    level: str,
    run_dir: Path,
    eval_fn: Callable[[Path], Tuple[Sequence[Mapping[str, float]], float]],
    options: Optional[EncodeOptions] = None,
    subsets: Optional[Mapping[str, Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Score every level up to ``level`` on the model decoded from its artifact.

    Per level: write ``run_dir / ARTIFACTS[lvl]``, read it back, decode,
    load into ``model``, then call ``eval_fn(run_dir / "eval_views" / lvl)``,
    which renders the eval split into that directory and returns
    ``(rows, fps)``: one ``{"psnr", "ssim", "lpips"}`` dict per eval view, in
    eval order, and the render throughput. ``subsets`` (group -> positions
    in eval order, e.g. ``Scene.eval_subsets``) adds each group's mean
    quality, from the same renders, as ``metrics.<lvl>.by_subset.<group>``.

    Post-training only: the model is put in eval mode and left there, its
    parameters are replaced by the decoded tensors (finally the raw
    snapshot again) and its splat buffers are compacted to the active rows,
    so it is fit for rendering, not for resuming training.

    Returns the ``metrics`` / ``codec`` / ``artifacts`` blocks of run.json
    v2 (``evals/RUN_SCHEMA.md``).
    """
    _check_level(level)
    model.eval()
    options = options or EncodeOptions()
    run_dir = Path(run_dir)
    live = {k: v.detach().cpu().clone() for k, v in raw_state_dict(model.state_dict()).items()}
    metrics: Dict[str, Dict[str, Any]] = {}
    artifacts: Dict[str, Dict[str, str]] = {"checkpoints": {}, "eval_views": {}}
    codec: Optional[Dict[str, Any]] = None
    try:
        for lvl in LEVELS[: LEVELS.index(level) + 1]:
            blob, info = encode(live, lvl, options)
            path = run_dir / ARTIFACTS[lvl]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            model.load_state_dict(decode(path.read_bytes(), lvl), strict=False)
            views = run_dir / "eval_views" / lvl
            views.mkdir(parents=True, exist_ok=True)
            rows, fps = eval_fn(views)
            n = int(model.num_points)
            size = path.stat().st_size
            metrics[lvl] = {
                **_mean_quality(rows),
                "size_bytes": size,
                "bpg": bits_per_gaussian(size, n),
                "fps": float(fps),
                "n_gaussians": n,
            }
            if subsets:
                metrics[lvl]["by_subset"] = {
                    g: {**_mean_quality([rows[i] for i in idx]), "n_views": len(idx)}
                    for g, idx in subsets.items() if idx}  # an empty group has no mean
            artifacts["checkpoints"][lvl] = ARTIFACTS[lvl]
            artifacts["eval_views"][lvl] = f"eval_views/{lvl}"
            if lvl == "codec":
                used = info["position_format"]
                codec = {
                    "options": asdict(options),
                    "position_format_used": used,
                    "tmc13_version": tmc13.tmc13_version_string() if used == "tmc13" else None,
                }
    finally:
        model.load_state_dict(live, strict=False)
    return {"metrics": metrics, "codec": codec, "artifacts": artifacts}
