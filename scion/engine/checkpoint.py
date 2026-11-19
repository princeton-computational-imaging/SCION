"""Native checkpoints and run directories.

Layout written by :class:`scion.engine.trainer.Trainer`::

    <run_dir>/
    ├── checkpoints/
    │   ├── step-000029999.ckpt   # compact model state + optimizer / resume state
    │   ├── meta.json             # model config, step, git SHA, scene + split
    │   ├── raw.ckpt              # compression level artifacts
    │   └── quantized.ckpt        #   (scion.compression.levels)
    ├── codec.omg
    └── run.json                  # paper worker only (evals/RUN_SCHEMA.md)

The ``.ckpt`` is a ``torch.save``-d dict: ``step``, ``pipeline`` (the
model's compact ``state_dict()`` under a ``_model.`` key prefix),
``optimizers`` and, from the trainer, ``trainer_state`` (resume state).

:func:`load` reads either the latest ``step-*.ckpt`` (``level=None``) or
one compression level's artifact (``level="raw" | "quantized" | "codec"``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

    from scion.data.scene import Scene
    from scion.models.codebook import CodebookModel

logger = logging.getLogger(__name__)

CKPT_FILENAME_TMPL = "step-{step:09d}.ckpt"
META_FILENAME = "meta.json"
CHECKPOINTS_SUBDIR = "checkpoints"
# Frozen networks / init-only state, rebuilt identically when a model is
# constructed: never written (load is non-strict). ~122 MB per checkpoint.
FROZEN_PREFIXES = ("lpips_loss.", "lpips.", "psnr.", "ssim.", "discriminator.backbone.",
                   "dino_oracle.")


def trainable_state(state_dict: dict) -> dict:
    """``state_dict`` without the :data:`FROZEN_PREFIXES` entries."""
    return {k: v for k, v in state_dict.items() if not k.startswith(FROZEN_PREFIXES)}


@dataclass
class LoadedCheckpoint:
    """Return shape of :func:`load`: ``model`` on ``device`` in ``eval()``
    mode; ``scene`` is the training scene (same split) or None when it could
    not be rebuilt; ``step`` is the last checkpointed training step."""

    model: "CodebookModel"
    scene: Optional["Scene"]
    step: int
    meta: dict


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _git_provenance() -> dict:
    """``git_sha`` / ``git_dirty`` of the checkout containing ``scion``
    (not the cwd), as run.json records them; None when it is not a git
    checkout (e.g. a wheel install)."""
    from scion.util import provenance

    if not (provenance.REPO_ROOT / ".git").exists():  # not some enclosing repo's sha
        return {"git_sha": None, "git_dirty": None}
    try:
        return {"git_sha": provenance.git_sha(), "git_dirty": provenance.tree_is_dirty()}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}


def _atomic_torch_save(obj: Any, dst: Path) -> None:
    """Write ``obj`` via ``torch.save`` to a temp file and rename atomically."""
    import torch

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, dst)


def _atomic_json_dump(obj: Any, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)
    os.replace(tmp, dst)


def save(
    *,
    model: "nn.Module",
    optimizers: Optional[dict[str, "Optimizer"]] = None,
    step: int,
    output_dir: Path,
    config_snapshot: Optional[dict] = None,
    scene_path: Optional[Path] = None,
    scene_kwargs: Optional[dict] = None,
    trainer_state: Optional[dict] = None,
    config_sha256: Optional[str] = None,
) -> Path:
    """Atomically write ``<output_dir>/checkpoints/step-NNN.ckpt`` + ``meta.json``.

    ``config_snapshot`` (the model config as a JSON-able dict) and
    ``scene_path`` / ``scene_kwargs`` (:func:`scion.data.load_scene`
    arguments) let :func:`load` rebuild the model and its training scene.
    ``trainer_state`` is the trainer's resume state; ``config_sha256`` (the
    resolved run config's hash) is what a resume must match. Returns the
    ``.ckpt`` path.
    """
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / CHECKPOINTS_SUBDIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "step": int(step),
        "pipeline": {f"_model.{k}": v for k, v in trainable_state(model.state_dict()).items()},
        "optimizers": {name: opt.state_dict() for name, opt in (optimizers or {}).items()},
    }
    if trainer_state is not None:
        payload["trainer_state"] = trainer_state

    primary = ckpt_dir / CKPT_FILENAME_TMPL.format(step=int(step))
    _atomic_torch_save(payload, primary)
    meta = {
        "format_version": 1,
        "step": int(step),
        "primary_ckpt": str(primary.relative_to(output_dir)),
        **_git_provenance(),
        "config_sha256": config_sha256,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_config": config_snapshot,
        "scene_path": str(scene_path) if scene_path else None,
        "scene_kwargs": scene_kwargs,
    }
    _atomic_json_dump(meta, ckpt_dir / META_FILENAME)
    return primary


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _strip_model_prefix(sd: dict) -> dict:
    return {(k[len("_model."):] if k.startswith("_model.") else k): v for k, v in sd.items()}


def _find_latest_native_ckpt(ckpt_dir: Path) -> Optional[Path]:
    """Return the highest-step ``step-NNN.ckpt`` under ``ckpt_dir``."""
    candidates: list[tuple[int, Path]] = []
    for entry in ckpt_dir.glob("step-*.ckpt"):
        stem = entry.name[len("step-"):-len(".ckpt")]
        if stem.isdigit():
            candidates.append((int(stem), entry))
    return max(candidates)[1] if candidates else None


def _resolve_ckpt_path(path: Path) -> Path:
    """A ``step-*.ckpt`` file, a ``checkpoints/`` dir, or a run dir -> the ckpt."""
    path = Path(path)
    if path.is_file():
        if path.suffix != ".ckpt":
            raise FileNotFoundError(f"Not a .ckpt file: {path}")
        return path
    for ckpt_dir in (path / CHECKPOINTS_SUBDIR, path):
        if ckpt_dir.is_dir():
            latest = _find_latest_native_ckpt(ckpt_dir)
            if latest is not None:
                return latest
    raise FileNotFoundError(f"No checkpoint found at {path}")


def load(
    path: Path,
    *,
    device: str = "cuda",
    level: Optional[str] = None,
    artifact: Optional[Path] = None,
) -> LoadedCheckpoint:
    """Load a trained model.

    ``level=None``: ``path`` is a ``step-*.ckpt``, a ``checkpoints/`` dir or a
    run dir; the latest step checkpoint is loaded. ``level="raw" |
    "quantized" | "codec"``: ``path`` is a run dir; the model is built from
    its ``checkpoints/meta.json`` and loaded from that level's artifact
    (:data:`scion.compression.levels.ARTIFACTS`; codec decodes
    ``codec.omg``), or from ``artifact`` (e.g. a re-encoded bundle).
    """
    import torch

    if level is not None:
        return _load_level(Path(path), level, device, artifact)
    ckpt_path = _resolve_ckpt_path(Path(path))
    meta = _read_meta(ckpt_path.parent)
    # weights_only=True: tensors + builtin containers only, so a downloaded
    # checkpoint cannot execute code on load.
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = _build_model_from_config_snapshot(meta["model_config"])
    model.load_state_dict(_strip_model_prefix(payload["pipeline"]), strict=False)
    return LoadedCheckpoint(model=model.to(device).eval(), scene=_scene_of(meta),
                            step=int(payload.get("step", meta["step"])), meta=meta)


def _read_meta(ckpt_dir: Path) -> dict:
    meta_path = ckpt_dir / META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing: not a SCION checkpoint dir"
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("model_config") is None:
        raise RuntimeError(f"{meta_path} has no 'model_config'; cannot rebuild the model")
    return meta


def _scene_of(meta: dict) -> Optional["Scene"]:
    if not meta.get("scene_path"):
        return None
    return _try_rebuild_scene(Path(meta["scene_path"]), meta.get("scene_kwargs"))


def _load_level(run_dir: Path, level: str, device: str,
                artifact: Optional[Path]) -> LoadedCheckpoint:
    """``level``'s artifact in a model built from ``run_dir``'s meta.json."""
    from scion.compression import levels

    if level not in levels.LEVELS:
        raise ValueError(f"level must be one of {levels.LEVELS}, got {level!r}")
    artifact = Path(artifact) if artifact else run_dir / levels.ARTIFACTS[level]
    if not artifact.is_file():
        raise FileNotFoundError(f"{artifact} missing (was the run trained to level {level}?)")
    meta = _read_meta(run_dir / CHECKPOINTS_SUBDIR)
    model = _build_model_from_config_snapshot(meta["model_config"])
    model.load_state_dict(levels.decode(artifact.read_bytes(), level), strict=False)
    return LoadedCheckpoint(model=model.to(device).eval(), scene=_scene_of(meta),
                            step=int(meta.get("step", -1)),
                            meta={**meta, "level": level, "artifact": str(artifact)})


def _build_model_from_config_snapshot(cfg_snap: dict) -> "CodebookModel":
    """A :class:`CodebookModel` for ``cfg_snap`` (a ``CodebookModelConfig`` as a
    dict), ready for ``load_state_dict``: the seed / DINO init and the
    discriminator are skipped, since every learned tensor comes from the
    checkpoint and rendering needs no discriminator."""
    from scion.losses.discriminator import DiscriminatorConfig
    from scion.models.codebook import CodebookModel, CodebookModelConfig

    valid = {f.name for f in dataclasses.fields(CodebookModelConfig)}
    cfg = {k: v for k, v in cfg_snap.items() if k in valid}
    disc = cfg.get("discriminator")
    if isinstance(disc, dict):
        disc = DiscriminatorConfig(**disc)
    cfg["discriminator"] = dataclasses.replace(disc or DiscriminatorConfig(), enabled=False)
    cfg.update(random_init=True, init_assignment_mode="random_uniform")
    return CodebookModel(config=CodebookModelConfig(**cfg))


def _try_rebuild_scene(scene_path: Path, scene_kwargs: Optional[dict]) -> Optional["Scene"]:
    """The training :class:`Scene` (same split / downscale), or None (logged)
    when it cannot be loaded: rendering and stats work without it."""
    try:
        from scion.data import load_scene

        if not Path(scene_path).exists():
            logger.warning("scene_path %s does not exist; skipping scene reload", scene_path)
            return None
        return load_scene(scene_path, **(scene_kwargs or {}))
    except Exception as e:  # noqa: BLE001
        logger.warning("Scene reload from %s failed: %s", scene_path, e)
        return None
