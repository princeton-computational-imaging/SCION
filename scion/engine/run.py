"""Hydra config -> scene, model, :class:`Trainer`; one training run end to end.

The single training path: ``python -m scion.train`` / ``train.py`` /
``scion-train`` (:func:`scion.cli.train_main`) and the paper
worker (``scripts/train_eval_render_one.py``) both call :func:`fit` then
:func:`score`.

* :func:`scene_from_config` — ``cfg.data`` + ``cfg.dataparser`` -> :class:`Scene`
  (triangulates a Synthetic scene's seed points with COLMAP on its first run).
* :func:`model_config_from_hydra` — ``cfg.method.model`` + ``cfg.discriminator``
  + the selected strategies' ``config/strategies`` defaults.
* :func:`learning_rates` — ``cfg.method.optimizer`` with ``scene_lr_scale``
  (``auto`` = 1.1 x the largest train-camera distance from their mean)
  applied to ``splat_means`` / ``instance_positions``.
* :func:`fit` — preflight the compression level, then train;
  :func:`score` — score every level up to ``cfg.compression.level`` on its
  decoded artifact (:func:`scion.compression.levels.evaluate_levels`).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import DictConfig, OmegaConf

from scion.data import Scene, load_scene
from scion.engine.trainer import Trainer, TrainerConfig, seed_everything

logger = logging.getLogger(__name__)

_SPLIT_KEYS = ("eval_mode", "train_split_fraction", "eval_interval", "downscale_factor")


def scene_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """:func:`load_scene` keyword arguments from ``cfg.dataparser``."""
    dp = OmegaConf.to_container(cfg.get("dataparser") or {}, resolve=True)
    return {k: dp[k] for k in _SPLIT_KEYS if dp.get(k) is not None}


def ensure_synthetic_seed(scene_dir: Path) -> None:
    """Triangulate a Synthetic (Blender) scene's seed points with COLMAP when it
    has none (:mod:`scion.data.synthetic`); instances start at them."""
    from scion.data.synthetic import SEED_NAME, write_seed

    scene_dir = Path(scene_dir)
    if (scene_dir / "transforms_train.json").exists() and not (scene_dir / SEED_NAME).exists():
        logger.info("[seed:%s] no %s; triangulating it with COLMAP", scene_dir.name, SEED_NAME)
        write_seed(scene_dir)


def scene_from_config(cfg: DictConfig) -> Scene:
    scene_dir = Path(cfg.data)
    ensure_synthetic_seed(scene_dir)
    return load_scene(scene_dir, **scene_kwargs(cfg))


def model_config_from_hydra(cfg: DictConfig):
    from scion.losses.discriminator import DiscriminatorConfig
    from scion.models.codebook import CodebookModelConfig
    from scion.util.hydra import apply_strategy_defaults

    model = OmegaConf.to_container(cfg.method.get("model", {}), resolve=True)
    model["discriminator"] = DiscriminatorConfig(
        **OmegaConf.to_container(cfg.discriminator, resolve=True)
    )
    # Strategy YAMLs provide defaults; explicit method.model keys win.
    apply_strategy_defaults(model)
    return CodebookModelConfig(**model)


def camera_radius_scale(scene: Scene) -> float:
    """INRIA 3DGS ``getNerfppNorm``: 1.1 x max train-camera distance from their mean."""
    centers = torch.stack([c.cam_to_world[:3, 3] for c in scene.train_cameras()])
    radius = 1.1 * float((centers - centers.mean(dim=0, keepdim=True)).norm(dim=-1).max())
    if not radius > 0.0:
        raise ValueError(f"non-positive scene radius {radius}")
    return radius


def learning_rates(cfg: DictConfig, scene: Scene) -> Dict[str, float]:
    lrs = OmegaConf.to_container(cfg.method.optimizer, resolve=True)
    sls = lrs.pop("scene_lr_scale", "auto")
    if isinstance(sls, str):
        if sls.lower() != "auto":
            raise ValueError(f"optimizer.scene_lr_scale must be 'auto' or a number, got {sls!r}")
        scale = camera_radius_scale(scene)
    else:
        scale = float(sls)
    logger.info("scene_lr_scale=%.6g (%s): scales splat_means and instance_positions LRs",
                scale, sls)
    for name in ("splat_means", "instance_positions"):
        lrs[name] = lrs[name] * scale
    return {k: float(v) for k, v in lrs.items()}


def trainer_config(cfg: DictConfig, output_dir: Path) -> TrainerConfig:
    t = cfg.trainer
    return TrainerConfig(
        max_steps=int(t.max_num_iterations),
        eval_every=int(t.get("steps_per_eval_all_images", 1000) or 0),
        save_every=int(t.get("steps_per_save", 2000) or 0),
        log_every=int(t.get("log_every", 100) or 0),
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=int(cfg.machine.seed),
        output_dir=Path(output_dir),
        vis=str(t.get("vis", "tensorboard")),
    )


def build_trainer(cfg: DictConfig, output_dir: Path) -> Trainer:
    """Seed, load the scene, build the model (seed cloud + DINO init on the
    train split) and a set-up :class:`Trainer`."""
    from scion.models.codebook import CodebookModel

    seed_everything(int(cfg.machine.seed))
    scene = scene_from_config(cfg)
    seed = None
    if scene.sparse_points_3d is not None:
        seed = (scene.sparse_points_3d, scene.sparse_points_rgb)
    model = CodebookModel(
        model_config_from_hydra(cfg),
        num_train_data=len(scene.train_indices),
        seed_points=seed,
        train_scene=scene,
    )
    trainer = Trainer(
        model, model.codebook_strategy, scene, trainer_config(cfg, output_dir),
        lrs=learning_rates(cfg, scene),
        run_config=OmegaConf.to_container(cfg, resolve=True),
        scene_source={"scene_path": Path(cfg.data).resolve(), "scene_kwargs": scene_kwargs(cfg)},
        config_sha256=run_config_sha256(cfg),
    )
    trainer.setup()
    return trainer


def run_config_sha256(cfg: DictConfig) -> str:
    """``config_sha256`` of ``cfg`` without ``trainer.load_checkpoint``: a
    resumed run hashes like the run it resumes (and like its run.json)."""
    from scion.util.provenance import config_sha256_of

    cfg = copy.deepcopy(cfg)
    cfg.trainer.load_checkpoint = None
    return config_sha256_of(cfg)


def fit(cfg: DictConfig, output_dir: Path) -> Trainer:
    """Preflight the compression level (fail in seconds, not after
    training), then train ``cfg`` into ``output_dir``."""
    from scion.compression import levels

    levels.preflight(*levels.from_config(cfg.compression))
    trainer = build_trainer(cfg, output_dir)
    resume = cfg.trainer.get("load_checkpoint")
    if resume:
        logger.info("resuming from %s at step %d", resume, trainer.resume(Path(resume)))
    trainer.fit()
    return trainer


def score(cfg: DictConfig, trainer: Trainer, output_dir: Path) -> Dict[str, Any]:
    """Write every compression level up to ``cfg.compression.level`` and score
    each on its decoded artifact: :func:`evaluate_levels`' run.json blocks
    (``metrics``, ``codec``, ``artifacts``)."""
    from scion.compression import levels
    from scion.engine.eval import eval_views

    level, options = levels.from_config(cfg.compression)
    model, scene = trainer.model, trainer.scene
    return levels.evaluate_levels(
        model, level=level, options=options, run_dir=Path(output_dir),
        eval_fn=lambda out_dir: eval_views(model, scene, out_dir),
        subsets=scene.eval_subsets,
    )
