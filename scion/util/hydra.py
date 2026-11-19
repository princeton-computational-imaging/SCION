"""Hydra config-loading helpers (the Hydra -> trainer glue is
:mod:`scion.engine.run`).

* :func:`load_hydra(method, overrides)` — compose
  ``config/method/<method>.yaml`` on top of ``config/config.yaml``.
* :func:`load_yaml(subdir, name)` — read a single YAML file from
  ``config/<subdir>/<name>.yaml`` without going through Hydra.
* :func:`apply_strategy_defaults(model_cfg)` — merge the selected strategies'
  ``config/strategies/`` YAML defaults into a model-config dict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Path to the YAML config directory (project root / config/)
CONF_DIR = str(Path(__file__).resolve().parent.parent.parent / "config")


# ---------------------------------------------------------------------------
# Hydra config loading
# ---------------------------------------------------------------------------


def load_hydra(method: str, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load and compose Hydra config for a given method.

    Args:
        method: Method name matching ``config/method/{method}.yaml``
            (e.g. ``"codebook"``).
        overrides: Optional list of Hydra overrides
            (e.g. ``["trainer.max_num_iterations=50000"]``).

    Returns:
        Composed Hydra :class:`DictConfig`.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(
            config_name="config",
            overrides=[f"method={method}"] + (overrides or []),
        )


def load_yaml(subdir: str, name: str) -> dict:
    """Load a single YAML file from ``config/{subdir}/{name}.yaml``.

    Args:
        subdir: Subdirectory within ``config/`` (e.g. ``"material"``,
            ``"optimizer"``).
        name: File name without extension (e.g. ``"grass"``,
            ``"default"``).

    Returns:
        Parsed YAML as a dict.
    """
    path = Path(CONF_DIR) / subdir / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def apply_strategy_defaults(model_cfg: dict) -> dict:
    """Fill ``model_cfg`` from ``config/strategies/<level>/<name>.yaml`` in place.

    Explicit keys win (``setdefault``). An import-path strategy
    (``"pkg.mod:Class"``) uses the YAML named by its class's ``config_name``
    (set on the built-ins and inherited by their subclasses), or none.
    """
    from scion.models.codebook import CodebookModelConfig
    from scion.strategies.base import strategy_class

    for level in ("splat", "instance"):
        key = f"{level}_strategy"
        name = model_cfg.get(key, getattr(CodebookModelConfig, key))
        if ":" in name:
            name = strategy_class(name, key).config_name
            if name is None:
                continue
        for k, v in (load_yaml(f"strategies/{level}", name).get("model") or {}).items():
            model_cfg.setdefault(k, v)
    return model_cfg


__all__ = [
    "CONF_DIR",
    "load_hydra",
    "load_yaml",
    "apply_strategy_defaults",
]
