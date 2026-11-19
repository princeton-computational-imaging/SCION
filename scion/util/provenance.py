"""Run provenance shared by the paper worker and the reproduce runner.

* :func:`config_sha256` — the idempotency key of a job (run.json ``config_sha256``)
* :func:`git_sha` / :func:`tree_is_dirty` — the code a record was produced
  with (code paths only: run records and docs don't count)
* :func:`eval_images_sha256` — run.json ``split.eval_images_sha256``
* :func:`data_sha256` — run.json ``split.data_sha256``
"""

from __future__ import annotations

import functools
import hashlib
import subprocess
from pathlib import Path
from typing import Iterable, List

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def compose(overrides: List[str]) -> DictConfig:
    """The Hydra config ``config/config.yaml`` + ``overrides`` (no ``hydra`` node)."""
    from hydra import compose as _compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from scion.util.hydra import CONF_DIR

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return _compose(config_name="config", overrides=list(overrides))


def config_sha256_of(cfg: DictConfig) -> str:
    text = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_sha256(overrides: List[str]) -> str:
    """Hex SHA-256 of the resolved config composed from ``overrides``
    (memoised per override list: ~40 ms per compose)."""
    return _config_sha256(tuple(overrides))


@functools.lru_cache(maxsize=None)
def _config_sha256(overrides: tuple) -> str:
    return config_sha256_of(compose(list(overrides)))


# The code identity of a job: what it trains and evaluates with. Run records
# (evals/runs) and docs are outputs or prose, so committing them must not
# invalidate finished jobs. An untracked (not ignored) file here also counts,
# e.g. a new config/ yaml would silently change what a job runs.
_CODE = ("--", "config", "scion", "scion-cuda", "scripts", "experiments",
         "pyproject.toml", "train.py", "Makefile", ":(exclude,glob)**/*.md")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(repo), text=True).strip()


def git_sha(repo: Path = REPO_ROOT) -> str:
    """Full 40-char SHA of the last commit that changed code (``_CODE``)."""
    return _git(repo, "rev-list", "-1", "HEAD", *_CODE)


def tree_is_dirty(repo: Path = REPO_ROOT) -> bool:
    """True iff code (``_CODE``) differs from ``HEAD`` or holds an untracked, non-ignored file."""
    return bool(_git(repo, "status", "--porcelain", "--untracked-files=normal", *_CODE))


def eval_images_sha256(names: Iterable[str]) -> str:
    """SHA-256 of the sorted, newline-joined eval image file names."""
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def data_sha256(scene_dir: Path) -> str:
    """SHA-256 over the scene's camera model: the COLMAP ``sparse/0/*.bin`` or
    the Blender ``transforms*.json``, plus a Blender scene's ``colmap_points.ply``
    seed cloud (name, size and bytes of each, sorted by name). Images are not
    hashed (cost); ``eval_images_sha256`` covers the eval frame names."""
    from scion.data.synthetic import SEED_NAME

    scene_dir = Path(scene_dir)
    files = sorted(scene_dir.glob("transforms*.json")) + sorted((scene_dir / "sparse" / "0").glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"{scene_dir}: no sparse/0/*.bin or transforms*.json to hash")
    ply = scene_dir / SEED_NAME
    h = hashlib.sha256()
    for f in files + ([ply] if ply.is_file() else []):
        data = f.read_bytes()
        h.update(f"{f.name}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()
