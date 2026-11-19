"""Implicit setup for ``reproduce.py``, run only for scenes a scheduled job needs:
scene data (``scripts/download_data.sh``), the DINO feature cache, and the
pinned ``tmc3`` build. Every step is idempotent.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

REPO = Path(__file__).resolve().parents[2]

# dataset config -> scripts/download_data.sh argument
SOURCES = {"mipnerf360": "mipnerf360", "tandt": "tandt", "deepblending": "tandt", "synthetic": "synthetic",
           "campus": "campus"}


class SetupError(RuntimeError):
    pass


def ready(scene_dir: Path) -> bool:
    return (scene_dir / "sparse" / "0" / "cameras.bin").is_file() or (scene_dir / "transforms_train.json").is_file()


def link_data_root(data_root: Path) -> None:
    """Hydra configs read ``data/<scene>`` relative to the repo; point it at DATA_ROOT."""
    link = REPO / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    if link.exists() and link.resolve() == data_root.resolve():
        return
    if link.exists() or link.is_symlink():
        raise SetupError(f"{link} exists and is not DATA_ROOT={data_root}; remove it, or set DATA_ROOT={link}")
    link.symlink_to(data_root.resolve(), target_is_directory=True)


def prepare_data(scenes: Iterable[Tuple[str, str]], data_root: Path) -> List[str]:
    """Make every ``(dataset, scene)`` present under ``data_root``; returns the
    errors (one actionable line each) instead of stopping at the first."""
    try:
        link_data_root(data_root)
    except SetupError as e:
        return [str(e)]
    errors, todo = [], set()
    for dataset, scene in sorted(set(scenes)):
        if ready(data_root / scene):
            continue
        if dataset not in SOURCES:
            errors.append(f"{data_root / scene} is missing: no source known for {dataset}")
        else:
            todo.add(SOURCES[dataset])
    for name in sorted(todo):
        try:
            subprocess.run(["bash", str(REPO / "scripts" / "download_data.sh"), name], cwd=REPO, check=True,
                           env={**os.environ, "DATA_ROOT": str(data_root)})
        except (subprocess.CalledProcessError, OSError) as e:
            errors.append(f"scripts/download_data.sh {name} failed ({e}); see the output above")
    return errors


def dino_argv(scene: str, data_root: Path) -> List[str]:
    """The per-scene step before its jobs: a Synthetic scene's seed points
    (colmap_points.ply) if missing, then the DINO cache. Idempotent. Training
    would create both too, but concurrent jobs would race on the files."""
    return [sys.executable, str(REPO / "scripts" / "preprocess_dino.py"),
            "--scene", scene, "--data-root", str(data_root)]


def tmc3() -> str:
    """Path of the pinned tmc3 (found or built by scripts/setup/build_tmc13.sh)."""
    try:
        out = subprocess.run(["bash", str(REPO / "scripts" / "setup" / "build_tmc13.sh")], cwd=REPO,
                             check=True, stdout=subprocess.PIPE, text=True).stdout
    except subprocess.CalledProcessError as e:
        raise SetupError(f"scripts/setup/build_tmc13.sh failed (exit {e.returncode}); it needs git, cmake "
                         "and a C++ compiler, see its output above") from None
    return out.strip().splitlines()[-1]
