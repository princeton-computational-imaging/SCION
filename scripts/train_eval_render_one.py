#!/usr/bin/env python3
"""Train, evaluate and render one paper job; write its run.json (schema v2).

    python scripts/train_eval_render_one.py --variant V [--ablation A] \\
        --dataset D --scene S --seed N [--tag T] [--runs-root DIR] \\
        [--override K=V ...] [--allow-dirty]

The job's Hydra overrides are rebuilt in registry order
(``experiments/paper/targets.py``): ``method=codebook variant=V dataset=D
scene=S machine.seed=N [ablation=A]`` then every ``--override``; training
runs in-process through :mod:`scion.engine.run`. Output
(``evals/RUN_SCHEMA.md``)::

    <runs-root>/<slot>/<dataset>/<scene>/seed<N>/
        run.json  train.log  checkpoints/{step-*,raw,quantized}.ckpt  codec.omg
        eval_views/<level>/eval_img_<NNNN>.png   # GT | render
        candidates/template_<k>.png  isolation_<k>.jpg

An existing run.json is deleted first, so a rerun that fails early never
leaves the old record standing. Refuses a dirty tree
(``scion.util.provenance.tree_is_dirty``) unless
``--allow-dirty`` or ``SCION_ALLOW_DIRTY=1`` (then ``git_dirty: true``
is recorded). Exit 0 iff the record is ``DONE``; 2 = refused or training
failed, 3 = evaluation failed (the run.json carries the error); SIGTERM /
SIGINT record ``FAILED_*`` and then exit 143 / raise. Failed candidate
renders only drop ``artifacts.templates``.
"""
from __future__ import annotations

import argparse
import contextlib
import faulthandler
import json
import logging
import os
import platform
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("train_eval_render_one")

RUNS_ROOT = REPO_ROOT / "evals" / "runs"
SCHEMA_VERSION = 2


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", required=True, help="config/variant/<V>.yaml")
    p.add_argument("--ablation", default=None, help="config/ablation/<A>.yaml")
    p.add_argument("--dataset", required=True, help="config/dataset/<D>.yaml")
    p.add_argument("--scene", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tag", default=None,
                   help="Environment-only arm label (e.g. a SCION_BACKEND_* "
                        "override); appended to the path slot.")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--override", action="append", default=[],
                   help="Extra Hydra override, appended last (repeatable).")
    p.add_argument("--allow-dirty", action="store_true",
                   help="Run with uncommitted code changes (recorded as git_dirty).")
    return p.parse_args(argv)


def job_overrides(args: argparse.Namespace) -> List[str]:
    """The job's Hydra overrides, in registry order."""
    ov = ["method=codebook", f"variant={args.variant}", f"dataset={args.dataset}",
          f"scene={args.scene}", f"machine.seed={args.seed}"]
    if args.ablation:
        ov.append(f"ablation={args.ablation}")
    return ov + list(args.override)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _gpu_name() -> str:
    """GPU model name (non-identifying), e.g. 'NVIDIA GeForce RTX 5090'."""
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""


def _versions() -> Dict[str, Optional[str]]:
    import gsplat
    import torch

    import scion
    from scion.compression.levels import library_versions

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gsplat": gsplat.__version__,
        "scion": scion.__version__,
        **library_versions(),
    }


def split_record(cfg, scene) -> Dict[str, Any]:
    """run.json ``split``: what the eval numbers were scored on."""
    from scion.util.provenance import data_sha256, eval_images_sha256

    if (Path(cfg.data) / "transforms_train.json").exists():
        mode = "blender_test"  # the scene's fixed transforms_test.json split
    else:
        mode = cfg.dataparser.eval_mode
    return {
        "eval_mode": mode,
        "eval_interval": int(cfg.dataparser.eval_interval) if mode == "interval" else None,
        "train_split_fraction": (float(cfg.dataparser.train_split_fraction)
                                 if mode == "fraction" else None),
        "downscale_factor": int(scene.downscale_factor),
        "n_train": len(scene.train_indices),
        "n_eval": len(scene.eval_indices),
        "eval_images_sha256": eval_images_sha256(
            Path(scene.image_paths[i]).name for i in scene.eval_indices),
        "data_sha256": data_sha256(Path(cfg.data)),
    }


def running_record(key, overrides: List[str], cfg, *, dirty: bool) -> Dict[str, Any]:
    """The ``RUNNING`` run.json written before training starts."""
    from scion.util.provenance import config_sha256_of, git_sha

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "error": None,
        "job": {"variant": key.variant, "ablation": key.ablation, "dataset": key.dataset,
                "scene": key.scene, "seed": key.seed, "tag": key.tag},
        "slot": key.slot,
        "overrides": list(overrides),
        "config_sha256": config_sha256_of(cfg),
        "git_sha": git_sha(),
        **({"git_dirty": True} if dirty else {}),
        "gpu": _gpu_name(),
        "versions": _versions(),
        "started_at": _now(),
        "ended_at": None,
    }


def done_record(base: Dict[str, Any], *, cfg, trainer, model_stats: Dict[str, Any],
                levels: Dict[str, Any]) -> Dict[str, Any]:
    """The ``DONE`` run.json from the RUNNING one plus the finished run."""
    from scion.ops.backend import resolved_tiers

    rec = dict(base)
    rec.update({
        "status": "DONE",
        "error": None,
        "backend_tiers": resolved_tiers(),
        "ended_at": _now(),
        "train_time_s": trainer.train_time_s,
        "median_step_s": trainer.median_step_s,
        "peak_vram_bytes": trainer.peak_vram_bytes,
        "split": split_record(cfg, trainer.scene),
        "n_gaussians": levels["metrics"]["raw"]["n_gaussians"],
        "model_stats": model_stats,
        "compression_level": cfg.compression.level,
        **levels,
    })
    rec["artifacts"] = {**levels["artifacts"], "templates": "candidates", "train_log": "train.log"}
    return rec


# ---------------------------------------------------------------------------
# Renders (figure targets) + model stats
# ---------------------------------------------------------------------------


def _save_png_uint8(img, path: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image
    if isinstance(img, torch.Tensor):
        arr = (img.clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8")
    else:
        arr = (np.clip(img, 0, 1) * 255).astype("uint8")
    Image.fromarray(arr).save(path, format="PNG")


def _render_templates(model, *, out_dir: Path) -> List[str]:
    """Render every candidate's splat template, side-view auto-framed
    (same recipe as ``scripts/build_figure1_panels._render_candidate_template``)."""
    import torch
    from scion.util.vis import render_single_candidate, setup_side_view_camera

    out_dir.mkdir(parents=True, exist_ok=True)
    K = int(model._num_candidates)
    device = model.splat_means.device
    bg = model._get_background_color()
    paths: List[str] = []
    pad = max(4, len(str(K)))
    for k in range(K):
        rows = model.get_candidate_rows(k)
        out_path = out_dir / f"template_{k:0{pad}d}.png"
        if rows.numel() == 0:
            _save_png_uint8(torch.ones(64, 64, 3, device=device), out_path)  # keep paths aligned
        else:
            means = model.splat_means.data[rows]
            scales = model.splat_scales.data[rows]
            opacities = model.splat_opacities.data[rows]
            scales_exp = torch.exp(scales)
            min_b = (means - 3.0 * scales_exp).min(dim=0).values
            max_b = (means + 3.0 * scales_exp).max(dim=0).values
            extent = (max_b - min_b).clamp(min=0.01)
            ctr = (min_b + max_b) / 2
            viewmat, K_mat, W, H = setup_side_view_camera(ctr - extent / 2, ctr + extent / 2, 256, device)
            img = render_single_candidate(
                means=means, scales=scales, quats=model.splat_quats.data[rows],
                colors=torch.sigmoid(model.splat_features_dc.data[rows]),
                opacities=opacities + torch.full_like(opacities, 5.0),
                viewmat=viewmat, K=K_mat, width=W, height=H,
                background_color=tuple(bg.tolist()),
            )
            _save_png_uint8(img, out_path)
        paths.append(str(out_path.relative_to(out_dir.parent)))
    return paths


@contextlib.contextmanager
def _filtered_assembly(model, keep_inst):
    """Render only the instances in ``keep_inst`` (sliced out of the assembled bundle)."""
    from scion.models.codebook_base import GaussianBundle

    original = model._get_assembled
    saved = (model._gauss_cache, model._gauss_cache_step, model._last_assignment_per_gaussian)
    filtered: list = []

    def _filtered():
        if not filtered:
            bundle = original()
            keep_gauss = keep_inst.to(model._cached_inst_idx.device)[model._cached_inst_idx]
            filtered.append(GaussianBundle(
                means=bundle.means[keep_gauss], scales=bundle.scales[keep_gauss],
                quats=bundle.quats[keep_gauss], opacities=bundle.opacities[keep_gauss],
                features_dc=bundle.features_dc[keep_gauss],
                features_rest=bundle.features_rest[keep_gauss],
            ))
            model._last_assignment_per_gaussian = None
        return filtered[0]

    model._last_assignment_per_gaussian = None
    model._gauss_cache, model._gauss_cache_step = None, -1
    model._get_assembled = _filtered
    try:
        yield
    finally:
        model._get_assembled = original
        model._gauss_cache, model._gauss_cache_step, model._last_assignment_per_gaussian = saved


def _render_isolations(model, *, camera, out_dir: Path) -> List[str]:
    """Render the scene once per candidate k with only its instances."""
    import torch
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    assignments = model.instance_assignments.detach()
    K = int(model._num_candidates)
    paths: List[str] = []
    pad = max(4, len(str(K)))
    for k in range(K):
        keep = assignments == k
        if not bool(keep.any()):
            continue
        with torch.no_grad(), _filtered_assembly(model, keep):
            rgb = model.get_outputs(camera)["rgb"]
        arr = (torch.nan_to_num(rgb, nan=0.0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        out_path = out_dir / f"isolation_{k:0{pad}d}.jpg"
        Image.fromarray(arr).save(out_path, format="JPEG", quality=90, optimize=True)
        paths.append(str(out_path.relative_to(out_dir.parent)))
    return paths


def _model_stats(model) -> Dict[str, Any]:
    """Lightweight per-checkpoint summary of the codebook geometry."""
    import torch
    K = int(model._num_candidates)
    rows_per_k = [int(model.get_candidate_rows(k).numel()) for k in range(K)]
    stats: Dict[str, Any] = {
        "num_candidates": K,
        "num_instances": int(model._num_instances),
        "num_splats": int(sum(rows_per_k)),
        "mean_splats_per_candidate": float(sum(rows_per_k) / K) if K else 0.0,
        "active_candidates": int(sum(1 for n in rows_per_k if n > 0)),
    }
    # Reuse statistics (init ablation): exp(entropy) of instance->candidate usage.
    counts = model.candidate_usage_counts().double()
    used = counts[counts > 0]
    if used.numel() > 0:
        p = used / used.sum()
        stats["effective_candidates"] = float(torch.exp(-(p * p.log()).sum()))
        stats["instances_per_candidate"] = float(used.sum() / used.numel())
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


class _Tee:
    """Mirror a stream into the run's train.log."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, s):
        self._stream.write(s)
        self._log.write(s)
        return len(s)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    from scion.util.provenance import compose, tree_is_dirty
    from scion.util.run_index import JobKey

    key = JobKey(args.variant, args.ablation, args.dataset, args.scene, args.seed, args.tag)
    run_json = key.path(args.runs_root.resolve())
    # A rerun that fails before its RUNNING record (dirty tree, compose, versions)
    # must not leave the previous record standing.
    run_json.unlink(missing_ok=True)
    dirty = tree_is_dirty()
    allow_dirty = args.allow_dirty or os.environ.get("SCION_ALLOW_DIRTY") == "1"
    if dirty and not allow_dirty:
        print("train_eval_render_one: the code has uncommitted changes (modified or untracked "
              "files under the code paths, see scion/util/provenance.py); commit "
              "first (or pass --allow-dirty / SCION_ALLOW_DIRTY=1)", file=sys.stderr)
        return 2

    overrides = job_overrides(args)
    cfg = compose(overrides)
    run_dir = run_json.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(run_dir / "train.log", "w", buffering=1)
    faulthandler.enable(file=log_file)  # segfaults / CUDA aborts leave a traceback
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, log_file), _Tee(sys.__stderr__, log_file)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    # Slurm / the runner stop jobs with SIGTERM: unwind (and record FAILED_*), exit 143.
    prev_sigterm = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        return _run_job(running_record(key, overrides, cfg, dirty=dirty), cfg, run_dir, run_json)
    finally:  # leave the process as we found it (in-process callers, tests)
        signal.signal(signal.SIGTERM, prev_sigterm)
        faulthandler.enable(file=sys.__stderr__)
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        logging.basicConfig(level=logging.INFO, force=True)
        log_file.close()


def _run_job(record: Dict[str, Any], cfg, run_dir: Path, run_json: Path) -> int:
    from scion.engine import run

    _atomic_write_json(run_json, record)

    def _fail(status: str, exc: BaseException, code: int) -> int:
        logger.error("%s", status, exc_info=exc)
        record.update(status=status, ended_at=_now(),
                      error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        _atomic_write_json(run_json, record)
        if not isinstance(exc, Exception):
            raise exc  # SIGINT / SIGTERM: the caller must still see the interrupt
        return code

    try:
        trainer = run.fit(cfg, run_dir)
    except BaseException as exc:
        return _fail("FAILED_TRAIN", exc, 2)
    try:
        model = trainer.model
        stats = _model_stats(model)
        levels = run.score(cfg, trainer, run_dir)
        record = done_record(record, cfg=cfg, trainer=trainer, model_stats=stats, levels=levels)
    except BaseException as exc:
        return _fail("FAILED_EVAL", exc, 3)
    # DONE now: a kill or crash during the figure-only renders below keeps the metrics.
    _atomic_write_json(run_json, record)
    try:
        _render_templates(model, out_dir=run_dir / "candidates")
        if trainer.scene.eval_indices:
            camera = trainer.scene.eval_cameras()[0].to(model.device)
            _render_isolations(model, camera=camera, out_dir=run_dir / "candidates")
    except Exception:
        logger.warning("candidate template / isolation renders failed; recorded without them",
                       exc_info=True)
        record["artifacts"].pop("templates")
        _atomic_write_json(run_json, record)
    lvl = record["compression_level"]
    logger.info("DONE %s  %s PSNR=%.3f", run_json, lvl, record["metrics"][lvl]["psnr"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
