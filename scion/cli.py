"""Entry points of the ``scion-{train,eval,render}`` scripts.

Heavy imports are deferred into the entry points so ``import
scion.cli`` stays cheap.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


# Exit code for a tolerated NaN / device-assert abort (opt-in via
# SCION_TOLERATE_NAN_ABORT=1). Distinct and non-zero so every shell caller
# still sees failure, while sweep drivers can recognize and requeue-skip it.
NAN_ABORT_EXIT_CODE = 42


def _is_tolerable_nan_abort(exc: BaseException) -> bool:
    """True only for the NaN / non-finite / device-assert failure family.

    CUDA out-of-memory is *never* tolerable: an OOM'd machine reporting
    success is exactly how contaminated numbers happen. Matching is on
    exception type plus a narrow keyword set, so OOM and unrelated messages
    never match.
    """
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return False
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    if "out of memory" in msg:
        return False
    return any(
        kw in msg
        for kw in ("non-finite", "nan", "device-side assert", "sigfpe")
    )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train_main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for training: ``scion-train`` /
    ``python -m scion.train`` / ``python train.py``.

    Composes the in-repo ``config/`` tree with Hydra and runs
    :func:`scion.engine.run.fit` + ``score`` into Hydra's run directory: checkpoints, every compression
    level up to ``compression=<level>`` with its eval renders, and
    ``results.json`` (metrics per level, timing, peak VRAM). Exits non-zero
    on a non-finite loss / output or a CUDA error.
    """
    import hydra
    from omegaconf import DictConfig

    repo_root = Path(__file__).resolve().parent.parent
    config_path = str(repo_root / "config")
    if not Path(config_path).is_dir():
        raise SystemExit(
            "scion-train: Hydra config tree not found at "
            f"{config_path}. The training CLI needs the repo's config/ "
            "directory, which ships with the source tree but not with a "
            "non-editable wheel. Install from a checkout instead: "
            "git clone <repo> && pip install -e ."
        )

    @hydra.main(version_base=None, config_path=config_path, config_name="config")
    def _main(cfg: DictConfig) -> None:
        import json as _json

        from hydra.core.hydra_config import HydraConfig

        from scion.engine.run import fit, score

        out_dir = Path(HydraConfig.get().runtime.output_dir)
        try:
            trainer = fit(cfg, out_dir)
            viewer = trainer.viewer
            # Scoring loads each level's decoded parameters into the model: no viewer renders meanwhile.
            with viewer.lock if viewer is not None else contextlib.nullcontext():
                summary = {
                    **score(cfg, trainer, out_dir),
                    "train_time_s": trainer.train_time_s,
                    "median_step_s": trainer.median_step_s,
                    "peak_vram_bytes": trainer.peak_vram_bytes,
                }
            (out_dir / "results.json").write_text(_json.dumps(summary, indent=2))
            logger.info("wrote %s", out_dir / "results.json")
            if viewer is not None:
                viewer.complete()
                print("Training done; the viewer keeps serving the final model. Ctrl+C to exit.", flush=True)
                with contextlib.suppress(KeyboardInterrupt):
                    while True:
                        time.sleep(3600)
        except RuntimeError as e:
            # NaN/device-assert aborts on degenerate scenes can optionally
            # be tolerated (sweep runs want the partial trajectory kept:
            # SCION_TOLERATE_NAN_ABORT=1). Even then the process exits with a
            # distinct non-zero code and drops a marker file, so a failed
            # run is never indistinguishable from a completed one.
            import os as _os

            tolerate = _os.environ.get("SCION_TOLERATE_NAN_ABORT", "") in (
                "1", "true", "yes",
            )
            if not (tolerate and _is_tolerable_nan_abort(e)):
                raise
            msg = str(e)
            logger.exception(
                "Training aborted by non-finite / device assert; exiting "
                "with code %d (SCION_TOLERATE_NAN_ABORT=1).",
                NAN_ABORT_EXIT_CODE,
            )
            try:
                marker = {
                    "aborted_early": True,
                    "abort_reason": msg[:500],
                    "exit_code": NAN_ABORT_EXIT_CODE,
                }
                (out_dir / "aborted_early.json").write_text(_json.dumps(marker, indent=2))
            except Exception:
                logger.exception("could not write aborted_early.json marker")
            sys.exit(NAN_ABORT_EXIT_CODE)

    if argv is not None:
        old_argv = sys.argv
        sys.argv = [old_argv[0], *argv]
        try:
            _main()
        finally:
            sys.argv = old_argv
    else:
        _main()


# ---------------------------------------------------------------------------
# eval (A3b)
# ---------------------------------------------------------------------------


def eval_main(argv: Optional[list[str]] = None) -> None:
    """``python -m scion.eval --ckpt PATH [--level L] [--artifact FILE] [--out DIR]``:
    score a trained model on its eval split (the paper's metrics,
    :func:`scion.engine.eval.eval_views`); JSON to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="python -m scion.eval",
        description="Evaluate a trained SCION model on its eval split.",
    )
    parser.add_argument(
        "--ckpt", type=Path, required=True,
        help="Run dir (or a step-*.ckpt / checkpoints/ dir when --level is not given).",
    )
    parser.add_argument(
        "--level", choices=("raw", "quantized", "codec"), default=None,
        help="Score this compression level's artifact in the run dir "
             "(default: the latest step checkpoint).",
    )
    parser.add_argument(
        "--artifact", type=Path, default=None,
        help="With --level: score this artifact file instead of the run's own "
             "(e.g. a codec.omg re-encoded by scripts/compress_omg_posthoc.py).",
    )
    parser.add_argument("--scene", type=Path, default=None,
                        help="Scene directory, if the run's scene cannot be found.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the GT | render eval images here.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)
    sys.exit(_eval_ckpt_mode(args))


def _eval_ckpt_mode(args: argparse.Namespace) -> int:
    import json as _json

    from scion.data import load_scene
    from scion.engine.checkpoint import load as _ckpt_load
    from scion.engine.eval import eval_views

    if args.artifact is not None and args.level is None:
        print("ERROR: --artifact needs --level", file=sys.stderr)
        return 2
    if args.level == "codec":
        from scion.compression import levels
        levels.preflight("codec", None, skip_hint="--level quantized")
    loaded = _ckpt_load(args.ckpt, device=args.device, level=args.level, artifact=args.artifact)
    scene = (load_scene(args.scene, **(loaded.meta.get("scene_kwargs") or {}))
             if args.scene else loaded.scene)
    if scene is None:
        print("ERROR: the run's scene could not be loaded; pass --scene <dir>.", file=sys.stderr)
        return 2
    rows, fps = eval_views(loaded.model, scene, args.out)
    out = {k: sum(r[k] for r in rows) / len(rows) for k in ("psnr", "ssim", "lpips")}
    out.update(step=loaded.step, level=args.level, n_views=len(rows), fps=fps,
               n_gaussians=int(loaded.model.num_points))
    print(_json.dumps(out, indent=2))
    return 0


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def render_main(argv: Optional[list[str]] = None) -> None:
    """``python -m scion.render --ckpt PATH [--level L] --camera-idx N --out PNG
    [--scene DIR]``: render one camera of the run's scene (all frames, in
    file-name order) to a PNG."""
    parser = argparse.ArgumentParser(
        prog="python -m scion.render",
        description="Render a single image from a trained SCION model.",
    )
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Run dir (or a step-*.ckpt when --level is not given).")
    parser.add_argument("--level", choices=("raw", "quantized", "codec"), default=None,
                        help="Render this compression level's artifact (default: latest step checkpoint).")
    parser.add_argument("--camera-idx", type=int, default=0,
                        help="Camera index within the scene's full image list.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output PNG path (parent dirs created if missing).")
    parser.add_argument("--scene", type=Path, default=None,
                        help="Scene directory, if the run's scene cannot be found.")
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    args = parser.parse_args(argv)

    import torch
    from PIL import Image

    from scion.data import load_scene
    from scion.engine.checkpoint import load

    loaded = load(args.ckpt, device=args.device, level=args.level)
    scene = (load_scene(args.scene, **(loaded.meta.get("scene_kwargs") or {}))
             if args.scene else loaded.scene)
    if scene is None:
        raise SystemExit("python -m scion.render: the run's scene could not be loaded; pass --scene")
    if args.camera_idx >= len(scene.cameras):
        raise IndexError(
            f"camera_idx={args.camera_idx} out of range; scene has "
            f"{len(scene.cameras)} cameras."
        )
    camera = scene.cameras[args.camera_idx].to(args.device)
    model = loaded.model

    with torch.no_grad():
        outputs = model.get_outputs(camera)

    rgb = outputs["rgb"].clamp(0, 1).detach().cpu().mul(255).to(torch.uint8).numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.out)
    print(f"wrote {args.out}")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
