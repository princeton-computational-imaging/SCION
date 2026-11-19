"""Training loop for :class:`CodebookModel`.

One iteration (tests/test_trainer.py, tests/test_config_snapshot.py):

1. ``model.step = step``; strategy ``step_pre_forward`` hooks (instance
   resize, MH splat births) on the previous step's camera / batch, then
   optimizer rebinding if a hook resized a buffer;
2. zero grads; render the next train camera; L1 + SSIM + LPIPS +
   regularisers (:meth:`CodebookModel.get_loss_dict`) plus, after the
   discriminator warm-up, one adversarial step (:func:`train_loss_dict`);
3. backward and one Adam step per parameter group;
4. strategy ``step_post_backward`` hooks (gradient / L1 accumulation; they
   see this step's gradients and the updated Adam moments).

A progress line on stdout every ``log_every`` steps, eval every
``eval_every`` steps, a checkpoint every ``save_every`` steps (only the
latest is kept) plus one at the end. Non-finite outputs or losses
raise ``RuntimeError`` (the CLI then exits non-zero).
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch

from scion.data.loader import FullImageLoader
from scion.data.scene import Scene
from scion.engine.optim import rebind_params

if TYPE_CHECKING:
    from scion.models.codebook import CodebookModel
    from scion.strategies.composite import CompositeStrategy

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Trainer knobs (Hydra ``trainer`` group + ``machine.seed``)."""

    max_steps: int = 30_000
    eval_every: int = 1000   # 0 = never
    save_every: int = 2000   # 0 = only the final checkpoint
    log_every: int = 100
    device: str = "cuda"
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("outputs/run"))
    # Telemetry: "tensorboard", "wandb", both ("tensorboard+wandb"), or "none";
    # add "viewer" (e.g. "tensorboard+viewer") for the live browser viewer.
    vis: str = "tensorboard"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_finite(tensors: Dict[str, Any], step: int, where: str) -> None:
    """Fail fast on NaN/Inf in a dict of tensors."""
    bad = []
    for key, val in tensors.items():
        if not isinstance(val, torch.Tensor) or not val.is_floating_point():
            continue
        if not torch.isfinite(val).all():
            n_nan = int(torch.isnan(val).sum().item())
            n_inf = int(torch.isinf(val).sum().item())
            bad.append(f"{key} (nan={n_nan}, inf={n_inf}, shape={tuple(val.shape)})")
    if bad:
        raise RuntimeError(f"Non-finite values in {where} at step {step}: " + "; ".join(bad))


def train_loss_dict(model: "CodebookModel", camera, batch: dict, step: int):
    """One render + every training loss, including the adversarial term.

    ``(outputs, loss_dict, metrics_dict)``; the total loss is the sum of
    ``loss_dict``.
    """
    model.step = step
    outputs = model.get_outputs(camera)
    _check_finite(outputs, step, "outputs")
    metrics_dict = model.get_metrics_dict(outputs, batch)
    loss_dict = model.get_loss_dict(outputs, batch, metrics_dict)
    _check_finite(loss_dict, step, "loss")

    if (
        model.discriminator is not None
        and model.training
        and step >= model.config.discriminator.warmup_step
    ):
        gt_img = model.composite_with_background(
            model.get_gt_img(batch["image"]), outputs["background"]
        )
        adv_mask = batch.get("mask", None)
        if adv_mask is not None:
            adv_mask = model._downscale_if_required(adv_mask).to(gt_img.device)
        adv_metrics, g_loss = model.adversarial_step(outputs["rgb"], gt_img, adv_mask, step)
        if g_loss is not None:
            loss_dict["adv_loss"] = g_loss
        model._adv_metrics = adv_metrics
    return outputs, loss_dict, metrics_dict


def _scalars(values: Dict[str, Any], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{k}": float(v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in values.items()
            if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1)}


class _OptimizerShim:
    """``.optimizers`` dict: the surface that ``CodebookMixin.replace_*_params``
    and the strategies read."""

    __slots__ = ("optimizers",)

    def __init__(self, optimizers: Dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers


class _Telemetry:
    """Scalars to TensorBoard and/or W&B (``TrainerConfig.vis``)."""

    def __init__(self, vis: str, log_dir: Path, run_config: Optional[dict]):
        self._tb = self._wandb = None
        if "tensorboard" in vis:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=str(log_dir))
            except ImportError:
                logger.warning("trainer.vis=%s but tensorboard is not installed; not logging", vis)
        if "wandb" in vis:
            import wandb
            self._wandb = wandb.init(dir=str(log_dir), config=run_config)

    def scalars(self, values: Dict[str, float], step: int) -> None:
        if self._tb is not None:
            for k, v in values.items():
                self._tb.add_scalar(k, v, step)
        if self._wandb is not None:
            self._wandb.log(values, step=step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()


def _viewer_render_fn(model: "CodebookModel", device: str):
    """nerfview ``render_fn``: a viewer camera (OpenCV c2w) -> uint8 HxWx3 render."""
    from scion.camera import Camera

    gl_from_cv = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))  # Camera is OpenGL

    @torch.no_grad()
    def render_fn(camera_state, img_wh):
        c2w = torch.from_numpy(camera_state.c2w).float() @ gl_from_cv
        K = torch.from_numpy(camera_state.get_K(img_wh)).float()
        model.eval()  # fixed background at full resolution; Trainer.step switches back
        rgb = model.get_outputs(Camera(c2w, K, *img_wh).to(device))["rgb"]
        return (rgb.clamp(0, 1) * 255).byte().cpu().numpy()

    return render_fn


def _start_viewer(model: "CodebookModel", device: str):
    """Live viser viewer (nerfview, as in gsplat's example trainer) on port 8080
    or the next free one. It renders only while holding ``viewer.lock``."""
    import nerfview
    import viser

    return nerfview.Viewer(server=viser.ViserServer(port=8080),
                           render_fn=_viewer_render_fn(model, device), mode="training")


_MODEL_SCALARS = ("T_max", "free_count", "num_candidates", "num_instances",
                  "_last_replaced_T_max", "_d_step_count")
_ADA_STATE = ("p", "real_acc_ema", "_update_counter")  # plain attributes, not in state_dict


class Trainer:
    """Training loop for :class:`CodebookModel`.

    ``lrs`` maps parameter-group names (``model.get_param_groups()``) to
    learning rates; each group gets its own Adam (eps 1e-15), no schedule.
    """

    def __init__(
        self,
        model: "CodebookModel",
        strategies: "CompositeStrategy",
        scene: Scene,
        config: TrainerConfig,
        *,
        lrs: Dict[str, float],
        run_config: Optional[dict] = None,
        scene_source: Optional[dict] = None,
        config_sha256: Optional[str] = None,
    ):
        self.model = model
        self.strategies = strategies
        self.scene = scene
        self.config = config
        self.lrs = dict(lrs)
        self.run_config = run_config  # resolved Hydra config (telemetry only)
        # {scene_path, scene_kwargs}: lets checkpoint.load rebuild the training scene.
        self.scene_source = dict(scene_source or {})
        # The run's config identity: saved in meta.json, compared on resume (None: no check).
        self.config_sha256 = config_sha256
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._opt_shim = _OptimizerShim(self.optimizers)
        self.strategy_state: Dict[str, Any] = {}
        self.train_loader: Optional[FullImageLoader] = None
        self.eval_loader: Optional[FullImageLoader] = None
        self._iter = None
        self._last_camera = None
        self._last_batch = None
        self.start_step = 0
        self.step_times: List[float] = []
        self.peak_vram_bytes: Optional[int] = None
        self.viewer = None  # nerfview.Viewer when cfg.vis has "viewer"

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        cfg = self.config
        self.model.to(cfg.device)
        self.model.train()
        # After .to(): strategies allocate device-side accumulators.
        self.strategy_state = self.strategies.initialize_state(self.model)
        self._build_optimizers()
        self.train_loader = FullImageLoader(
            self.scene, split="train", shuffle=True, device=cfg.device, seed=cfg.seed,
        )
        # Periodic eval is monitoring only (no checkpoint selection), so scoring
        # Blender's test split here leaks nothing into training.
        if self.scene.eval_indices:
            self.eval_loader = FullImageLoader(
                self.scene, split="eval", shuffle=False, device=cfg.device,
            )
        self._iter = iter(self.train_loader)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def _build_optimizers(self) -> None:
        missing = [n for n in self.model.get_param_groups() if n not in self.lrs]
        if missing:
            raise KeyError(f"no learning rate for parameter groups {missing}")
        self.optimizers.clear()
        for name, params in self.model.get_param_groups().items():
            self.optimizers[name] = torch.optim.Adam(params, lr=float(self.lrs[name]), eps=1e-15)

    # ------------------------------------------------------------------- step

    def step(self, step: int):
        """One iteration; returns ``(loss_dict, metrics_dict)`` as tensors
        (convert with :func:`_scalars` only when logging: each is a sync)."""
        model = self.model
        model.train()
        model.step = step

        # (1) structural moves on the previous step's view, then rebind.
        pre = self.strategies.step_pre_forward(
            model, step, self.strategy_state,
            camera=self._last_camera, batch=self._last_batch,
        )
        model._last_strategy_metrics = {**getattr(model, "_last_strategy_metrics", {}), **pre}
        rebind_params(model, self._opt_shim, self.strategy_state)

        # (2)-(3) forward, losses, backward, optimizer step.
        for opt in self.optimizers.values():
            opt.zero_grad()
        batch = next(self._iter)
        camera = batch["camera"]
        self._last_camera, self._last_batch = camera, batch
        _outputs, loss_dict, metrics_dict = train_loss_dict(model, camera, batch, step)
        loss = sum(loss_dict.values())
        loss.backward()
        for opt in self.optimizers.values():
            if any(p.grad is not None for g in opt.param_groups for p in g["params"]):
                opt.step()

        # (4) signal accumulation (reads grads + updated Adam moments).
        post = self.strategies.step_post_backward(
            model, step, self.strategy_state,
            optimizers=self._opt_shim,
            lr=self.optimizers["splat_means"].param_groups[0]["lr"],
        )
        if post:
            model._last_strategy_metrics = {**getattr(model, "_last_strategy_metrics", {}), **post}
        return {"loss": loss, **loss_dict}, metrics_dict

    # ------------------------------------------------------------------- eval

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Mean PSNR / SSIM / LPIPS over the eval split (the model's metrics)."""
        if self.eval_loader is None:
            return {}
        from scion.engine.eval import eval_views

        rows, _fps = eval_views(self.model, self.scene, out_dir=None, loader=self.eval_loader)
        self.model.train()
        return {f"eval/{k}": sum(r[k] for r in rows) / len(rows) for k in rows[0]}

    # ------------------------------------------------------------ checkpoint

    def save_checkpoint(self, step: int) -> Path:
        """``<output_dir>/checkpoints/step-NNN.ckpt`` (older ones are removed)."""
        from scion.engine.checkpoint import save, trainable_state

        model = self.model
        cfg_snap = asdict(model.config)
        trainer_state = {
            "model": trainable_state(torch.nn.Module.state_dict(model)),  # uncompacted
            "scalars": {k: int(getattr(model, k, 0)) for k in _MODEL_SCALARS},
            "strategy_state": self.strategy_state,
            "d_optimizer": model._d_optimizer.state_dict() if model._d_optimizer else None,
            "ada": ({k: getattr(model.discriminator.ada, k) for k in _ADA_STATE}
                    if model.discriminator is not None else None),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_loader": self.train_loader.state_dict(),
        }
        path = save(
            model=model,
            optimizers=self.optimizers,
            step=step,
            output_dir=Path(self.config.output_dir),
            config_snapshot=cfg_snap,
            trainer_state=trainer_state,
            config_sha256=self.config_sha256,
            **self.scene_source,
        )
        for old in path.parent.glob("step-*.ckpt"):
            if old != path:
                old.unlink()
        return path

    def resume(self, path: Path) -> int:
        """Restore model, optimizer, strategy, RNG and camera-order state;
        training continues at the saved step + 1. Call after :meth:`setup`.
        Refuses a checkpoint whose meta.json ``config_sha256`` is not this
        run's (unless the trainer was built without one)."""
        meta_path = Path(path).parent / "meta.json"
        saved = json.loads(meta_path.read_text()).get("config_sha256") if meta_path.is_file() else None
        if self.config_sha256 is not None and saved != self.config_sha256:
            raise ValueError(
                f"refusing to resume from {path}: it was trained with config_sha256 {saved}, "
                f"this run's config is {self.config_sha256}; resume with the same config and "
                "overrides (only trainer.load_checkpoint may differ)")
        payload = torch.load(path, map_location=self.config.device, weights_only=True)
        ts = payload["trainer_state"]
        model = self.model
        model.load_state_dict(ts["model"], strict=False)
        for k, v in ts["scalars"].items():
            setattr(model, k, v)
        model.invalidate_cache()
        self._build_optimizers()
        for name, sd in payload["optimizers"].items():
            self.optimizers[name].load_state_dict(sd)
        self.strategy_state = ts["strategy_state"]
        if ts["d_optimizer"] is not None:
            model._d_optimizer.load_state_dict(ts["d_optimizer"])
            for k, v in ts["ada"].items():
                setattr(model.discriminator.ada, k, v)
        torch.set_rng_state(ts["torch_rng"].cpu())
        if ts["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in ts["cuda_rng"]])
        self.train_loader.load_state_dict(ts["train_loader"])
        self.start_step = int(payload["step"]) + 1
        return self.start_step

    # -------------------------------------------------------------------- fit

    def fit(self) -> None:
        if self.train_loader is None:
            self.setup()
        cfg = self.config
        cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        tel = _Telemetry(cfg.vis, Path(cfg.output_dir), self.run_config)
        if "viewer" in cfg.vis and self.viewer is None:
            self.viewer = _start_viewer(self.model, cfg.device)
        viewer = self.viewer
        last = cfg.max_steps - 1
        try:
            for step in range(self.start_step, cfg.max_steps):
                while viewer is not None and viewer.state.status == "paused":
                    time.sleep(0.01)
                # The viewer renders between iterations, never mid-step or mid-eval.
                with viewer.lock if viewer is not None else contextlib.nullcontext():
                    t0 = time.perf_counter()
                    losses, metrics = self.step(step)
                    self.step_times.append(time.perf_counter() - t0)
                    rays = self.model.last_size[0] * self.model.last_size[1]
                    if cfg.log_every and step % cfg.log_every == 0:
                        values = {**_scalars(losses, "loss/"), **_scalars(metrics, "train/")}
                        window = self.step_times[-cfg.log_every:]
                        print(f"[train] step {step}/{cfg.max_steps}  loss {values['loss/loss']:.4f}  "
                              f"psnr {values.get('train/psnr', float('nan')):.2f}  "
                              f"gaussians {int(values.get('train/gaussian_count', 0)):,}  "
                              f"{len(window) / sum(window):.1f} it/s", flush=True)
                        tel.scalars(values, step)
                    if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
                        ev = self.evaluate()
                        logger.info("eval step=%d %s", step, ev)
                        tel.scalars(ev, step)
                    if step == last or (cfg.save_every and step > 0 and step % cfg.save_every == 0):
                        self.save_checkpoint(step)
                if viewer is not None:
                    viewer.state.num_train_rays_per_sec = rays / self.step_times[-1]
                    viewer.update(step, rays)
        finally:
            tel.close()
            if cuda:
                self.peak_vram_bytes = int(torch.cuda.max_memory_allocated())

    @property
    def train_time_s(self) -> float:
        """Wall-clock of the training iterations (no eval / checkpoints)."""
        return float(sum(self.step_times))

    @property
    def median_step_s(self) -> Optional[float]:
        return float(statistics.median(self.step_times)) if self.step_times else None
