"""Splat-level density control: dead-splat relocation + MH birth proposals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from scion.strategies.base import CodebookStrategy
from scion.ops import (
    relocate_dead_splats,
    mh_birth_splats,
    accumulate_splat_grad2d,
)

if TYPE_CHECKING:
    from scion.camera import Camera
    from scion.models.codebook_base import CodebookMixin


def cache_gradient_ema(model: CodebookMixin, state: dict, optimizers: Any) -> None:
    """Read Adam ``exp_avg`` for instance params into ``state['grad_ema']``.

    The cached dict is consumed by gradient-informed MH moves in the next
    step's ``step_pre_forward``.  Called from every splat
    strategy's ``step_post_backward`` (after ``optimizer.step()``).
    """
    if optimizers is None:
        state["grad_ema"] = {}
        model._cached_grad_ema = {}
        return

    cache = {}
    for key in ("instance_positions", "instance_scales"):
        param = getattr(model, key, None)
        if param is None:
            continue

        if hasattr(optimizers, 'optimizers'):
            if key not in optimizers.optimizers:
                continue
            opt = optimizers.optimizers[key]
            opt_state = opt.state.get(param)
        else:
            opt_state = optimizers.state.get(param)

        if opt_state is not None and "exp_avg" in opt_state:
            cache[key] = opt_state["exp_avg"].clone()

    state["grad_ema"] = cache
    model._cached_grad_ema = cache


class MHSplatStrategy(CodebookStrategy):
    """MH-3DGS splat-level density control: relocate + birth proposals.

    Each densification step:
      1. Unconditionally relocate dead splats (MCMC-style mass-preserving).
      2. Propose new splats via importance-weighted MH birth proposals
         (per-candidate z-scored ``sigmoid(w_opa·z(opa) + w_grad·z(grad)
         + w_l1·z(L1))``), with annealed coarse + fine batches and
         per-parent ``1/(1 + log(1 + denom))`` damping.

    Maintains per-row accumulators required by the importance signal:
      - ``splat_grad2d_accum`` / ``splat_grad2d_count``: 2D absgrad
        from the rasteriser (reverse-mapped via ``_assembly_global_idx``).
      - ``splat_l1_accum`` / ``splat_l1_count``: L1 error sampled at
        per-splat ``means2d`` from ``model._last_l1_map``.
      - ``splat_denom``: per-row visited counter for density damping.

    Uses the module-level ``cache_gradient_ema`` helper so the instance
    strategy sees the same Adam exp_avg signal.
    """

    config_name = "mh"

    def initialize_state(self, model: CodebookMixin) -> dict:
        T_max = model._T_max
        device = model.splat_means.device
        return {
            "splat_grad2d_accum": torch.zeros(T_max, device=device),
            "splat_grad2d_count": torch.zeros(T_max, device=device),
            "splat_l1_accum": torch.zeros(T_max, device=device),
            "splat_l1_count": torch.zeros(T_max, device=device),
            "splat_denom": torch.zeros(T_max, device=device),
        }

    @torch.no_grad()
    def step_pre_forward(
        self,
        model: CodebookMixin,
        step: int,
        state: dict,
        *,
        camera: Optional[Camera] = None,
        batch: Optional[dict] = None,
    ) -> Dict[str, float]:
        """Relocate dead splats + paper-faithful MH birth proposals.

        Emits per-tick metrics (n_relocated, n_births, accept_rate,
        Sk_mean/max, opacity quantiles) so MH behaviour is observable.
        """
        metrics: Dict[str, float] = {}
        cfg = model.config

        if step < cfg.splat_refine_start:
            return metrics
        if step % cfg.splat_refine_every != 0:
            return metrics
        if step >= cfg.splat_refine_stop:
            return metrics

        K = model._num_candidates
        splat_params = {k: v.data for k, v in model.get_splat_params_dict().items()}

        # 1. Unconditional relocate (paper step 1: recycle dead splats)
        n_relocated = relocate_dead_splats(
            splat_params,
            model.splat_candidate_id,
            K,
            cfg.splat_min_opacity,
            model._binoms,
            fw_offsets=model._fw_offsets,
            fw_sorted_rows=model._fw_sorted_rows,
        )

        # 2. Anneal proposal scales + batch sizes over the densification window.
        progress = (
            (step - cfg.splat_refine_start)
            / max(1, cfg.splat_refine_stop - cfg.splat_refine_start)
        )
        progress = max(0.0, min(1.0, progress))

        def _lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        c_scale = _lerp(
            cfg.mh_splat_coarse_scale_max,
            cfg.mh_splat_coarse_scale_min,
            progress,
        )
        f_scale = _lerp(
            cfg.mh_splat_fine_scale_max,
            cfg.mh_splat_fine_scale_min,
            progress,
        )
        voxel_size = _lerp(
            cfg.mh_splat_voxel_size_max,
            cfg.mh_splat_voxel_size_min,
            progress,
        )
        c_batch = max(0, int((1.0 - progress) * cfg.mh_splat_coarse_batch_per_cand))
        f_batch = max(0, int((1.0 - progress) * cfg.mh_splat_fine_batch_per_cand))

        # 3. MH birth proposals (paper step 2: importance-guided growth)
        old_free_count = model.free_count
        model.free_count, mh_metrics = mh_birth_splats(
            splat_params,
            model.splat_candidate_id,
            model.free_list,
            model.free_count,
            K,
            cap_max=cfg.max_splats_per_candidate,
            accept_threshold=cfg.splat_birth_accept_threshold,
            binoms=model._binoms,
            fw_offsets=model._fw_offsets,
            fw_sorted_rows=model._fw_sorted_rows,
            splat_grad_accum=state.get("splat_grad2d_accum"),
            splat_grad_count=state.get("splat_grad2d_count"),
            splat_l1_accum=state.get("splat_l1_accum"),
            splat_l1_count=state.get("splat_l1_count"),
            splat_denom=state.get("splat_denom"),
            w_opa=cfg.mh_splat_w_opa,
            w_grad=cfg.mh_splat_w_grad,
            w_l1=cfg.mh_splat_w_l1,
            coarse_scale=c_scale,
            fine_scale=f_scale,
            coarse_batch_per_cand=c_batch,
            fine_batch_per_cand=f_batch,
            voxel_size=voxel_size,
            voxel_alpha=cfg.mh_splat_voxel_alpha,
            use_voxel_penalty=cfg.mh_splat_use_voxel_penalty,
            use_density_factor=cfg.mh_splat_use_density_factor,
        )
        n_births = int(mh_metrics.get("n_births", 0.0))
        if model.free_count != old_free_count or n_relocated > 0:
            model.invalidate_cache()

        # --- 4. Observability metrics ---
        active_rows = model.get_active_rows()
        if active_rows.numel() > 0:
            counts = torch.bincount(
                model.splat_candidate_id[active_rows], minlength=K,
            )[:K].float()
            metrics["mh_splat/Sk_mean"] = float(counts.mean().item())
            metrics["mh_splat/Sk_max"] = float(counts.max().item())
            metrics["mh_splat/Sk_at_cap_frac"] = float(
                (counts >= cfg.max_splats_per_candidate).float().mean().item()
            )
            opa_sig = torch.sigmoid(
                splat_params["opacities"][active_rows, 0]
            )
            q = torch.quantile(
                opa_sig.detach(),
                torch.tensor([0.1, 0.5], device=opa_sig.device),
            )
            metrics["mh_splat/opa_p10"] = float(q[0])
            metrics["mh_splat/opa_p50"] = float(q[1])
            metrics["mh_splat/opa_below_thresh_frac"] = float(
                (opa_sig < cfg.splat_min_opacity).float().mean().item()
            )

        metrics["mh_splat/n_relocated"] = float(n_relocated)
        metrics["mh_splat/n_births"] = float(n_births)
        metrics["mh_splat/n_proposals_total"] = float(
            mh_metrics.get("n_proposals_total", 0.0)
        )
        metrics["mh_splat/accept_rate"] = float(
            mh_metrics.get("accept_rate", 0.0)
        )
        metrics["mh_splat/parent_imp_mean"] = float(
            mh_metrics.get("parent_imp_mean", 0.0)
        )
        metrics["mh_splat/free_count"] = float(model.free_count)

        # --- 5. Reset accumulators (paper's densification_postfix semantics) ---
        for key in (
            "splat_grad2d_accum",
            "splat_grad2d_count",
            "splat_l1_accum",
            "splat_l1_count",
            "splat_denom",
        ):
            buf = state.get(key)
            if buf is not None:
                buf.zero_()

        return metrics

    @torch.no_grad()
    def step_post_backward(
        self,
        model: CodebookMixin,
        step: int,
        state: dict,
        *,
        optimizers: Any = None,
        lr: float = 0.0,
    ) -> Dict[str, float]:
        """Accumulate per-splat 2D absgrad + L1 + denom; cache grad EMA."""
        # Cache gradient EMA so candidate/instance MH strategies can read it.
        cache_gradient_ema(model, state, optimizers)

        info = getattr(model, "info", None)
        global_idx = getattr(model, "_assembly_global_idx", None)
        if info is None or global_idx is None:
            return {}

        means2d = info.get("means2d")
        if means2d is None:
            return {}

        # --- 1. Per-splat 2D absgrad, reverse-mapped to buffer rows ---
        absgrad = getattr(means2d, "absgrad", None)
        if absgrad is not None:
            absgrad = absgrad.detach()
            if absgrad.dim() == 3 and absgrad.shape[0] == 1:
                absgrad = absgrad[0]  # [N, 2]
            accum = state.get("splat_grad2d_accum")
            count = state.get("splat_grad2d_count")
            if accum is not None and count is not None:
                accumulate_splat_grad2d(
                    absgrad, global_idx, accum, count, model._T_max,
                )

        # --- 2. Per-splat L1 sampled at means2d from cached prior-step
        #         L1 map (mean over RGB; populated by get_loss_dict). ---
        l1_map = getattr(model, "_last_l1_map", None)
        l1_accum = state.get("splat_l1_accum")
        l1_count = state.get("splat_l1_count")
        if (
            l1_map is not None
            and l1_accum is not None
            and l1_count is not None
        ):
            m2d = means2d.detach()
            if m2d.dim() == 3:  # [C, N, 2] — assume C=1
                m2d = m2d[0]
            H, W = l1_map.shape[-2:]
            x = m2d[..., 0].clamp(0, W - 1).long()
            y = m2d[..., 1].clamp(0, H - 1).long()
            on_screen = (
                (m2d[..., 0] >= 0) & (m2d[..., 0] < W)
                & (m2d[..., 1] >= 0) & (m2d[..., 1] < H)
            )
            valid = on_screen & (global_idx < model._T_max)
            idx = global_idx[valid]
            if idx.numel() > 0:
                per_gauss_err = l1_map[y[valid], x[valid]]
                l1_accum.scatter_add_(0, idx, per_gauss_err.to(l1_accum.dtype))
                l1_count.scatter_add_(0, idx, torch.ones_like(per_gauss_err))

        # --- 3. Per-row visited counter (paper's ``denom``) ---
        denom = state.get("splat_denom")
        if denom is not None:
            valid = global_idx < model._T_max
            idx = global_idx[valid]
            if idx.numel() > 0:
                denom.scatter_add_(
                    0, idx, torch.ones_like(idx, dtype=denom.dtype),
                )

        return {}
