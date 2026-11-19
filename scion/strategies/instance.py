"""Instance-level density control: gradient-driven clone / subdivide / cull."""

from __future__ import annotations

import math

import torch

from scion.strategies.base import CodebookStrategy
from scion.ops import (
    compute_instance_urgency,
    accumulate_instance_grad2d,
    clone_instances_by_mask,
    cull_instances_by_mask,
    subdivide_instance,
)


class DefaultInstanceStrategy(CodebookStrategy):
    """Gradient-based instance clone/cull adapted from 3DGS densification.

    Accumulates per-instance screen-space gradients (or uses Adam EMA as
    fallback). Periodically clones high-gradient instances and culls
    low-opacity ones.
    """

    config_name = "default"

    def initialize_state(self, model):
        N = model.instance_positions.shape[0]
        device = model.instance_positions.device
        return {
            "inst_grad2d_accum": torch.zeros(N, device=device),
            "inst_grad2d_count": torch.zeros(N, device=device),
        }

    @torch.no_grad()
    def step_post_backward(self, model, step, state, *, optimizers=None, lr=0.0):
        """Accumulate the per-instance screen-space gradient signal."""
        # Primary signal: absgrad from rasteriser
        info = getattr(model, "info", None)
        inst_idx = getattr(model, "_assembly_inst_idx", None)
        N = model.instance_positions.shape[0]

        # Resize accumulators if N changed (grow OR shrink — stale rows after a
        # cull-heavy refinement otherwise leak into the next firing).
        if state["inst_grad2d_accum"].shape[0] != N:
            device = state["inst_grad2d_accum"].device
            state["inst_grad2d_accum"] = torch.zeros(N, device=device)
            state["inst_grad2d_count"] = torch.zeros(N, device=device)

        if info is not None and inst_idx is not None:
            means2d = info.get("means2d")
            if means2d is not None:
                absgrad = getattr(means2d, "absgrad", None)
                if absgrad is not None:
                    absgrad = absgrad.detach()
                    # Non-packed mode: means2d is [C, N, 2], squeeze camera dim
                    if absgrad.dim() == 3 and absgrad.shape[0] == 1:
                        absgrad = absgrad[0]  # [N, 2]
                    accumulate_instance_grad2d(
                        absgrad,
                        inst_idx,
                        state["inst_grad2d_accum"],
                        state["inst_grad2d_count"],
                        N,
                    )

        return {}

    @torch.no_grad()
    def step_pre_forward(self, model, step, state, *, camera=None, batch=None):
        """Periodic cull / clone / split based on accumulated gradients."""
        metrics = {}
        cfg = model.config

        if step < cfg.instance_refine_start or step >= cfg.instance_refine_stop:
            return metrics

        if step % cfg.instance_refine_every != 0:
            return metrics

        N = model.instance_positions.shape[0]
        device = model.instance_positions.device

        # Get gradient signal
        accum = state.get("inst_grad2d_accum")
        count = state.get("inst_grad2d_count")

        if accum is not None and count is not None and accum.shape[0] >= N:
            avg_grad = accum[:N] / count[:N].clamp(min=1)
        elif "grad_ema" in state and "instance_positions" in state.get("grad_ema", {}) and state["grad_ema"]["instance_positions"].shape[0] == N:
            # Fallback: Adam EMA urgency. Emit a one-shot metric so the
            # degraded-signal path is visible in the logs.
            avg_grad = compute_instance_urgency(
                state["grad_ema"]["instance_positions"],
                state["grad_ema"].get("instance_scales"),
            )
            metrics["default_inst_signal_fallback"] = 1.0
        else:
            return metrics

        # Sanitise gradient signal: NaN compares False against any numeric
        # threshold, but ``torch.topk`` ranks NaN as *maximum* and would
        # prefer NaN-gradient instances for subdivide/clone — exactly the
        # ones we want to skip. Replace NaN/Inf with 0 before downstream
        # comparisons and top-k selection.
        avg_grad = torch.nan_to_num(avg_grad, nan=0.0, posinf=0.0, neginf=0.0)

        prune_opa = getattr(cfg, 'default_inst_prune_opa', 0.01)
        grow_grad = getattr(cfg, 'default_inst_grow_grad', 0.001)
        grow_scale = getattr(cfg, 'default_inst_grow_scale', 0.5)
        enable_clone = bool(getattr(cfg, "default_inst_enable_clone", True))
        enable_subdivide = bool(
            getattr(cfg, "default_inst_enable_subdivide", True)
        )
        max_scale_log = float(
            getattr(cfg, "default_inst_max_scale_log", 4.0)
        )  # exp(4) ≈ 55; guards against runaway Adam updates.
        cap = cfg.max_num_instances
        any_change = False

        # 1. Cull low-opacity instances
        opa_sigmoid = torch.sigmoid(model.instance_opacities.data[:, 0])
        cull_mask = opa_sigmoid < prune_opa
        # Also cull non-finite-parameter rows — if any of scales/opacities/
        # positions contains NaN/Inf (runaway Adam update or upstream bug),
        # the only safe thing is to remove the instance. Leaving it in drives
        # NaN into the rasteriser next forward and kills the run.
        bad = (
            ~torch.isfinite(model.instance_scales.data).all(dim=-1)
            | ~torch.isfinite(model.instance_opacities.data).all(dim=-1)
            | ~torch.isfinite(model.instance_positions.data).all(dim=-1)
        )
        if bad.any():
            cull_mask = cull_mask | bad
            metrics["default_inst_non_finite_culled"] = float(bad.sum().item())
        n_culled = int(cull_mask.sum().item())

        if cull_mask.any():
            new_pos, new_rot, new_scl, new_opa, new_asgn = cull_instances_by_mask(
                model.instance_positions.data,
                model.instance_rotations.data,
                model.instance_scales.data,
                model.instance_opacities.data,
                model.instance_assignments,
                cull_mask,
            )
            model.instance_positions.data = new_pos
            model.instance_rotations.data = new_rot
            model.instance_scales.data = new_scl
            model.instance_opacities.data = new_opa
            model.instance_assignments = new_asgn
            avg_grad = avg_grad[~cull_mask]
            N = new_pos.shape[0]
            any_change = True

        # 2. Grow high-gradient instances (absgrad > grow_grad, classic 3DGS
        # density control): clone small-scale, subdivide large.
        n_cloned = 0
        n_split = 0
        if N < cap:
            scale_log_col = model.instance_scales.data[:N, 0]
            # Clamp *for eligibility* only (we don't modify the Parameter).
            # Any row whose log-scale exceeds max_scale_log is treated as
            # ineligible for subdivide: its ``scale.exp()`` would push
            # children many scene-extents away via the position-noise term
            # and corrupt the next render.
            scale_vals = scale_log_col.exp()
            runaway = (scale_log_col > max_scale_log) | ~torch.isfinite(scale_log_col)
            high_grad = avg_grad[:N] > grow_grad
            clone_mask = high_grad & (scale_vals < grow_scale) & ~runaway
            split_mask = high_grad & (scale_vals >= grow_scale) & ~runaway
            if not enable_clone:
                clone_mask = torch.zeros_like(clone_mask)
            if not enable_subdivide:
                split_mask = torch.zeros_like(split_mask)
            metrics["default_inst_high_grad_n"] = float(high_grad.sum().item())
            if runaway.any():
                metrics["default_inst_runaway_skipped"] = float(runaway.sum().item())

            # 2a. Subdivide large, high-gradient instances (M children tile
            # the parent ellipsoid, mass-preserving). Fixes the "Default only
            # clones" gap.
            #
            # Bounded per-firing: a naive subdivide-all cascades because
            # children after scale/M^(1/3) often stay above grow_scale for
            # several refinements — N grows geometrically and blows memory.
            # ``default_inst_subdivide_max_parents`` caps the damage at a
            # constant per firing; the remaining eligible parents get picked
            # up in subsequent refinements.
            if split_mask.any():
                M = int(getattr(cfg, "default_inst_subdivide_M", 2))
                M = max(2, min(M, 8))
                max_parents = int(
                    getattr(cfg, "default_inst_subdivide_max_parents", 16)
                )
                src_idx = split_mask.nonzero(as_tuple=False).squeeze(-1)
                budget = (cap - N) // max(M - 1, 1)
                budget = min(budget, max_parents)
                if budget > 0:
                    # Rank by gradient when capping — subdivide the highest-
                    # gradient parents first, deferring the rest.
                    if src_idx.numel() > budget:
                        top_grad = avg_grad[:N][src_idx]
                        _, order = top_grad.topk(budget)
                        src_idx = src_idx[order]
                    # position_noise_scale: 1.0 (the "tile the parent
                    # ellipsoid" setting) places children at σ = exp(scale).
                    # For large-scale high-gradient instances that's many
                    # units from the parent and frequently outside the scene
                    # AABB, corrupting the next render. 0.2 keeps children
                    # inside the parent while still separating them.
                    noise_scale = float(
                        getattr(cfg, "default_inst_subdivide_noise_scale", 0.2)
                    )
                    new_pos, new_rot, new_scl, new_opa, new_asgn = subdivide_instance(
                        model.instance_positions.data,
                        model.instance_rotations.data,
                        model.instance_scales.data,
                        model.instance_opacities.data,
                        model.instance_assignments,
                        src_idx, M,
                        keep_parent=True,
                        position_noise_scale=noise_scale,
                    )
                    model.instance_positions.data = new_pos
                    model.instance_rotations.data = new_rot
                    model.instance_scales.data = new_scl
                    model.instance_opacities.data = new_opa
                    model.instance_assignments = new_asgn
                    n_split = int(src_idx.numel())
                    N = new_pos.shape[0]
                    any_change = True

            # 2b. Clone small, high-gradient instances. Kept opacity-preserving
            # (both source and clone retain the source's opacity) — mass-
            # preserving insert halves opacity on both, which
            # empirically causes a rendered-brightness dip and slows PSNR
            # recovery. The per-step scale-free brightness boost here is the
            # regime DefaultInstance was tuned for in 3DGS ADC.
            if clone_mask.numel() >= N:
                clone_mask = clone_mask[:N]
            if N < cap and clone_mask.any():
                # Cap clones per firing for the same reason as subdivide:
                # unbounded clone on high-grad masks doubles N in a single
                # refinement and spikes memory for assembly-index tensors.
                max_clones = int(
                    getattr(cfg, "default_inst_clone_max_batch", 512)
                )
                clone_budget = min(cap - N, max_clones)
                if int(clone_mask.sum().item()) > clone_budget:
                    clone_idx = clone_mask.nonzero(as_tuple=False).squeeze(-1)
                    top_grad = avg_grad[:N][clone_idx]
                    _, order = top_grad.topk(clone_budget)
                    kept = clone_idx[order]
                    clone_mask = torch.zeros_like(clone_mask)
                    clone_mask[kept] = True
                opa_offset = float(
                    getattr(cfg, "default_inst_clone_opa_offset", 0.0)
                )
                # Floor clones above the prune threshold so they aren't
                # culled on the next refine tick.
                min_opa_logit = math.log(
                    max(float(getattr(cfg, "instance_min_opacity", 5e-3)), 1e-6)
                ) - math.log(
                    max(1.0 - float(getattr(cfg, "instance_min_opacity", 5e-3)), 1e-6)
                )
                new_pos, new_rot, new_scl, new_opa, new_asgn = clone_instances_by_mask(
                    model.instance_positions.data,
                    model.instance_rotations.data,
                    model.instance_scales.data,
                    model.instance_opacities.data,
                    model.instance_assignments,
                    clone_mask,
                    cap,
                    opa_offset=opa_offset,
                    min_opa_logit=min_opa_logit,
                    noise_scale_factor=float(
                        getattr(cfg, "default_inst_clone_noise_scale", 0.1)
                    ),
                )
                added = new_pos.shape[0] - N
                if added > 0:
                    model.instance_positions.data = new_pos
                    model.instance_rotations.data = new_rot
                    model.instance_scales.data = new_scl
                    model.instance_opacities.data = new_opa
                    model.instance_assignments = new_asgn
                    n_cloned = added
                    N = new_pos.shape[0]
                    any_change = True

        # Always flag size change when any structural op fired — not just on
        # net-delta. A cull + clone that net to zero still rewrote rows and
        # invalidated the Adam state indexed against them.
        if any_change:
            model._instance_size_changed = True
        model._num_instances = N

        model.invalidate_cache()

        # Reset accumulators to the current N.
        state["inst_grad2d_accum"] = torch.zeros(N, device=device)
        state["inst_grad2d_count"] = torch.zeros(N, device=device)

        metrics["default_inst_culled"] = float(n_culled)
        metrics["default_inst_cloned"] = float(n_cloned)
        metrics["default_inst_split"] = float(n_split)
        return metrics
