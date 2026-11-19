"""CodebookMixin: shared codebook functionality for CodebookModel.

Plain Python class (not nn.Module) providing codebook init, assembly,
caching and optimizer management. Designed
to be mixed into classes that *are* nn.Modules (CodebookModel)
without introducing diamond inheritance.

The host class must:
  - Inherit from nn.Module (provides register_buffer, register_parameter).
  - Call ``init_codebook(...)`` during its own __init__ / populate_modules.
  - Implement ``get_splat_params_dict()`` returning a dict of nn.Parameters
    keyed by ``{"means", "scales", "rotations", "opacities", "features_dc"}``.
  - Have ``instance_positions``, ``instance_rotations``, ``instance_scales``,
    ``instance_opacities`` as nn.Parameters on ``self``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.nn import Parameter

from scion.ops.common import SPLAT_FREE, build_binoms


@dataclass
class GaussianBundle:
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor

    def detach(self) -> "GaussianBundle":
        return type(self)(
            **{
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in self.__dict__.items()
            }
        )

    def to(self, *args, **kwargs) -> "GaussianBundle":
        return type(self)(
            **{
                k: (v.to(*args, **kwargs) if torch.is_tensor(v) else v)
                for k, v in self.__dict__.items()
            }
        )

    def clone(self) -> "GaussianBundle":
        return type(self)(
            **{
                k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in self.__dict__.items()
            }
        )

    @classmethod
    def concatenate(cls, bundles: "List[GaussianBundle]") -> "GaussianBundle":
        """Concatenate multiple GaussianBundles along dim 0."""
        if not bundles:
            raise ValueError("Cannot concatenate empty list of bundles")
        if len(bundles) == 1:
            return bundles[0]
        return cls(
            means=torch.cat([b.means for b in bundles], dim=0),
            scales=torch.cat([b.scales for b in bundles], dim=0),
            quats=torch.cat([b.quats for b in bundles], dim=0),
            opacities=torch.cat([b.opacities for b in bundles], dim=0),
            features_dc=torch.cat([b.features_dc for b in bundles], dim=0),
            features_rest=torch.cat([b.features_rest for b in bundles], dim=0),
        )

# Minimum world-space gaussian scale (linear, not log).  Clamped into every
# assembled ``world_scales`` to prevent denormal covariances from reaching
# gsplat's CUDA projection kernel (divide-by-zero → SIGFPE).  Set to
# roughly the effective pixel size of a small scene; tighter than gsplat's
# own threshold but loose enough to leave room for gradient descent.
_MIN_WORLD_SCALE_LOG = math.log(1e-4)


class CodebookMixin:
    """Mixin providing the codebook layout, assembly and parameter swaps."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_codebook(
        self,
        *,
        spatial_dim: int,
        rot_dim: int,
        volume_exp: float,
        p_splat: int,
        p_inst: int,
        K: int,
        N: int,
        T_max: int,
        register_buffers: bool = True,
    ) -> None:
        """Initialize shared codebook attributes and buffers.

        Must be called during the host class's initialization. The host
        must be an ``nn.Module`` so that ``register_buffer`` is available.

        Args:
            spatial_dim: Dimensionality of positions (2 or 3).
            rot_dim: Dimensionality of rotations (1 for 2D angle, 4 for 3D quat).
            volume_exp: Exponent for volume/area computation (1/3 for 3D, 1/2 for 2D).
            p_splat: Number of free parameters per splat (for MDL).
            p_inst: Number of free parameters per instance (for MDL).
            K: Initial number of candidates.
            N: Number of instances.
            T_max: Total pre-allocated splat buffer size.
            register_buffers: If True (default), register ``splat_candidate_id``,
                ``free_list``, and ``instance_assignments`` as nn.Module buffers.
                Set to False when the host class already creates these buffers.
        """
        self.spatial_dim = spatial_dim
        self.rot_dim = rot_dim
        self.volume_exp = volume_exp
        self.p_splat = p_splat
        self.p_inst = p_inst

        self.num_candidates = K
        self.num_instances = N
        self.T_max = T_max

        self.instance_size_changed = False

        # Non-persistent buffer: auto-moves with model.to(device) but
        # excluded from state_dict (cheap to rebuild from scratch).
        self.register_buffer("_binoms_buf", build_binoms(), persistent=False)

        self.gauss_cache = None
        self.gauss_cache_step = -1

        self.cached_grad_ema: Dict[str, Any] = {}

        # Precomputed assembly index tensors (populated by recompute_assembly_indices)
        self._cached_inst_idx: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._cached_global_idx: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._cached_total: int = 0
        self._cached_T_active: int = 0

        # Forward-map caches (populated by _rebuild_layout)
        self._fw_sorted_rows: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._fw_offsets: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._candidate_sizes: torch.Tensor = torch.empty(0, dtype=torch.long)

        # Buffers (self must be an nn.Module)
        if register_buffers:
            # Reverse mapping: splat row -> candidate ID (SPLAT_FREE = free)
            self.register_buffer(
                "splat_candidate_id",
                torch.full((T_max,), SPLAT_FREE, dtype=torch.long),
            )
            # Free-list stack (LIFO): free_list[:free_count] are available rows
            self.register_buffer(
                "free_list", torch.arange(T_max, dtype=torch.long)
            )
            self.free_count = T_max
            self.register_buffer(
                "instance_assignments", torch.zeros(N, dtype=torch.long)
            )

        # --- Layout buffers for the fused assemble kernel ---
        # Registered unconditionally (independent of `register_buffers`):
        # the host class may have already created splat_candidate_id /
        # instance_assignments itself, but the layout buffers are
        # always built here.
        # Flat per-instance splat-row table.
        self.register_buffer(
            "_cand_splat_rows",
            torch.empty(0, dtype=torch.int32),
            persistent=False,
        )
        # Prefix-sum into _cand_splat_rows. Shape (N+1,).
        self.register_buffer(
            "_inst_offsets",
            torch.zeros(N + 1, dtype=torch.int32),
            persistent=False,
        )
        # -1 marks the layout as stale: it is rebuilt before the next read.
        self._layout_total: int = -1

    # ------------------------------------------------------------------
    # Row lookup
    # ------------------------------------------------------------------

    def get_candidate_rows(self, k: int) -> torch.Tensor:
        """Return splat row indices owned by candidate *k*.

        Uses the cached forward map if available, otherwise scans
        ``splat_candidate_id`` directly.
        """
        if self._fw_sorted_rows.numel() > 0 and self._fw_offsets.numel() > 0:
            s = int(self._fw_offsets[k])
            e = int(self._fw_offsets[k + 1])
            return self._fw_sorted_rows[s:e].clone()
        return torch.where(self.splat_candidate_id == k)[0]

    def get_active_rows(self) -> torch.Tensor:
        """Return all active (non-free) splat row indices."""
        if self._fw_sorted_rows.numel() > 0:
            return self._fw_sorted_rows
        return torch.where(self.splat_candidate_id >= 0)[0]

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def recompute_assembly_indices(self) -> None:
        """Rebuild the layout and assembly index tensors. Idempotent."""
        self._rebuild_layout()

    # ------------------------------------------------------------------
    # Layout for the fused assemble kernel
    # ------------------------------------------------------------------

    def _rebuild_layout(self) -> None:
        """Recompute the layout buffers from current state.

        Reads ``splat_candidate_id`` and ``instance_assignments`` and
        repopulates:

        Fused-assemble layout (int32):
          - ``_cand_splat_rows``[total]
          - ``_inst_offsets``   [N+1]     (prefix sum)
          - ``_layout_total`` (Python int)

        Forward map + assembly indices (int64, read by strategies / eval):
          - ``_fw_sorted_rows`` [T_active]
          - ``_fw_offsets``     [K+1]
          - ``_candidate_sizes``[K]
          - ``_cached_inst_idx`` [total]
          - ``_cached_global_idx`` [total]
          - ``_cached_total``, ``_cached_T_active``

        Idempotent. Pure PyTorch on purpose: it runs only after structural
        changes and costs < 0.1% of training time (garden, 30k steps).
        """
        cid = self.splat_candidate_id          # int64 [T_max]
        K = int(self.num_candidates)
        dev = cid.device

        # ---- Forward map: active rows compacted, sorted by candidate ----
        active_mask = cid >= 0
        active_rows = torch.where(active_mask)[0]  # int64 [T_active]
        T_active = int(active_rows.shape[0])

        if T_active == 0:
            # Empty layout — clear everything to canonical empty state.
            fw_sorted_rows = torch.empty(0, dtype=torch.long, device=dev)
            fw_offsets = torch.zeros(K + 1, dtype=torch.long, device=dev)
            candidate_sizes = torch.zeros(K, dtype=torch.long, device=dev)
        else:
            active_cids = cid[active_rows]  # int64 [T_active]
            sorted_order = torch.argsort(active_cids, stable=True)
            fw_sorted_rows = active_rows[sorted_order]
            sorted_cids = active_cids[sorted_order]
            candidate_sizes = torch.bincount(sorted_cids, minlength=K).to(torch.long)
            fw_offsets = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=dev), candidate_sizes.cumsum(0)]
            )

        # ---- Per-instance flat lookup table ----
        k = self.instance_assignments              # int64 [N]
        N = int(k.shape[0])

        sizes = fw_offsets[k + 1] - fw_offsets[k]  # int64 [N]

        inst_offsets_i64 = torch.zeros(N + 1, dtype=torch.long, device=dev)
        inst_offsets_i64[1:] = sizes.cumsum(0)
        total = int(inst_offsets_i64[-1].item())

        if total == 0:
            cached_inst_idx = torch.empty(0, dtype=torch.long, device=dev)
            cached_global_idx = torch.empty(0, dtype=torch.long, device=dev)
            cand_splat_rows = torch.empty(0, dtype=torch.int32, device=dev)
        else:
            cached_inst_idx = torch.repeat_interleave(
                torch.arange(N, device=dev), sizes,
            )
            starts = fw_offsets[k]
            local_pos = torch.arange(total, device=dev) - torch.repeat_interleave(
                sizes.cumsum(0) - sizes, sizes,
            )
            fw_positions = torch.repeat_interleave(starts, sizes) + local_pos
            cached_global_idx = fw_sorted_rows[fw_positions]
            cand_splat_rows = cached_global_idx.to(torch.int32)

        inst_offsets_i32 = inst_offsets_i64.to(torch.int32)

        # ---- Commit state ----
        # Layout buffers (int32) — assigning to a registered buffer slot
        # is how we replace it (PyTorch supports `module.buf = new_tensor`
        # provided the name was registered).
        self._cand_splat_rows = cand_splat_rows
        self._inst_offsets = inst_offsets_i32
        self._layout_total = total

        # int64 forward map + assembly indices.
        self._fw_sorted_rows = fw_sorted_rows
        self._fw_offsets = fw_offsets
        self._candidate_sizes = candidate_sizes
        self._cached_inst_idx = cached_inst_idx
        self._cached_global_idx = cached_global_idx
        self._cached_total = total
        self._cached_T_active = T_active

    def _ensure_layout(self) -> None:
        """Lazy rebuild trigger — call before reading any layout buffer."""
        if self._layout_total < 0:
            self._rebuild_layout()

    def assemble_gaussians(self) -> Dict[str, torch.Tensor]:
        """Assemble world-space gaussians from codebook + instance transforms.

        Calls ``get_splat_params_dict()`` (which the host must implement) to
        obtain the flat splat buffer, then applies per-instance position,
        rotation, scale, and opacity transforms.

        CUDA tensors dispatch through ``scion.ops.assemble.assemble_fused``
        (CUDA > Triton > Python tiers; the CUDA kernel matches the Python
        reference to within ~1e-5 in fp32). CPU tensors use the Python
        reference directly.

        Returns:
            Dict with keys ``{"means", "scales", "rotations", "opacities",
            "features_dc"}`` containing world-space tensors.
        """
        params = self.get_splat_params_dict()

        # Ensure layout is fresh. After this, _cand_splat_rows and
        # _inst_offsets are the source of truth for assembly indices.
        self._ensure_layout()
        total = self._layout_total

        # Empty-bundle short circuit (the reference returns the same, but this
        # skips dispatch resolution on the rare empty layout).
        device = self._cand_splat_rows.device
        if total == 0:
            empty_means = torch.zeros((0, 3), device=device)
            empty_scales = torch.zeros((0, 3), device=device)
            empty_rot = torch.zeros((0, 4), device=device)
            empty_opac = torch.zeros((0, 1), device=device)
            empty_feat = torch.zeros((0, 3), device=device)
            return {
                "means": empty_means,
                "scales": empty_scales,
                "rotations": empty_rot,
                "opacities": empty_opac,
                "features_dc": empty_feat,
            }

        # Dispatch through the backend registry. With ``scion_cuda``
        # installed this runs the CUDA kernel (substantially faster than the
        # python tier on synthetic microbenches — see docs/cuda_architecture.md);
        # otherwise the registry resolves to ``ops._ref.assemble.assemble_fused``.
        # The autograd Function wired in ``ops/assemble.py`` covers both
        # forward and backward.
        #
        # CPU tensors stay on the python reference: the CUDA tier asserts
        # ``is_cuda`` and would raise; the env override would also bypass
        # this, but on CUDA hardware the default resolution is what we want.
        if params["means"].is_cuda:
            from scion.ops.assemble import assemble_fused as _assemble_fused
            world_means, world_scales, world_rot, world_opac, world_feat = _assemble_fused(
                params["means"],
                params["scales"],
                params["rotations"],
                params["opacities"],
                params["features_dc"],
                self.instance_positions,
                self.instance_rotations,
                self.instance_scales,
                self.instance_opacities,
                self._cand_splat_rows,
                self._inst_offsets,
                _MIN_WORLD_SCALE_LOG,
                5.0,
                20.0,
            )
        else:
            # CPU-only path: use the reference body directly.
            from scion.ops._ref.assemble import (
                assemble_fused as _assemble_fused_ref,
            )
            world_means, world_scales, world_rot, world_opac, world_feat = (
                _assemble_fused_ref(
                    params["means"],
                    params["scales"],
                    params["rotations"],
                    params["opacities"],
                    params["features_dc"],
                    self.instance_positions,
                    self.instance_rotations,
                    self.instance_scales,
                    self.instance_opacities,
                    self._cand_splat_rows,
                    self._inst_offsets,
                    _MIN_WORLD_SCALE_LOG,
                    5.0,
                    20.0,
                )
            )

        # Final finite check — if a NaN/Inf slipped through (e.g. from an
        # Adam update on a frozen row), surface a clean RuntimeError rather
        # than letting gsplat's CUDA kernel SIGFPE. The tensor->bool branch
        # is a device sync and data-dependent, so it is skipped inside
        # torch.compile graphs (compiled callers keep gsplat's own error).
        if not torch.compiler.is_compiling():
            if not torch.isfinite(world_means).all():
                raise RuntimeError(
                    "Non-finite values in world_means before rasterization"
                )
            if not torch.isfinite(world_rot).all():
                raise RuntimeError(
                    "Non-finite values in world_rot before rasterization"
                )

        return {
            "means": world_means,
            "scales": world_scales,
            "rotations": world_rot,
            "opacities": world_opac,
            "features_dc": world_feat,
        }

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Clear the assembly cache and rebuild the layout.

        The rebuild is eager so callers that read the index tensors
        (``_cached_global_idx`` etc.) right after see fresh data.
        """
        self.gauss_cache = None
        self.gauss_cache_step = -1
        self._rebuild_layout()

    # ------------------------------------------------------------------
    # Abstract interface (host must implement)
    # ------------------------------------------------------------------

    def get_splat_params_dict(self) -> Dict[str, torch.Tensor]:
        """Return flat splat buffer as a dict.

        Must be implemented by the host class. Expected keys:
        ``{"means", "scales", "rotations", "opacities", "features_dc"}``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared concrete helpers
    # ------------------------------------------------------------------

    def get_instance_param_names(self) -> List[str]:
        """Return attribute names for instance nn.Parameters."""
        return [
            "instance_positions",
            "instance_rotations",
            "instance_scales",
            "instance_opacities",
        ]

    def replace_instance_params(self, optimizers: Any) -> None:
        """Re-create instance nn.Parameters after an instance resize.

        Handles both an object with an ``.optimizers`` dict (one optimizer
        per parameter name, as the trainer passes) and a plain
        ``torch.optim.Optimizer``.

        Args:
            optimizers: Optimizer(s) managing instance parameters.
        """
        for key in self.get_instance_param_names():
            old_param = getattr(self, key)
            new_param = Parameter(old_param.data)
            setattr(self, key, new_param)

            if hasattr(optimizers, "optimizers"):
                # Per-parameter optimizer dict
                if key in optimizers.optimizers:
                    opt = optimizers.optimizers[key]
                    if old_param in opt.state:
                        del opt.state[old_param]
                    for pg in opt.param_groups:
                        pg["params"] = [
                            new_param if p is old_param else p
                            for p in pg["params"]
                        ]
            else:
                # Plain torch optimizer
                if old_param in optimizers.state:
                    del optimizers.state[old_param]
                for pg in optimizers.param_groups:
                    pg["params"] = [
                        new_param if p is old_param else p
                        for p in pg["params"]
                    ]
