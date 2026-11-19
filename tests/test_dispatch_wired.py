"""Assert ``assemble_gaussians`` consumes the dispatch table.

``CodebookMixin.assemble_gaussians`` runs through
``scion.ops.assemble.assemble_fused``, so a training run uses the
``scion.ops.backend`` registry (and the CUDA kernel when built).

This test spies on ``backend.dispatch`` and asserts ``"assemble_fused"``
fires at least once during a single ``_assemble_gaussians`` call on a
minimal CodebookModel. With CUDA available the spy also confirms the
op resolves to the CUDA tier.
"""
from __future__ import annotations

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from scion.models.codebook import CodebookModel, CodebookModelConfig
from scion.ops import backend, assemble as assemble_mod
from scion.ops.common import SPLAT_FREE


def _make_tiny_codebook_model(
    *,
    K: int = 4,
    S: int = 3,
    N: int = 6,
    T_max: int = 32,
    N_max: int = 64,
    device: str = "cpu",
) -> CodebookModel:
    """Minimal CodebookModel — mirrors tests/test_compact_checkpoint.py.

    Kept local to keep the test self-contained.
    """
    T_active = K * S
    assert T_active <= T_max and N <= N_max

    model = CodebookModel.__new__(CodebookModel)
    nn.Module.__init__(model)

    cfg = CodebookModelConfig()
    cfg.num_candidates = K
    cfg.initial_splats_per_candidate = S
    cfg.max_candidates = K
    cfg.max_splats_per_candidate = S
    cfg.num_instances = N
    cfg.max_num_instances = N_max
    cfg.max_num_iterations = 100
    model.config = cfg
    model.step = 0

    torch.manual_seed(0)
    splat_means = torch.randn(T_max, 3, device=device) * 0.05
    splat_scales = torch.randn(T_max, 3, device=device) * 0.1 - 2.0
    splat_quats = F.normalize(torch.randn(T_max, 4, device=device), dim=-1)
    splat_opacities = torch.logit(torch.full((T_max, 1), 0.5, device=device))
    splat_features_dc = torch.rand(T_max, 3, device=device)

    model.register_parameter("splat_means", Parameter(splat_means))
    model.register_parameter("splat_scales", Parameter(splat_scales))
    model.register_parameter("splat_quats", Parameter(splat_quats))
    model.register_parameter("splat_opacities", Parameter(splat_opacities))
    model.register_parameter("splat_features_dc", Parameter(splat_features_dc))

    cid = torch.full((T_max,), SPLAT_FREE, dtype=torch.long, device=device)
    for k in range(K):
        cid[k * S : (k + 1) * S] = k
    model.register_buffer("splat_candidate_id", cid)

    free_list = torch.zeros(T_max, dtype=torch.long, device=device)
    n_free = T_max - T_active
    free_list[:n_free] = torch.arange(T_active, T_max, device=device)
    model.register_buffer("free_list", free_list)
    model.free_count = n_free

    model.register_parameter(
        "instance_positions", Parameter(torch.randn(N, 3, device=device))
    )
    model.register_parameter(
        "instance_rotations",
        Parameter(F.normalize(torch.randn(N, 4, device=device), dim=-1).contiguous()),
    )
    model.register_parameter(
        "instance_scales", Parameter(torch.zeros(N, 1, device=device))
    )
    model.register_parameter(
        "instance_opacities", Parameter(torch.zeros(N, 1, device=device))
    )
    assignments = (torch.arange(N, dtype=torch.long, device=device) % K)
    model.register_buffer("instance_assignments", assignments)

    model.init_codebook(
        spatial_dim=3,
        rot_dim=4,
        volume_exp=1.0 / 3.0,
        p_splat=14,
        p_inst=9,
        K=K, N=N, T_max=T_max,
        register_buffers=False,
    )

    # Hook state — assemble_gaussians only needs a small subset, but the
    # _assemble_gaussians wrapper around it touches several caches.
    model._pending_compact_load = None
    model._assembly_global_idx = None
    model._assembly_inst_idx = None
    model._last_assignment_per_gaussian = None
    model.dino_oracle = None
    model.discriminator = None
    model._d_optimizer = None
    model._adv_metrics = {}
    model._d_last_metrics = {}
    model._d_forward = None
    model.free_count = n_free
    model.num_candidates = K
    model.num_instances = N

    model.recompute_assembly_indices()
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_assemble_gaussians_routes_through_dispatch(monkeypatch):
    """The rank-1 hotspot must dispatch through the backend registry.

    Monkeypatches ``backend.dispatch`` (the entry point that
    ``ops.assemble.AssembleFusedFunction`` calls) to record op-name
    invocations, then runs a forward on a CUDA-resident tiny model and
    asserts ``"assemble_fused"`` fired.

    CPU forwards skip the dispatch (the CUDA kernel asserts ``is_cuda``
    and would raise), so the wiring only applies on CUDA.
    """
    seen: list[str] = []
    original_dispatch = backend.dispatch

    def _spy(op_name: str, *args, **kwargs):
        seen.append(op_name)
        return original_dispatch(op_name, *args, **kwargs)

    # The autograd Function imports ``backend`` at the module level
    # (``from scion.ops import backend`` in ops/assemble.py).
    # That binds ``backend`` as an attribute of the module; replacing
    # ``backend.dispatch`` in the registry module reaches that binding.
    monkeypatch.setattr(backend, "dispatch", _spy)

    model = _make_tiny_codebook_model(device="cuda")

    # Single forward through the assemble path. ``_assemble_gaussians``
    # wraps ``assemble_gaussians`` and is the
    # entry-point everything else in the model funnels through.
    bundle = model._assemble_gaussians()

    assert "assemble_fused" in seen, (
        f"assemble_fused was never dispatched during a forward; "
        f"saw dispatch calls = {seen!r}. "
        f"CodebookMixin.assemble_gaussians did not route through "
        f"scion.ops.assemble.assemble_fused."
    )

    # Sanity: the output shape must match the layout count, i.e. K*S
    # active rows assembled into N instances (assignments cycle modulo K).
    K, S, N = 4, 3, 6
    expected_total = N * S  # each instance points at a candidate with S rows
    assert bundle.means.shape == (expected_total, 3), bundle.means.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_assemble_dispatch_resolves_to_cuda_when_available():
    """When ``scion_cuda`` is installed, dispatch must resolve
    to the CUDA tier (not silently fall back to Python)."""
    import scion

    # The package-level flag.
    assert scion.HAS_CUDA, (
        "HAS_CUDA flag is False but torch.cuda.is_available() returned True; "
        "scion_cuda is not installed."
    )

    backend.reset_backend_cache()
    fn = backend._resolve("assemble_fused")
    # Resolved callable must be the CUDA op, not the python reference.
    table = backend._REGISTRY["assemble_fused"]
    assert fn is table["cuda"], (
        f"assemble_fused did not resolve to the CUDA tier despite "
        f"HAS_CUDA=True. resolved={fn!r}, cuda={table['cuda']!r}, "
        f"python={table['python']!r}"
    )


def test_stale_cuda_extension_raises_clear_error(monkeypatch):
    """A scion_cuda build whose version != the checkout's must not run."""
    import importlib
    import importlib.metadata
    import sys
    import types

    import scion._cuda as cuda_mod

    fake = types.ModuleType("scion_cuda")
    fake.assemble_fused = lambda *a: None
    monkeypatch.setitem(sys.modules, "scion_cuda", fake)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.0")
    try:
        importlib.reload(cuda_mod)
        assert cuda_mod.HAS_CUDA
        with pytest.raises(RuntimeError, match="stale build.*pip install -e scion-cuda/"):
            cuda_mod.OPS["assemble_fused"]()
    finally:
        monkeypatch.undo()
        importlib.reload(cuda_mod)
