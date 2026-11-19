"""``assemble_fused``: CUDA tier vs the pure-PyTorch reference.

Asserts the CUDA tier of ``assemble_fused`` matches the pure-PyTorch
reference (forward + backward) and runs ``torch.autograd.gradcheck`` on a
small fp64 model.

Checks:
- **C1**: forward max-abs ``< 1e-5`` against ``_ref`` over a random batch.
- **C2**: ``torch.autograd.gradcheck`` passes (against the fp64-capable
  python tier so we don't pin gradcheck to a fp32-only kernel).

Additionally we run a CUDA-backward-vs-python-fallback parity check so the
CUDA tier's ``assemble_fused_bwd`` (when registered) is exercised under
realistic upstream gradients.
"""
from __future__ import annotations

import math

import pytest
import torch

import scion  # noqa: F401  — ensures _cuda / _triton import
from scion.ops import backend
from scion.ops._ref import assemble as ref_mod
from scion.ops.assemble import assemble_fused


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-only"
)

_FP32_TOL = 1e-5
# Backward path goes through atomicAdd scatter into per-row accumulators,
# whose accumulation order is non-deterministic — so the kernel and the
# autograd-reference can differ by a few ULPs even when the math is
# identical. 5e-5 corresponds to ~6 ULPs at gradient magnitudes O(1).
_FWD_BWD_TOL = 5e-5


# --------------------------------------------------------------------------- helpers
def _rand_inputs(
    T_max: int,
    N: int,
    rows_per_inst: int,
    device: str,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    requires_grad: bool = False,
):
    """Build a self-consistent set of inputs for ``assemble_fused``."""
    g = torch.Generator(device=device).manual_seed(seed)

    splat_means = torch.randn(T_max, 3, device=device, dtype=dtype, generator=g)
    splat_scales = torch.randn(T_max, 3, device=device, dtype=dtype, generator=g) * 0.1
    splat_quats = torch.nn.functional.normalize(
        torch.randn(T_max, 4, device=device, dtype=dtype, generator=g), dim=-1,
    )
    splat_opa = torch.randn(T_max, 1, device=device, dtype=dtype, generator=g)
    splat_fd = torch.randn(T_max, 3, device=device, dtype=dtype, generator=g)
    inst_pos = torch.randn(N, 3, device=device, dtype=dtype, generator=g)
    inst_rot = torch.nn.functional.normalize(
        torch.randn(N, 4, device=device, dtype=dtype, generator=g), dim=-1,
    )
    inst_scl = torch.randn(N, 1, device=device, dtype=dtype, generator=g) * 0.1
    inst_opa = torch.randn(N, 1, device=device, dtype=dtype, generator=g)
    cand_splat_rows = torch.randint(
        0, T_max, (N * rows_per_inst,), device=device, generator=g, dtype=torch.int32,
    )
    inst_offsets = torch.arange(
        0, (N + 1) * rows_per_inst, rows_per_inst, device=device, dtype=torch.int32,
    )

    if requires_grad:
        # Clone-detach-set on each tensor we want grads through.
        splat_means = splat_means.detach().clone().requires_grad_(True)
        splat_scales = splat_scales.detach().clone().requires_grad_(True)
        splat_quats = splat_quats.detach().clone().requires_grad_(True)
        splat_opa = splat_opa.detach().clone().requires_grad_(True)
        splat_fd = splat_fd.detach().clone().requires_grad_(True)
        inst_pos = inst_pos.detach().clone().requires_grad_(True)
        inst_rot = inst_rot.detach().clone().requires_grad_(True)
        inst_scl = inst_scl.detach().clone().requires_grad_(True)
        inst_opa = inst_opa.detach().clone().requires_grad_(True)

    return (
        splat_means, splat_scales, splat_quats, splat_opa, splat_fd,
        inst_pos, inst_rot, inst_scl, inst_opa,
        cand_splat_rows, inst_offsets,
    )


# --------------------------------------------------------------------------- C1
@needs_cuda
def test_forward_parity_matches_ref():
    """C1 — forward max-abs < 1e-5 over a random batch."""
    cuda_fn = backend._REGISTRY.get("assemble_fused", {}).get("cuda")
    if cuda_fn is None:
        pytest.skip("CUDA tier not built")

    inp = _rand_inputs(
        T_max=256, N=64, rows_per_inst=4,
        device="cuda", seed=42,
    )
    (sm, sscl, sq, so, sf, ip, ir, iscl, iopa,
     cand_splat_rows, inst_offsets) = inp

    # Force CUDA tier for the dispatched call.
    out_cuda = cuda_fn(
        sm, sscl, sq, so, sf, ip, ir, iscl, iopa,
        cand_splat_rows, inst_offsets,
        math.log(1e-4), 5.0, 20.0,
    )
    out_ref = ref_mod.assemble_fused(
        sm, sscl, sq, so, sf, ip, ir, iscl, iopa,
        cand_splat_rows, inst_offsets,
        math.log(1e-4), 5.0, 20.0,
    )

    names = ["world_means", "world_scales", "world_rot", "world_opa", "world_feat"]
    for n, a, b in zip(names, out_cuda, out_ref):
        max_abs = (a - b).abs().max().item()
        assert max_abs < _FP32_TOL, (
            f"output {n}: max|Δ| = {max_abs:.3e} ≥ {_FP32_TOL:.0e}"
        )


# --------------------------------------------------------------------------- C2
@needs_cuda
def test_gradcheck_fp64(monkeypatch):
    """C2 — torch.autograd.gradcheck on a tiny fp64 model.

    Pinned to the python tier because the CUDA kernel is fp32-only;
    gradcheck dispatching to a fp32 kernel would lose precision and fail
    regardless of math correctness. This still validates the autograd
    Function's wrapping (forward+backward connectivity).
    """
    monkeypatch.setenv("SCION_BACKEND_assemble_fused", "python")
    monkeypatch.setenv("SCION_BACKEND_assemble_fused_bwd", "python")
    backend.reset_backend_cache()

    T_max, N, rows_per_inst = 8, 3, 2
    inp = _rand_inputs(
        T_max=T_max, N=N, rows_per_inst=rows_per_inst,
        device="cuda", dtype=torch.float64, seed=11, requires_grad=True,
    )
    (sm, sscl, sq, so, sf, ip, ir, iscl, iopa,
     cand_splat_rows, inst_offsets) = inp

    diff_inputs = (sm, sscl, sq, so, sf, ip, ir, iscl, iopa)

    def fn(*tensors):
        out = assemble_fused(
            *tensors, cand_splat_rows, inst_offsets,
            math.log(1e-4), 5.0, 20.0,
        )
        return tuple(out)

    assert torch.autograd.gradcheck(
        fn, diff_inputs, eps=1e-6, atol=1e-4, rtol=1e-3, fast_mode=True,
    )


# --------------------------------------------------------------------------- backward parity
@needs_cuda
def test_backward_parity_vs_autograd_fallback(monkeypatch):
    """Compares the dispatched (CUDA-bwd if present, otherwise fallback)
    gradient to the explicit python-tier autograd fallback. Tolerance
    matches the forward gate (1e-5 fp32)."""
    # First: gradient via whatever tier dispatch picks (CUDA if built).
    backend.reset_backend_cache()
    inp_dispatch = _rand_inputs(
        T_max=128, N=24, rows_per_inst=3, device="cuda",
        seed=7, requires_grad=True,
    )
    (sm, sscl, sq, so, sf, ip, ir, iscl, iopa,
     cand_splat_rows, inst_offsets) = inp_dispatch

    out = assemble_fused(
        sm, sscl, sq, so, sf, ip, ir, iscl, iopa,
        cand_splat_rows, inst_offsets,
        math.log(1e-4), 5.0, 20.0,
    )
    # Fixed scalar reduction with non-trivial mixing across the five outputs.
    loss = (
        out[0].pow(2).sum()
        + 1.3 * out[1].sum()
        + 0.7 * out[2].pow(2).sum()
        + 2.1 * out[3].sum()
        + 0.5 * out[4].abs().sum()
    )
    loss.backward()

    grads_dispatch = [
        (t.grad.detach().clone() if (t is not None and t.grad is not None) else None)
        for t in (sm, sscl, sq, so, sf, ip, ir, iscl, iopa)
    ]

    # Second: force python tier (forward + backward) to get the reference.
    monkeypatch.setenv("SCION_BACKEND_assemble_fused", "python")
    monkeypatch.setenv("SCION_BACKEND_assemble_fused_bwd", "python")
    backend.reset_backend_cache()

    inp_ref = _rand_inputs(
        T_max=128, N=24, rows_per_inst=3, device="cuda",
        seed=7, requires_grad=True,
    )
    (sm2, sscl2, sq2, so2, sf2, ip2, ir2, iscl2, iopa2,
     car2, io2) = inp_ref
    out2 = assemble_fused(
        sm2, sscl2, sq2, so2, sf2, ip2, ir2, iscl2, iopa2,
        car2, io2,
        math.log(1e-4), 5.0, 20.0,
    )
    loss2 = (
        out2[0].pow(2).sum()
        + 1.3 * out2[1].sum()
        + 0.7 * out2[2].pow(2).sum()
        + 2.1 * out2[3].sum()
        + 0.5 * out2[4].abs().sum()
    )
    loss2.backward()

    grads_ref = [
        (t.grad.detach().clone() if (t is not None and t.grad is not None) else None)
        for t in (sm2, sscl2, sq2, so2, sf2, ip2, ir2, iscl2, iopa2)
    ]

    names = [
        "splat_means", "splat_scales", "splat_quats", "splat_opa", "splat_feat_dc",
        "inst_pos", "inst_rot", "inst_scl", "inst_opa",
    ]
    for name, a, b in zip(names, grads_dispatch, grads_ref):
        if a is None and b is None:
            continue
        assert a is not None and b is not None, (
            f"grad mismatch (None vs tensor) for {name}"
        )
        max_abs = (a - b).abs().max().item()
        assert max_abs < _FWD_BWD_TOL, (
            f"grad {name}: max|Δ| = {max_abs:.3e} ≥ {_FWD_BWD_TOL:.0e}"
        )

    # Reset cache so other tests see fresh env.
    backend.reset_backend_cache()


# --------------------------------------------------------------------------- regression: shared splat row
@needs_cuda
def test_backward_shared_splat_row_atomicadd():
    """When multiple Gaussians share a splat row, the CUDA backward kernel must atomicAdd their
    contributions — last-writer-wins would silently zero some grads.

    Setup: identity rotations + zero scales / positions → forward is the
    identity transform, and ``d_world_scales = ones`` flows straight back
    to ``d_splat_scales[shared_row]`` summed across the 4 Gaussians.
    """
    cuda_bwd = backend._REGISTRY.get("assemble_fused_bwd", {}).get("cuda")
    if cuda_bwd is None:
        pytest.skip("CUDA tier not built")

    device = "cuda"
    T_max, N, R = 4, 1, 4
    splat_means = torch.zeros(T_max, 3, device=device)
    splat_scales = torch.zeros(T_max, 3, device=device)
    splat_quats = torch.zeros(T_max, 4, device=device); splat_quats[:, 0] = 1.0
    splat_opa = torch.zeros(T_max, 1, device=device)
    splat_fd = torch.zeros(T_max, 3, device=device)
    inst_pos = torch.zeros(N, 3, device=device)
    inst_rot = torch.zeros(N, 4, device=device); inst_rot[:, 0] = 1.0
    inst_scl = torch.zeros(N, 1, device=device)
    inst_opa = torch.zeros(N, 1, device=device)
    cand_splat_rows = torch.zeros(R, device=device, dtype=torch.int32)
    inst_offsets = torch.tensor([0, R], device=device, dtype=torch.int32)

    d_wm = torch.zeros(R, 3, device=device)
    d_ws = torch.ones(R, 3, device=device)
    d_wr = torch.zeros(R, 4, device=device)
    d_wo = torch.zeros(R, 1, device=device)
    d_wf = torch.zeros(R, 3, device=device)

    grads = cuda_bwd(
        splat_means, splat_scales, splat_quats, splat_opa, splat_fd,
        inst_pos, inst_rot, inst_scl, inst_opa,
        cand_splat_rows, inst_offsets,
        d_wm, d_ws, d_wr, d_wo, d_wf,
        math.log(1e-4), 5.0, 20.0,
    )
    # 4 Gaussians × d_world_scales=ones, all hitting splat row 0.
    assert torch.allclose(
        grads[1][0], torch.full((3,), 4.0, device=device)
    ), f"expected [4,4,4] from atomicAdd of 4 ones, got {grads[1][0].tolist()}"
    # Rows 1..3 should remain zero.
    assert grads[1][1:].abs().max().item() == 0.0


# --------------------------------------------------------------------------- regression: non-contiguous upstream
@needs_cuda
def test_backward_handles_noncontiguous_upstream_grad():
    """Regression test: PyTorch creates stride-0 expanded ``ones_like``
    upstream grads when the loss contains plain ``out.sum()`` terms. The
    CUDA backward must ``.contiguous()`` these or the raw ``data_ptr``
    reads garbage memory."""
    inp = _rand_inputs(
        T_max=32, N=8, rows_per_inst=2, device="cuda", seed=3,
    )
    (sm, sscl, sq, so, sf, ip, ir, iscl, iopa,
     cand_splat_rows, inst_offsets) = inp

    # Forge a stride-0 upstream grad (mimics what autograd hands us for
    # ``out.sum()``) and a contiguous one, and check we get the same
    # numerical answer.
    total = int(inst_offsets[-1].item())
    d_ws_expanded = torch.tensor(1.3, device="cuda").expand(total, 3)
    d_ws_contig   = torch.full((total, 3), 1.3, device="cuda")
    assert d_ws_expanded.stride() == (0, 0)
    assert d_ws_contig.is_contiguous()

    cuda_bwd = backend._REGISTRY.get("assemble_fused_bwd", {}).get("cuda")
    if cuda_bwd is None:
        pytest.skip("CUDA tier not built")

    zeros_wm = torch.zeros(total, 3, device="cuda")
    zeros_wr = torch.zeros(total, 4, device="cuda")
    zeros_wo = torch.zeros(total, 1, device="cuda")
    zeros_wf = torch.zeros(total, 3, device="cuda")

    g_exp = cuda_bwd(
        sm, sscl, sq, so, sf, ip, ir, iscl, iopa,
        cand_splat_rows, inst_offsets,
        zeros_wm, d_ws_expanded, zeros_wr, zeros_wo, zeros_wf,
        math.log(1e-4), 5.0, 20.0,
    )
    g_con = cuda_bwd(
        sm, sscl, sq, so, sf, ip, ir, iscl, iopa,
        cand_splat_rows, inst_offsets,
        zeros_wm, d_ws_contig, zeros_wr, zeros_wo, zeros_wf,
        math.log(1e-4), 5.0, 20.0,
    )
    for name, a, b in zip(
        ["sm","sscl","sq","so","sf","ip","ir","iscl","iopa"],
        g_exp, g_con,
    ):
        assert torch.allclose(a, b, atol=0.0), (
            f"non-contiguous vs contiguous mismatch for {name}: "
            f"max|Δ|={(a-b).abs().max().item():.3e}"
        )
