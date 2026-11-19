"""Autograd-aware wrapper for assemble_fused.

The forward dispatches through ``scion.ops.backend`` → CUDA / Triton
/ pure-PyTorch tier. The backward dispatches through the CUDA tier's
explicit gradient kernel (``assemble_fused_bwd``), and falls back to
PyTorch's autograd if only the python tier is available (the pure-PyTorch
reference at ``_ref/assemble.py`` is differentiable by construction).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from scion.ops import backend


# Number of leading inputs to `AssembleFusedFunction.forward` that are
# tensors with potential gradients. Backward must return that many tensor
# gradients followed by ``None`` for the non-tensor scalar / index inputs.
_NUM_TENSOR_INPUTS = 11   # 9 float tensors + 2 int layout buffers
_NUM_SCALAR_INPUTS = 3    # min_world_scale_log, max_world_scale_log, opac_clamp


def _backward_backend_available() -> bool:
    """True if any tier (respecting env overrides) can serve
    ``assemble_fused_bwd``."""
    try:
        backend._resolve("assemble_fused_bwd")
        return True
    except (KeyError, RuntimeError):
        return False


class AssembleFusedFunction(torch.autograd.Function):
    """Differentiable wrapper around the ``assemble_fused`` op."""

    @staticmethod
    def forward(
        ctx,
        splat_means: Tensor,
        splat_scales: Tensor,
        splat_quats: Tensor,
        splat_opa: Tensor,
        splat_feat_dc: Tensor,
        inst_pos: Tensor,
        inst_rot: Tensor,
        inst_scl: Tensor,
        inst_opa: Tensor,
        cand_splat_rows: Tensor,
        inst_offsets: Tensor,
        min_world_scale_log: float,
        max_world_scale_log: float,
        opac_clamp: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        ctx.min_world_scale_log = min_world_scale_log
        ctx.max_world_scale_log = max_world_scale_log
        ctx.opac_clamp = opac_clamp
        ctx.save_for_backward(
            splat_means, splat_scales, splat_quats, splat_opa, splat_feat_dc,
            inst_pos, inst_rot, inst_scl, inst_opa,
            cand_splat_rows, inst_offsets,
        )

        out = backend.dispatch(
            "assemble_fused",
            splat_means, splat_scales, splat_quats, splat_opa, splat_feat_dc,
            inst_pos, inst_rot, inst_scl, inst_opa,
            cand_splat_rows, inst_offsets,
            min_world_scale_log, max_world_scale_log, opac_clamp,
        )
        # CUDA tier returns a std::vector → tuple; reference returns a
        # Python tuple. Normalise.
        return tuple(out)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            splat_means, splat_scales, splat_quats, splat_opa, splat_feat_dc,
            inst_pos, inst_rot, inst_scl, inst_opa,
            cand_splat_rows, inst_offsets,
        ) = ctx.saved_tensors
        d_world_means, d_world_scales, d_world_rot, d_world_opa, d_world_feat = grad_outputs

        # Prefer the CUDA backward when registered; otherwise rebuild the
        # gradient via autograd through the pure-PyTorch reference.
        if _backward_backend_available():
            grads = backend.dispatch(
                "assemble_fused_bwd",
                splat_means, splat_scales, splat_quats, splat_opa, splat_feat_dc,
                inst_pos, inst_rot, inst_scl, inst_opa,
                cand_splat_rows, inst_offsets,
                d_world_means, d_world_scales, d_world_rot, d_world_opa, d_world_feat,
                ctx.min_world_scale_log,
                ctx.max_world_scale_log,
                ctx.opac_clamp,
            )
            return _grads_to_function_outputs(grads)
        return _autograd_fallback_backward(
            splat_means, splat_scales, splat_quats, splat_opa, splat_feat_dc,
            inst_pos, inst_rot, inst_scl, inst_opa,
            cand_splat_rows, inst_offsets,
            d_world_means, d_world_scales, d_world_rot, d_world_opa, d_world_feat,
            ctx.min_world_scale_log,
            ctx.max_world_scale_log,
            ctx.opac_clamp,
        )


def _grads_to_function_outputs(grads):
    """Map a backward kernel's tuple of input-gradients to the autograd
    Function's expected output signature.

    The Function has 14 inputs (11 tensors + 3 scalars). The backward
    must return 14 entries, with ``None`` for non-differentiable inputs.
    """
    (
        g_splat_means, g_splat_scales, g_splat_quats, g_splat_opa, g_splat_feat_dc,
        g_inst_pos, g_inst_rot, g_inst_scl, g_inst_opa,
    ) = grads

    return (
        g_splat_means, g_splat_scales, g_splat_quats, g_splat_opa, g_splat_feat_dc,
        g_inst_pos, g_inst_rot, g_inst_scl, g_inst_opa,
        None,  # cand_splat_rows (int32, no grad)
        None,  # inst_offsets    (int32, no grad)
        None,  # min_world_scale_log (float)
        None,  # max_world_scale_log (float)
        None,  # opac_clamp (float)
    )


def _autograd_fallback_backward(
    splat_means, splat_scales, splat_quats, splat_opa, splat_feat_dc,
    inst_pos, inst_rot, inst_scl, inst_opa,
    cand_splat_rows, inst_offsets,
    d_world_means, d_world_scales, d_world_rot, d_world_opa, d_world_feat,
    min_world_scale_log, max_world_scale_log, opac_clamp,
):
    """Re-run the pure-PyTorch reference under autograd to derive the
    Function's gradients when no CUDA backward kernel is registered.

    Detaches + clones every differentiable input, sets
    ``requires_grad=True``, runs the reference forward, then back-props
    the supplied upstream gradients.
    """
    # Inside Function.backward, autograd is disabled by default. We need
    # to re-enable it so the reference's PyTorch ops record a graph that
    # backward() can traverse.
    with torch.enable_grad():
        sm = splat_means.detach().clone().requires_grad_(True)
        sscl = splat_scales.detach().clone().requires_grad_(True)
        sq = splat_quats.detach().clone().requires_grad_(True)
        so = splat_opa.detach().clone().requires_grad_(True)
        sf = splat_feat_dc.detach().clone().requires_grad_(True)
        ip = inst_pos.detach().clone().requires_grad_(True)
        ir = inst_rot.detach().clone().requires_grad_(True)
        iscl = inst_scl.detach().clone().requires_grad_(True)
        iopa = inst_opa.detach().clone().requires_grad_(True)

        from scion.ops._ref.assemble import assemble_fused as _ref
        out = _ref(
            sm, sscl, sq, so, sf, ip, ir, iscl, iopa,
            cand_splat_rows, inst_offsets,
            min_world_scale_log, max_world_scale_log, opac_clamp,
        )

        # Pair each output with its upstream grad. Skip outputs whose
        # upstream grad is None (gradcheck flows one output at a time) —
        # passing None to torch.autograd.backward errors on outputs that
        # don't require grad in the current sub-graph.
        pairs = [
            (o, g) for o, g in zip(
                out, (d_world_means, d_world_scales, d_world_rot,
                      d_world_opa, d_world_feat),
            )
            if g is not None and o.requires_grad
        ]
        if pairs:
            outs, grads_in = zip(*pairs)
            torch.autograd.backward(list(outs), grad_tensors=list(grads_in))

    return (
        sm.grad, sscl.grad, sq.grad, so.grad, sf.grad,
        ip.grad, ir.grad, iscl.grad, iopa.grad,
        None, None,        # cand_splat_rows / inst_offsets
        None, None, None,  # scalars
    )


def assemble_fused(
    splat_means: Tensor,
    splat_scales: Tensor,
    splat_quats: Tensor,
    splat_opa: Tensor,
    splat_feat_dc: Tensor,
    inst_pos: Tensor,
    inst_rot: Tensor,
    inst_scl: Tensor,
    inst_opa: Tensor,
    cand_splat_rows: Tensor,
    inst_offsets: Tensor,
    min_world_scale_log: float = -9.21034,
    max_world_scale_log: float = 5.0,
    opac_clamp: float = 20.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Differentiable ``assemble_fused``. Dispatches through the backend
    registry (CUDA > Triton > Python).

    Returns ``(world_means, world_scales, world_rot, world_opa,
    world_feat)`` -- five contiguous fp32 tensors of length ``total =
    inst_offsets[-1]``. See ``scion.ops._ref.assemble.assemble_fused``
    for full documentation of the math and shapes.
    """
    return AssembleFusedFunction.apply(
        splat_means, splat_scales, splat_quats, splat_opa, splat_feat_dc,
        inst_pos, inst_rot, inst_scl, inst_opa,
        cand_splat_rows, inst_offsets,
        min_world_scale_log, max_world_scale_log, opac_clamp,
    )
