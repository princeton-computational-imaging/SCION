"""Triton-tier backend resolution for scion.

Triton kernels live under ``scion-cuda/python/scion_cuda/triton/``
but are imported as ``scion_cuda.triton.*`` once the wheel is
installed. When Triton itself is missing OR the wheel isn't built,
``HAS_TRITON = False`` and the dispatch registry falls through to the
python reference tier.
"""
from __future__ import annotations

try:
    import triton  # noqa: F401  -- presence check
    from scion_cuda.triton import assemble_fused as _assemble_fused_triton
    HAS_TRITON = True
    OPS: dict = {
        "assemble_fused": _assemble_fused_triton,
    }
except ImportError:
    HAS_TRITON = False
    OPS = {}
