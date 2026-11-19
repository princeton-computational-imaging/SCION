"""Triton-tier kernels for scion.

Tier strategy: forward-only, no atomic reductions. The backward pass
delegates to the CUDA tier (or python fallback) -- see
``scion.ops.assemble``.
"""
from __future__ import annotations

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

if _HAS_TRITON:
    from .assemble_fused import assemble_fused
    __all__ = ["assemble_fused"]
else:
    __all__ = []
