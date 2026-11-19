"""CUDA kernels for scion. Built via torch.utils.cpp_extension."""
from __future__ import annotations

# Importing torch first triggers libtorch / libc10 loading, which the C
# extension depends on. Without this, `import scion_cuda` from a
# fresh interpreter fails with `ImportError: libc10.so: cannot open
# shared object file`.
import torch as _torch  # noqa: F401

try:
    from . import _C  # the compiled extension
    HAS_CUDA = True
except ImportError as e:
    HAS_CUDA = False
    raise ImportError(
        "scion_cuda._C did not import - was the C extension built? "
        "Try: pip install -e scion-cuda/"
    ) from e

# Re-export public symbols from _C so scion._cuda can resolve them
# via `import scion_cuda as _C` and call _C.<op>.
assemble_fused = _C.assemble_fused
assemble_fused_bwd = _C.assemble_fused_bwd

__all__ = [
    "HAS_CUDA",
    "assemble_fused",
    "assemble_fused_bwd",
]
