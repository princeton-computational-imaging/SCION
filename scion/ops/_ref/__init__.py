"""Pure-PyTorch reference implementations of the hotpath ops.

These are the fallback tier -- used when neither CUDA nor Triton backends
are available, AND as the parity gold-standard for testing the CUDA/Triton
implementations.

Each public function name is the same as the registered op name so
backend.py can populate the registry by introspection.
"""
from scion.ops._ref.assemble import assemble_fused

__all__ = [
    "assemble_fused",
]
