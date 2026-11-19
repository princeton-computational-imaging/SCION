# scion-cuda

CUDA kernels for [SCION](https://github.com/princeton-computational-imaging/SCION)
(the `scion` package). This is a **separate distribution** from the
core: the core stays pip-installable on CPU-only and non-CUDA machines, and
this package ships the compiled C++/CUDA extension for the codebook assembly
hot path, `assemble_fused` (forward and backward), plus a Triton forward
variant.

## Installation

This distribution is not published on PyPI; install it from a checkout of
the main repo (after installing the core):

```bash
pip install -e scion-cuda/ --no-build-isolation
```

The build uses `torch.utils.cpp_extension.BuildExtension` (JIT/AOT). You
need a CUDA toolkit matching your PyTorch build (`nvcc` on `PATH`,
`CUDA_HOME` set if non-standard).

To disable `--use_fast_math` for reproducibility runs:

```bash
NO_FAST_MATH=1 pip install -e scion-cuda/ --no-build-isolation
```

### Supported GPU architectures

Without `TORCH_CUDA_ARCH_LIST`, the build emits cubins for a multi-arch
default plus PTX for forward compatibility:

| Architecture | sm | Compiled by default |
|---|---|---|
| Turing (RTX 20xx / T4) | 75 | yes |
| Ampere (A100 / RTX 30xx) | 80, 86 | yes |
| Ada (RTX 40xx / L4) | 89 | yes |
| Hopper (H100) | 90 | yes |
| Blackwell (RTX 50xx / B200) | 120 | only with CUDA ≥ 12.8 |
| newer | — | via embedded PTX (JIT) |

Set `TORCH_CUDA_ARCH_LIST` to override (a single arch builds much
faster for development, e.g. `TORCH_CUDA_ARCH_LIST="8.6"`). The CUDA
toolkit must match your PyTorch build; toolkits older than 12.8 simply
skip the Blackwell cubin rather than failing.

## How it relates to the core

`scion._cuda` tries `import scion_cuda as _C` when
`scion` is imported. On success it sets `HAS_CUDA = True` and
registers `assemble_fused` / `assemble_fused_bwd` with
`scion.ops.backend`; a build whose version differs from the
checkout's `REQUIRED_CUDA_EXT_VERSION` raises a rebuild hint on first use.
On import failure the core falls back to its pure-PyTorch reference path
(standard PyTorch ops, still on the GPU; same numerical semantics).

## Backend selection

Per-op overrides via env vars:

```bash
SCION_BACKEND_assemble_fused=triton  # Triton forward (CUDA backward)
SCION_BACKEND_assemble_fused=cuda    # CUDA forward
# pure PyTorch: override the backward too, or the CUDA backward still runs
SCION_BACKEND_assemble_fused=python SCION_BACKEND_assemble_fused_bwd=python
```

Default resolution order: `cuda -> triton -> python`, skipping unavailable
tiers.

## Smoke test

```python
import scion
print(scion.HAS_CUDA)   # True if the extension built + imported
```

The kernels are parity-tested against the pure-PyTorch reference
(`pytest tests/test_assemble_fused_parity.py` on a GPU). Measured speedups
and the end-to-end benchmark: `docs/cuda_architecture.md` in the main repo.
