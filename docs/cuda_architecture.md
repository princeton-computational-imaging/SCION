# CUDA tier

`scion-cuda` is an optional extension of the `scion` core.
It ships one op, `assemble_fused`, as a CUDA forward and backward kernel and a
Triton forward variant. The op expands every instance into its primitive's
world-space Gaussians on each training step. Without the extension the core
runs the same op as standard PyTorch operations, still on the GPU, with the
same numerical semantics. "PyTorch" in this document means that path, not a
CPU path.

## Install

`pixi install` builds and installs it with the rest of the environment. In an
environment of your own (the extension is not on PyPI), install it after the
core:

```bash
pip install -e scion-cuda/ --no-build-isolation
```

To rebuild it, run `pixi reinstall scion-cuda` in the pixi environment, or the
`pip` command above in your own.

You need an `nvcc` that matches your PyTorch build. By default the build
targets Turing through Hopper, adds Blackwell on CUDA ≥ 12.8, and embeds PTX
for newer GPUs. To override, set `TORCH_CUDA_ARCH_LIST`; the matrix is in
[scion-cuda/README.md](../scion-cuda/README.md).

Check what loaded:

```bash
python -c "import scion; print(scion.HAS_CUDA, scion.HAS_TRITON)"
```

- **`HAS_CUDA`** is `False` when the extension is not installed or failed to
  build. Training then uses the PyTorch path.
- **`HAS_TRITON`** also needs `triton` to be importable.
- **Stale builds.** A build whose version doesn't match the checkout's
  `REQUIRED_CUDA_EXT_VERSION` raises an error on first use. Rebuild it as above.

## Dispatch

[`scion.ops.backend`](../scion/ops/backend.py) resolves each op
once, in the order `cuda > triton > python`, and caches the result.

| op | CUDA | Triton | PyTorch |
|---|---|---|---|
| `assemble_fused` (forward) | `scion_cuda._C` | `scion_cuda.triton.assemble_fused` | `scion.ops._ref.assemble` |
| `assemble_fused_bwd` (backward) | `scion_cuda._C` | — | autograd through the PyTorch forward |

The autograd wrapper is
[`scion.ops.assemble`](../scion/ops/assemble.py), called from
`CodebookMixin.assemble_gaussians`. It runs the backward kernel whenever one is
available, even when the forward is forced to another tier. To force a tier
for one op, set an environment variable:

```bash
# pure PyTorch end to end: forward AND backward
SCION_BACKEND_assemble_fused=python \
SCION_BACKEND_assemble_fused_bwd=python \
python -m scion.train dataset=mipnerf360 variant=ours_small scene=garden
```

The values are `cuda`, `triton` and `python`. Two cases need care:

- **Only the forward forced.** Setting `SCION_BACKEND_assemble_fused`
  alone still uses the CUDA backward.
- **Unregistered tiers.** An override that names a tier with no registered op
  raises on first dispatch. The one exception is `assemble_fused_bwd=python`,
  which selects the autograd fallback.

`run.json` records the tier each op resolved to (`backend_tiers`).

## Measured

**Microbenchmark.** `scripts/bench_assemble.py --ckpt <run>` times the op on a
trained model's real state, per tier, with CUDA events. On garden
`ours_small` after 30k steps (1,735,393 Gaussians from 54,814 instances) on an
RTX 5090:

| `assemble_fused` | PyTorch | CUDA | speedup |
|---|---:|---:|---:|
| forward | 2.42 ms | 0.105 ms | 23× |
| forward + backward | 10.86 ms | 0.79 ms | 13.7× |

`tests/bench/test_assemble_perf.py` is a synthetic-input microbenchmark. It is
informational only and excluded from the default test run.

**End to end.** `make reproduce TARGETS=compute` measures whole-run
training per backend. It trains garden `ours_small` three times (seeds
0, 1, 2) on each of two arms:

- `cuda`, the default dispatch;
- `python`, with both overrides above.

The renderer refuses arms whose records report a different tier or mix GPU
models. It writes `tables/table_compute.tex` with wall-clock, median step time,
peak VRAM and PSNR per backend. Run the target alone on the machine so nothing
else shares the GPU.

Measured (`make reproduce TARGETS=compute`):

| garden `ours_small`, RTX 5090, 3 seeds | PyTorch | CUDA | speedup |
|---|---:|---:|---:|
| training time (30k steps) | 18.4 ± 0.2 min | 14.4 ± 0.1 min | 1.28× |
| median step | 28.8 ± 0.8 ms | 19.5 ± 0.2 ms | 1.48× |
| peak VRAM | 6.0 GiB | 6.0 GiB | — |
| PSNR (quantized) | 21.98 ± 0.13 | 21.90 ± 0.08 | parity |

## Diagnosing

1. `HAS_CUDA` is `False`: rebuild as above (add `-v` to the `pip` command)
   and look for nvcc or `cpp_extension` errors in the log.
2. To check which tier ran, look at `backend_tiers` in a run's `run.json`, or
   call `scion.ops.backend.resolved_tiers()` after a step.
3. To compare tiers on your own model, run
   `python scripts/bench_assemble.py --ckpt <run dir> --out bench.json`.

## Adding an op

1. **PyTorch reference.** Write it in `scion/ops/_ref/<op>.py` and list
   it in `scion.ops._ref.__all__`. The registry discovers it from
   there, and it is the reference the other tiers are tested against.
2. **CUDA kernel.** Put it in `scion-cuda/src/csrc/`, bind it in
   `src/ext.cpp`, add it to `setup.py`'s `sources`, and add its name to the op
   tuple in `scion/_cuda.py`. If the C++ signature changes, bump the
   extension version in `scion-cuda/pyproject.toml` and
   `REQUIRED_CUDA_EXT_VERSION` together.
3. **Triton variant (optional).** Put it under
   `scion-cuda/python/scion_cuda/triton/` and register it in
   `scion/_triton.py`'s `OPS`.
4. **Parity test.** Add one in the style of
   `tests/test_assemble_fused_parity.py`: run every tier through
   `SCION_BACKEND_<op>` and compare against the reference.
5. **Call site.** Call it via `backend.dispatch("<op>", ...)` and measure its
   share of a real run before relying on it.
