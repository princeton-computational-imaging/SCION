# Contributing

Thanks for your interest! This is a research codebase; contributions
that fix bugs, improve documentation, or harden the tooling are very
welcome. For new features or algorithmic changes, please open an issue
first so we can discuss scope.

## Development setup

```bash
git clone https://github.com/princeton-computational-imaging/SCION
cd SCION
pip install -e '.[dev]'    # Synthetic-NeRF scenes and your own data also need `colmap` on PATH
# optional CUDA/Triton kernels (needs nvcc matching your torch build):
pip install -e scion-cuda/
```

The tested stack is the one `pixi install` pins in `pixi.lock`: Python
3.10, torch 2.8.0+cu128, gsplat 1.5.3, COLMAP 3.13. See the README's
Installation and Data sections for datasets and external binaries (`tmc3`).

## Running the tests

```bash
pytest              # CPU-safe subset runs everywhere; GPU tests skip without CUDA
pytest tests/bench  # microbenchmarks (informational, not run by default)
```

A bare `pytest` must stay green on CPU-only machines, so GPU-dependent
tests must skip cleanly: guard them with a `skipif` on
`torch.cuda.is_available()` whose reason mentions CUDA (like `needs_cuda` in
`tests/test_assemble_fused_parity.py`). `tests/conftest.py` tags such tests
with the `gpu` marker, so `pytest -m "not gpu"` deselects them.

`tests/test_assemble_fused_parity.py` is the contract for any change
touching the assembly path: the pure-PyTorch tier is the reference, and the
CUDA/Triton tiers must match it within the test's tolerance. The paper-table
renderers have golden tests (`tests/test_paper_render.py`) against
`tests/fixtures/paper/expected/`.

## Code style

Match the surrounding code. There is no enforced formatter; keep diffs
minimal and focused. Comments should state constraints the code
can't express, not narrate the code.

## Adding a CUDA/Triton op

Follow the recipe in
[docs/cuda_architecture.md](docs/cuda_architecture.md#adding-an-op):
pure-PyTorch reference first (the parity reference), then the CUDA kernel,
then a parity test; measure the op's share of a real run before wiring it
in.

## Pull requests

- One concern per PR.
- Include the test that would have caught the bug you're fixing.
- CI must pass (clean install + import + CPU test suite).
