"""Shared fixtures for the CUDA kernel perf benchmarks.

Provides:
- synthetic_codebook_state: a SYNTHETIC model state (torch.randn /
  randint at small shapes). It is NOT a captured training snapshot —
  benchmarks measured on it (or on the per-file _build_inputs helpers,
  which are also synthetic) say nothing about real candidate-reuse
  distributions or memory locality, and the published speedup ratios
  must be labelled accordingly. Real-snapshot numbers come from
  scripts/bench_assemble.py (docs/cuda_architecture.md).
- benchmark_device: 'cuda' or 'cpu' depending on GPU availability AND
  the SCION_BENCH_DEVICE env var.
- gpu_free: skip marker that's True iff the running GPU has >2 GiB free.
"""
from __future__ import annotations

import os
from typing import Optional

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    # Everything under tests/bench/ is a microbenchmark (informational —
    # there is no committed baseline to gate against); mark accordingly
    # so `-m "not bench"` works even when the directory is selected
    # explicitly.
    for item in items:
        item.add_marker(pytest.mark.bench)


def _gpu_has_room(min_free_gib: float = 2.0) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:
        return False
    return free / (1024**3) >= min_free_gib


@pytest.fixture(scope="session")
def benchmark_device() -> str:
    override = os.environ.get("SCION_BENCH_DEVICE")
    if override:
        return override
    if torch.cuda.is_available() and _gpu_has_room():
        return "cuda"
    return "cpu"


@pytest.fixture(scope="session")
def gpu_free() -> bool:
    return _gpu_has_room()


def _synth_codebook_state(*, T_max: int = 16_384, K: int = 64, N: int = 2_048,
                          device: str = "cuda", seed: int = 0xC0DEB001) -> dict:
    """Synthesize a plausible codebook state for benchmarking.

    Random tensors, not a captured training snapshot. Shapes and dtypes
    match the CodebookModel parameters and buffers of the same names.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    return {
        "splat_means":        torch.randn((T_max, 3), generator=g, device=device),
        "splat_scales":       torch.randn((T_max, 3), generator=g, device=device) * 0.3,
        "splat_quats":        torch.nn.functional.normalize(
            torch.randn((T_max, 4), generator=g, device=device), dim=-1
        ),
        "splat_opacities":    torch.randn((T_max, 1), generator=g, device=device),
        "splat_features_dc":  torch.randn((T_max, 3), generator=g, device=device),
        "splat_candidate_id": torch.randint(0, K, (T_max,), generator=g, device=device,
                                            dtype=torch.int32),
        "instance_positions":   torch.randn((N, 3), generator=g, device=device),
        "instance_rotations":   torch.nn.functional.normalize(
            torch.randn((N, 4), generator=g, device=device), dim=-1
        ),
        "instance_scales":      torch.randn((N, 1), generator=g, device=device) * 0.2,
        "instance_opacities":   torch.randn((N, 1), generator=g, device=device),
        "instance_assignments": torch.randint(0, K, (N,), generator=g, device=device,
                                              dtype=torch.int32),
        "K": K, "T_max": T_max, "N": N,
    }


@pytest.fixture(scope="session")
def synthetic_codebook_state(benchmark_device):
    # Synthetic stand-in: keeps the harness importable and runnable without
    # a captured training snapshot.
    if benchmark_device == "cpu":
        return _synth_codebook_state(device="cpu")
    return _synth_codebook_state(device=benchmark_device)
