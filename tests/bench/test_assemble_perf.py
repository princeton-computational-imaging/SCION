"""Benchmark for the ``assemble_fused`` kernel.

Compares the dispatched (CUDA) ``assemble_fused`` against the pure-PyTorch
reference at garden scale and asserts ≥ 4× speedup.
"""
from __future__ import annotations

import math

import pytest
import torch


def _build_inputs(
    T_max: int = 128_000,
    N: int = 4_000,
    rows_per_inst: int = 50,
    device: str = "cuda",
    seed: int = 0,
):
    """Garden-scale plausible inputs. Total world Gaussians = N *
    rows_per_inst ≈ 200k, matching garden mid-training."""
    g = torch.Generator(device=device).manual_seed(seed)
    splat_means = torch.randn(T_max, 3, device=device, generator=g)
    splat_scales = torch.randn(T_max, 3, device=device, generator=g) * 0.1
    splat_quats = torch.nn.functional.normalize(
        torch.randn(T_max, 4, device=device, generator=g), dim=-1,
    )
    splat_opa = torch.randn(T_max, 1, device=device, generator=g)
    splat_fd = torch.randn(T_max, 3, device=device, generator=g)
    inst_pos = torch.randn(N, 3, device=device, generator=g)
    inst_rot = torch.nn.functional.normalize(
        torch.randn(N, 4, device=device, generator=g), dim=-1,
    )
    inst_scl = torch.randn(N, 1, device=device, generator=g) * 0.1
    inst_opa = torch.randn(N, 1, device=device, generator=g)
    cand_splat_rows = torch.randint(
        0, T_max, (N * rows_per_inst,), device=device, generator=g, dtype=torch.int32,
    )
    inst_offsets = torch.arange(
        0, (N + 1) * rows_per_inst, rows_per_inst, device=device, dtype=torch.int32,
    )
    return (
        splat_means, splat_scales, splat_quats, splat_opa, splat_fd,
        inst_pos, inst_rot, inst_scl, inst_opa,
        cand_splat_rows, inst_offsets,
    )


def _time_op(op, inputs, warmup: int = 3, iters: int = 20) -> float:
    """Returns mean ms/iter for ``op(*inputs, *bounds)``."""
    extra = (math.log(1e-4), 5.0, 20.0)
    for _ in range(warmup):
        _ = op(*inputs, *extra)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = op(*inputs, *extra)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@pytest.mark.benchmark(group="assemble")
def test_assemble_python_reference(benchmark, gpu_free):
    """Baseline timing for the pure-PyTorch reference."""
    if not gpu_free:
        pytest.skip("GPU busy")
    from scion.ops._ref.assemble import assemble_fused as ref_fn
    inputs = _build_inputs()
    ms = _time_op(ref_fn, inputs)
    benchmark.extra_info["ms_per_iter"] = ms
    print(f"\n[assemble_fused/ref] {ms:.3f} ms/iter")


@pytest.mark.benchmark(group="assemble")
def test_assemble_speedup_gate(benchmark, gpu_free):
    """CUDA tier must be ≥ 4× faster than the python reference."""
    if not gpu_free:
        pytest.skip("GPU busy")
    import scion
    if not scion.HAS_CUDA:
        pytest.skip("scion_cuda not installed")

    from scion.ops import backend
    if backend._REGISTRY.get("assemble_fused", {}).get("cuda") is None:
        pytest.skip("CUDA tier for assemble_fused not registered")

    from scion.ops._ref.assemble import assemble_fused as ref_fn
    from scion.ops.assemble import assemble_fused as disp_fn

    inputs = _build_inputs()
    ms_ref = _time_op(ref_fn, inputs)
    ms_cuda = _time_op(disp_fn, inputs)
    speedup = ms_ref / ms_cuda

    benchmark.extra_info["ms_ref"] = ms_ref
    benchmark.extra_info["ms_cuda"] = ms_cuda
    benchmark.extra_info["speedup"] = speedup

    print(
        f"\n[assemble_fused] cuda {ms_cuda:.3f}ms vs ref {ms_ref:.3f}ms "
        f"= {speedup:.2f}x speedup"
    )
    assert speedup >= 4.0, f"speedup {speedup:.2f}x < target 4x"
