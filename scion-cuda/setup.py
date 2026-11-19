"""Build script for scion-cuda.

Compiles the CUDA kernels for the codebook hotpath (``assemble_fused``
forward + backward) into ``scion_cuda._C``.

Architecture selection:
- If ``TORCH_CUDA_ARCH_LIST`` is set, PyTorch's build machinery honours
  it and no explicit ``-gencode`` flags are added here.
- Otherwise a multi-arch default is used: Turing/Ampere/Ada/Hopper
  cubins, plus Blackwell (sm_120) when the CUDA toolkit is >= 12.8, and
  PTX for the newest arch so future GPUs can JIT.

Env knobs: ``NO_FAST_MATH=1`` disables ``--use_fast_math`` (the parity
tests are the reference for the numerical impact); ``TORCH_CUDA_ARCH_LIST``
overrides the arch list (e.g. ``TORCH_CUDA_ARCH_LIST="8.6"`` for a fast
single-arch dev build).
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src_dir = Path(__file__).parent / "src"
csrc_dir = src_dir / "csrc"
include_dirs = [str(src_dir / "include")]

sources = [
    "src/ext.cpp",
    "src/csrc/AssembleFused.cu",
    "src/csrc/AssembleFusedBwd.cu",
]


def _cuda_version() -> tuple:
    """(major, minor) of the CUDA toolkit torch was built against."""
    try:
        import torch

        if torch.version.cuda:
            major, minor = torch.version.cuda.split(".")[:2]
            return int(major), int(minor)
    except Exception:
        pass
    return (0, 0)


def _default_gencode_flags() -> list:
    """Multi-arch default when TORCH_CUDA_ARCH_LIST is unset.

    sm_75 (Turing) through sm_90 (Hopper) always; sm_120 (Blackwell)
    only when the toolkit understands compute_120 (CUDA >= 12.8) —
    older toolkits fail outright on unknown architectures. PTX is
    embedded for the newest compiled arch so not-yet-released GPUs can
    JIT instead of failing to load.
    """
    archs = ["75", "80", "86", "89", "90"]
    if _cuda_version() >= (12, 8):
        archs.append("120")
    flags = []
    for a in archs:
        flags += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    flags += ["-gencode", f"arch=compute_{archs[-1]},code=compute_{archs[-1]}"]
    return flags


nvcc_flags = [
    "-O3",
    "--use_fast_math" if not os.environ.get("NO_FAST_MATH") else "",
    "-std=c++17",
]
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    nvcc_flags += _default_gencode_flags()

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    # Strip empty strings (the conditional fast-math flag).
    "nvcc": [s for s in nvcc_flags if s],
}

ext_modules = [
    CUDAExtension(
        name="scion_cuda._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="scion-cuda",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=["scion_cuda"],
    package_dir={"scion_cuda": "python/scion_cuda"},
    zip_safe=False,
)
