"""CUDA-tier backend resolution for scion.

Imports the optional scion_cuda extension (built separately via
``pip install -e scion-cuda/``, not on PyPI) and exposes its op table. When
the extension is not installed, HAS_CUDA = False and OPS is empty -- core
falls back to the pure-PyTorch reference at scion.ops._ref.
"""
from __future__ import annotations

# The kernels' C++ signatures change with this version; keep it equal to
# scion-cuda/pyproject.toml.
REQUIRED_CUDA_EXT_VERSION = "0.2.0a1"

try:
    import scion_cuda as _C
    HAS_CUDA = True
    OPS: dict = {}
    for _name in ("assemble_fused", "assemble_fused_bwd"):
        if hasattr(_C, _name):
            OPS[_name] = getattr(_C, _name)

    # A stale build would fail later with an opaque pybind TypeError; make
    # every CUDA op raise a clear error instead (CPU / python-tier use keeps working).
    from importlib.metadata import PackageNotFoundError, version as _dist_version
    try:
        _built = _dist_version("scion-cuda")
    except PackageNotFoundError:  # in-place build without dist metadata: nothing to compare
        _built = REQUIRED_CUDA_EXT_VERSION
    if _built != REQUIRED_CUDA_EXT_VERSION:
        def _stale(*_args, **_kwargs):
            raise RuntimeError(
                f"scion_cuda {_built} is a stale build (this checkout needs "
                f"{REQUIRED_CUDA_EXT_VERSION}); rebuild: pixi reinstall scion-cuda, or "
                "pip install -e scion-cuda/ --no-build-isolation"
            )
        OPS = {name: _stale for name in OPS}
except ImportError:
    HAS_CUDA = False
    OPS = {}
