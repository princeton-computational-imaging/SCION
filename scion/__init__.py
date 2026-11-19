"""Public surface for the SCION core.

Importing the top-level package gives the primary entry points:

  * :class:`Camera` — the gsplat-friendly camera dataclass.
  * :func:`rasterize` — the only gsplat call site.
  * :class:`CodebookModel` — the flat-codebook splatting model.
  * :class:`Trainer` / :class:`TrainerConfig` — the training loop.
"""

from scion.camera import Camera
from scion.rasterizer import rasterize
from scion.models.codebook import CodebookModel
from scion.engine.trainer import Trainer, TrainerConfig

try:
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    __version__ = _dist_version("scion")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.2.0a0"

__all__ = [
    "Camera",
    "CodebookModel",
    "rasterize",
    "Trainer",
    "TrainerConfig",
    "__version__",
]

from scion._cuda import HAS_CUDA  # noqa: F401
from scion._triton import HAS_TRITON  # noqa: F401
