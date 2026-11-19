"""Flat-buffer codebook operations — split into sub-modules.

Re-exports every public symbol so that imports of the form
``from scion.ops import X`` work without reaching into sub-modules.
"""

from __future__ import annotations

from .common import (
    SPLAT_FREE,
    rows_for_candidate,
    build_binoms,
    compute_relocation,
    compute_instance_urgency,
    compute_gradient_disagreement,
    codebook_description_length,
    codebook_compression_ratio,
)

from .mh import (
    mh_birth_splats,
)

from .default import (
    accumulate_splat_grad2d,
    accumulate_instance_grad2d,
    clone_instances_by_mask,
    cull_instances_by_mask,
    relocate_dead_splats,
    subdivide_instance,
)

__all__ = [
    # common
    "rows_for_candidate",
    "build_binoms",
    "compute_relocation",
    "compute_instance_urgency",
    "compute_gradient_disagreement",
    "codebook_description_length",
    "codebook_compression_ratio",
    # mh
    "mh_birth_splats",
    # default
    "accumulate_splat_grad2d",
    "accumulate_instance_grad2d",
    "clone_instances_by_mask",
    "cull_instances_by_mask",
    "relocate_dead_splats",
    "subdivide_instance",
]
