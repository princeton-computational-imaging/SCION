"""Save-time compression for the SCION codebook. Nothing here runs during training.

Three levels, each a strict superset of the work of the previous one
(:mod:`scion.compression.levels`):

* ``raw`` — fp32 learned params (:func:`quantize.raw_state_dict`)
* ``quantized`` — fp16 geometry, uint8 colour/opacity (:mod:`.quantize`)
* ``codec`` (default) — quantized + OMG-style SVQ, MPEG G-PCC positions,
  range coding and LZMA (:mod:`.omg_posthoc`, :mod:`.tmc13`)

Select with Hydra ``compression=raw|quantized|codec``; see docs/compression.md.
"""

from scion.compression import tmc13  # noqa: F401
from scion.compression import omg_posthoc  # noqa: F401
