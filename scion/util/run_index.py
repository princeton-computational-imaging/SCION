"""Job keys, on-disk layout and reading of ``run.json`` v2 records.

See ``evals/RUN_SCHEMA.md``. A job is keyed by
``(variant, ablation, dataset, scene, seed, tag)`` and its record lives at::

    <runs_root>/<slot>/<dataset>/<scene>/seed<seed>/run.json

with ``<slot> = <variant>[_abl_<ablation>][_<tag>]``. ``tag`` marks
environment-only differences the Hydra config does not capture (e.g. the
pure-PyTorch backend arm of the compute benchmark).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

#: Compression levels, in evaluation order (``raw`` < ``quantized`` < ``codec``).
LEVELS = ("raw", "quantized", "codec")


@dataclass(frozen=True)
class JobKey:
    variant: str
    ablation: Optional[str]
    dataset: str
    scene: str
    seed: int
    tag: Optional[str] = None

    @property
    def slot(self) -> str:
        slot = self.variant
        if self.ablation:
            slot += f"_abl_{self.ablation}"
        if self.tag:
            slot += f"_{self.tag}"
        return slot

    def path(self, runs_root: Path | str) -> Path:
        return (Path(runs_root) / self.slot / self.dataset / self.scene
                / f"seed{self.seed}" / "run.json")


def key_of(record: Dict[str, Any]) -> JobKey:
    """The job key a record claims in its ``job`` block."""
    j = record["job"]
    return JobKey(j["variant"], j.get("ablation"), j["dataset"], j["scene"],
                  int(j["seed"]), j.get("tag"))


def flatten(record: Dict[str, Any]) -> Dict[str, float]:
    """Numeric fields as flat keys: ``<level>.<metric>`` for every evaluated
    level (e.g. ``quantized.psnr``, ``codec.size_bytes``), plus
    ``train_time_s``, ``median_step_s``, ``peak_vram_bytes``, ``n_gaussians`` and
    ``model_stats.<field>``; eval groups as ``<level>.by_subset.<group>.<metric>``."""
    out: Dict[str, float] = {}
    for level, block in (record.get("metrics") or {}).items():
        for name, value in block.items():
            if name == "by_subset":
                out.update({f"{level}.by_subset.{g}.{k}": float(v)
                            for g, m in value.items() for k, v in m.items()})
            else:
                out[f"{level}.{name}"] = float(value)
    for name in ("train_time_s", "median_step_s", "peak_vram_bytes", "n_gaussians"):
        if record.get(name) is not None:
            out[name] = float(record[name])
    for name, value in (record.get("model_stats") or {}).items():
        out[f"model_stats.{name}"] = float(value)
    return {k: v for k, v in out.items() if math.isfinite(v)}


@dataclass
class Run:
    key: JobKey
    path: Path
    record: Dict[str, Any]
    metrics: Dict[str, float]

    @property
    def status(self) -> str:
        return self.record.get("status")

    @property
    def git_sha(self) -> str:
        return self.record.get("git_sha")

    @property
    def eval_images_sha256(self) -> Optional[str]:
        return (self.record.get("split") or {}).get("eval_images_sha256")

    @property
    def data_sha256(self) -> Optional[str]:
        return (self.record.get("split") or {}).get("data_sha256")


def read(path: Path | str) -> Run:
    """Parse one v2 ``run.json``; raises ``ValueError`` naming the file for
    other versions and for malformed records."""
    path = Path(path)
    try:
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise TypeError("not a JSON object")
        if record.get("schema_version") != 2:
            raise ValueError(f"schema_version={record.get('schema_version')!r}, expected 2")
        metrics = record.get("metrics")
        if not (isinstance(record.get("job"), dict) and isinstance(metrics, dict)
                and all(isinstance(b, dict) for b in metrics.values())
                and isinstance(record.get("model_stats"), dict)):
            raise TypeError("job, metrics.<level> and model_stats must be objects")
        return Run(key_of(record), path, record, flatten(record))
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ValueError(f"{path}: {e}") from None
