"""Loader for ``targets.yaml``: paper items -> training jobs -> renderers.

    reg = load_registry()
    reg.targets["main_results"].jobs     # [Job(key, overrides, env), ...]
    reg.jobs(["main_results", "capacity"])  # deduplicated union
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from scion.util.run_index import JobKey

REGISTRY = Path(__file__).with_name("targets.yaml")
_ENTRY_KEYS = {"variants", "ablations", "datasets", "scenes", "seeds", "tag", "env"}


@dataclass(frozen=True)
class Job:
    key: JobKey
    overrides: Tuple[str, ...]          # Hydra overrides, in order
    env: Tuple[Tuple[str, str], ...] = ()


@dataclass
class Target:
    name: str
    spec: dict
    jobs: List[Job]

    renderer = property(lambda self: self.spec.get("renderer"))
    command = property(lambda self: self.spec.get("command"))
    manual = property(lambda self: self.spec.get("manual"))

    def unique(self, field: str):
        """The single value of a job-key field across this target's jobs."""
        values = {getattr(j.key, field) for j in self.jobs}
        if len(values) != 1:
            raise ValueError(f"target {self.name}: expected one {field}, got {sorted(map(str, values))}")
        return values.pop()

    def present(self, field: str, order: Iterable) -> list:
        """Values of a job-key field used by this target, in ``order``."""
        used = {getattr(j.key, field) for j in self.jobs}
        return [v for v in order if v in used]


@dataclass
class Registry:
    seeds: int
    variants: Dict[str, str]            # name -> LaTeX label
    datasets: Dict[str, dict]
    targets: Dict[str, Target]

    def jobs(self, names: Optional[Iterable[str]] = None) -> List[Job]:
        names = list(self.targets) if names is None else list(names)
        return _dedup(j for n in names for j in self.targets[n].jobs)


def load_registry(path: Path = REGISTRY) -> Registry:
    raw = yaml.safe_load(Path(path).read_text())
    reg = Registry(int(raw["seeds"]), dict(raw["variants"]), dict(raw["datasets"]), {})
    specs = raw["targets"]
    resolved: Dict[str, List[Job]] = {}

    def expand(name: str, stack: Tuple[str, ...] = ()) -> List[Job]:
        if name in stack:
            raise ValueError(f"targets.yaml: job reuse cycle {stack + (name,)}")
        if name not in resolved:
            jobs: List[Job] = []
            for entry in specs[name].get("jobs") or []:
                if "target" in entry:
                    if set(entry) != {"target"} or entry["target"] not in specs:
                        raise ValueError(f"targets.yaml: {name}: reuse entry {entry} must be "
                                         "exactly {target: <existing target>}")
                    jobs += expand(entry["target"], stack + (name,))
                else:
                    jobs += _expand_entry(reg, name, entry)
            resolved[name] = _dedup(jobs)
        return resolved[name]

    for name, spec in specs.items():
        reg.targets[name] = Target(name, spec, expand(name))

    # One key per run path (_dedup already enforces one definition per key).
    by_path: Dict[Path, JobKey] = {}
    for job in reg.jobs():
        other = by_path.setdefault(job.key.path(""), job.key)
        if other != job.key:
            raise ValueError(f"targets.yaml: {job.key} and {other} share run path {job.key.path('')}")
    return reg


def _expand_entry(reg: Registry, target: str, entry: dict) -> List[Job]:
    unknown = set(entry) - _ENTRY_KEYS
    if unknown:
        raise ValueError(f"targets.yaml: {target}: unknown job fields {sorted(unknown)}")
    datasets = entry["datasets"]
    if "scenes" in entry and len(datasets) != 1:
        raise ValueError(f"targets.yaml: {target}: `scenes` needs exactly one dataset")
    seeds = list(entry.get("seeds", range(reg.seeds)))
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"targets.yaml: {target}: duplicate seeds {seeds}")
    env = tuple(sorted((entry.get("env") or {}).items()))
    jobs = []
    for v in entry.get("variants", list(reg.variants)):
        if v not in reg.variants:
            raise ValueError(f"targets.yaml: {target}: unknown variant {v!r}")
        for a in entry.get("ablations", [None]):
            for d in datasets:
                ds = reg.datasets[d]
                for s in entry.get("scenes", ds["scenes"]):
                    if s not in ds["scenes"]:
                        raise ValueError(f"targets.yaml: {target}: {s!r} is not a {d} scene")
                    for seed in seeds:
                        overrides = ["method=codebook", f"variant={v}", f"dataset={d}",
                                     f"scene={s}", f"machine.seed={seed}"]
                        if a:
                            overrides.append(f"ablation={a}")
                        overrides += (ds.get("overrides") or {}).get(v, [])
                        jobs.append(Job(JobKey(v, a, d, s, int(seed), entry.get("tag")),
                                        tuple(overrides), env))
    return jobs


def _dedup(jobs: Iterable[Job]) -> List[Job]:
    seen: Dict[JobKey, Job] = {}
    for j in jobs:
        if seen.setdefault(j.key, j) != j:
            raise ValueError(f"targets.yaml: {j.key} defined with different overrides/env")
    return list(seen.values())
