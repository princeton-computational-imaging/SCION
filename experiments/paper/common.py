"""Shared pieces of the paper renderers: fail-closed record loading, seed
aggregation, cell formatting, best/second highlighting, baselines.

Aggregation rule (every table): a per-scene cell is mean +/- sample std
(ddof=1) over the seeds. A dataset-average cell first averages the scenes
within each seed, then takes mean +/- std over the per-seed dataset means.
Std is never computed over pooled scene x seed values.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import yaml

from scion.util.provenance import config_sha256
from scion.util.run_index import JobKey, Run, read
from scion.util.stats import mean_std
from scion.util.units import MiB

REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO / "evals" / "baselines" / "baselines.yaml"

GiB = 2 ** 30

MISSING = "---"
MB_HDR = r"MB$\downarrow$"  # MB = 2^20 bytes (stated in the paper text)
QUALITY = ("psnr", "ssim", "lpips", "mb")

#: metric -> (column header, decimals, direction). decimals None = integer
#: with LaTeX thousands separators. direction +1 higher is better, -1 lower.
METRICS = {
    "psnr": (r"PSNR$\uparrow$", 2, 1),
    "ssim": (r"SSIM$\uparrow$", 3, 1),
    "lpips": (r"LPIPS$\downarrow$", 3, -1),
    "mb": (MB_HDR, 2, -1),
    "bpg": (r"\bpg$\downarrow$", 1, -1),
    "n_gaussians": (r"\# Total Splats", 2, 0),          # shown in millions
    "num_instances": (r"\# Instances", None, 0),
    "num_splats": (r"\# Local splats", None, 0),
    "active_candidates": (r"Active \cands{}", None, 0),
    "effective_candidates": (r"Effective \cands{}", 1, 0),
    "instances_per_candidate": (r"\insts{}/\cand{}", 1, 0),
    "train_min": (r"Train (min)$\downarrow$", 1, -1),
    "step_ms": (r"Median step (ms)$\downarrow$", 1, -1),
    "vram_gib": (r"Peak VRAM (GiB)$\downarrow$", 1, -1),
}
#: How the paper names a compression level (default: the level itself). codec is
#: "w/ param." (parameter compression), the paper macro ``\wparam``.
LEVEL_TEX = {"codec": r"\wparam{}"}


def level_tex(level: str) -> str:
    """A level as a column-group header."""
    return LEVEL_TEX.get(level, level)


def level_suffix(level: str) -> str:
    r"""A level as a row-label suffix after a variant macro: ``{} (raw)``, ``{} \wparam{}``."""
    return "{} " + LEVEL_TEX.get(level, f"({level})")


_PER_LEVEL = {"psnr", "ssim", "lpips", "n_gaussians", "size_bytes"}
_MODEL_STATS = {"num_instances", "num_splats", "active_candidates", "effective_candidates",
                "instances_per_candidate"}


class RenderError(Exception):
    """A table cannot be rendered faithfully; the message names the records."""


# ---------------------------------------------------------------- records

def value(run: Run, metric: str, level: str, subset: Optional[str] = None) -> float:
    """One number from one record (sizes in MiB, times in minutes/hours);
    ``subset``: the quality of one eval group (``by_subset``)."""
    m = run.metrics
    try:
        if subset is not None:
            return m[f"{level}.by_subset.{subset}.{metric}"]
        if metric == "mb":
            return m[f"{level}.size_bytes"] / MiB
        if metric == "bpg":  # recomputed, not the record's own field
            return m[f"{level}.size_bytes"] * 8 / m[f"{level}.n_gaussians"]
        if metric in _PER_LEVEL:
            return m[f"{level}.{metric}"]
        if metric in _MODEL_STATS:
            return m[f"model_stats.{metric}"]
        if metric == "step_ms":
            return m["median_step_s"] * 1000
        if metric == "train_min":
            return m["train_time_s"] / 60
        if metric == "train_hours":
            return m["train_time_s"] / 3600
        if metric == "vram_gib":
            return m["peak_vram_bytes"] / GiB
    except KeyError as e:
        raise RenderError(f"  - {run.path}: no {e.args[0]}") from None
    except ZeroDivisionError:
        raise RenderError(f"  - {run.path}: {level}.n_gaussians is 0") from None
    raise ValueError(f"unknown metric {metric!r}")


class Pool:
    """The DONE records of a target's jobs, validated before any cell is
    computed: exactly the registry seeds per job group, schema v2, DONE, the
    requested levels present, one git_sha, and one eval split and one
    data_sha256 per scene."""

    def __init__(self, runs_root: Path, jobs, seeds: int, levels: Iterable[str]):
        self.seeds = list(range(seeds))
        self.runs: Dict[JobKey, Run] = {}
        errors: List[str] = []
        groups: Dict[Tuple, Set[int]] = defaultdict(set)
        for job in jobs:
            groups[_group(job.key)].add(job.key.seed)
        for group, wanted in groups.items():
            gdir = JobKey(*group[:4], 0, group[4]).path(runs_root).parent.parent
            if wanted != set(self.seeds):
                errors.append(f"registry: {gdir} lists seeds {sorted(wanted)}, expected {self.seeds}")
            found = {p.name for p in gdir.glob("seed*")} if gdir.is_dir() else set()
            if found != {f"seed{s}" for s in self.seeds}:
                errors.append(f"{gdir}: expected seed dirs {self.seeds}, found {sorted(found) or 'none'}")
        shas = defaultdict(list)
        splits = defaultdict(lambda: defaultdict(list))
        datas = defaultdict(lambda: defaultdict(list))
        positions = defaultdict(list)
        for job in jobs:
            path = job.key.path(runs_root)
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            try:
                run = read(path)
                problems = _check(run, job, levels)
                position = str((run.record.get("codec") or {}).get("position_format_used"))
                split, data = run.eval_images_sha256, run.data_sha256
            except (ValueError, TypeError, AttributeError) as e:
                errors.append(f"{path}: malformed record ({e})")
                continue
            errors += [f"{path}: {p}" for p in problems]
            self.runs[job.key] = run
            shas[str(run.git_sha)].append(path)
            splits[(run.key.dataset, run.key.scene)][str(split)].append(path)
            datas[(run.key.dataset, run.key.scene)][str(data)].append(path)
            if "codec" in levels:
                positions[position].append(path)
        if len(positions) > 1:
            errors.append("mixed codec.position_format_used: " + "; ".join(
                f"{f} (e.g. {p[0]})" for f, p in positions.items()))
        if len(shas) > 1:
            errors.append("mixed git_sha: " + "; ".join(
                f"{sha} ({len(p)} records, e.g. {p[0]})" for sha, p in shas.items()))
        for field_, per_scene in (("eval_images_sha256", splits), ("data_sha256", datas)):
            for (dataset, scene), by_sha in per_scene.items():
                if len(by_sha) > 1:
                    errors.append(f"mixed split.{field_} for {dataset}/{scene}: " + "; ".join(
                        f"{sha} (e.g. {p[0]})" for sha, p in by_sha.items()))
        if errors:
            raise RenderError("\n".join(f"  - {e}" for e in errors))
        self.git_sha = next(iter(shas), None)

    def run(self, key: JobKey) -> Run:
        if key not in self.runs:
            raise RenderError(f"  - {key} is not a job of this target (check targets.yaml)")
        return self.runs[key]

    def runs_of(self, jobs) -> List[Run]:
        return [self.run(j.key) for j in jobs]

    def seed_means(self, metric, level, variant, dataset, scenes, ablation=None,
                   tag=None, subset=None) -> List[float]:
        """Per seed: the mean over ``scenes`` (one number per seed)."""
        return [statistics.fmean(value(self.run(JobKey(variant, ablation, dataset, s, seed, tag)),
                                       metric, level, subset) for s in scenes)
                for seed in self.seeds]

    def cell(self, metric, level, variant, dataset, scenes, decimals=None, **kw) -> "Cell":
        return pm_cell(*mean_std(self.seed_means(metric, level, variant, dataset, scenes, **kw)),
                       metric, decimals)


def _group(key: JobKey) -> Tuple:
    return (key.variant, key.ablation, key.dataset, key.scene, key.tag)


def _check(run: Run, job, levels) -> List[str]:
    """Why ``run`` cannot stand in for ``job`` (empty if it can)."""
    rec, out = run.record, []
    if run.key != job.key:
        out.append(f"record claims job {run.key}")
    if rec.get("slot") != job.key.slot:
        out.append(f"slot {rec.get('slot')!r}, expected {job.key.slot!r}")
    recorded = rec.get("overrides")
    lost = [o for o in job.overrides if not isinstance(recorded, list) or o not in recorded]
    if lost:
        out.append(f"trained without the job's overrides {lost}")
    elif rec.get("config_sha256") != config_sha256(list(job.overrides)):
        out.append("config_sha256 is not the job's resolved config (a smoke run with extra "
                   "overrides, or config/ changed since)")
    if run.status != "DONE":
        out.append(f"status {run.status}")
    if rec.get("git_dirty"):
        out.append("made from a dirty tree (--allow-dirty)")
    if (not isinstance(run.git_sha, str) or len(run.git_sha) != 40 or not run.eval_images_sha256
            or not run.data_sha256):
        out.append("needs a 40-char git_sha, split.eval_images_sha256 and split.data_sha256")
    out += [f"no metrics.{level}" for level in levels if level not in rec["metrics"]]
    return out


# ---------------------------------------------------------------- cells

@dataclass
class Cell:
    text: str
    rank: Optional[float] = None  # the displayed mean, for best/second


def _decimals(metric: str, v: float, decimals: Optional[int], measured: bool = False) -> Optional[int]:
    if decimals is not None:
        return decimals
    if metric == "mb" and v >= 100:
        return 1
    if metric == "mb" and measured and v < 1:  # quoted baselines keep their source's precision
        return 3
    return METRICS[metric][1]


def fmt_num(v: float, d: Optional[int]) -> str:
    t = f"{v:,.0f}".replace(",", "{,}") if d is None else f"{v:.{d}f}"
    return t[1:] if t.startswith("-") and float(t.replace("{,}", "")) == 0 else t  # no "-0.00"


def pm_cell(mean: float, std: float, metric: str, decimals: Optional[int] = None) -> Cell:
    scale = 1e-6 if metric == "n_gaussians" else 1.0
    d = _decimals(metric, mean, decimals, measured=True)
    m, s = fmt_num(mean * scale, d), fmt_num(std * scale, d)
    return Cell(f"${m}_{{\\pm {s}}}$" + ("M" if metric == "n_gaussians" else ""),
                float(m.replace("{,}", "")))


def plain_cell(v: Optional[float], metric: str, decimals: Optional[int] = None) -> Cell:
    if v is None:
        return Cell(MISSING)
    t = fmt_num(v, _decimals(metric, v, decimals))
    return Cell(t, float(t.replace("{,}", "")))


def highlight(rows: Sequence[List[Cell]], directions: Sequence[int]) -> None:
    """Wrap per-column best / second-best cells (ties shared) in the paper's
    ``\\best{}`` / ``\\second{}`` macros, in place."""
    for c, direction in enumerate(directions):
        if not direction:
            continue
        col = [(i, row[c].rank * direction) for i, row in enumerate(rows) if row[c].rank is not None]
        for macro in ("best", "second"):
            if not col:
                break
            top = max(v for _, v in col)
            for i, _ in (x for x in col if x[1] == top):
                rows[i][c].text = f"\\{macro}{{{rows[i][c].text}}}"
            col = [x for x in col if x[1] != top]


def bold(text: str) -> str:
    return "{\\bfseries\\boldmath " + text + "}"


# ---------------------------------------------------------------- LaTeX

def grouped_header(first: str, groups: Sequence[Tuple[str, int]], sub: Sequence[str],
                   start: int = 2) -> List[str]:
    """Two-level header: a multicolumn label per group, cmidrules, then ``sub``.
    ``start``: the first column after the row-label columns (1-based)."""
    top, rules, col = [], [], start
    for label, n in groups:
        top.append(f"\\multicolumn{{{n}}}{{c}}{{{label}}}" if label else "")
        if label:
            rules.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    return ["& " * (start - 1) + " & ".join(top) + r" \\", " ".join(rules),
            first + " & " + " & ".join(sub) + r" \\"]


def tabular(colspec: str, header: Sequence[str],
            groups: Sequence[Sequence[Tuple[str, List[Cell]]]], comment: str) -> str:
    lines = [comment.rstrip("\n"), f"\\begin{{tabular}}{{{colspec}}}", "  \\toprule"]
    lines += [f"  {h}" for h in header]
    lines.append("  \\midrule")
    for i, rows in enumerate(groups):
        if i:
            lines.append("  \\midrule")
        lines += [f"  {label} & " + " & ".join(c.text for c in cells) + r" \\"
                  for label, cells in rows]
    lines += ["  \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


# ---------------------------------------------------------------- baselines

SIZE_UNITS = {"MiB": 1.0, "MB_1e6": 1e6 / MiB}  # factor to MiB


class Baselines:
    """``evals/baselines/baselines.yaml``. ``cell`` returns the value in table
    units (sizes in MiB) or None for an explicit ``~`` (not reported). A gap
    (missing key or ``source``, ``TODO`` value or size unit) fails a required
    entry and renders as ``---`` with a warning for an entry marked
    ``required: false``. A ``TODO`` inside the ``source`` prose is not a gap.
    A ``[mean, std]`` value (an author rerun over seeds) is returned as a tuple."""

    def __init__(self, path: Path = BASELINES):
        self.path = Path(path)
        self.data = yaml.safe_load(self.path.read_text()) or {}
        self.warnings: Set[str] = set()

    def label(self, name: str) -> str:
        return self._entry(name).get("label", name)

    def _entry(self, name: str) -> dict:
        if name not in self.data:
            raise RenderError(f"  - {self.path}: no entry {name!r}")
        return self.data[name]

    def cell(self, name: str, dataset: str, scene: str, metric: str,
             group: Optional[str] = None) -> Optional[Union[float, Tuple[float, float]]]:
        """``group``: an eval group under the scene (``<scene>: {<group>: {...}}``)."""
        e = self._entry(name)
        where = " / ".join(x for x in (name, dataset, scene, group, metric) if x)
        try:
            if not e.get("source"):
                raise LookupError("no source")
            row = e["datasets"]
            for k in (dataset, scene, group):
                if k is not None:
                    row = row[k]
                    if row is None:
                        return None
            v = row[metric]
            if v is None:
                return None
            if v == "TODO":
                raise LookupError("value TODO")
            scale = 1.0
            if metric == "mb":
                unit = e.get("size_unit")
                if unit not in SIZE_UNITS:
                    raise LookupError(f"size_unit {unit!r}")
                scale = SIZE_UNITS[unit]
            if isinstance(v, list):
                mean, std = v
                return float(mean) * scale, float(std) * scale
            return float(v) * scale
        except (KeyError, TypeError, LookupError) as err:
            reason = f"{err.args[0] if err.args else 'missing'}"
            if e.get("required", True):
                raise RenderError(f"  - {self.path}: {where}: {reason} (required)") from None
            self.warnings.add(f"{self.path.name}: {name}: {reason}; affected cells render as {MISSING}")
            return None


# ---------------------------------------------------------------- context

@dataclass
class Ctx:
    registry: "object"
    target: "object"
    runs_root: Path
    out: Path
    level: str
    baselines: Baselines
    git_sha: Optional[str] = field(default=None)

    def pool(self, levels: Iterable[str], jobs=None) -> Pool:
        pool = Pool(self.runs_root, self.target.jobs if jobs is None else jobs,
                    self.registry.seeds, set(levels))
        self.git_sha = pool.git_sha
        return pool

    def comment(self, extra: str = "", level: Optional[str] = None) -> str:
        lines = [f"% Generated by `python -m experiments.paper.render {self.target.name}` -- do not edit.",
                 f"% level={level or self.level}; git_sha={self.git_sha}",
                 "% Cells: mean +/- sample std (ddof=1) over seeds; dataset averages first average "
                 "the scenes within each seed, then take mean +/- std over the per-seed means."]
        return "\n".join(lines + ([extra] if extra else []))
