"""Compute table: training wall-clock, peak VRAM and PSNR per backend arm
(job ``tag``), on one scene and GPU. Each arm's records must report the
backend tier its tag names, on a single GPU model."""
from __future__ import annotations

from typing import Dict

from experiments.paper.common import METRICS, Cell, RenderError, tabular
from scion.util.run_index import JobKey

COLUMNS = ("train_min", "step_ms", "vram_gib", "psnr")


def render(ctx) -> Dict[str, str]:
    t = ctx.target
    pool = ctx.pool([ctx.level])
    variant, dataset, scene = t.unique("variant"), t.unique("dataset"), t.unique("scene")
    rows = []
    for r in t.spec["rows"]:
        runs = [pool.run(JobKey(variant, None, dataset, scene, s, r["tag"])) for s in pool.seeds]
        tiers = {tier for run in runs for tier in (run.record.get("backend_tiers") or {}).values()}
        gpus = {run.record.get("gpu") for run in runs}
        if tiers != {r["tag"]} or len(gpus) != 1 or None in gpus:
            raise RenderError(f"  - compute arm {r['tag']}: backend tiers {sorted(tiers)}, "
                              f"GPUs {sorted(map(str, gpus))} (want tier {r['tag']} on one GPU)")
        rows.append((r["label"], [Cell(gpus.pop())] + [
            pool.cell(m, ctx.level, variant, dataset, [scene], tag=r["tag"]) for m in COLUMNS]))
    header = ["Backend & GPU & " + " & ".join(METRICS[m][0] for m in COLUMNS) + r" \\"]
    return {"tables/table_compute.tex": tabular("l l " + "c" * len(COLUMNS), header, [rows],
                                                ctx.comment(f"% Scene: {dataset}/{scene}."))}
