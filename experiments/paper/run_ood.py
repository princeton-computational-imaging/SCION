"""tab:ood: campus test groups (ID / sideways / opposite). Per scene, the
baselines' author reruns, then ours at the target's fixed level; cells are
mean +/- std over seeds, highlighted per scene and column."""
from __future__ import annotations

from typing import Dict

from experiments.paper.common import (METRICS, MISSING, Cell, grouped_header, highlight, level_suffix,
                                      plain_cell, pm_cell, tabular)

Q = ("psnr", "ssim", "lpips")


def _quoted(v, metric: str) -> Cell:
    if v is None:
        return Cell(MISSING)
    return pm_cell(*v, metric) if isinstance(v, tuple) else plain_cell(v, metric)


def render(ctx) -> Dict[str, str]:
    reg, t, bl = ctx.registry, ctx.target, ctx.baselines
    level, groups, labels = t.spec["level"], t.spec["groups"], t.spec["scene_labels"]
    pool = ctx.pool({level})
    variant, dataset = t.unique("variant"), t.unique("dataset")
    cols = [("mb", None)] + [(m, g) for g in groups for m in Q]
    header = grouped_header("Scene & Method", [("", 1)] + [(lab, len(Q)) for lab in groups.values()],
                            [METRICS[m][0] for m, _ in cols], start=3)
    body = []
    for scene in t.present("scene", reg.datasets[dataset]["scenes"]):
        rows = [(bl.label(b), [_quoted(bl.cell(b, dataset, scene, m, g), m) for m, g in cols])
                for b in t.spec["baselines"]]
        rows.append((reg.variants[variant] + level_suffix(level),
                     [pool.cell(m, level, variant, dataset, [scene], subset=g) for m, g in cols]))
        highlight([c for _, c in rows], [METRICS[m][2] for m, _ in cols])
        body.append([((labels[scene] if i == 0 else "") + " & " + name, cells)
                     for i, (name, cells) in enumerate(rows)])
    colspec = "ll c " + " ".join(["ccc"] * len(groups))
    return {"tables/table_ood.tex": tabular(colspec, header, body, ctx.comment(
        f"% Groups: {', '.join(groups)} (split.json). Baselines: evals/baselines/baselines.yaml "
        "(author reruns, mean +/- std over the same seeds).", level=level))}
