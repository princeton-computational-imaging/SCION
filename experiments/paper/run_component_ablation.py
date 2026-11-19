"""table_2: cumulative component ablation on one scene, one column group per variant."""
from __future__ import annotations

from typing import Dict

from experiments.paper.common import METRICS, QUALITY, grouped_header, highlight, level_tex, tabular
from experiments.paper.run_main_results import headline_levels


def render(ctx) -> Dict[str, str]:
    reg, t = ctx.registry, ctx.target
    pool = ctx.pool(headline_levels(ctx))
    variants = t.present("variant", reg.variants)
    dataset, scene = t.unique("dataset"), t.unique("scene")
    header = grouped_header(
        "Ablation", [(f"{reg.variants[v]}\\ ($n{{=}}{reg.seeds}$)", 4) for v in variants],
        [METRICS[m][0] for m in QUALITY] * len(variants))

    def row(label, ablation, level):
        return (label, [pool.cell(m, level, v, dataset, [scene], ablation=ablation)
                        for v in variants for m in QUALITY])

    rows = [row(r["label"], r["ablation"], ctx.level) for r in t.spec["rows"]]
    if ctx.level != "codec":  # the full method once more, at the codec level
        last = t.spec["rows"][-1]
        rows.append(row(f"{last['label']}, {level_tex('codec')}", last["ablation"], "codec"))
    highlight([c for _, c in rows], [METRICS[m][2] for m in QUALITY] * len(variants))
    return {"tables/table_2_component_ablation.tex": tabular(
        "l " + " ".join(["cccc"] * len(variants)), header, [rows], ctx.comment())}
