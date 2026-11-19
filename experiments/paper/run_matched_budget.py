"""table_6: matched-storage comparison on one dataset (dataset means)."""
from __future__ import annotations

from typing import Dict

from experiments.paper.common import METRICS, QUALITY, highlight, level_suffix, plain_cell, tabular
from experiments.paper.run_main_results import headline_levels

MB_DECIMALS = {"mb": 3}  # sizes cluster around 1.5 MB, so one more digit


def render(ctx) -> Dict[str, str]:
    reg, t, bl = ctx.registry, ctx.target, ctx.baselines
    levels = headline_levels(ctx)
    pool = ctx.pool(levels)
    variant, dataset = t.unique("variant"), t.unique("dataset")
    scenes = reg.datasets[dataset]["scenes"]
    header = ["Method & " + " & ".join(METRICS[m][0] for m in QUALITY) + r" \\"]
    base = [(bl.label(b), [plain_cell(bl.cell(b, dataset, "_dataset_mean", m), m, MB_DECIMALS.get(m))
                           for m in QUALITY]) for b in t.spec["baselines"]]
    ours = [(reg.variants[variant] + ("" if lv == ctx.level else level_suffix(lv)),
             [pool.cell(m, lv, variant, dataset, scenes, MB_DECIMALS.get(m)) for m in QUALITY])
            for lv in levels]
    highlight([c for _, c in base + ours], [METRICS[m][2] for m in QUALITY])
    return {"tables/table_6_matched_budget.tex": tabular(
        "lcccc", header, [base, ours],
        ctx.comment("% Baselines: evals/baselines/baselines.yaml (author reruns)."))}
