"""Main results: table_1 (dataset means), per-scene tables 1a-1d, and the
compression-level table 1e (raw / quantized / codec). Layouts: LAYOUTS.md."""
from __future__ import annotations

from typing import Dict, List

from scion.util.run_index import LEVELS

from experiments.paper.common import (METRICS, QUALITY, Cell, grouped_header, highlight, level_suffix,
                                      plain_cell, tabular, tex_escape)

PER_SCENE = {
    "mipnerf360": "table_1a_per_scene_mipnerf360.tex",
    "tandt": "table_1b_per_scene_tandt.tex",
    "synthetic": "table_1c_per_scene_synthetic.tex",
    "deepblending": "table_1d_per_scene_deepblending.tex",
}
SUB = [METRICS[m][0] for m in QUALITY]
DIRS = [METRICS[m][2] for m in QUALITY]


def ours_rows(ctx, pool, variants, dataset_scenes, levels) -> List:
    """One row per (variant, level); cells over ``dataset_scenes`` [(dataset, scenes)]."""
    reg = ctx.registry
    return [(reg.variants[v] + ("" if lv == ctx.level else level_suffix(lv)),
             [pool.cell(m, lv, v, d, scenes) for d, scenes in dataset_scenes for m in QUALITY])
            for lv in levels for v in variants]


def headline_levels(ctx) -> List[str]:
    return [ctx.level] + ([] if ctx.level == "codec" else ["codec"])


def render(ctx) -> Dict[str, str]:
    reg, t, bl = ctx.registry, ctx.target, ctx.baselines
    pool = ctx.pool(LEVELS)
    variants = t.present("variant", reg.variants)
    datasets = t.present("dataset", reg.datasets)
    out = {}

    # table_1: dataset means.
    full = [(d, reg.datasets[d]["scenes"]) for d in datasets]
    header = grouped_header("Method", [(reg.datasets[d]["label"], 4) for d in datasets],
                            SUB * len(datasets))
    groups = [[(bl.label(b), [plain_cell(bl.cell(b, d, "_dataset_mean", m), m)
                              for d in datasets for m in QUALITY]) for b in group]
              for group in t.spec["baselines"]]
    groups.append(ours_rows(ctx, pool, variants, full, headline_levels(ctx)))
    highlight([cells for g in groups for _, cells in g], DIRS * len(datasets))
    colspec = "l " + " ".join(["cccc"] * len(datasets))
    out["tables/table_1_main_results.tex"] = tabular(
        colspec, header, groups, ctx.comment("% Baselines: evals/baselines/baselines.yaml (quoted)."))

    # table_1a-1d: per scene, in blocks of at most 3 scenes stacked in one tabular (legible at \linewidth).
    def per_scene(d, scenes, comment):
        header = grouped_header("Method", [(tex_escape(s), 4) for s in scenes], SUB * len(scenes))
        rows = []
        for b in t.spec["per_scene_baselines"]:
            cells: List[Cell] = [plain_cell(bl.cell(b, d, s, m), m) for s in scenes for m in QUALITY]
            if any(c.rank is not None for c in cells):
                rows.append((bl.label(b), cells))
        ours = ours_rows(ctx, pool, variants, [(d, [s]) for s in scenes], headline_levels(ctx))
        groups = [g for g in (rows, ours) if g]
        highlight([cells for g in groups for _, cells in g], DIRS * len(scenes))
        return tabular("l " + " ".join(["cccc"] * len(scenes)), header, groups, comment)

    for d in datasets:
        scenes = reg.datasets[d]["scenes"]
        blocks = [scenes[i:i + 3] for i in range(0, len(scenes), 3)]
        if len(blocks) == 1:
            text = per_scene(d, scenes, ctx.comment())
        else:
            parts = [per_scene(d, sc, "") for sc in blocks]
            text = (ctx.comment().rstrip("\n") + "\n\\begin{tabular}{@{}c@{}}\n"
                    + "\\\\[3mm]\n".join(p.lstrip("\n") for p in parts) + "\\end{tabular}\n")
        out[f"tables/{PER_SCENE[d]}"] = text

    # table_1e: every level, dataset means, ours only.
    header = grouped_header("Method", [(reg.datasets[d]["label"], 4) for d in datasets],
                            SUB * len(datasets))
    groups = [[(reg.variants[v] + level_suffix(lv), [pool.cell(m, lv, v, d, sc) for d, sc in full
                                                   for m in QUALITY]) for lv in LEVELS]
              for v in variants]
    out["tables/table_1e_compression_levels.tex"] = tabular(colspec, header, groups, ctx.comment())
    return out
