"""Appendix ablation tables on one scene (init, operator, compactness): one row
per ablation arm in the target's ``rows`` order, columns from its ``columns``
(``metric`` at --level or ``level:metric``). Optional target keys:
``row_header`` (first-column header) and ``highlight: true`` (best/second)."""
from __future__ import annotations

from itertools import groupby
from typing import Dict

from experiments.paper.common import METRICS, grouped_header, highlight, level_tex, tabular


def render(ctx) -> Dict[str, str]:
    t = ctx.target
    cols = [c.split(":") if ":" in c else [ctx.level, c] for c in t.spec["columns"]]
    pool = ctx.pool({lv for lv, _ in cols})
    variant, dataset, scene = t.unique("variant"), t.unique("dataset"), t.unique("scene")
    first = t.spec.get("row_header", "")
    sub = [METRICS[m][0] for _, m in cols]
    if any(":" in c for c in t.spec["columns"]):  # several levels: group the header by level
        header = grouped_header(first, [(level_tex(lv), len(list(g)))
                                         for lv, g in groupby(lv for lv, _ in cols)], sub)
    else:
        header = [first + " & " + " & ".join(sub) + r" \\"]
    rows = [(r["label"], [pool.cell(m, lv, variant, dataset, [scene], ablation=r["ablation"])
                          for lv, m in cols]) for r in t.spec["rows"]]
    if t.spec.get("highlight"):
        highlight([c for _, c in rows], [METRICS[m][2] for _, m in cols])
    return {f"tables/{t.spec['output']}": tabular("l " + "c" * len(cols), header, [rows], ctx.comment())}
