"""table_3 capacity sweep + figures/pareto_capacity.pdf, a rate-distortion plot
(MB vs PSNR / SSIM / LPIPS) with one row per sweep (K, then S) at the ``codec``
level, each row with its own non-dominated frontier. Quality and size always share one level; the
table shows ``--level`` and, beside it, ``codec``."""
from __future__ import annotations

import sys
from typing import Dict, List, Sequence, Tuple

from scion.util.stats import mean_std

from experiments.paper.common import METRICS, Cell, bold, grouped_header, level_tex, tabular

COLUMNS = ("psnr", "ssim", "lpips", "mb", "bpg", "n_gaussians")
CODEC_COLUMNS = ("psnr", "ssim", "lpips", "mb", "bpg")
RD_METRICS = ("psnr", "ssim", "lpips")


def _levels(ctx) -> List[str]:
    return [ctx.level] if ctx.level == "codec" else [ctx.level, "codec"]


def frontier(points: Sequence[Tuple[float, float]], direction: int) -> List[int]:
    """Indices of the non-dominated (size, quality) points, by increasing size.
    ``direction`` is +1 when higher quality is better, -1 when lower is."""
    keep = []
    for i, (mb, q) in enumerate(points):
        dominated = any(
            mb2 <= mb and q2 * direction >= q * direction and (mb2 < mb or q2 * direction > q * direction)
            for j, (mb2, q2) in enumerate(points) if j != i)
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda i: points[i][0])


def render(ctx) -> Dict[str, str]:
    t = ctx.target
    levels = _levels(ctx)
    pool = ctx.pool(levels)
    variant, dataset, scene = t.unique("variant"), t.unique("dataset"), t.unique("scene")
    cols = [(ctx.level, m) for m in COLUMNS] + [("codec", m) for m in CODEC_COLUMNS if len(levels) > 1]

    def cells(ablation) -> List[Cell]:
        return [pool.cell(m, lv, variant, dataset, [scene], ablation=ablation) for lv, m in cols]

    # The leading ("", 1) group spans the Value column so the level rules start after it.
    header = grouped_header("Knob & Value", [("", 1)] + [(level_tex(lv), sum(c[0] == lv for c in cols)) for lv in levels],
                            [METRICS[m][0] for _, m in cols])
    groups = []
    for sweep in t.spec["sweeps"]:
        arms = sweep["arms"]
        rows = []
        for i, (val, ablation) in enumerate(arms.items()):
            c = cells(ablation)
            label = f"\\multirow{{{len(arms)}}}{{*}}{{{sweep['label']}}}" if i == 0 else ""
            value = str(val)
            if val == sweep["default"]:
                value, c = f"\\textbf{{{value}}}", [Cell(bold(x.text)) for x in c]
            rows.append((label, [Cell(value)] + c))
        groups.append(rows)
    out = {"tables/table_3_capacity_ablation.tex": tabular(
        "l c " + "c" * len(cols), header, groups,
        ctx.comment("% Bold: the shared default arm of each sweep."))}
    _plot(ctx, pool, variant, dataset, scene)
    return out


def _place_labels(ax, labels, fontsize: float = 5) -> None:
    """Annotate (x, y, text, alpha) points without overlapping text: a label that would collide
    with one already placed moves down until it is free and gets a leader line to its point."""
    ax.figure.canvas.draw()  # final layout, so data -> pixel transforms are fixed
    px = ax.figure.dpi / 72
    h = 1.25 * fontsize * px
    placed: List[Tuple[float, float, float]] = []
    for x, y, text, alpha in sorted(labels, key=lambda l: -ax.transData.transform((l[0], l[1]))[1]):
        X, Y = ax.transData.transform((x, y))
        w = 0.55 * fontsize * px * len(text)
        x0, y0 = X + 3 * px, Y + 3 * px
        while any(x0 < bx + bw and bx < x0 + w and y0 < by + h and by < y0 + h for bx, by, bw in placed):
            y0 -= h
        placed.append((x0, y0, w))
        dy = (y0 - Y) / px
        ax.annotate(text, (x, y), xytext=(3, dy), textcoords="offset points", fontsize=fontsize, alpha=alpha,
                    arrowprops=dict(arrowstyle="-", lw=0.3, alpha=0.5) if dy < 2.9 else None)


def _plot(ctx, pool, variant, dataset, scene) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[capacity] matplotlib not installed: skipping figures/pareto_capacity.pdf", file=sys.stderr)
        return
    sweeps = ctx.target.spec["sweeps"]
    rows = [[(f"{sw['short']}{val}", {m: mean_std(pool.seed_means(m, "codec", variant, dataset, [scene], ablation=a))
                                      for m in ("mb",) + RD_METRICS}) for val, a in sw["arms"].items()]
            for sw in sweeps]
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.5, 2.1 * len(rows)), squeeze=False)
    for pts, row in zip(rows, axes):
        for ax, m in zip(row, RD_METRICS):
            ax.errorbar([v["mb"][0] for _, v in pts], [v[m][0] for _, v in pts],
                        xerr=[v["mb"][1] for _, v in pts], yerr=[v[m][1] for _, v in pts], fmt="o", ms=4, capsize=2)
            front = frontier([(v["mb"][0], v[m][0]) for _, v in pts], METRICS[m][2])
            ax.plot([pts[i][1]["mb"][0] for i in front], [pts[i][1][m][0] for i in front], "k--", lw=0.8)
            ax.set_xlabel("MB", fontsize=9)
            ax.set_ylabel(METRICS[m][0], fontsize=9)
            ax.tick_params(labelsize=8)
            ax.margins(x=0.15)  # room for the rightmost point's label
            ax.grid(True, alpha=0.3)
            ax.set_box_aspect(1)  # square panels
    fig.tight_layout()
    for pts, row in zip(rows, axes):
        for ax, m in zip(row, RD_METRICS):
            _place_labels(ax, [(v["mb"][0], v[m][0], label, 1.0) for label, v in pts], fontsize=7)
    path = ctx.out / "figures" / "pareto_capacity.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
