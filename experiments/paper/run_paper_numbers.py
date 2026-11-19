"""tables/numbers.tex: inline paper numbers as ``\\newcommand`` macros.

Dataset numbers follow the table rule (scenes averaged within each seed,
then the mean over seeds) at --level. Add a macro by adding a line below.
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

from experiments.paper.common import RenderError, fmt_num, tex_escape, value
from scion.util.run_index import LEVELS

ABBR = {"mipnerf360": "Mip", "tandt": "Tnt", "deepblending": "Db", "synthetic": "Syn"}
RTX = "RTX 5090"


def render(ctx) -> Dict[str, str]:
    reg = ctx.registry
    pool = ctx.pool(LEVELS)
    L = ctx.level

    def mean(metric, variant, dataset, scenes=None, ablation=None, level=L):
        scenes = scenes or reg.datasets[dataset]["scenes"]
        return statistics.fmean(pool.seed_means(metric, level, variant, dataset, scenes, ablation=ablation))

    def hours(target, tag=None) -> Tuple[List[float], str]:
        runs = [r for r in pool.runs_of(reg.targets[target].jobs) if r.key.tag == tag]
        gpus = {r.record.get("gpu") for r in runs}
        if len(gpus) != 1 or None in gpus:  # a time range across GPU models means nothing
            raise RenderError(f"  - train hours of {target}: several GPUs {sorted(map(str, gpus))}")
        return [value(r, "train_hours", L) for r in runs], str(gpus.pop())

    nums: List[Tuple[str, float, int]] = [
        ("nSeeds", reg.seeds, 0),
        ("oursSmallMipMBShort", mean("mb", "ours_small", "mipnerf360"), 1),
    ]
    ratios = []
    for d, ab in ABBR.items():
        nums += [(f"oursSmall{ab}MB", mean("mb", "ours_small", d), 2),
                 (f"oursSmall{ab}PSNR", mean("psnr", "ours_small", d), 2),
                 (f"oursSmall{ab}LPIPS", mean("lpips", "ours_small", d), 3)]
        ref = ctx.baselines.cell("3DGS", d, "_dataset_mean", "mb")
        if ref is None:
            raise RenderError(f"  - storage ratio needs a 3DGS size for {d}")
        ratios.append(ref / mean("mb", "ours_small", d))
    # Storage ratio: 3DGS size over Ours-Small size, per dataset.
    nums += [("storageRatioMin", min(ratios), 0), ("storageRatioMax", max(ratios), 0)]
    garden = dict(dataset="mipnerf360", scenes=["garden"])

    def lpips3(ablation=None):  # rounded as in the table, so the quoted difference matches its cells
        return round(mean("lpips", "ours_small", ablation=ablation, **garden), 3)

    nums += [  # component ablation on garden: full - (+LPIPS), (+LPIPS) - (+inst DC)
        ("ablSmallAdvDeltaLPIPS", lpips3() - lpips3("c"), 3),
        ("ablSmallLpipsDeltaLPIPS", lpips3("c") - lpips3("b"), 3),
    ]
    main_runs = pool.runs_of(reg.targets["main_results"].jobs)
    vram = max(value(r, "vram_gib", L) for r in main_runs)
    (main_h, main_gpu), (rtx_h, rtx_gpu) = hours("main_results"), hours("compute", tag="cuda")
    if RTX not in rtx_gpu:
        raise RenderError(f"  - compute (cuda) records are on {rtx_gpu}, not an {RTX}")
    nums += [("peakVramGB", vram, 1), ("trainHoursMin", min(main_h), 2),
             ("trainHoursMax", max(main_h), 2), ("trainHoursMaxRTX", max(rtx_h), 2)]

    lines = [ctx.comment("% Sizes in MB = 2^20 bytes; VRAM in GiB; peak VRAM and train hours "
                         "over the main_results runs (RTX: the compute CUDA arm).")]
    texts = [(name, fmt_num(v, d)) for name, v, d in nums] + [("mainGpuName", tex_escape(main_gpu))]
    for name, text in texts:
        assert name.isalpha(), name
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    return {"tables/numbers.tex": "\n".join(lines) + "\n"}
