"""Compose the ablation-study figure grid (PDF) from finished runs.

One row per ROI, one column per ablation arm (plus a leading GT column).
Each cell is the ROI cropped from the job's eval render
``<runs-root>/<slot>/<dataset>/<scene>/seed<N>/eval_views/<level>/eval_img_<view>.png``
(GT | render, side by side); captions are PSNR / SSIM / LPIPS / MB of that
level from the job's run.json (schema v2, 1 MB = 2^20 bytes).

    python scripts/compose_ablation_grid.py --scene garden --view 7 \\
        --rois 1031,326,127,88 70,344,131,91 --variant ours_large --seed 0 \\
        --out figures/04_ablation_figure.pdf

Arms are ablation ids (``config/ablation/<id>.yaml``); ``full`` is the full
method (no ablation). The view and ROIs were picked by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # pure-file backend; no display needed
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # this checkout's scion
from scion.util.run_index import JobKey  # noqa: E402

SERIF_FALLBACKS = ("Times New Roman", "Liberation Serif", "Nimbus Roman", "Nimbus Roman No9 L",
                   "DejaVu Serif")
MiB = 2 ** 20
Roi = Tuple[int, int, int, int]


def dataset_for_scene(scene: str, runs_root: Path) -> str:
    """The one dataset directory holding ``scene`` under ``runs_root``."""
    found = {p.parent.name for p in runs_root.glob(f"*/*/{scene}") if p.is_dir()}
    if len(found) != 1:
        raise ValueError(f"scene {scene!r} found under datasets {sorted(found)} in {runs_root}; "
                         "pass --dataset")
    return found.pop()


def _eval_image(run_dir: Path, level: str, view: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(gt, render)`` halves of the job's eval image ``view``."""
    pair = np.asarray(Image.open(run_dir / "eval_views" / level / f"eval_img_{view:04d}.png")
                      .convert("RGB"))
    w = pair.shape[1] // 2
    return pair[:, :w], pair[:, w:]


def _crop(img: np.ndarray, roi: Roi) -> np.ndarray:
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def _caption(run_dir: Path, level: str) -> str:
    m = json.loads((run_dir / "run.json").read_text())["metrics"][level]
    return (f"PSNR={m['psnr']:.2f}\nSSIM={m['ssim']:.3f}\nLPIPS={m['lpips']:.3f}\n"
            f"MB={m['size_bytes'] / MiB:.2f}")


def compose_grid(
    *,
    run_dirs: Sequence[Path],
    headers: Sequence[str],
    view: int,
    rois: Sequence[Roi],
    level: str,
    out_path: Path,
    font_size: float = 10.0,
    show_metrics: bool = True,
    h_spacing_pt: float = 2.0,
    v_spacing_pt: float = 2.0,
    label_position: str = "bottom",
) -> Path:
    """Write the grid PDF: column 0 is GT (from the first run), then one
    column per run; ``headers`` has one entry per column including GT."""
    if len(headers) != len(run_dirs) + 1:
        raise ValueError(f"{len(headers)} headers for {len(run_dirs) + 1} columns")
    available = {f.name for f in fm.fontManager.ttflist}
    font = next((f for f in SERIF_FALLBACKS if f in available), "serif")
    plt.rcParams.update({"font.family": "serif", "font.serif": [font, *SERIF_FALLBACKS],
                         "font.size": float(font_size), "pdf.fonttype": 42, "ps.fonttype": 42})

    halves = [_eval_image(d, level, view) for d in run_dirs]
    columns = [halves[0][0]] + [render for _gt, render in halves]
    images = [[_crop(img, roi) for img in columns] for roi in rois]
    captions = [None] + [_caption(d, level) for d in run_dirs] if show_metrics else None

    dpi, pt = 200.0, 1.0 / 72.0
    target_h = max(img.shape[0] for row in images for img in row)
    scale = [target_h / max(row[0].shape[0], 1) for row in images]  # one ROI size per row
    col_w = [max(images[r][c].shape[1] * scale[r] for r in range(len(rois))) / dpi
             for c in range(len(columns))]
    row_h = target_h / dpi
    gap_x, gap_y = h_spacing_pt * pt, v_spacing_pt * pt
    header_h = 1.4 * font_size * pt
    cap_h = (4 * 1.2 * font_size * pt + 0.05) if show_metrics else 0.0
    top_h = header_h if label_position in ("top", "both") else 0.0
    bot_h = header_h if label_position in ("bottom", "both") else 0.0
    fig_w = max(sum(col_w) + gap_x * (len(columns) - 1), 2.0)
    fig_h = max(top_h + len(rois) * (row_h + cap_h + gap_y) + bot_h, 1.5)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    lefts = np.concatenate([[0.0], np.cumsum(np.array(col_w) + gap_x)[:-1]])

    def label_row(y_in: float) -> None:
        for c, text in enumerate(headers):
            fig.text((lefts[c] + col_w[c] / 2) / fig_w, y_in / fig_h, text,
                     ha="center", va="center", fontsize=font_size, fontweight="bold")

    if top_h:
        label_row(fig_h - 0.55 * top_h)
    top = fig_h - top_h - (gap_y if top_h else 0.0)
    for r, row in enumerate(images):
        for c, img in enumerate(row):
            w_in, h_in = img.shape[1] * scale[r] / dpi, img.shape[0] * scale[r] / dpi
            ax = fig.add_axes([(lefts[c] + (col_w[c] - w_in) / 2) / fig_w,
                               (top - row_h + (row_h - h_in) / 2) / fig_h, w_in / fig_w, h_in / fig_h])
            ax.imshow(img, interpolation="none", aspect="auto")
            ax.set_axis_off()
            if captions and captions[c]:
                fig.text((lefts[c] + col_w[c] / 2) / fig_w, (top - row_h - cap_h / 2) / fig_h,
                         captions[c], ha="center", va="center", fontsize=font_size, linespacing=1.1)
        top -= row_h + cap_h + gap_y
    if bot_h:
        label_row(top - 0.45 * bot_h)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path.resolve()


def _parse_roi(s: str) -> Roi:
    try:
        x, y, w, h = (int(p) for p in s.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ROI must be 'x,y,w,h' ints, got {s!r}") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"ROI w,h must be positive: {s!r}")
    return x, y, w, h


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--view", required=True, type=int, help="Eval-view index.")
    p.add_argument("--rois", required=True, nargs="+", type=_parse_roi, help="'x,y,w,h' each.")
    p.add_argument("--variant", default="ours_large")
    p.add_argument("--ablations", nargs="+", default=["a", "b", "c", "full"],
                   help="Arms in column order; `full` = the full method.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", choices=("raw", "quantized", "codec"), default="quantized",
                   help="Which level's eval renders and metrics to show.")
    p.add_argument("--dataset", default=None, help="Default: the runs-root dataset holding --scene.")
    p.add_argument("--runs-root", type=Path, default=REPO_ROOT / "evals" / "runs")
    p.add_argument("--font-size", type=float, default=10.0)
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True,
                   help="PSNR/SSIM/LPIPS/MB caption under each non-GT cell.")
    p.add_argument("--h-spacing", type=float, default=2.0, help="Column gutter (pt).")
    p.add_argument("--v-spacing", type=float, default=2.0, help="Row gutter (pt).")
    p.add_argument("--column-names", nargs="+", default=None,
                   help="Headers, GT first (default: GT + the arm ids).")
    p.add_argument("--label-position", choices=("top", "bottom", "both"), default="bottom")
    p.add_argument("--out", type=Path, required=True, help="Output PDF.")
    args = p.parse_args(argv)

    dataset = args.dataset or dataset_for_scene(args.scene, args.runs_root)
    run_dirs = [JobKey(args.variant, None if a == "full" else a, dataset, args.scene, args.seed)
                .path(args.runs_root).parent for a in args.ablations]
    written = compose_grid(
        run_dirs=run_dirs, headers=args.column_names or ["GT", *args.ablations],
        view=args.view, rois=args.rois, level=args.level, out_path=args.out,
        font_size=args.font_size, show_metrics=args.metrics, h_spacing_pt=args.h_spacing,
        v_spacing_pt=args.v_spacing, label_position=args.label_position,
    )
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
