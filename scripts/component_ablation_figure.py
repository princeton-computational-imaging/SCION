"""Fig. 4 (fig:component_ablation_renders): one garden test view, two crops, GT then the cumulative arms of Table 2.

    python component_ablation_figure.py --runs <runs-root> --out 04_ablation_figure.pdf

Ours-Large, quantized, seed 0: ours_large_abl_{a,b,c} and ours_large (full). Same no-hand-picking rule as
supp_qualitative.py: the view is the test view with the (lower) median full-model PSNR over the scene's test views; the crops are the two non-overlapping windows (1/4 of the image height, 1.5:1) with the most GT gradient energy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from PIL import Image  # noqa: E402


def rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(-10 * np.log10(np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)))


ASPECT = 1.5  # crop width : height

ARMS = [("Splat DC", "ours_large_abl_a"), ("+ Instance DC", "ours_large_abl_b"), ("+ LPIPS", "ours_large_abl_c"),
        ("+ Adversarial", "ours_large")]


def windows(gt: np.ndarray, n: int = 2) -> list[tuple[int, int, int, int]]:
    """(y, x, h, w) of the n non-overlapping windows with the most gradient energy, on a stride of h/4."""
    g = gt.astype(np.float32).mean(-1)
    e = np.zeros_like(g)
    e[:, 1:] += np.abs(np.diff(g, axis=1))
    e[1:, :] += np.abs(np.diff(g, axis=0))
    h = gt.shape[0] // 4
    w = round(ASPECT * h)
    ii = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    cand = sorted(((float(ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]), y, x)
                   for y in range(0, gt.shape[0] - h + 1, h // 4) for x in range(0, gt.shape[1] - w + 1, h // 4)),
                  reverse=True)
    out = []
    for _, y, x in cand:
        if all(abs(y - y2) >= h or abs(x - x2) >= w for y2, x2, _, _ in out):
            out.append((y, x, h, w))
        if len(out) == n:
            return out
    raise SystemExit("no non-overlapping windows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--level", default="quantized")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    views = {key: sorted((a.runs / key / "mipnerf360/garden/seed0/eval_views" / a.level).glob("eval_img_*.png"))
             for _, key in ARMS}
    assert len({len(v) for v in views.values()}) == 1, "arms have different test sets"
    full = [rgb(f) for f in views["ours_large"]]
    half = full[0].shape[1] // 2
    scores = [psnr(p[:, half:], p[:, :half]) for p in full]
    view = int(np.argsort(scores)[(len(scores) - 1) // 2])  # lower median: no float tie between the middle two
    gt = full[view][:, :half]
    renders = {}
    for label, key in ARMS:
        p = rgb(views[key][view])
        if not np.array_equal(p[:, :half], gt):
            raise SystemExit(f"{key}: GT of view {view} differs")
        renders[label] = p[:, half:]
    boxes = windows(gt)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"]})
    cols = [("Ground Truth", gt)] + list(renders.items())
    fig, axes = plt.subplots(len(boxes), len(cols), figsize=(7.0, 7.0 / len(cols) / ASPECT * len(boxes) + 0.2),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    for r, (y, x, h, w) in enumerate(boxes):
        for c, (label, im) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(im[y:y + h, x:x + w], interpolation="lanczos")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(label, fontsize=9, pad=3)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.01, dpi=300)
    a.out.with_suffix(".json").write_text(json.dumps(
        {"view": view, "full_psnr": scores[view], "median_psnr": float(np.median(scores)), "boxes_yxhw": boxes,
         "crop_psnr": {label: [psnr(im[y:y + h, x:x + w], gt[y:y + h, x:x + w]) for y, x, h, w in boxes]
                       for label, im in renders.items()}}, indent=1))
    print("wrote", a.out, "view", view)


if __name__ == "__main__":
    main()
