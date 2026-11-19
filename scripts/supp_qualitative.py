"""Supplementary qualitative figure: every MipNeRF360 scene, matched-storage methods, zoomed crops.

    python supp_qualitative.py --scion <runs-root> --baselines <baseline runs> --out supp_qualitative.pdf
    # main-paper Fig. 3: the submission's two views and crops, pinned
    python supp_qualitative.py ... --scenes garden bicycle --pin garden:0:165,257,107,176 \
        --pin bicycle:21:26,19,197,312 --out qualitative_compression_eval.pdf

Per scene (rows): the ground-truth test frame with the crop box, then the crop for GT, SCION (Ours-Small,
quantized, seed 0), OMG, HAC++, ContextGS (Table 6 runs), each labelled with the method's model size.
No hand-picking: the view is the test view with SCION's (lower) median PSNR over the scene's test views; the crop is the square window (1/4 of the image height) with the most ground-truth gradient energy.
SCION eval_views/<level>/eval_img_NNNN.png hold GT | render; baselines test/ours_*/{renders,gt}/NNNNN.png, same
test order (every 8th image, sorted names). Each baseline's GT must equal SCION's GT pixel for pixel.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
MiB = 2 ** 20


def rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(-10 * np.log10(np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)))


def crop_box(gt: np.ndarray) -> tuple[int, int, int]:
    """(y, x, size) of the square window with the most gradient energy, on a stride of size/4."""
    g = gt.astype(np.float32).mean(-1)
    e = np.zeros_like(g)
    e[:, 1:] += np.abs(np.diff(g, axis=1))
    e[1:, :] += np.abs(np.diff(g, axis=0))
    s = gt.shape[0] // 4
    ii = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best = (-1.0, 0, 0)
    for y in range(0, gt.shape[0] - s + 1, s // 4):
        for x in range(0, gt.shape[1] - s + 1, s // 4):
            v = ii[y + s, x + s] - ii[y, x + s] - ii[y + s, x] + ii[y, x]
            best = max(best, (float(v), y, x))
    return best[1], best[2], s


def baseline_size(run: Path) -> float | None:
    if (run / "comp.xz").exists():  # OMG
        return (run / "comp.xz").stat().st_size / MiB
    if (run / "bitstreams/mlp.pt").exists():  # ContextGS: bitstreams/ incl. the MLPs
        return sum(f.stat().st_size for f in (run / "bitstreams").rglob("*") if f.is_file()) / MiB
    if (run / "outputs.log").exists():  # HAC++: logged encoded total incl. MLPs
        m = re.findall(r"Encoded sizes in MB:.*?Total ([0-9.]+)", (run / "outputs.log").read_text())
        return float(m[-1]) if m else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scion", type=Path, required=True, help="runs root holding ours_small/mipnerf360/<scene>/seed0")
    ap.add_argument("--baselines", type=Path, required=True, help="holds omg|hac|contextgs/<scene>")
    ap.add_argument("--level", default="quantized")
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--pin", action="append", default=[], metavar="SCENE:VIEW:Y,X,H,W",
                    help="fixed view and crop instead of the rule (main-paper Fig. 3)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"]})
    methods = [("OMG", "omg"), ("HAC++", "hac"), ("ContextGS", "contextgs")]
    cols = ["Ground truth", "GT crop", "Ours-Small"] + [m for m, _ in methods]
    pins = {sc: (int(v), tuple(int(n) for n in box.split(","))) for sc, v, box in (p.split(":") for p in a.pin)}
    widths = [0.94, 1, 1, 1, 1, 1] if pins else [1.5, 1, 1, 1, 1, 1]  # 0.94: frames as tall as the ~1.6:1 crops
    heights = [pins[sc][1][2] / pins[sc][1][3] if sc in pins else 1.0 for sc in a.scenes]
    slack = 1.0 if pins else 1.06  # square crops next to the wider GT column need a little vertical room
    fig = plt.figure(figsize=(7.0, 7.0 / sum(widths) * slack * sum(heights) + 0.25))
    gs = fig.add_gridspec(len(a.scenes), len(cols), width_ratios=widths, height_ratios=heights,
                          wspace=0.03, hspace=0.06)
    meta = {}
    for r, scene in enumerate(a.scenes):
        run = a.scion / "ours_small" / "mipnerf360" / scene / "seed0"
        files = sorted((run / "eval_views" / a.level).glob("eval_img_*.png"))
        pairs = [rgb(f) for f in files]
        half = pairs[0].shape[1] // 2
        scores = [psnr(p[:, half:], p[:, :half]) for p in pairs]
        view = pins[scene][0] if scene in pins else int(np.argsort(scores)[(len(scores) - 1) // 2])  # lower median
        gt, ours = pairs[view][:, :half], pairs[view][:, half:]
        rec = json.loads((run / "run.json").read_text())
        sizes = {"Ours-Small": rec["metrics"][a.level]["size_bytes"] / MiB}
        crops = {"Ours-Small": ours}
        for label, key in methods:
            d = sorted((a.baselines / key / scene / "test").glob("ours_*"))[-1]
            bgt = rgb(d / "gt" / f"{view:05d}.png")
            if not np.array_equal(bgt, gt):
                raise SystemExit(f"{key}/{scene}: GT of view {view} differs from SCION's")
            crops[label] = rgb(d / "renders" / f"{view:05d}.png")
            sizes[label] = baseline_size(a.baselines / key / scene)
        if scene in pins:
            y, x, h, w = pins[scene][1]
        else:
            y, x, h = crop_box(gt)
            w = h
        meta[scene] = {"view": view, "scion_psnr": scores[view], "crop": [y, x, h, w], "sizes_mib": sizes}
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(gt)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="red", lw=0.8))
        ax.set_ylabel(scene, fontsize=7)
        panels = [("GT crop", gt, None)] + [(m, crops[m], sizes[m]) for m in cols[2:]]
        for c, (label, im, mb) in enumerate(panels, start=1):
            ax_c = fig.add_subplot(gs[r, c])
            ax_c.imshow(im[y:y + h, x:x + w], interpolation="nearest")
            if mb is not None:
                ax_c.text(0.97, 0.03, f"{mb:.2f} MB", transform=ax_c.transAxes, ha="right", va="bottom",
                          fontsize=5.5, color="white", bbox=dict(fc="black", alpha=0.55, pad=0.8, lw=0))
            ax_c.set_xticks([]), ax_c.set_yticks([])
            if r == 0:
                ax_c.set_title(label, fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])
        if r == 0:
            ax.set_title(cols[0], fontsize=7)
    fig.savefig(a.out, bbox_inches="tight", dpi=300)
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
