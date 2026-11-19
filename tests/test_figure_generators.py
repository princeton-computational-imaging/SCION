"""Figure generators read v2 run dirs (evals/runs/<slot>/<dataset>/<scene>/seed<N>)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ablation_grid_from_v2_runs(tmp_path):
    grid = _script("compose_ablation_grid")
    runs = tmp_path / "runs"
    for i, slot in enumerate(["ours_large_abl_a", "ours_large_abl_b", "ours_large"]):
        d = runs / slot / "mipnerf360" / "garden" / "seed0"
        (d / "eval_views" / "quantized").mkdir(parents=True)
        pair = np.zeros((40, 120, 3), np.uint8)
        pair[:, 60:] = 50 * (i + 1)                    # GT | render
        Image.fromarray(pair).save(d / "eval_views" / "quantized" / "eval_img_0007.png")
        (d / "run.json").write_text(json.dumps({"metrics": {"quantized": {
            "psnr": 20.0 + i, "ssim": 0.5, "lpips": 0.4, "size_bytes": 3 * 2 ** 20}}}))
    out = tmp_path / "grid.pdf"
    assert grid.main(["--scene", "garden", "--view", "7", "--rois", "5,5,20,10", "0,0,30,30",
                      "--ablations", "a", "b", "full", "--runs-root", str(runs),
                      "--out", str(out)]) == 0
    assert out.read_bytes().startswith(b"%PDF")
    assert grid._caption(runs / "ours_large" / "mipnerf360" / "garden" / "seed0", "quantized") \
        == "PSNR=22.00\nSSIM=0.500\nLPIPS=0.400\nMB=3.00"


def test_ablation_figure_windows_pick_detail_without_overlap():
    fig = _script("component_ablation_figure")
    gt = np.zeros((80, 200, 3), np.uint8)
    gt[::2, 150:] = 255                                # stripes: all the gradient energy is on the right
    (y1, x1, h, w), (y2, x2, _, _) = fig.windows(gt)
    assert (h, w) == (20, 30) and x1 + w > 150
    assert abs(y1 - y2) >= h or abs(x1 - x2) >= w


def test_rd_labels_do_not_overlap():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from experiments.paper.run_capacity_sweep import _place_labels
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    _place_labels(ax, [(0.5, 0.5, name, 1.0) for name in ("Full", "K128", "Clone only")])  # one spot
    fig.canvas.draw()
    boxes = [Text.get_window_extent(t) for t in ax.texts]  # text only; Annotation adds its leader line
    assert len(boxes) == 3
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    plt.close(fig)
