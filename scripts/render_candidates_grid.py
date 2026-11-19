"""Render every learned primitive (codebook candidate template) in one grid.

    python scripts/render_candidates_grid.py \
        --run-dir evals/runs/ours_large/mipnerf360/garden/seed0 [--level raw] \
        --output figures/primitives_garden.png

Each tile is a candidate's splats seen from the side, auto-framed (the
``B_candidates_template/`` recipe of ``scripts/build_figure1_panels.py``),
ordered by usage (most-used first); empty candidates are skipped.
"""
import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's scion

import torch
from PIL import Image

from build_figure1_panels import _render_candidate_template, _to_uint8_rgb
from scion.engine.checkpoint import load
from scion.util.vis import tile_images_into_grid


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run dir (evals/runs/<slot>/<dataset>/<scene>/seed<N>).")
    p.add_argument("--level", choices=["raw", "quantized", "codec"], default="raw")
    p.add_argument("--output", "-o", type=Path, default=Path("candidates_grid.png"))
    p.add_argument("--resolution", "-r", type=int, default=256, help="Tile side (px).")
    p.add_argument("--background", "-b", default="white", choices=["white", "black"])
    p.add_argument("--device", "-d", default="cuda")
    args = p.parse_args()

    model = load(args.run_dir, device=args.device, level=args.level).model
    model.config.background_color = args.background
    counts = model.candidate_usage_counts()
    order = [k for k in torch.argsort(counts, descending=True).tolist() if counts[k] > 0]
    tiles = [_render_candidate_template(model, k, resolution=args.resolution) for k in order]
    bg = 1.0 if args.background == "white" else 0.0
    grid = tile_images_into_grid(tiles, args.resolution, args.resolution, bg_value=bg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8_rgb(grid)).save(args.output)
    print(f"Saved {len(tiles)} primitives to {args.output}")


if __name__ == "__main__":
    main()
