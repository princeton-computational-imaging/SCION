"""Build Figure 1 panels for a trained run.

Given a run directory (and optionally an eval-image index), saves the six panels
needed to assemble Figure 1 as individual PNGs in an Adobe-friendly folder
layout:

    <out_dir>/
        A_render.png             # rendered scene (model prediction)
        B_candidates/
            candidate_0000.png   # one PNG per active candidate, rendered in
            candidate_0001.png   # isolation from the same camera (background
            ...                  # = scene background; only this candidate's
                                 # instances are rasterized in the scene).
        B_candidates_template/
            candidate_0000.png   # one PNG per active candidate, rendered as
            candidate_0001.png   # the isolated codebook *template* (single
            ...                  # synthetic instance, side view, auto-framed).
        C_assignment.png         # per-pixel candidate-id colormap
        D_frequency.png          # candidate-frequency map (greyscale)
        E_gt.png                 # ground truth
        F_depth.png              # 8-bit normalized depth visualization
        F_depth_raw_uint16.png   # 16-bit raw depth (linearly scaled)
        metadata.json            # camera/scene/eval-index/run provenance

Usage:
    python scripts/build_figure1_panels.py \\
        --run-dir evals/runs/ours_large/mipnerf360/garden/seed0 [--level raw] \\
        --eval-index 0 \\
        --output figures/fig1_<scene>

The model is loaded from the run's ``--level`` artifact
(:func:`scion.engine.checkpoint.load`).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's scion

import numpy as np
import torch
from PIL import Image

from scion.util.vis import (
    extract_camera_params,
    index_to_color,
    rasterize_gaussians,
)

logger = logging.getLogger("figure1")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _to_uint8_rgb(img: torch.Tensor) -> np.ndarray:
    """Convert a [H, W, 3] float tensor in [0, 1] to a uint8 numpy array."""
    arr = img.detach().clamp(0.0, 1.0).cpu().numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_png(img: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8_rgb(img)).save(path)


def _normalize_gt_image(img: torch.Tensor) -> torch.Tensor:
    """Coerce an eval image to [H, W, 3] float in [0,1].

    Handles channel-first vs channel-last layouts and uint8 vs float dtypes.
    Drops any alpha channel.
    """
    x = img
    if x.dim() == 4:
        x = x[0]
    if x.shape[-1] not in (3, 4) and x.shape[0] in (3, 4):
        x = x.permute(1, 2, 0)
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
    return x


# ---------------------------------------------------------------------------
# Eval cameras
# ---------------------------------------------------------------------------


def _scene(loaded):
    if loaded.scene is None or not loaded.scene.eval_indices:
        raise RuntimeError("the run's scene (with an eval split) could not be loaded")
    return loaded.scene


def _resolve_eval_camera_and_gt(loaded, eval_index: int):
    """``(camera, gt_image)`` of the eval-index'th eval view."""
    scene = _scene(loaded)
    if not 0 <= eval_index < len(scene.eval_indices):
        raise IndexError(f"--eval-index {eval_index} out of range [0, {len(scene.eval_indices)})")
    return scene.eval_cameras()[eval_index], scene.load_image(scene.eval_indices[eval_index])


def _iter_eval_gt_images(loaded):
    """Ground-truth eval images for the overview contact sheet."""
    scene = _scene(loaded)
    for scene_idx in scene.eval_indices:
        yield scene.load_image(scene_idx)


# ---------------------------------------------------------------------------
# Per-panel rendering
# ---------------------------------------------------------------------------

@torch.no_grad()
def _render_per_candidate(
    model,
    camera,
    candidate_ids: Iterable[int],
) -> List[Tuple[int, torch.Tensor]]:
    """Render each candidate's instances in isolation.

    Returns a list of ``(candidate_id, image)`` tuples — one image per
    candidate that has at least one rendered Gaussian. Skipped (empty)
    candidates are omitted.
    """
    bundle = model._get_assembled()
    if bundle.means.shape[0] == 0:
        return []

    asgn = model._last_assignment_per_gaussian
    if asgn is None:
        return []

    device = bundle.means.device
    viewmat, K_mat, W, H = extract_camera_params(
        camera, model._get_downscale_factor(), device
    )
    bg = model._get_background_color()

    out: List[Tuple[int, torch.Tensor]] = []
    for k in candidate_ids:
        mask = asgn == int(k)
        if not bool(mask.any()):
            continue

        render, alpha, _ = rasterize_gaussians(
            bundle.means[mask], bundle.quats[mask], bundle.scales[mask],
            bundle.opacities[mask],
            torch.sigmoid(bundle.features_dc[mask]),
            viewmat, K_mat, W, H,
            rasterize_mode=model.config.rasterize_mode,
        )
        rgb = render.squeeze(0)
        a = alpha.squeeze(0)
        img = (rgb + (1 - a) * bg).clamp(0, 1)
        out.append((int(k), img))
    return out


def _depth_to_visualization(depth: torch.Tensor) -> torch.Tensor:
    """Map a [H, W, 1] depth tensor to a grayscale [H, W, 3] image.

    Uses inverse depth (disparity = 1/d) — the convention for unbounded
    NeRF/3DGS-style scenes (close objects keep contrast, sky compresses
    to black). Encoding: white = close, black = far. Disparity is then
    percentile-clipped (1st/99th) and min-max normalized to [0, 1].
    Non-finite or non-positive depth pixels render as black.
    """
    d = depth.detach().squeeze(-1).cpu().numpy().astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return torch.zeros(d.shape[0], d.shape[1], 3)

    disp = np.zeros_like(d)
    disp[valid] = 1.0 / d[valid]

    lo = float(np.percentile(disp[valid], 1.0))
    hi = float(np.percentile(disp[valid], 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    disp_norm = np.clip((disp - lo) / (hi - lo), 0.0, 1.0)
    disp_norm = np.where(valid, disp_norm, 0.0)
    return torch.from_numpy(np.repeat(disp_norm[..., None], 3, axis=-1))


def _qualitative_palette(
    cmap_name: str, n_slots: Optional[int] = None,
) -> np.ndarray:
    """Return an ``[n, 3]`` palette sampled from a matplotlib colormap.

    For ``ListedColormap`` (e.g. ``tab20b``, ``Set3``) the natural slot
    count is used; for continuous colormaps (``viridis``, ``YlOrBr``) the
    palette is sampled at ``n_slots`` evenly-spaced positions (default
    20).
    """
    import matplotlib

    cmap = matplotlib.colormaps[cmap_name]
    if hasattr(cmap, "colors") and len(cmap.colors) <= 64:
        palette = np.asarray(cmap.colors, dtype=np.float32)
        if palette.shape[1] == 4:
            palette = palette[:, :3]
        return palette
    n = int(n_slots or 20)
    return np.array(
        [cmap(i / max(n - 1, 1))[:3] for i in range(n)], dtype=np.float32,
    )


@torch.no_grad()
def _render_assignment_map(
    model, camera, *, cmap_name: str = "tab20b",
    cmap_slots: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Per-pixel candidate-id colormap with frequency-sorted slot assignment.

    The ``N`` distinct slots of ``cmap_name`` are assigned to candidates in
    descending frequency order — i.e. the most-used candidate gets slot 0,
    second-most slot 1, etc. Beyond ``N`` candidates the assignment cycles
    (``rank % N``); the cycle is between high-frequency and very-low-frequency
    candidates, so visual collisions land on rare candidates.

    Default ``tab20b`` is a 20-slot qualitative palette with a muted, earth-
    leaning aesthetic — the warmest categorical option matplotlib ships
    that still keeps neighbouring slots clearly distinct. Override with e.g.
    ``cmap_name="tab20c"`` or any sequential colormap (``YlOrBr``, ``copper``).
    """
    bundle = model._get_assembled()
    if bundle.means.shape[0] == 0:
        return None
    asgn = model._last_assignment_per_gaussian
    if asgn is None:
        return None

    device = bundle.means.device
    K = model._num_candidates
    counts = model.candidate_usage_counts().to(device)
    # rank[k] = position of candidate k after sorting by descending frequency.
    order = torch.argsort(counts, descending=True)
    rank = torch.empty(K, dtype=torch.long, device=device)
    rank[order] = torch.arange(K, device=device)

    palette_np = _qualitative_palette(cmap_name, cmap_slots)
    n_slots = int(palette_np.shape[0])
    palette = torch.from_numpy(palette_np).to(device)
    cand_color = palette[rank % n_slots]                              # [K, 3]
    splat_color = cand_color[asgn]                                    # [M, 3]

    viewmat, K_mat, W, H = extract_camera_params(
        camera, model._get_downscale_factor(), device,
    )
    bg = model._get_background_color()

    render, alpha, _ = rasterize_gaussians(
        bundle.means, bundle.quats, bundle.scales, bundle.opacities,
        splat_color, viewmat, K_mat, W, H,
        rasterize_mode=model.config.rasterize_mode,
    )
    return (render.squeeze(0) + (1 - alpha.squeeze(0)) * bg).clamp(0, 1)


@torch.no_grad()
def _render_candidate_template(
    model, k: int, *, resolution: int = 256,
) -> torch.Tensor:
    """Render candidate ``k``'s splat template, side-view auto-framed."""
    from scion.util.vis import setup_side_view_camera, render_single_candidate

    device = model.splat_means.device
    rows = model.get_candidate_rows(k)
    if rows.numel() == 0:
        return torch.ones(resolution, resolution, 3, device=device)

    means = model.splat_means.data[rows]
    scales = model.splat_scales.data[rows]
    quats = model.splat_quats.data[rows]
    opacities = model.splat_opacities.data[rows]
    features = model.splat_features_dc.data[rows].clone()           # [S, 3]

    # Frame the splats — same logic as model._render_candidate_on_plane.
    scales_exp = torch.exp(scales)
    min_bounds = (means - 3.0 * scales_exp).min(dim=0).values
    max_bounds = (means + 3.0 * scales_exp).max(dim=0).values
    extent = (max_bounds - min_bounds).clamp(min=0.01)
    center = (min_bounds + max_bounds) / 2
    min_bounds = center - extent / 2
    max_bounds = center + extent / 2
    viewmat, K_mat, W, H = setup_side_view_camera(
        min_bounds, max_bounds, resolution, device,
    )

    # Compensate for the missing per-instance opacity boost (sigmoid(5)≈0.99).
    full_instance_opac = torch.full_like(opacities, 5.0)
    combined_opac = opacities + full_instance_opac

    bg = model._get_background_color()
    return render_single_candidate(
        means=means,
        scales=scales,
        quats=quats,
        colors=torch.sigmoid(features),
        opacities=combined_opac,
        viewmat=viewmat,
        K=K_mat,
        width=W,
        height=H,
        background_color=tuple(bg.tolist()),
    )


@torch.no_grad()
def _render_frequency_map(
    model, camera, *, scale: str = "log",
) -> Optional[torch.Tensor]:
    """Render the candidate-frequency map with a sequential colormap.

    Each splat is coloured by its candidate's usage count (the count from
    ``model.candidate_usage_counts()``, broadcast to splats via
    ``_last_assignment_per_gaussian``), mapped through the ``turbo`` colormap.

    Args:
        scale: ``"log"`` (default) applies ``log1p(count)`` before min-max
            normalization — recommended because usage histograms are
            heavy-tailed, so linear scaling collapses most candidates into
            the bottom of the colormap. ``"linear"`` uses raw counts.
    """
    import matplotlib

    bundle = model._get_assembled()
    if bundle.means.shape[0] == 0:
        return None
    asgn = model._last_assignment_per_gaussian
    if asgn is None:
        return None

    device = bundle.means.device
    counts = model.candidate_usage_counts().to(device).float()
    if scale == "log":
        signal = torch.log1p(counts)
    elif scale == "linear":
        signal = counts
    else:
        raise ValueError(f"Unknown frequency scale: {scale!r}")
    denom = (signal.max() - signal.min()).clamp(min=1e-6)
    intensity_per_cand = (signal - signal.min()) / denom            # [K] in [0,1]
    intensity = intensity_per_cand[asgn].cpu().numpy().astype(np.float32)

    # Sequential perceptually-uniform colormap: blue (low) → red (high).
    # `turbo` is preferred over diverging colormaps like `coolwarm` because
    # frequency is one-sided — there's no meaningful midpoint to anchor a
    # diverging palette, and `coolwarm`'s white centre hides the mid-range.
    cmap = matplotlib.colormaps["turbo"]
    colors_np = cmap(intensity)[..., :3].astype(np.float32)
    colors = torch.from_numpy(colors_np).to(device)

    viewmat, K_mat, W, H = extract_camera_params(
        camera, model._get_downscale_factor(), device
    )
    bg = model._get_background_color()

    render, alpha, _ = rasterize_gaussians(
        bundle.means, bundle.quats, bundle.scales, bundle.opacities,
        colors, viewmat, K_mat, W, H,
        rasterize_mode=model.config.rasterize_mode,
    )
    return (render.squeeze(0) + (1 - alpha.squeeze(0)) * bg).clamp(0, 1)


def _depth_to_uint16(depth: torch.Tensor) -> np.ndarray:
    """Linearly scale a [H, W, 1] depth tensor to a 16-bit grayscale image."""
    d = depth.detach().squeeze(-1).cpu().numpy().astype(np.float32)
    finite = np.isfinite(d)
    finite_max = float(d[finite].max()) if finite.any() else 0.0
    d = np.where(finite, d, finite_max)
    lo, hi = float(d.min()), float(d.max())
    if hi <= lo:
        hi = lo + 1e-6
    d_norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    return (d_norm * 65535.0 + 0.5).astype(np.uint16)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_figure1_panels(
    run_dir: Path,
    eval_index: int,
    output_dir: Path,
    *,
    max_candidates: Optional[int] = None,
    candidate_order: str = "frequency",
    template_resolution: int = 256,
    frequency_scale: str = "log",
    background: str = "scene",
    assignment_cmap: str = "tab20b",
    assignment_cmap_slots: Optional[int] = None,
    level: str = "raw",
) -> Path:
    """Render Figure 1 panels for the run at ``eval_index``.

    Args:
        run_dir: Run directory (``evals/runs/<slot>/<dataset>/<scene>/seed<N>``).
        level: Compression level whose artifact is rendered.
        eval_index: Index into the eval dataset.
        output_dir: Destination folder. Created if missing.
        max_candidates: If set, cap the number of per-candidate renders
            (after sorting by ``candidate_order``). ``None`` = all active.
        candidate_order: ``"frequency"`` (most-used first), ``"id"``
            (ascending candidate index).
        template_resolution: Side length (px) for the per-candidate template
            renders saved under ``B_candidates_template/``.
    """
    from scion.engine.checkpoint import load as _ckpt_load

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s (level %s)", run_dir, level)
    loaded = _ckpt_load(Path(run_dir), level=level)
    model = loaded.model
    step = loaded.step

    # Override the rasterization background once: the model reads
    # `config.background_color` (str or tensor) at render time, so flipping
    # the string flows through to A/B/B_template/C/D uniformly.
    if background != "scene":
        if background not in ("white", "black"):
            raise ValueError(f"Unknown background: {background!r}")
        model.config.background_color = background
        logger.info("Forcing background_color = %s", background)

    camera, gt_image = _resolve_eval_camera_and_gt(loaded, eval_index)
    device = next(model.parameters()).device
    if hasattr(camera, "to"):
        camera = camera.to(device)

    # Forward pass — populates _last_assignment_per_gaussian and depth.
    outputs = model.get_outputs(camera)

    # ---- Panel A: rendered scene ----
    _save_png(outputs["rgb"], output_dir / "A_render.png")
    logger.info("wrote A_render.png")

    # ---- Panel E: ground truth ----
    gt_img = _normalize_gt_image(gt_image)
    _save_png(gt_img.to(outputs["rgb"].device), output_dir / "E_gt.png")
    logger.info("wrote E_gt.png")

    # ---- Panel C: assignment colormap ----
    amap = _render_assignment_map(
        model, camera,
        cmap_name=assignment_cmap, cmap_slots=assignment_cmap_slots,
    )
    if amap is not None:
        _save_png(amap, output_dir / "C_assignment.png")
        logger.info(
            "wrote C_assignment.png (cmap=%s, %d slots)",
            assignment_cmap,
            int(_qualitative_palette(assignment_cmap, assignment_cmap_slots).shape[0]),
        )
    else:
        logger.warning("Assignment map could not be rendered; skipping panel C.")

    # ---- Panel D: frequency / usage map (blue → red) ----
    # Always emit both scales so the figure assembler can pick. The
    # unsuffixed ``D_frequency.png`` mirrors the chosen default
    # (`frequency_scale`).
    primary: Optional[torch.Tensor] = None
    for sc in ("log", "linear"):
        fmap = _render_frequency_map(model, camera, scale=sc)
        if fmap is None:
            continue
        path = output_dir / f"D_frequency_{sc}.png"
        _save_png(fmap, path)
        logger.info("wrote %s", path.name)
        if sc == frequency_scale:
            primary = fmap
    if primary is not None:
        _save_png(primary, output_dir / "D_frequency.png")
        logger.info("wrote D_frequency.png (mirror of D_frequency_%s.png)",
                    frequency_scale)

    # ---- Panel F: depth ----
    depth = outputs.get("depth")
    if depth is not None:
        _save_png(_depth_to_visualization(depth), output_dir / "F_depth.png")
        Image.fromarray(_depth_to_uint16(depth), mode="I;16").save(
            output_dir / "F_depth_raw_uint16.png"
        )
        logger.info("wrote F_depth.png + F_depth_raw_uint16.png")
    else:
        logger.warning("No depth output from get_outputs; skipping panel F.")

    # ---- Panel B: per-candidate isolation renders ----
    counts = model.candidate_usage_counts()
    active = torch.nonzero(counts > 0, as_tuple=False).squeeze(-1).tolist()
    if candidate_order == "frequency":
        active.sort(key=lambda k: -int(counts[k].item()))
    elif candidate_order == "id":
        active.sort()
    else:
        raise ValueError(f"Unknown candidate_order: {candidate_order!r}")
    if max_candidates is not None:
        active = active[: int(max_candidates)]

    cand_dir = output_dir / "B_candidates"
    cand_dir.mkdir(exist_ok=True)
    rendered = _render_per_candidate(model, camera, active)
    pad = max(4, len(str(model._num_candidates)))
    for k, img in rendered:
        _save_png(img, cand_dir / f"candidate_{k:0{pad}d}.png")
    logger.info("wrote %d candidate panels under %s", len(rendered), cand_dir)

    # ---- Panel B (template): each candidate's template splats, side-view ----
    template_dir = output_dir / "B_candidates_template"
    template_dir.mkdir(exist_ok=True)
    n_template = 0
    for k in active:
        img = _render_candidate_template(
            model, int(k), resolution=template_resolution,
        )
        _save_png(img, template_dir / f"candidate_{k:0{pad}d}.png")
        n_template += 1
    logger.info("wrote %d template panels under %s", n_template, template_dir)

    # ---- Provenance ----
    metadata = {
        "run_dir": str(Path(run_dir)),
        "level": level,
        "checkpoint_step": int(step),
        "eval_index": int(eval_index),
        "num_candidates_total": int(model._num_candidates),
        "num_candidates_active": int(sum(1 for _, _ in rendered)),
        "candidate_order": candidate_order,
        "candidate_usage_counts": [int(c) for c in counts.tolist()],
        "image_size": [int(outputs["rgb"].shape[0]), int(outputs["rgb"].shape[1])],
        "scene": Path(run_dir).parent.name,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("wrote metadata.json")
    return output_dir


@torch.no_grad()
def build_eval_overview(
    run_dir: Path,
    output_path: Path,
    *,
    thumb_width: int = 320,
) -> Path:
    """Write a contact sheet of every eval image, labeled with its index.

    Use this to choose a good ``--eval-index`` before running the full
    Figure 1 pipeline. No model forward is performed.
    """
    from scion.engine.checkpoint import load as _ckpt_load
    from PIL import ImageDraw, ImageFont

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s (overview only)", run_dir)
    loaded = _ckpt_load(Path(run_dir), level="raw")
    gt_images = list(_iter_eval_gt_images(loaded))
    n = len(gt_images)
    if n == 0:
        raise RuntimeError("Eval dataset is empty.")

    sample = _normalize_gt_image(gt_images[0])
    H, W = int(sample.shape[0]), int(sample.shape[1])
    th_w = int(thumb_width)
    th_h = int(round(H * th_w / W))
    label_h = 22
    ncols = max(1, int(round(np.sqrt(n))))
    nrows = (n + ncols - 1) // ncols

    sheet = Image.new(
        "RGB",
        (ncols * th_w, nrows * (th_h + label_h)),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except IOError:
        font = ImageFont.load_default()

    for idx, gt_raw in enumerate(gt_images):
        gt = _normalize_gt_image(gt_raw)
        thumb_arr = _to_uint8_rgb(gt)
        thumb = Image.fromarray(thumb_arr).resize((th_w, th_h), Image.LANCZOS)
        c, r = idx % ncols, idx // ncols
        x, y = c * th_w, r * (th_h + label_h)
        sheet.paste(thumb, (x, y))
        draw.rectangle((x, y + th_h, x + th_w, y + th_h + label_h),
                       fill=(30, 30, 30))
        draw.text((x + 6, y + th_h + 3), f"index {idx}",
                  fill=(255, 255, 255), font=font)

    sheet.save(output_path)
    logger.info("wrote eval overview (%d images, %d×%d) -> %s",
                n, ncols, nrows, output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Run dir (evals/runs/<slot>/<dataset>/<scene>/seed<N>).")
    p.add_argument("--level", choices=["raw", "quantized", "codec"], default="raw",
                   help="Compression level whose artifact is rendered (default raw).")
    p.add_argument("--eval-index", type=int, default=0,
                   help="Index into the eval dataset (default: 0).")
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory (created if missing).")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Cap on B/ panels (default: render all active).")
    p.add_argument("--candidate-order", choices=["frequency", "id"],
                   default="frequency",
                   help="Sort order for per-candidate filenames "
                        "(default: frequency, most-used first).")
    p.add_argument("--template-resolution", type=int, default=256,
                   help="Side length (px) for B_candidates_template/* renders "
                        "(default 256).")
    p.add_argument("--frequency-scale", choices=["log", "linear"],
                   default="log",
                   help="Default scale for the unsuffixed D_frequency.png "
                        "mirror (default: log). Both scales are always "
                        "written to D_frequency_log.png and "
                        "D_frequency_linear.png.")
    p.add_argument("--background", choices=["scene", "white", "black"],
                   default="scene",
                   help="Override rasterization background (default: scene "
                        "— use whatever the trained config specified).")
    p.add_argument("--assignment-cmap", default="tab20b",
                   help="Matplotlib colormap for the assignment map "
                        "(default: tab20b — 20 muted earth-leaning slots). "
                        "Try `tab20c`, `Set3`, `Paired`, `YlOrBr`, `copper`.")
    p.add_argument("--assignment-cmap-slots", type=int, default=None,
                   help="Number of distinct colors to sample from the cmap. "
                        "Default: cmap.N for ListedColormap, else 20. "
                        "Top-N candidates by frequency get unique slots; "
                        "the rest cycle (collisions land on rare candidates).")
    p.add_argument("--mode", choices=["panels", "overview", "both"],
                   default="panels",
                   help="`panels` (default) writes Figure 1; `overview` writes "
                        "only an eval contact sheet; `both` does both.")
    p.add_argument("--overview-output", type=Path, default=None,
                   help="Path for the eval overview PNG (default: "
                        "<output>/eval_overview.png in panels/both mode, or "
                        "<output> itself in overview mode if it ends in .png).")
    p.add_argument("--overview-thumb-width", type=int, default=320,
                   help="Thumbnail width (px) for eval overview (default 320).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if args.mode in ("panels", "both"):
        out = build_figure1_panels(
            run_dir=args.run_dir,
            level=args.level,
            eval_index=args.eval_index,
            output_dir=args.output,
            max_candidates=args.max_candidates,
            candidate_order=args.candidate_order,
            template_resolution=args.template_resolution,
            frequency_scale=args.frequency_scale,
            background=args.background,
            assignment_cmap=args.assignment_cmap,
            assignment_cmap_slots=args.assignment_cmap_slots,
        )
        print(f"Figure 1 panels written to {out}")

    if args.mode in ("overview", "both"):
        if args.overview_output is not None:
            ov = args.overview_output
        elif args.mode == "overview" and args.output.suffix.lower() == ".png":
            ov = args.output
        else:
            ov = args.output / "eval_overview.png"
        build_eval_overview(
            run_dir=args.run_dir,
            output_path=ov,
            thumb_width=args.overview_thumb_width,
        )
        print(f"Eval overview written to {ov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
