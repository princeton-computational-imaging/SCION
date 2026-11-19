"""Animate a trained CodebookModel via DINO-text instance grouping.

Single-command pipeline:
  1. Load a trained run's :class:`CodebookModel` (the ``--level`` artifact,
     via :func:`scion.engine.checkpoint.load`) and assemble the
     codebook's world-space Gaussians.
  2. Sample fp32 DINO features at every assembled splat in every view of
     the checkpoint's scene (train + eval) and opacity-average them per
     instance — the construction the cz animation runs used.
  3. Classify each instance against user-provided text labels using the
     DINOv3 DINOtxt head (:mod:`scion.util.dinotxt`).
  4. Look up a per-label :class:`AnimationConfig` (YAML) under
     ``config/animation_codebook/``, falling back to ``default.yaml``
     (``--anim-config-mode default`` forces ``default.yaml`` for every label).
  5. Render a camera-path video, mutating the assembled Gaussian bundle in
     place with per-group wind.  Instances not confidently matched, and
     instances matched to a ``--static-groups`` absorber label, stay static.
  6. Optionally (``--save-motion-maps DIR``, static camera only) write
     per-instance flow maps via :mod:`scion.util.motion_diagnostics`.

Compared with cz: the default behaves like cz's ``--no-coherent-label-motion``,
and ``--save-motion-maps`` takes the output directory instead of a flag.

Example (the paper's animation-figure flow maps, flowers scene)
---------------------------------------------------------------

    python scripts/animate_codebook.py \
        --run-dir evals/runs/ours_large/mipnerf360/flowers/seed0 \
        --camera-path-filename evals/camera_paths/flowers.json \
        --groups grass,leaves,flower --static-groups sky,dirt,wood \
        --label-thresholds grass=0.08,leaves=0.10,flower=0.08 \
        --anim-config-mode default \
        --duration 6.0 --fps 10 \
        --output-path renders/flowers_wind/animation.mp4 \
        --save-motion-maps renders/flowers_wind
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's scion

import numpy as np
import torch
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Camera resolution helpers
# ---------------------------------------------------------------------------


def _scene_split_cameras(loaded, split: str):
    """``list[scion.Camera]`` of the run's train (``train``) or eval
    (``val`` / ``test``: the scene's single eval split) cameras."""
    if loaded.scene is None:
        raise RuntimeError("the run's scene could not be loaded; cannot resolve its cameras")
    return loaded.scene.train_cameras() if split == "train" else loaded.scene.eval_cameras()


def _load_camera_path_json(path: Path):
    """Parse a nerfstudio-viewer-exported camera path JSON into ``list[Camera]``.

    Follows :func:`nerfstudio.cameras.camera_paths.get_path_from_json` for
    pinhole cameras; unsupported camera types (fisheye / equirect / vr180)
    fall back to perspective intrinsics.
    """
    import math as _math
    import torch as _torch
    from scion.camera import Camera as _Camera

    with open(path, "r") as f:
        data = json.load(f)

    H = int(data["render_height"])
    W = int(data["render_width"])
    cx = W / 2.0
    cy = H / 2.0
    cams: List[_Camera] = []
    for cam in data["camera_path"]:
        c2w = _torch.tensor(cam["camera_to_world"], dtype=_torch.float32).view(4, 4)
        fov = float(cam.get("fov", 50.0))
        # Three.js focal length convention: pp_h / tan(fov/2)  (image_height
        # principal-point half-height; fov is vertical, in degrees).
        focal = (H / 2.0) / _math.tan(_math.radians(fov) / 2.0)
        K = _torch.tensor(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            dtype=_torch.float32,
        )
        cams.append(_Camera(cam_to_world=c2w, K=K, width=W, height=H))
    return cams


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_group_spec(
    label: str, anim_dir: Path, force_default: bool = False,
):
    """Load an AnimationConfig + group-level knobs for a text label.

    Falls back to ``default.yaml`` if ``<label>.yaml`` is missing, or always
    uses ``default.yaml`` when ``force_default`` is true.
    """
    from scion.util.animate import AnimationConfig
    from scion.util.animate_codebook import GroupAnimationSpec

    yaml_path = anim_dir / f"{label}.yaml"
    if force_default or not yaml_path.exists():
        yaml_path = anim_dir / "default.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"No YAML for label '{label}' and no default.yaml in {anim_dir}."
            )

    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    cfg_fields = dict(data.get("animation_config") or {})
    # wind_direction can be a list in YAML; AnimationConfig expects a tuple.
    if "wind_direction" in cfg_fields and isinstance(cfg_fields["wind_direction"], list):
        cfg_fields["wind_direction"] = tuple(cfg_fields["wind_direction"])

    anim = AnimationConfig(**cfg_fields)
    return GroupAnimationSpec(
        label=label,
        animation_config=anim,
        bend_axis_mode=str(data.get("bend_axis", "instance_local")),
        per_splat_flutter=bool(data.get("per_splat_flutter", False)),
        size_impact_alpha=float(data.get("size_impact_alpha", 0.0)),
        size_impact_min=float(data.get("size_impact_min", 0.2)),
        size_impact_max=float(data.get("size_impact_max", 5.0)),
        pivot_mode=str(data.get("pivot_mode", "base")),
        pivot_down_offset=float(data.get("pivot_down_offset", 0.0)),
    )


def _parse_label_thresholds(
    spec: Optional[str],
    label_names: List[str],
    default_threshold: float,
) -> List[float]:
    """Build one DINOtxt match threshold per label.

    ``spec`` is a comma-separated list of ``label=value`` overrides. Labels
    not mentioned inherit ``default_threshold``.
    """
    thresholds = [float(default_threshold)] * len(label_names)
    if not spec:
        return thresholds

    label_to_idx = {name: i for i, name in enumerate(label_names)}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"--label-thresholds entries must be label=value, got {item!r}"
            )
        name, value = (x.strip() for x in item.split("=", 1))
        if name not in label_to_idx:
            raise ValueError(
                f"--label-thresholds contains unknown label {name!r}; "
                f"valid labels are {label_names}"
            )
        try:
            thresholds[label_to_idx[name]] = float(value)
        except ValueError as exc:
            raise ValueError(
                f"--label-thresholds value for {name!r} must be a float, "
                f"got {value!r}"
            ) from exc
    return thresholds


def _freeze_static_groups(
    labels: torch.Tensor, label_names: List[str], static_group_names: List[str],
) -> torch.Tensor:
    """Set instances matched to an absorber label to STATIC (-1), in place."""
    for name in static_group_names:
        labels[labels == label_names.index(name)] = -1
    return labels


def _is_static_path(cameras) -> bool:
    """True when every camera shares one pose, intrinsics and image size."""
    c0 = cameras[0]
    return all(
        torch.allclose(c.cam_to_world, c0.cam_to_world, rtol=1e-5, atol=1e-6)
        and torch.allclose(c.K, c0.K, rtol=1e-5, atol=1e-6)
        and (c.width, c.height) == (c0.width, c0.height)
        for c in cameras
    )


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------


@torch.no_grad()
def _render_group_overlay(
    model,
    camera,
    labels: torch.Tensor,
    cached_inst_idx: torch.Tensor,
    num_groups_with_static: int,
) -> np.ndarray:
    """Render a single-frame coloured-by-group preview.

    Splats inherit their instance's group color. STATIC instances render
    in grey via :func:`util.vis.index_to_color`'s 0-index convention.
    """
    from scion.util.vis import (
        extract_camera_params, index_to_color, rasterize_gaussians,
    )

    bundle = model._get_assembled()
    device = bundle.means.device
    # Shift STATIC (-1) → 0 so index_to_color's "label 0 = gray" applies.
    shifted = (labels + 1).to(device=device, dtype=torch.long)
    splat_labels = shifted[cached_inst_idx]
    colors = index_to_color(splat_labels, num_groups_with_static)

    viewmat, K_mat, W, H = extract_camera_params(
        camera, model._get_downscale_factor(), device,
    )
    bg = model._get_background_color().to(device)
    render, alpha, _ = rasterize_gaussians(
        bundle.means, bundle.quats, bundle.scales, bundle.opacities,
        colors, viewmat, K_mat, W, H,
        rasterize_mode=model.config.rasterize_mode,
    )
    out = (render.squeeze(0) + (1 - alpha.squeeze(0)) * bg).clamp(0, 1)
    return (out.cpu().numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def compute_instance_dino_features(
    model, scene, dino_model: str, resolution: int, cache_dir: Path, device: str,
) -> torch.Tensor:
    """Opacity-weighted per-instance mean of DINO features sampled at every splat.

    Matches the cz construction: fp32 DINO maps for every scene view (train +
    eval), bilinear ``align_corners=False`` sampling at each splat centre,
    equal weight per view, unseen splats left zero. Returns ``[N, D_dino]``.
    """
    from scion.util.animate_codebook import (
        aggregate_splat_features_to_instances,
    )
    from scion.util.dino_init import (
        extract_dino_features_cached, project_features_at_positions,
    )

    cams = scene.cameras
    W, H = cams[0].width, cams[0].height
    if any((c.width, c.height) != (W, H) for c in cams):
        raise ValueError("Scene cameras have mixed image sizes; expected one W x H.")
    feature_maps = extract_dino_features_cached(
        [str(path) for path in scene.image_paths],
        cache_dir=cache_dir,
        model_name=dino_model,
        resolution=resolution,
        device=device,
        bf16_autocast=False,
    )
    bundle = model._get_assembled()
    print(f"Sampling DINO features at {bundle.means.shape[0]:,} splats "
          f"over {len(cams)} views…")
    splat_feats = project_features_at_positions(
        bundle.means,
        [c.viewmat() for c in cams],
        [c.K for c in cams],
        feature_maps,
        W, H,
        device=device,
        align_corners=False,
        fill_unseen=False,
    )
    opa = torch.sigmoid(bundle.opacities.detach().squeeze(-1))
    return aggregate_splat_features_to_instances(
        splat_feats, model._cached_inst_idx.to(device), opa,
        model.instance_positions.shape[0],
    )


def compute_instance_groups(
    model,
    scene,
    dino_model: str,
    resolution: int,
    match_thresholds: List[float],
    label_names: List[str],
    dinotxt_weights: Optional[str],
    cache_dir: Path,
    device: str,
) -> torch.Tensor:
    """Classify each instance against ``label_names`` with DINOtxt.

    Returns ``[N]`` long labels in ``[-1, len(label_names) - 1]`` (-1 = STATIC).
    """
    from scion.util.dinotxt import classify_instances, load_dinotxt

    inst_dino = compute_instance_dino_features(
        model, scene, dino_model, resolution, cache_dir, device,
    )
    print("Loading DINOtxt head…")
    head = load_dinotxt(weights_path=dinotxt_weights, device=device)
    labels, _sim = classify_instances(
        head, inst_dino, label_names, match_threshold=match_thresholds,
    )
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Animate a trained CodebookModel via DINO-text instance groups.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run dir (evals/runs/<slot>/<dataset>/<scene>/seed<N>).")
    parser.add_argument("--level", choices=("raw", "quantized", "codec"), default="raw",
                        help="Compression level whose artifact is animated (default raw).")
    parser.add_argument("--camera-path-filename", type=str, default=None,
                        help="Camera trajectory JSON (nerfstudio viewer export). "
                             "Required unless --eval-camera-index is set.")
    parser.add_argument("--eval-camera-index", type=int, default=None,
                        help="Render every frame from this camera in the "
                             "eval split (pulled via the dataparser). Static "
                             "clip — wind animates, view does not.")
    parser.add_argument("--eval-split", type=str, default="val",
                        choices=("val", "test", "train"),
                        help="Split to draw --eval-camera-index from.")
    parser.add_argument("--output-path", type=str, default="renders/codebook_animated.mp4")

    # Labelling + DINO
    parser.add_argument("--groups", type=str, default=None,
                        help="Comma-separated text labels (e.g. grass,leaves,trunk). "
                             "Mutually exclusive with --single-group.")
    parser.add_argument("--single-group", type=str, default=None,
                        help="Skip DINO/DINOtxt entirely and assign every "
                             "instance to a single group (looked up under "
                             "--anim-config-dir). Useful for a first smoke "
                             "render without a text-aligned head.")
    parser.add_argument("--match-threshold", type=float, default=0.05,
                        help="Min cosine similarity to assign a label; else STATIC.")
    parser.add_argument("--label-thresholds", type=str, default=None,
                        help="Per-label threshold overrides as label=value,... "
                             "(e.g. grass=0.08,flower=0.03). Labels not listed "
                             "use --match-threshold.")
    parser.add_argument("--static-groups", type=str, default=None,
                        help="Comma-separated absorber labels that compete in "
                             "classification but are never animated, e.g. "
                             "sky,dirt,wood.")
    parser.add_argument("--anim-config-dir", type=str,
                        default="config/animation_codebook",
                        help="Directory of per-label animation YAMLs.")
    parser.add_argument("--anim-config-mode", type=str, default="per-label",
                        choices=("per-label", "default"),
                        help="'per-label' uses <label>.yaml with fallback to "
                             "default.yaml; 'default' uses default.yaml for "
                             "every animated label.")
    parser.add_argument("--dino-model", type=str, default="dinov3_vitl16",
                        help="DINO model key (see scion/util/dino_models.py).")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--dinotxt-weights", type=str, default=None,
                        help="Optional local path for the DINOtxt head weights.")

    # Render
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--static-camera-index", type=int, default=None,
                        help="If set, lock the view to this camera index in "
                             "the camera path for the whole render (wind still "
                             "animates, viewpoint does not).")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache dir for the per-image DINO feature maps. "
                             "Defaults to <run-dir>/animate_codebook.")
    parser.add_argument("--preview-groups-image", type=str, default=None,
                        help="Optional path to write a single-frame "
                             "group-overlay PNG before the main render.")
    parser.add_argument("--save-motion-maps", type=str, default=None,
                        metavar="DIR",
                        help="Write per-instance flow maps to DIR: "
                             "instance_flow.mp4 and reference_instance_flow/"
                             "from_first_frame/*.png (white = still, hue = "
                             "direction). Needs a static view: "
                             "--eval-camera-index, --static-camera-index, or "
                             "a single-pose camera path.")
    parser.add_argument("--motion-map-stride", type=int, default=1,
                        help="instance_flow.mp4 shows flow t -> t+stride.")
    parser.add_argument("--motion-map-step", type=int, default=1,
                        help="instance_flow.mp4 start-frame step.")
    parser.add_argument("--motion-map-reference-every", type=int, default=10,
                        help="Reference flow maps 0 -> k, 2k, ... and the last frame.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the per-instance phase/frequency jitter.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.single_group and args.groups:
        raise ValueError("Pass either --groups or --single-group, not both.")
    static_group_names: List[str] = []
    if args.single_group:
        label_names = [args.single_group.strip()]
        print(f"Single-group mode: all instances → {label_names[0]!r} "
              f"(DINO + DINOtxt skipped)")
    else:
        if not args.groups:
            raise ValueError("Pass --groups (comma-separated labels) or --single-group.")
        label_names = [s.strip() for s in args.groups.split(",") if s.strip()]
        if not label_names:
            raise ValueError("--groups must contain at least one label.")
        if args.static_groups:
            static_group_names = [
                s.strip() for s in args.static_groups.split(",") if s.strip()
            ]
            label_names = label_names + static_group_names
        label_thresholds = _parse_label_thresholds(
            args.label_thresholds, label_names, args.match_threshold,
        )
        print("Labels / thresholds: " + ", ".join(
            f"{n}={t:.3f}" for n, t in zip(label_names, label_thresholds)
        ))
        if static_group_names:
            print(f"  Absorber labels (kept static): {static_group_names}")

    run_dir = Path(args.run_dir).resolve()
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir
        else run_dir / "animate_codebook"
    )
    anim_dir = Path(args.anim_config_dir)

    # --- Load model ---
    from scion.engine.checkpoint import load as _ckpt_load
    print(f"Loading {run_dir} (level {args.level})…")
    loaded = _ckpt_load(run_dir, device=args.device, level=args.level)
    model = loaded.model
    step = loaded.step
    print(f"  loaded (step={step}). model={type(model).__name__}")

    # Sanity-check that this is a CodebookModel before touching its internals
    from scion.models.codebook import CodebookModel
    if not isinstance(model, CodebookModel):
        raise TypeError(
            f"animate_codebook.py expects a CodebookModel; got "
            f"{type(model).__name__}."
        )

    # --- Build group assignments ---
    if args.single_group:
        # All instances → group 0 (label_names[0]); no DINO/DINOtxt.
        N = model.instance_positions.shape[0]
        labels = torch.zeros(N, dtype=torch.long, device=args.device)
    else:
        if loaded.scene is None:
            raise RuntimeError(
                "DINOtxt grouping needs the run's scene to sample DINO "
                "features; it could not be loaded. Use --single-group to bypass."
            )
        with torch.no_grad():
            labels = compute_instance_groups(
                model=model,
                scene=loaded.scene,
                dino_model=args.dino_model,
                resolution=args.resolution,
                match_thresholds=label_thresholds,
                label_names=label_names,
                dinotxt_weights=args.dinotxt_weights,
                cache_dir=cache_dir,
                device=args.device,
            )
        _freeze_static_groups(labels, label_names, static_group_names)

    # Print histogram
    counts = {name: int((labels == i).sum().item()) for i, name in enumerate(label_names)}
    counts["STATIC"] = int((labels == -1).sum().item())
    print("Instance group histogram:")
    for name, n in counts.items():
        print(f"    {name:>12s}: {n}")

    # --- Resolve per-label specs (skip labels with zero members) ---
    from scion.util.animate_codebook import (
        GroupAnimationSpec, apply_wind_to_bundle, instance_heights_along_up,
        instance_up_axes, write_video,
    )
    specs_by_label: Dict[int, GroupAnimationSpec] = {}
    for idx, name in enumerate(label_names):
        if counts.get(name, 0) == 0:
            continue
        specs_by_label[idx] = _load_group_spec(
            name, anim_dir, force_default=args.anim_config_mode == "default",
        )
    states_by_label: Dict[int, object] = {idx: None for idx in specs_by_label}

    # --- Cameras: either a viewer-exported path or the scene's eval split.
    # ``cameras`` is normalised to a ``list[scion.Camera]`` so the
    # render loop below can index it uniformly regardless of source.
    if args.eval_camera_index is not None:
        cameras = _scene_split_cameras(loaded, args.eval_split)
        num_cameras = len(cameras)
        print(
            f"Using eval-split '{args.eval_split}' cameras "
            f"({num_cameras} total); locked to index {args.eval_camera_index}"
        )
    else:
        if not args.camera_path_filename:
            raise ValueError(
                "Pass --camera-path-filename or --eval-camera-index."
            )
        cameras = _load_camera_path_json(Path(args.camera_path_filename))
        num_cameras = len(cameras)
    device = args.device
    cached_inst_idx = model._cached_inst_idx.to(device)

    if args.eval_camera_index is not None:
        if not (0 <= args.eval_camera_index < num_cameras):
            raise ValueError(
                f"--eval-camera-index {args.eval_camera_index} out of range "
                f"[0, {num_cameras})."
            )
        fixed_cam_idx = args.eval_camera_index
    elif args.static_camera_index is not None:
        if not (0 <= args.static_camera_index < num_cameras):
            raise ValueError(
                f"--static-camera-index {args.static_camera_index} out of "
                f"range [0, {num_cameras})."
            )
        fixed_cam_idx = args.static_camera_index
        print(f"Static camera: locked to index {args.static_camera_index}")
    elif _is_static_path(cameras):
        fixed_cam_idx = 0
    else:
        fixed_cam_idx = None
    if args.save_motion_maps and fixed_cam_idx is None:
        raise ValueError(
            "--save-motion-maps needs a static camera: pass "
            "--eval-camera-index, --static-camera-index, or a single-pose "
            "camera path."
        )

    if args.preview_groups_image:
        preview_path = Path(args.preview_groups_image)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cam = cameras[fixed_cam_idx or 0].to(device)
        img = _render_group_overlay(
            model, cam, labels, cached_inst_idx,
            num_groups_with_static=len(label_names) + 1,
        )
        from PIL import Image as _Image
        _Image.fromarray(img).save(preview_path)
        print(f"Wrote preview: {preview_path}")

    # --- Render loop ---
    num_frames = max(1, int(args.duration * args.fps))
    print(f"Rendering {num_frames} frames @ {args.fps} fps "
          f"({num_frames / args.fps:.2f}s)")

    # Cache geometry that doesn't change frame-to-frame.
    bundle0 = model._get_assembled()
    inst_pos = model.instance_positions.detach()
    inst_rot = model.instance_rotations.detach()
    inst_log_scales = model.instance_scales.detach().squeeze(-1)
    up_axes = instance_up_axes(inst_rot)
    inst_heights = instance_heights_along_up(
        bundle0.means.detach(), cached_inst_idx, inst_pos, up_axes,
    )
    dt = 1.0 / float(args.fps)

    frames: List[np.ndarray] = []
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Motion maps track opacity-weighted instance centroids per frame.
    traj = None
    if args.save_motion_maps:
        n_inst = inst_pos.shape[0]
        splat_opa = torch.sigmoid(bundle0.opacities.detach().squeeze(-1))
        opa_sum = torch.zeros(n_inst, device=device).index_add_(
            0, cached_inst_idx, splat_opa,
        ).clamp(min=1e-8)
        traj = torch.zeros(n_inst, num_frames, 3, device=device)

    # Seed right before the loop: the wind states (phase / frequency jitter)
    # are drawn on frame 0, so jitter depends only on --seed.
    torch.manual_seed(args.seed)
    with torch.no_grad():
        for frame_idx in tqdm(range(num_frames), desc="Render"):
            cam_idx = (
                fixed_cam_idx
                if fixed_cam_idx is not None
                else frame_idx % num_cameras
            )
            cam = cameras[cam_idx].to(device)
            t = frame_idx / args.fps

            bundle = model._get_assembled()
            orig_means = bundle.means.clone()
            orig_quats = bundle.quats.clone()

            apply_wind_to_bundle(
                bundle,
                cached_inst_idx=cached_inst_idx,
                inst_pos=inst_pos,
                inst_rot=inst_rot,
                inst_heights=inst_heights,
                inst_log_scales=inst_log_scales,
                labels=labels,
                specs_by_label=specs_by_label,
                states_by_label=states_by_label,
                t=t,
                dt=dt,
            )
            if traj is not None:
                traj[:, frame_idx] = torch.zeros_like(traj[:, 0]).index_add_(
                    0, cached_inst_idx, bundle.means * splat_opa[:, None],
                ) / opa_sum[:, None]

            outputs = model.get_outputs(cam)
            rgb = outputs["rgb"].clamp(0.0, 1.0)
            frames.append((rgb.cpu().numpy() * 255).astype(np.uint8))

            # Restore the bundle so the next-frame assembly starts clean
            # (the cache is invalidated by .step changes).
            bundle.means.copy_(orig_means)
            bundle.quats.copy_(orig_quats)

    print(f"Writing video to {args.output_path}…")
    write_video(frames, Path(args.output_path), fps=int(args.fps))
    if traj is not None:
        from scion.util.motion_diagnostics import (
            save_instance_motion_diagnostics,
        )
        print(f"Writing motion maps to {args.save_motion_maps}…")
        save_instance_motion_diagnostics(
            model,
            cameras[fixed_cam_idx].to(device),
            cached_inst_idx,
            traj,
            Path(args.save_motion_maps),
            fps=args.fps,
            flow_frame_stride=args.motion_map_stride,
            flow_frame_step=args.motion_map_step,
            reference_flow_every=args.motion_map_reference_every,
        )
    print("Done.")


if __name__ == "__main__":
    main()
