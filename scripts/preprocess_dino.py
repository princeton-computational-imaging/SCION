"""Run DINOv3 over a scene's training images and cache per-image features.

Pre-processing step for codebook training: fills exactly the cache that
training reads (``data/{scene}/dino/``), keyed by the train split, DINO
model, resolution, PCA dim and downscale factor. The split and DINO
settings come from the Hydra config of ``scene=<scene> dataset=<dataset>``;
``--dataset`` defaults to the dataset that lists the scene in
``experiments/paper/targets.yaml``. A Synthetic-NeRF scene without seed
points gets them first (COLMAP, :mod:`scion.data.synthetic`).
Idempotent: skips cached scenes.

    python scripts/preprocess_dino.py --scene flowers
    python scripts/preprocess_dino.py --scene lego --resolution 832 1664 2496
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


from scion.data import load_scene  # noqa: E402
from scion.engine.run import ensure_synthetic_seed, scene_kwargs  # noqa: E402
from scion.util.dino_init import train_feature_maps  # noqa: E402
from scion.util.provenance import compose  # noqa: E402


def registry_dataset(scene: str):
    """The registry dataset listing ``scene`` (None when unlisted)."""
    datasets = yaml.safe_load((REPO / "experiments" / "paper" / "targets.yaml").read_text())["datasets"]
    hits = [name for name, ds in datasets.items() if scene in ds["scenes"]]
    return hits[0] if len(hits) == 1 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True, help="Scene name (data/{scene}/)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--dataset", default=None,
                    help="config/dataset/<name>.yaml (split + downscale); "
                         "default: the registry dataset of --scene")
    ap.add_argument("--dino-model", default=None, help="default: method.model.dino_direct_model")
    ap.add_argument("--resolution", type=int, nargs="+", default=None,
                    help="Long-side resolutions (each writes a separate cache file); "
                         "default: method.model.dino_direct_resolution.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-cache", action="store_true",
                    help="Force re-extraction even if cache exists.")
    args = ap.parse_args()

    dataset = args.dataset or registry_dataset(args.scene)
    cfg = compose(["method=codebook", f"scene={args.scene}"]
                  + ([f"dataset={dataset}"] if dataset else []))
    model_cfg = cfg.method.model
    dino_model = args.dino_model or model_cfg.dino_direct_model
    resolutions = args.resolution or [int(model_cfg.dino_direct_resolution)]

    scene_dir = Path(args.data_root) / args.scene
    # Synthetic scenes: triangulate the seed points here, once per scene before
    # its jobs (the runner runs this step first), so concurrent jobs don't each run COLMAP.
    ensure_synthetic_seed(scene_dir)
    scene = load_scene(scene_dir, **scene_kwargs(cfg))
    image_paths = [str(scene.image_paths[i]) for i in scene.train_indices]

    cache_dir = scene_dir / "dino"
    cache_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = {
        "schema_version": 1,
        "scene": args.scene,
        "dataset": dataset,
        "split": "train",
        "image_count": len(image_paths),
        "image_names_sha256": hashlib.sha256(
            "\n".join(sorted(Path(p).name for p in image_paths)).encode("utf-8")
        ).hexdigest(),
        "image_paths": image_paths,
        "dino_model": dino_model,
        "resolutions": list(resolutions),
    }
    manifest_path = cache_dir / "input_manifest.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(input_manifest, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)

    for resolution in resolutions:
        print(f"[preprocess_dino] scene={args.scene} dataset={dataset} model={dino_model} "
              f"res={resolution} train images={len(image_paths)}")
        train_feature_maps(
            scene, scene_dir, model_name=dino_model, resolution=resolution,
            pca_dim=int(model_cfg.dino_direct_pca_dim), device=args.device,
            no_cache=args.no_cache,
        )


if __name__ == "__main__":
    main()
