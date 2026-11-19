# `run.json` schema, version 2

One file per training job, written by `scripts/train_eval_render_one.py`
(the only worker) at

```
evals/runs/<slot>/<dataset>/<scene>/seed<seed>/run.json
```

where `<slot>` is `<variant>[_abl_<ablation>][_<tag>]` (for example
`ours_small`, `ours_small_abl_no_inst_dc`, `ours_small_python`). A job is keyed by
`(variant, ablation, dataset, scene, seed, tag)`; `tag` (default `null`) marks
environment-only differences the Hydra config does not capture (e.g. the
pure-PyTorch backend arm of the compute benchmark); `seed` doubles as the repetition
index. Renderers read these files and nothing else (baselines come from
`evals/baselines/baselines.yaml`). An example lives at
`tests/fixtures/run_json_v2/example_run.json`.

## Fields

| field | type | meaning |
|---|---|---|
| `schema_version` | int | `2` |
| `status` | str | `RUNNING`, `DONE`, `FAILED_TRAIN` or `FAILED_EVAL`. Renderers use `DONE` only. |
| `error` | str \| null | failure message for the `FAILED_*` states |
| `job` | object | `{variant, ablation, dataset, scene, seed, tag}`; `ablation` / `tag` are `null` for the full method |
| `slot` | str | the `<slot>` path component above |
| `overrides` | list[str] | the Hydra overrides the worker passed to training, in order |
| `config_sha256` | str | hex SHA-256 of the resolved Hydra config (`OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)`, UTF-8). The idempotency key: a `DONE` record with the same value is skipped. |
| `git_sha` | str | full 40-char SHA of the last commit that changed *code* (`config/`, `scion/`, `scion-cuda/`, `scripts/`, `experiments/`, `pyproject.toml`, `train.py`, `Makefile`; Markdown excluded), so committing records or docs doesn't invalidate finished jobs (`scion.util.provenance.git_sha`) |
| `git_dirty` | bool \| absent | `true` only for a run made with `--allow-dirty` from a tree with uncommitted tracked changes; absent otherwise. Renderers refuse such records. |
| `gpu` | str | GPU model name, e.g. `NVIDIA GeForce RTX 5090` (no host names or serials) |
| `backend_tiers` | object | resolved tier per dispatched op, e.g. `{"assemble_fused": "cuda", "assemble_fused_bwd": "cuda"}` |
| `versions` | object | `{python, torch, cuda, gsplat, scion, constriction, scikit_learn}` |
| `started_at`, `ended_at` | str | UTC ISO-8601, `YYYY-MM-DDTHH:MM:SSZ` |
| `train_time_s` | float | wall-clock training time (excludes eval and compression) |
| `median_step_s` | float | median wall-clock time of one training iteration |
| `peak_vram_bytes` | int | `torch.cuda.max_memory_allocated()` over training |
| `split` | object | see below |
| `n_gaussians` | int | world-space Gaussians of the trained (raw) model |
| `model_stats` | object | `{num_candidates, active_candidates, num_instances, num_splats, mean_splats_per_candidate, effective_candidates, instances_per_candidate}` (`num_splats` = active rows of the shared splat buffer): `effective_candidates` = exp(entropy of the instance→candidate usage distribution); `instances_per_candidate` = `num_instances` / candidates with ≥ 1 instance (mean reuse) |
| `compression_level` | str | configured level: `raw`, `quantized` or `codec` |
| `metrics` | object | one block per evaluated level, see below |
| `codec` | object \| null | present when `codec` was evaluated, see below |
| `artifacts` | object | run-dir-relative paths, see below |

### `split`

```json
{"eval_mode": "interval", "eval_interval": 8, "train_split_fraction": null,
 "downscale_factor": 4, "n_train": 161, "n_eval": 24,
 "eval_images_sha256": "<hex>", "data_sha256": "<hex>"}
```

`eval_images_sha256` hashes the sorted, newline-joined eval image file names,
so two records with the same value were scored on the same frames.
`data_sha256` hashes the bytes of the scene's camera model (COLMAP
`sparse/0/*.bin`, or Blender `transforms*.json`) and `colmap_points.ply` when
present (each prefixed by its name and size, sorted by name;
`scion.util.provenance.data_sha256`): the poses, intrinsics and seed cloud
training read. The images themselves are **not** hashed
(reading every image of every scene per run costs too much), so a re-encoded
image with an unchanged name goes undetected. Renderers refuse a table that
mixes either hash within one scene. `eval_mode` is `blender_test` for
Synthetic scenes; `train_split_fraction` is set only for `eval_mode: fraction`.

### `metrics`

Every level up to the configured one is evaluated on the model **decoded from
that level's artifact** (`raw` < `quantized` < `codec`), so a `codec` run has
all three blocks:

```json
"metrics": {
  "raw":       {"psnr": 21.86, "ssim": 0.553, "lpips": 0.367, "size_bytes": 2625000, "bpg": 12.08, "fps": 1.01, "n_gaussians": 1738894},
  "quantized": {...},
  "codec":     {...}
}
```

| key | meaning |
|---|---|
| `psnr`, `ssim`, `lpips` | mean over the eval split, scored like 3DGS's `metrics.py` (`scion.engine.eval.image_metrics`): SSIM with an 11x11 Gaussian window and zero padding, LPIPS-VGG on [0, 1] inputs |
| `size_bytes` | byte length of the file a user downloads for this level (`artifacts.checkpoints.<level>`) |
| `bpg` | bits per Gaussian, `size_bytes * 8 / n_gaussians` (`scion.util.units.bits_per_gaussian`) |
| `fps` | render-only throughput over the eval split, frames per second (excludes metrics and image writes) |
| `n_gaussians` | world-space Gaussians of the decoded model at this level |
| `by_subset` | only when the split names test groups (`eval_mode: split`, e.g. the campus `id` / `sideways` / `opposite`): `{<group>: {psnr, ssim, lpips, n_views}}`, each group's mean from the same renders; the level's `psnr` / `ssim` / `lpips` stay the mean over all eval views |

Sizes are bytes in the record. Tables show MB with **1 MB = 2^20 bytes**
(`scion.util.units.MiB`), labelled "MB (2^20 bytes)". Flattened
records (`scion.util.run_index`) use `<level>.<metric>` keys, e.g.
`quantized.psnr`, `codec.size_bytes`, and `<level>.by_subset.<group>.<metric>`.

### `codec`

```json
"codec": {
  "options": {"M": 3, "B": 256, "position_format": "tmc13", "svq_instance_attrs": true, "...": "dataclasses.asdict(EncodeOptions)"},
  "position_format_used": "tmc13",
  "tmc13_version": "release-v23.0-rc2 a3d15c5"
}
```

`position_format_used` (`tmc13` or `fp16`) is what was actually written, so a record can never
claim G-PCC while holding a fallback. Level `codec` fails loudly when `tmc3`
is missing; it never falls back silently.

### `artifacts`

```json
"artifacts": {
  "checkpoints": {"raw": "checkpoints/raw.ckpt", "quantized": "checkpoints/quantized.ckpt", "codec": "codec.omg"},
  "eval_views":  {"raw": "eval_views/raw", "quantized": "eval_views/quantized", "codec": "eval_views/codec"},
  "templates":   "candidates",
  "train_log":   "train.log"
}
```

Run directories are outputs and gitignored.

