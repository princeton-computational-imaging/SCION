# Reproducing the paper

One command regenerates every SCION number and figure panel in the paper:

```bash
make dry-run                    # what would run; checks every job's config composes
make reproduce                  # local GPUs: setup -> train/eval jobs -> tables + figures
make reproduce BACKEND=slurm    # same jobs as a Slurm array, then a render job
make render                     # tables + figures from existing records, no training
make tables                     # tables only, on CPU
```

Variables: `TARGETS` (default `all`), `BACKEND`, `GPUS`, `JOBS_PER_GPU`,
`PAPER_DIR`, and `ARGS` for other flags such as `--allow-dirty`. Each target
runs `python -m experiments.paper.reproduce` (`--help` lists its flags). Tables land in `<paper-dir>/tables/*.tex` and
figure panels in `<paper-dir>/figures/` (default `paper_artifacts/`,
gitignored). Baseline rows are not reproduced: they are quoted from the
literature or from our reruns of the baselines' code, and every entry in
`evals/baselines/baselines.yaml` cites its source.

## Targets

A target is a paper item: a set of training jobs plus a renderer (tables) or a
generator command (figures). The registry is
[`experiments/paper/targets.yaml`](../experiments/paper/targets.yaml); table
layouts are in [`experiments/paper/LAYOUTS.md`](../experiments/paper/LAYOUTS.md).
A job is keyed by `(variant, ablation, dataset, scene, seed, tag)`; a job shared
by several targets runs once. Every cell is mean ± std over seeds 0, 1, 2.

| target | paper item | kind | jobs |
|---|---|---|---:|
| `main_results` | `tab:main_results`, per-scene tables, `tab:levels` | table | 126 |
| `matched_budget` | `tab:matched_budget` (our row) | table | 27 |
| `component_ablation` | `tab:component_ablation` (a → b → c → full) | table | 24 |
| `capacity` | `tab:capacity_ablation` + `fig:rd` | table | 21 |
| `init_ablation` | `tab:init` | table | 12 |
| `operator_ablation` | `tab:operators` | table | 15 |
| `compactness_ablation` | `tab:compactness` | table | 12 |
| `ood` | `tab:ood` (campus ID / sideways / opposite, our rows) | table | 6 |
| `compute` | `tab:compute` (CUDA vs PyTorch `assemble_fused`) | table | 6 |
| `paper_numbers` | inline numbers → `tables/numbers.tex` | table | 150 |
| `qual_compression` | `fig:compression_eval` | figure | 2 |
| `ablation_renders` | `fig:component_ablation_renders` | figure | 4 |
| `supp_qualitative` | `fig:supp_qualitative` | figure | 42 |
| `primitives` | learned primitive renders | figure | 1 |
| `figure1_panels` | `fig:overview` and `fig:process` insets | figure | 1 |
| `animation` | `fig:animation_demo` flow maps | figure | 1 |

`all` is 207 distinct jobs of 30k steps each, so the full matrix is a
cluster-sized job. For scale: garden `ours_small` trained in about 36 minutes
with two jobs sharing one RTX 5090 and peaked at 6.4 GiB of allocated memory; larger
scenes and `ours_large` take more. The `compute` target measures this
properly. Table headline numbers use compression level `quantized` plus a
`codec` row (labelled `\wparam{}`, "w/ param.", as in the paper); every run evaluates `raw`, `quantized` and `codec` (see
[compression.md](compression.md)).

## Setup

- A CUDA GPU with ≥ 12 GB and the environment from the README: `pixi install`,
  or `pip install -e .` plus `pip install -e scion-cuda/ --no-build-isolation` for the CUDA kernels
  (the `compute` target measures both tiers) and `colmap` on `PATH` for
  Synthetic-NeRF.
- Optional `reproduce.env` (copy `reproduce.env.example`): `DATA_ROOT`,
  `TMC13_BINARY` and the `SLURM_*` settings. The Makefile reads it.

Everything else is prepared on first need, only for scenes a scheduled job uses:

| step | how | notes |
|---|---|---|
| scene data | `scripts/download_data.sh <dataset>` | archives are kept in `$DATA_ROOT/_archives/`, downloaded when missing, and SHA256-checked (a mismatch stops the run) |
| DINO cache | `scripts/preprocess_dino.py --scene S` | once per scene before its jobs, on a GPU |
| `tmc3` | `scripts/setup/build_tmc13.sh` | when a job's compression level is `codec`; the pinned MPEG TMC13 `release-v23.0-rc2` (needs git, cmake, a C++ compiler) |

| archive | source | SHA256 |
|---|---|---|
| `360_v2.zip` (Mip-NeRF 360) | downloaded from the [official page](https://jonbarron.info/mipnerf360/) | `77332bf4eba3b8ca0c7f70130849b1e394efdd60d8f20efa6f217081d08a8b2a` |
| `360_extra_scenes.zip` (Mip-NeRF 360 `flowers`, `treehill`) | downloaded, same page | `f8d42b372d7cc589928c3d22849f958c1e2ae948c21e89c6a32c904cefba2fa4` |
| `tandt_db.zip` (Tanks & Temples, Deep Blending) | downloaded ([3DGS bundle](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)) | `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1` |
| `nerf_synthetic.zip` (Synthetic-NeRF) | **manual**: [NeRF Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), saved in `$DATA_ROOT/_archives/` | `554fed5d4884028f95e0da422f99f5fc071fffefc9f2fc72c5f9361801df2599` |

The pins live in `scripts/download_data.sh`. Scenes end up at
`$DATA_ROOT/<scene>/` (default `data/`) in the layout the archives ship
(README, Data); when `DATA_ROOT` is elsewhere, the repo's `data` becomes a
symlink to it, because the configs read `data/<scene>`. A Synthetic-NeRF
scene gets its seed points (`colmap_points.ply`) before its first job: COLMAP
triangulates SIFT matches across the training views with the dataset's poses
held fixed (`scion.data.synthetic`, CPU, about two minutes per scene). The
Synthetic-NeRF rows of HAC++ and ContextGS come from their code started from
the same points (copied to `points3d.ply`) rather than from random points.

## Local backend (default)

```bash
make reproduce TARGETS=compute                  # first, alone on the machine
make reproduce GPUS=0,1 JOBS_PER_GPU=2
```

Run `compute` on its own first: its CUDA-vs-PyTorch wall-clock, step-time and
VRAM numbers are only meaningful when nothing else shares the GPU. The later
`all` run skips those six jobs.

A pool over `GPUS` (default: every visible GPU) with `JOBS_PER_GPU`
concurrent jobs per card (2 fit on a 32 GB card). Each job is one run of
`scripts/train_eval_render_one.py`, retried once on failure, and counts as done
only when its `run.json` is `DONE` with the expected `config_sha256` and
`git_sha`. The log shows one line per job start / finish; per-job output goes to
`evals/logs/<slot>/<dataset>/<scene>/seed<N>.log` (DINO steps:
`evals/logs/dino/<scene>.log`). Ctrl-C stops scheduling new jobs; re-running
the same command continues. The run ends with a summary (done / skipped /
failed) and the render phase, and exits non-zero if any job failed.

## Slurm backend

```bash
make reproduce BACKEND=slurm
```

Needs `SLURM_GRES` and `SLURM_TIME`; `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_CPUS` (cpus-per-task) and `SLURM_MEM` are
passed when set, and `SLURM_MAXPAR` (default 16) caps running array tasks.
There are no cluster defaults. Data and `tmc3` are prepared on the submitting
host (put the repo and `DATA_ROOT` on a shared filesystem). It writes
`evals/slurm/<timestamp>/` and submits three jobs, printing their IDs:

1. `dino.sbatch`: an array, one GPU task per scene, building the DINO cache;
2. `jobs.sbatch`: an array over `jobs.txt` (one worker invocation per line,
   with its env), each task retried once, after (1);
3. `render.sbatch`: a CPU job running the render phase after (2).
   It writes the tables and prints the figure generators; run
   `make render` on a GPU host afterwards for the panels.

Task logs go to `evals/slurm/<timestamp>/logs/`.

## Idempotency and provenance

A job is skipped when its `evals/runs/<slot>/<dataset>/<scene>/seed<N>/run.json`
(schema: [`evals/RUN_SCHEMA.md`](../evals/RUN_SCHEMA.md)) is `DONE` **and**
records the same `config_sha256` (the resolved Hydra config) **and** the same
`git_sha` as the current checkout. Anything else reruns, so re-running after a
crash continues where it stopped.

`git_sha` and the dirty-tree check cover **code paths only**: `config/`,
`scion/`, `scion-cuda/`, `scripts/`, `experiments/`,
`pyproject.toml`, `train.py` and `Makefile`, with Markdown excluded
(`scion.util.provenance`). `git_sha` is the last commit that changed
one of them. Committing docs therefore invalidates no finished job, while any code commit reruns every job: tag the commit before a full run.
`make reproduce` refuses to train when those paths hold uncommitted changes or
untracked files, because the records would claim a `git_sha` that does not
describe the code. `ARGS=--allow-dirty` overrides this with a warning; its records
carry `git_dirty: true`, are never skipped as done, and the renderers refuse
them.
The renderers also refuse tables that mix `git_sha` values, eval splits, scene
data (`split.data_sha256`: the scene's camera model and seed points, not
its images), or incomplete seeds. A figure generator runs only when every
one of its input runs is `DONE`, clean and made from the current config, and
all of them share one `git_sha` (it need not be the current one, so finished
records still render after later code commits); otherwise the render phase
names each blocking record and why.

Run records are outputs: `evals/runs/` is gitignored, and the repository
ships none.

## Camera paths

`scripts/animate_codebook.py` renders from one of the run's test cameras
(`--eval-camera-index N`) or from a camera path exported from the viewer.
`evals/camera_paths/` holds two:

- `flowers.json`: the single view of the paper's animation figure (the
  `animation` target uses it).
- `garden_teaser.json`: the overview orbit of the teaser video (garden, 10 s,
  1920x1080, 30 fps); the video's opening close-ups are not part of it.

```bash
python scripts/animate_codebook.py --run-dir <run dir> \
    --camera-path-filename <path-to-camera-path.json> --groups grass,leaves ...
```

The camera path is a JSON file in the nerfstudio viewer's export format, in the
trained model's coordinate frame (the frame `load_scene` builds from the
scene's COLMAP model):

```json
{
  "render_width": 1280,
  "render_height": 720,
  "camera_path": [
    {"camera_to_world": [16 floats: a row-major 4x4 matrix], "fov": 50.0},
    ...
  ]
}
```

Each entry is one frame. The pose uses OpenGL axes (x right, y up, z back),
as `scion.camera.Camera` does. `fov` is the vertical field of view in degrees
(default 50) and the principal point is the image centre. Other keys the viewer
writes (`camera_type`, `aspect`, ...) are ignored; every camera is rendered as
a pinhole camera.

## What stays manual

- Figure composites. Generators (`figure1_panels`, `primitives`,
  `ablation_renders`, `animation`) run in the render phase once their input
  runs are `DONE` and a GPU is visible; otherwise the command is printed.
  Layout, crops and insets are assembled by hand; each figure target's
  `manual:` note in `targets.yaml` says what (e.g. the ablation view and ROIs
  were picked by hand). `qual_compression` and `supp_qualitative` have no
  generator: crops come from `eval_views/<level>/` of their runs.
- Baseline numbers (`evals/baselines/baselines.yaml`): nothing here reruns them.
- Copying `tables/*.tex` into the paper: pass `--paper-dir <paper checkout>`.
