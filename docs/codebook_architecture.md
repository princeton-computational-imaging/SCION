# CodebookModel architecture

This document describes how `CodebookModel` (`scion/models/codebook.py`)
represents a scene and trains it. It covers three levels: **instances**,
**candidates** (codebook entries) and **splats** (Gaussians within a
candidate). Density control, which adds and removes splats and instances,
is documented in [codebook_strategies.md](codebook_strategies.md).

---

## 1. Representation

```
Splats (Gaussians)            ← candidate-local 3D Gaussians
  └─ grouped into Candidates  ← K reusable templates (the codebook)
       └─ placed by Instances ← N world-space placements, one candidate each
```

Each instance has a hard integer assignment to exactly one candidate. A
candidate is a variable-length set of splats. At render time every instance
is expanded into its candidate's splats, transformed to world space and
rasterized as one Gaussian cloud.

### Storage layout

| level | tensors | notes |
|---|---|---|
| splat | `splat_means [T,3]`, `splat_scales [T,3]` (log), `splat_quats [T,4]` (wxyz), `splat_opacities [T,1]` (logit), `splat_features_dc [T,3]` | pre-allocated `[T_max, ·]` buffers, never resized during training |
| splat → candidate | `splat_candidate_id [T_max]` (long, `-1` = free), `free_list [T_max]` + `free_count` | rows are allocated and freed via a LIFO free-list |
| instance | `instance_positions [N,3]`, `instance_rotations [N,4]` (wxyz), `instance_scales [N,1]` (log), `instance_opacities [N,1]` (logit), `instance_assignments [N]` | resized when instances are added or removed |

`T_max = max(1.5·K·S_init + 1024, max_candidates · max_splats_per_candidate)`.
Splat births and relocation write into free rows instead of reallocating, so
the optimizer never sees a splat buffer change shape.

`_rebuild_layout()` derives the views every consumer reads from these
buffers:
- the forward map `_fw_sorted_rows` / `_fw_offsets`, which holds the live
  rows grouped by candidate;
- the per-Gaussian assembly tables `_cand_splat_rows` / `_inst_offsets`.

`invalidate_cache()` calls it after any structural change.

### Assembly

`assemble_gaussians()` (`models/codebook_base.py`) dispatches
`scion.ops.assemble.assemble_fused`, which resolves to the CUDA,
Triton or PyTorch tier, in that order. For instance `i` with assignment `k`
and each splat row `r` of candidate `k`:

```
world_mean  = pos_i + R(rot_i) · (mean_r · exp(scale_i))
world_scale = clamp(scale_i + scale_r, log 1e-4, 5)           # log space
world_quat  = normalize(rot_i ⊗ quat_r)
world_opac  = clamp(opac_i + opac_r, -20, 20)                 # logit space
world_color = features_dc_r                                    # shared by all instances of k
```

The result is cached per step. `_assemble_gaussians()` also records:
- `_assembly_global_idx`: the splat row of each rasterized Gaussian;
- `_assembly_inst_idx`: its instance;
- `_last_assignment_per_gaussian`: its candidate.

Strategies use these to scatter screen-space signals back onto rows and
instances.

---

## 2. Initialization

From the scene's seed point cloud (the COLMAP model's points, or for a
Synthetic-NeRF scene the points COLMAP triangulates from its known poses,
`colmap_points.ply`):

- **Instances.** `num_instances` sets the count: `-1` takes every point,
  `(0, 1]` takes that fraction, and a larger number is an absolute count.
  Each instance sits on a point and starts with identity rotation, unit
  scale and opacity 0.5.
- **Assignments.** `init_assignment_mode` chooses how instances are
  clustered into the `K = num_candidates` candidates:
  - `dino` (default): spherical k-means on DINOv3 features. The features
    come from projecting each instance into the training views
    (`DinoFeatureOracle.from_training_views`; precompute with
    `scripts/preprocess_dino.py`).
  - `position_color`: k-means on position + weighted RGB.
  - `color`: k-means on RGB only.
  - `random_uniform`: IID uniform random assignment.
- **Candidates.** Each starts with `initial_splats_per_candidate` splats.
  - Means, rotations and scales are random in local coordinates, using
    `splat_init_*` and a kNN-scaled size.
  - Opacity is 0.1.
  - The colour is the mean colour of the candidate's instances, plus
    noise.

---

## 3. Training step

Training runs through `scion.engine.Trainer` (`python -m scion.train`
builds it from the Hydra config, `scion.engine.run`). One step `s`:

```
step_pre_forward(s)      instance strategy: cull / subdivide / clone
                         splat strategy:    relocate dead splats + MH births
rebind                   if a resize was flagged: new nn.Parameters, rebind Adam state
forward                  assemble → rasterize (gsplat) → losses (§4)
adversarial step         after discriminator.warmup_step: D update + G loss
backward
optimizer step           one Adam per parameter group (eps 1e-15, no LR schedule)
step_post_backward(s)    splat strategy:    accumulate absgrad / L1 / visit counts, cache Adam EMA
                         instance strategy: accumulate per-instance absgrad
```

`step_post_backward` runs after the optimizer step, so it sees this step's
gradients and the updated Adam moments.

Every splat and instance parameter is trained by gradient descent on every
step. The shipped stack refines splats on every 100th step in
`[500, 25000)` and instances on every 100th step in `[1060, 25000)`. After
step 25000 the structure is fixed and only gradient descent runs.

---

## 4. Losses

| loss | config key (shipped value) | level |
|---|---|---|
| (1 − λ)·L1 + λ·(1 − SSIM) | `ssim_lambda` (0.2) | pixel |
| LPIPS (VGG) | `lpips_lambda` (0.05), every `lpips_every` (4) steps | pixel |
| adversarial (projected VGG discriminator) | `discriminator.loss_weight`, from `discriminator.warmup_step` | patch |
| splat opacity sparsity | `splat_opacity_reg` (0) | splat |
| splat opacity tail (relu² above `splat_opa_logit_cap`) | `splat_opa_cap_weight` (0.1) | splat |
| splat aspect ratio (above `max_gauss_ratio`), every 5 steps | `splat_scale_reg` (0.1) | splat |
| within-candidate compactness | `compactness_reg_weight` (2e-4) | splat |
| instance opacity sparsity | `instance_opacity_reg` (0) | instance |
| instance scale toward 1 | `instance_scale_reg` (0.1) | instance |
| instance scale tail (relu² above `instance_scale_log_cap`) | `instance_scale_cap_weight` (0.1) | instance |

---

## 5. Density control

Density control is pluggable. `CompositeStrategy` holds one splat strategy
and one instance strategy:
- the shipped stack is `MHSplatStrategy` + `DefaultInstanceStrategy`;
- `none` disables a level;
- `pkg.module:Class` selects your own class.

The candidate count `K` stays fixed during training.

- What the shipped strategies do: [codebook_strategies.md](codebook_strategies.md).
- How to write your own: [extending_strategies.md](extending_strategies.md).
