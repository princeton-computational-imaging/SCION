# Codebook density-control strategies

Reference for the shipped strategies: `MHSplatStrategy`,
`DefaultInstanceStrategy` and the `CompositeStrategy` that runs them. To
write your own, see [extending_strategies.md](extending_strategies.md). For
the flat-buffer data layout, see
[codebook_architecture.md](codebook_architecture.md).

## Overview

```
CodebookStrategy                  (strategies/base.py)
 ├── MHSplatStrategy              (strategies/splat.py)     splat_strategy: mh
 ├── DefaultInstanceStrategy      (strategies/instance.py)  instance_strategy: default
 └── CompositeStrategy            (strategies/composite.py) one splat + instance slot(s)
```

| level | `mh` / `default` | `none` | `pkg.module:Class` |
|---|---|---|---|
| splat (`method.model.splat_strategy`) | `MHSplatStrategy` | no splat density control | user class |
| instance (`method.model.instance_strategy`) | `DefaultInstanceStrategy` | no instance density control | user class |

The name → class table lives in `CodebookModel._populate_strategies`
(`models/codebook.py`). Each class's `config_name` (`"mh"`, `"default"`)
names the defaults YAML `config/strategies/{splat,instance}/<config_name>.yaml`,
which `util.hydra.apply_strategy_defaults` merges in. Keys set in
`config/method/codebook.yaml`, a variant, dataset or ablation override
them.

### Step order

```
step_pre_forward:    instance strategy (cull / clone / subdivide)
                     splat strategy    (relocate dead + MH births)
(trainer rebinds optimizer state if a resize was flagged)
forward → loss → backward
step_post_backward:  splat strategy    (accumulate absgrad / L1 / visits, cache Adam EMA)
                     instance strategy (accumulate per-instance absgrad)
```

Both strategies share one state dict (see the hook contract in
[extending_strategies.md](extending_strategies.md)).

## MHSplatStrategy

Splat density control within each candidate, after MH-3DGS (Kim, Jung &
Park, 2025) with 3DGS-MCMC-style relocation (Kheradmand et al., 2024). It runs on every `splat_refine_every`-th step in
`[splat_refine_start, splat_refine_stop)`.

1. **Relocate dead splats** (`ops.relocate_dead_splats`). Rows with
   `sigmoid(opacity) < splat_min_opacity` are overwritten with copies of live
   splats of the same candidate, so the per-candidate splat count does not
   change. The source and its copies use the mass-preserving update
   `o' = 1 − (1 − o)^(1/n)`, `s' = s / n^(1/3)`.
2. **Importance-guided births** (`ops.mh_birth_splats`). Each live splat
   gets the importance `sigmoid(w_opa·z(opa) + w_grad·z(grad) + w_l1·z(L1))`,
   z-scored per candidate. `grad` is the mean 2D absgrad and `L1` is the
   per-pixel L1 error sampled at the splat's projected centre, both
   accumulated since the last tick. Per candidate, a coarse and a fine
   batch of children are proposed around importance-sampled parents. The
   proposal std is `scale · parent_scale / (1 + log(1 + visits))`. A child
   is accepted with probability
   `sigmoid(importance − splat_birth_accept_threshold) / (1 + voxel_alpha · V)`,
   where `V` counts same-candidate splats already in the child's voxel.
   Proposal scales, batch sizes and voxel size anneal linearly from their
   max to their min over the refine window. Each candidate is capped at
   `max_splats_per_candidate`.
3. Accumulators are reset after each tick.

`step_post_backward` fills the per-row accumulators (`splat_grad2d_*`,
`splat_l1_*`, `splat_denom`) and caches the Adam `exp_avg` of
`instance_positions` / `instance_scales` into `state["grad_ema"]`.

| key | default | where |
|---|---|---|
| `splat_refine_start` / `splat_refine_every` / `splat_refine_stop` | 500 / 100 / 25000 | `strategies/splat/mh.yaml` |
| `splat_min_opacity` | 0.01 | `strategies/splat/mh.yaml` |
| `splat_birth_accept_threshold` | 0.5 | `strategies/splat/mh.yaml` |
| `mh_splat_w_opa` / `mh_splat_w_grad` / `mh_splat_w_l1` | 0.8 / 0.5 / 0.5 | `CodebookModelConfig` |
| `mh_splat_coarse_scale_max` → `mh_splat_coarse_scale_min` | 10 → 5 | `CodebookModelConfig` |
| `mh_splat_fine_scale_max` → `mh_splat_fine_scale_min` | 2 → 1 | `CodebookModelConfig` |
| `mh_splat_coarse_batch_per_cand` / `mh_splat_fine_batch_per_cand` | 2 / 6 (annealed to 0) | `CodebookModelConfig` |
| `mh_splat_voxel_alpha` | 5.0 | `CodebookModelConfig` |
| `mh_splat_voxel_size_max` → `mh_splat_voxel_size_min` | 0.05 → 0.005 | `CodebookModelConfig` |
| `mh_splat_use_voxel_penalty` / `mh_splat_use_density_factor` | true / true | `CodebookModelConfig` |
| `max_splats_per_candidate` | 128 | `config/method/codebook.yaml` |

## DefaultInstanceStrategy

Gradient-driven instance density control, the 3DGS ADC rule applied to
instances. `step_post_backward` accumulates each instance's mean 2D
absgrad over its rasterized splats. If that signal is missing, the Adam
EMA "urgency" is used instead. On every `instance_refine_every`-th step in
`[instance_refine_start, instance_refine_stop)`, `step_pre_forward` does:

1. **Cull** instances with `sigmoid(opacity) < default_inst_prune_opa`, and
   any instance with a non-finite parameter.
2. **Subdivide** high-gradient instances (`avg_grad > default_inst_grow_grad`)
   with `exp(scale) ≥ default_inst_grow_scale`. Each is replaced by
   `default_inst_subdivide_M` children at jittered positions. Children keep
   the parent's opacity and shrink scale by `M^(1/3)`, because they sit at
   disjoint positions and are not α-composited. At most
   `default_inst_subdivide_max_parents` parents per tick.
3. **Clone** high-gradient instances with `exp(scale) < default_inst_grow_scale`,
   keeping their opacity. At most `default_inst_clone_max_batch` per tick,
   up to `max_num_instances` in total.

Instances whose log-scale exceeds `default_inst_max_scale_log` are never
grown. `default_inst_enable_clone` / `default_inst_enable_subdivide` gate
steps 3 and 2 (used by the `clone_only` / `subdiv_only` ablations).

| key | default | where |
|---|---|---|
| `instance_refine_start` / `instance_refine_every` / `instance_refine_stop` | 1060 / 100 / 25000 | `strategies/instance/default.yaml` |
| `default_inst_prune_opa` | 1e-3 | `strategies/instance/default.yaml` |
| `default_inst_grow_grad` | 1e-4 | `strategies/instance/default.yaml` |
| `default_inst_grow_scale` | 1.0 | `strategies/instance/default.yaml` |
| `default_inst_enable_clone` / `default_inst_enable_subdivide` | true / true | `strategies/instance/default.yaml` |
| `default_inst_subdivide_M` | 2 | `CodebookModelConfig` |
| `default_inst_subdivide_max_parents` | 16 | `CodebookModelConfig` |
| `default_inst_subdivide_noise_scale` | 0.2 | `CodebookModelConfig` |
| `default_inst_clone_max_batch` | 512 | `CodebookModelConfig` |
| `default_inst_clone_noise_scale` | 0.1 | `CodebookModelConfig` |
| `default_inst_clone_opa_offset` | 0.0 | `CodebookModelConfig` |
| `default_inst_max_scale_log` | 4.0 | `CodebookModelConfig` |
| `max_num_instances` | 1e6 | `config/method/codebook.yaml` |

## CompositeStrategy

```python
CompositeStrategy(
    splat: Optional[CodebookStrategy],
    instance: Optional[Union[CodebookStrategy, List[CodebookStrategy]]],
)
```

`None` slots are skipped. The instance slot accepts a list; the shipped
config uses a single strategy. `initialize_state` merges every
strategy's state into one dict.
