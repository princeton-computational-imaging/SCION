# Extending density control: custom strategies

Density control (adding, moving and removing splats or instances during
training) is pluggable. The model owns one `CompositeStrategy` with two
slots:

| slot | config key | built-in names |
|---|---|---|
| splat (rows of the shared codebook) | `method.model.splat_strategy` | `mh` (default), `none` |
| instance (world-space placements) | `method.model.instance_strategy` | `default` (default), `none` |

Any other value of the form `package.module:ClassName` is imported and
instantiated with no arguments. What the built-ins do is described in
[codebook_strategies.md](codebook_strategies.md).

## Selecting a strategy

```bash
# my_project/strategies.py defines class MySplatStrategy(CodebookStrategy)
python -m scion.train method=codebook scene=garden \
    method.model.splat_strategy=my_project.strategies:MySplatStrategy
```

- The module must be importable from the training process (installed, or on
  `PYTHONPATH`).
- A custom strategy **replaces** the built-in in that slot. To extend the
  built-in instead, subclass it (`MHSplatStrategy`, `DefaultInstanceStrategy`)
  and call `super()` in the hooks you override.
- Strategy defaults are merged from
  `config/strategies/{splat,instance}/<config_name>.yaml`
  (`scion.util.hydra.apply_strategy_defaults`). Explicit config keys
  win. `config_name` is a class attribute: `"mh"` / `"default"` on the
  built-ins, so their subclasses keep those defaults; `None` on
  `CodebookStrategy`, so a plain custom strategy loads no YAML. Put your
  strategy's own hyperparameters on the class (class attributes or a
  subclass). Read shared knobs such as `splat_refine_start` from
  `model.config`.
- `none` disables a level entirely (used by the `no_splat_dc` / `no_inst_dc`
  ablations).

## The hook contract

Subclass `scion.strategies.CodebookStrategy` and override any of
its three methods. Each returns a value; anything you don't override is a
no-op.

```python
class CodebookStrategy:
    def initialize_state(self, model) -> dict: ...
    def step_pre_forward(self, model, step, state, *, camera=None, batch=None) -> Dict[str, float]: ...
    def step_post_backward(self, model, step, state, *, optimizers=None, lr=0.0) -> Dict[str, float]: ...
```

One training iteration at step `s` runs:

1. `step_pre_forward(model, s, state, camera=..., batch=...)` runs for the
   instance strategy first, then the splat strategy. This is the place for
   structural changes (adding, removing or relocating rows).
   `camera` / `batch` are the previous step's view (`None` on the first
   step). Use them if a move needs to re-render.
2. The trainer applies any pending resize (see below). It re-creates the
   affected `nn.Parameter`s and rebinds their Adam state.
3. Forward (assemble + rasterize), loss, `loss.backward()`, one Adam step
   per parameter group.
4. `step_post_backward(model, s, state, optimizers=..., lr=...)` runs for the
   splat strategy first, then the instance strategy. Gradients and the
   rasterizer's `absgrad` are populated at this point, so this is the place
   to accumulate per-row signals. It runs after the optimizer step, so the
   Adam moments are already updated for step `s`.

`initialize_state(model)` runs once, after the model has been moved to its
training device. Each strategy's dict is merged into **one shared state
dict** that every hook receives. Prefix your keys (for example `my_*`) so
they don't collide with the built-ins (`splat_grad2d_*`, `splat_l1_*`,
`splat_denom`, `inst_grad2d_*`, `grad_ema`).

The returned metric dicts (`{name: float}`) are merged into the step's
logged metrics (as `train/strategy/<name>`). Return `{}` when nothing
happened.

Hooks are called every step, so check your schedule (`step`) yourself.
Decorate them with `@torch.no_grad()` when they edit parameters in place.

## What you may mutate

The flat-buffer layout is described in
[codebook_architecture.md](codebook_architecture.md). In short:

| tensor | shape | meaning |
|---|---|---|
| `splat_means`, `splat_scales` (log), `splat_quats` (wxyz), `splat_opacities` (logit), `splat_features_dc` | `[T_max, ·]` | candidate-local splats; one row per splat |
| `splat_candidate_id` | `[T_max]` long | candidate owning each row, `-1` = free |
| `free_list`, `free_count` | `[T_max]` long, int | stack of free rows |
| `instance_positions`, `instance_rotations` (wxyz), `instance_scales` (log, `[N,1]`), `instance_opacities` (logit) | `[N, ·]` | per-instance transform |
| `instance_assignments` | `[N]` long | candidate of each instance |

Edit **`.data`** only. Never assign a new `nn.Parameter` yourself; the
trainer does that when you flag a resize. After an edit:

| you changed | then |
|---|---|
| splat values (`splat_*.data[rows] = …`) | `model.invalidate_cache()` |
| which rows are live (`splat_candidate_id`, `free_list`, `model.free_count`) | the same; keep all three consistent (`relocate_dead_splats` and `mh_birth_splats` in `scion.ops` show how) |
| instance values in place (same `N`) | `model.invalidate_cache()` |
| the instance count (replaced `instance_*.data` / `instance_assignments` with length `N'`) | set `model._num_instances = N'` and `model._instance_size_changed = True`, then `model.invalidate_cache()`. The trainer rebinds the optimizer and clears `state["grad_ema"]`. Resize any `[N]` accumulators in your state too. |
| the number of candidates `K` | keep `model._num_candidates` in sync, and remap `splat_candidate_id` / `instance_assignments` to `[0, K)` |

`invalidate_cache()` drops the per-step Gaussian cache and rebuilds the
assembly layout. Forgetting it renders, or reverse-maps gradients, against a
stale layout.

## What you may read

| attribute | content |
|---|---|
| `model.config` | the `CodebookModelConfig` (shared knobs such as `splat_refine_*`, `instance_refine_*`, `max_splats_per_candidate`, `max_num_instances`) |
| `model.info["means2d"]` | `[1, M, 2]` projected centres of the `M` Gaussians rasterized in the last forward. After backward, `.absgrad` holds `[1, M, 2]` absolute screen-space gradients. |
| `model._assembly_global_idx` | `[M]` splat-buffer row of each rasterized Gaussian (scatter target for per-row signals) |
| `model._assembly_inst_idx` | `[M]` instance index of each rasterized Gaussian |
| `model._last_assignment_per_gaussian` | `[M]` candidate id of each rasterized Gaussian |
| `model._last_l1_map` | `[H, W]` per-pixel L1 error (mean over RGB) of the last training render, detached |
| `model._num_candidates`, `model._num_instances`, `model._T_max` | `K`, `N`, buffer capacity |
| `model.get_active_rows()`, `model.get_candidate_rows(k)` | live row indices (all, or of candidate `k`) |
| `model._fw_offsets`, `model._fw_sorted_rows` | live rows grouped by candidate: rows of `k` are `_fw_sorted_rows[_fw_offsets[k]:_fw_offsets[k+1]]` |
| `state["splat_grad2d_accum"]` / `["splat_grad2d_count"]`, `["splat_l1_*"]`, `["splat_denom"]` | per-row accumulators maintained by `MHSplatStrategy` (reset at each of its refine ticks) |
| `state["inst_grad2d_accum"]` / `["inst_grad2d_count"]` | per-instance accumulators maintained by `DefaultInstanceStrategy` |
| `state["grad_ema"]` | `{"instance_positions", "instance_scales"}` Adam `exp_avg`, cached by `MHSplatStrategy` |
| `optimizers.optimizers[name]` | the per-group `torch.optim.Adam` (`"splat_means"`, …, `"instance_opacities"`) |

## Worked example

This splat strategy keeps its own screen-space gradient accumulator. At each
reset tick it clamps splat opacities to 0.01, the 3DGS opacity reset, and
logs the mean gradient seen since the last tick.

```python
# my_project/strategies.py
import math
import torch
from scion.ops import accumulate_splat_grad2d
from scion.strategies import CodebookStrategy


class OpacityResetSplatStrategy(CodebookStrategy):
    every, start, stop, value = 3000, 3000, 15000, 0.01   # hyperparameters

    def initialize_state(self, model):
        zeros = lambda: torch.zeros(model._T_max, device=model.splat_means.device)
        return {"my_grad_accum": zeros(), "my_grad_count": zeros()}

    @torch.no_grad()
    def step_post_backward(self, model, step, state, *, optimizers=None, lr=0.0):
        means2d = model.info.get("means2d")
        if means2d is None or getattr(means2d, "absgrad", None) is None:
            return {}
        accumulate_splat_grad2d(means2d.absgrad[0], model._assembly_global_idx,
                                state["my_grad_accum"], state["my_grad_count"])
        return {}

    @torch.no_grad()
    def step_pre_forward(self, model, step, state, *, camera=None, batch=None):
        if not (self.start <= step < self.stop) or step % self.every:
            return {}
        rows = model.get_active_rows()
        cap = math.log(self.value / (1.0 - self.value))
        model.splat_opacities.data[rows] = model.splat_opacities.data[rows].clamp(max=cap)
        model.invalidate_cache()
        grad = state["my_grad_accum"][rows] / state["my_grad_count"][rows].clamp(min=1)
        state["my_grad_accum"].zero_()
        state["my_grad_count"].zero_()
        return {"my_reset/mean_grad2d": float(grad.mean())}
```

Run it with
`method.model.splat_strategy=my_project.strategies:OpacityResetSplatStrategy`.
Because it replaces `MHSplatStrategy`, this run has no splat births. To
keep them, write `class OpacityResetMH(MHSplatStrategy)` and call
`super().step_pre_forward(...)` / `super().step_post_backward(...)` before
your own logic, merging the metric dicts the parent returns. Merge the
parent's `initialize_state` dict with yours too.
