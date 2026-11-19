from __future__ import annotations

import dataclasses

from omegaconf import OmegaConf

from scion.models.codebook import CodebookModelConfig
from scion.util.hydra import apply_strategy_defaults, load_hydra, load_yaml


COMMON = ["variant=ours_small", "dataset=mipnerf360", "scene=garden"]

# Each single-factor arm vs the plain run (the full-method control):
# the only effective CodebookModelConfig fields that may differ.
EXPECTED_DIFF = {
    "no_splat_dc": {"splat_strategy": "none"},
    "no_inst_dc": {"instance_strategy": "none"},
    "clone_only": {"default_inst_enable_subdivide": False},
    "subdiv_only": {"default_inst_enable_clone": False},
    "compact_off": {"compactness_reg_weight": 0.0},
    "init_random": {"init_assignment_mode": "random_uniform"},
    "init_rgb": {"init_assignment_mode": "color"},
    "init_rgb_pos": {"init_assignment_mode": "position_color"},
}


def _effective(ablation: str | None) -> dict:
    """Resolved model config incl. strategy-YAML defaults."""
    overrides = COMMON + ([f"ablation={ablation}"] if ablation else [])
    model = OmegaConf.to_container(load_hydra("codebook", overrides).method.model, resolve=True)
    return dataclasses.asdict(CodebookModelConfig(**apply_strategy_defaults(model)))


def test_single_factor_arms_differ_from_control_only_in_their_factor():
    control = _effective(None)
    # Turning a level off also drops that strategy's YAML defaults (inert).
    inert = {
        "no_splat_dc": set(load_yaml("strategies/splat", "mh")["model"]),
        "no_inst_dc": set(load_yaml("strategies/instance", "default")["model"]),
    }
    for arm, expected in EXPECTED_DIFF.items():
        cfg = _effective(arm)
        diff = {
            k: cfg[k] for k in cfg
            if cfg[k] != control[k] and k not in inert.get(arm, ())
        }
        assert diff == expected, arm


def test_operator_arms_keep_seed_and_schedule():
    cfg = load_hydra(
        "codebook",
        COMMON + ["ablation=no_inst_dc", "machine.seed=0", "trainer.max_num_iterations=30000"],
    )
    assert cfg.machine.seed == 0
    assert cfg.trainer.max_num_iterations == 30000
    assert cfg.method.model.num_candidates == 128
