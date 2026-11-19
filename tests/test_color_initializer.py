from types import SimpleNamespace
from unittest.mock import Mock

import torch
from omegaconf import OmegaConf

from scion.models.codebook import CodebookModel, CodebookModelConfig
from scion.util.hydra import load_hydra


def test_color_initializer_routes_only_rgb_to_euclidean_clustering():
    colors = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]])
    assignments = torch.tensor([0, 1])
    fake_model = SimpleNamespace(
        _cluster_features=Mock(return_value=assignments),
    )

    result = CodebookModel._cluster_color(fake_model, colors, K=2)

    assert result is assignments
    fake_model._cluster_features.assert_called_once_with(
        colors, 2, spherical=False
    )


def test_color_ablation_differs_from_control_only_by_assignment_mode():
    common = [
        "variant=ours_small",
        "dataset=mipnerf360",
        "scene=garden",
        "machine.seed=2",
        "trainer.max_num_iterations=30000",
    ]
    control = load_hydra("codebook", common)
    color = load_hydra("codebook", common + ["ablation=init_rgb"])
    control_model = OmegaConf.to_container(control.method.model, resolve=True)
    color_model = OmegaConf.to_container(color.method.model, resolve=True)

    differing = {
        key: (control_model.get(key), color_model.get(key))
        for key in control_model.keys() | color_model.keys()
        if control_model.get(key) != color_model.get(key)
    }
    assert differing == {"init_assignment_mode": ("dino", "color")}
    assert color.machine.seed == 2
    assert color.trainer.max_num_iterations == 30000
    CodebookModelConfig(**color_model)  # every key is a real field
