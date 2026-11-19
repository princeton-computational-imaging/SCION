"""Training must feed the per-pixel L1 map into MH splat births; without it
the L1 term of the MH importance is silently dropped.
"""
import torch

from scion.strategies.splat import MHSplatStrategy
from test_dispatch_wired import _make_tiny_codebook_model


def test_mh_birth_receives_l1_accumulator():
    model = _make_tiny_codebook_model()
    cfg = model.config
    cfg.lpips_lambda = 0.0
    cfg.use_scale_regularization = False
    cfg.compactness_reg_weight = 0.0
    model.ssim = lambda a, b: torch.tensor(1.0)
    model.device_indicator_param = torch.nn.Parameter(torch.empty(0))
    model._get_downscale_factor = lambda: 1
    model.train()

    H, W = 8, 8
    outputs = {"rgb": torch.zeros(H, W, 3), "background": torch.zeros(3)}
    model.get_loss_dict(outputs, {"image": torch.ones(H, W, 3)})
    assert model._last_l1_map is not None and model._last_l1_map.sum() > 0

    strategy = MHSplatStrategy()
    state = strategy.initialize_state(model)
    model._assemble_gaussians()  # sets _assembly_global_idx
    total = model._assembly_global_idx.numel()
    model.info = {"means2d": torch.full((total, 2), 3.0)}  # all on screen
    strategy.step_post_backward(model, 1, state)
    assert state["splat_l1_accum"].sum() > 0
