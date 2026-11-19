"""Trainer: one step against a frozen snapshot, checkpoint save -> load ->
resume, fail-fast on non-finite losses.

CPU only: ``_tiny_run.fake_rasterize`` replaces gsplat (CUDA-only); the
model, losses, LPIPS, discriminator and strategies are the real ones.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import scion.models.codebook as codebook_mod
from scion.engine.trainer import Trainer, TrainerConfig

from _tiny_run import LRS, fake_rasterize, tiny_model, tiny_scene

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _cpu_rasterizer(monkeypatch):
    monkeypatch.setattr(codebook_mod, "_rasterize", fake_rasterize)


def _trainer(model, scene, tmp_path, max_steps=10, config_sha256=None):
    t = Trainer(model, model.codebook_strategy, scene,
                TrainerConfig(max_steps=max_steps, eval_every=0, save_every=0, device="cpu",
                              output_dir=tmp_path, vis="none"), lrs=LRS,
                config_sha256=config_sha256)
    t.setup()
    return t


def test_one_step_matches_snapshot(tmp_path):
    """Fixed model / camera / batch / background -> the recorded loss dict and
    gradient statistics (adversarial step included); exact on the machine the
    snapshot was taken on. Reference: tests/fixtures/snapshot/one_step.json."""
    ref = json.loads((REPO / "tests" / "fixtures" / "snapshot" / "one_step.json").read_text())
    trainer = _trainer(tiny_model(), tiny_scene(tmp_path), tmp_path)
    # One Adam(eps=1e-15) per parameter group, no LR schedule.
    assert {g["eps"] for o in trainer.optimizers.values() for g in o.param_groups} == {1e-15}
    assert trainer.optimizers.keys() == LRS.keys()
    torch.manual_seed(7)
    losses, _ = trainer.step(0)
    losses = {k: float(v) for k, v in losses.items() if k != "loss"}
    assert losses.keys() == ref["losses"].keys() and {"adv_loss", "lpips_loss"} <= losses.keys()
    for k, v in ref["losses"].items():
        assert losses[k] == pytest.approx(v, rel=1e-5, abs=1e-8), k
    grads = {n: p.grad.double() for n, p in trainer.model.named_parameters() if p.grad is not None}
    assert grads.keys() == ref["grads"].keys()
    for n, (s, a, norm) in ref["grads"].items():
        g = grads[n]
        assert [float(g.sum()), float(g.abs().sum()), float(g.norm())] == pytest.approx(
            [s, a, norm], rel=1e-4, abs=1e-9), n


def _state(t: Trainer):
    m = t.model
    params = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = {name: copy.deepcopy(o.state_dict()["state"]) for name, o in t.optimizers.items()}
    return params, opt, (m.T_max, m.free_count, m.num_candidates, m.num_instances)


def _assert_same(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_checkpoint_resume_continues_identically(tmp_path):
    scene = tiny_scene(tmp_path, n=5)               # 3 steps leave the loader mid-epoch
    init = tiny_model()

    first = _trainer(copy.deepcopy(init), scene, tmp_path / "a", max_steps=3, config_sha256="cfg")
    first.fit()                                   # steps 0..2; saves step 2 at the end
    ckpt = next((tmp_path / "a" / "checkpoints").glob("step-*.ckpt"))
    assert ckpt.name == "step-000000002.ckpt" and (ckpt.parent / "meta.json").exists()
    meta = json.loads((ckpt.parent / "meta.json").read_text())
    assert meta["config_sha256"] == "cfg" and len(meta["git_sha"]) == 40
    assert isinstance(meta["git_dirty"], bool)
    saved = _state(first)

    other = _trainer(copy.deepcopy(init), scene, tmp_path / "c", max_steps=4, config_sha256="changed")
    with pytest.raises(ValueError, match="config_sha256 cfg"):
        other.resume(ckpt)

    second = _trainer(copy.deepcopy(init), scene, tmp_path / "b", max_steps=4, config_sha256="cfg")
    assert second.resume(ckpt) == 3
    _assert_same(_state(second), saved)
    _assert_same(second.strategy_state, first.strategy_state)

    rng = torch.get_rng_state()                   # restored by resume; both runs share it
    first.config.max_steps = 4
    first.start_step = 3
    first.fit()
    torch.set_rng_state(rng)
    second.fit()                                  # runs step 3 only
    assert second.model.step == 3 and len(second.step_times) == 1
    _assert_same(_state(second)[0], _state(first)[0])
    # The camera sequence continues, not restarts: the next 8 match (across an epoch boundary).
    assert [next(second._iter)["idx"] for _ in range(8)] == [next(first._iter)["idx"] for _ in range(8)]


def test_level_artifact_loads_into_a_fresh_model(tmp_path):
    """``engine.checkpoint.load(run_dir, level=...)`` rebuilds the model from the
    run's meta.json and loads that level's artifact; step checkpoints carry no
    frozen networks."""
    from scion.compression import levels
    from scion.engine.checkpoint import FROZEN_PREFIXES, load

    trainer = _trainer(tiny_model(), tiny_scene(tmp_path), tmp_path / "run", max_steps=2)
    trainer.fit()
    ckpt = next((tmp_path / "run" / "checkpoints").glob("step-*.ckpt"))
    saved = torch.load(ckpt, weights_only=True)
    assert not [k for k in saved["pipeline"] if k[len("_model."):].startswith(FROZEN_PREFIXES)]
    assert not [k for k in saved["trainer_state"]["model"] if k.startswith(FROZEN_PREFIXES)]

    want = levels.raw_state_dict(trainer.model.state_dict())
    blob, _ = levels.encode(want, "raw")
    (tmp_path / "run" / levels.ARTIFACTS["raw"]).write_bytes(blob)
    loaded = load(tmp_path / "run", device="cpu", level="raw")
    got = levels.raw_state_dict(loaded.model.state_dict())
    assert got.keys() == want.keys() and loaded.step == 1
    for k in want:
        torch.testing.assert_close(got[k].cpu(), want[k].cpu(), rtol=0, atol=0, msg=k)
    assert loaded.model.discriminator is None  # rendering needs no discriminator
    with pytest.raises(FileNotFoundError, match="quantized"):
        load(tmp_path / "run", device="cpu", level="quantized")

    # An artifact elsewhere (e.g. a re-encoded bundle) loads directly.
    other = tmp_path / "elsewhere.ckpt"
    other.write_bytes(levels.encode(want, "quantized")[0])
    q = load(tmp_path / "run", device="cpu", level="quantized", artifact=other)
    assert q.meta["artifact"] == str(other)
    torch.testing.assert_close(levels.raw_state_dict(q.model.state_dict())["instance_positions"],
                               want["instance_positions"], rtol=1e-3, atol=1e-2)


def test_non_finite_loss_raises(tmp_path, monkeypatch):
    def nan_rasterize(**kw):
        rgb, *rest = fake_rasterize(**kw)
        return (rgb * float("nan"), *rest)

    monkeypatch.setattr(codebook_mod, "_rasterize", nan_rasterize)
    trainer = _trainer(tiny_model(), tiny_scene(tmp_path), tmp_path)
    with pytest.raises(RuntimeError, match="Non-finite"):
        trainer.step(0)


def test_viewer_render_matches_the_camera_render(tmp_path, monkeypatch):
    """The live viewer's render_fn, given a viser (OpenCV) pose, renders what a
    scion Camera (OpenGL) at the same pose renders."""
    from types import SimpleNamespace

    from scion.engine.trainer import _viewer_render_fn

    model = tiny_model()
    cam = tiny_scene(tmp_path).cameras[0]
    model.eval()
    with torch.no_grad():
        want = (model.get_outputs(cam)["rgb"].clamp(0, 1) * 255).byte().numpy()
    c2w_cv = torch.cat([cam.cam_to_world, torch.tensor([[0.0, 0, 0, 1]])]) @ torch.diag(torch.tensor([1.0, -1, -1, 1]))
    state = SimpleNamespace(c2w=c2w_cv.double().numpy(), get_K=lambda img_wh: cam.K.double().numpy())
    got = _viewer_render_fn(model, "cpu")(state, (cam.width, cam.height))
    assert want.std() > 0  # the camera sees the seed points, not only background
    assert (got == want).all()
