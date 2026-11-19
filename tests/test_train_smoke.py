"""GPU smokes of the training path, one per loader family
(COLMAP, Blender), ~50 steps each, plus
one end-to-end worker run (run.json v2, codec level when tmc3 is present).
Skipped without CUDA or without the scene under data/."""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest
import torch

from scion.engine import run
from scion.util.provenance import compose

REPO = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")]
SHORT = ["trainer.max_num_iterations=50", "trainer.steps_per_eval_all_images=25",
         "trainer.steps_per_save=0", "trainer.vis=none"]


def _need(scene):
    if not (REPO / "data" / scene).is_dir():
        pytest.skip(f"data/{scene} missing")


@pytest.mark.parametrize("dataset,scene", [("mipnerf360", "garden"), ("synthetic", "lego")])
def test_fit_and_score(dataset, scene, tmp_path, monkeypatch):
    _need(scene)
    monkeypatch.chdir(REPO)
    cfg = compose(["method=codebook", "variant=ours_small", f"dataset={dataset}", f"scene={scene}",
                   "machine.seed=0", "compression=quantized", *SHORT])
    trainer = run.fit(cfg, tmp_path)
    assert trainer.model.step == 49 and len(trainer.step_times) == 50
    assert trainer.peak_vram_bytes > 0
    res = run.score(cfg, trainer, tmp_path)
    for level in ("raw", "quantized"):
        m = res["metrics"][level]
        assert math.isfinite(m["psnr"]) and m["psnr"] > 8 and m["size_bytes"] > 0
    assert next((tmp_path / "checkpoints").glob("step-000000049.ckpt"))
    if dataset == "synthetic":
        assert trainer.model.config.background_color == "white"
        assert trainer.model.config.num_candidates == 64
        assert len(trainer.scene.eval_indices) == 200


def test_worker_end_to_end(tmp_path, monkeypatch):
    _need("garden")
    monkeypatch.chdir(REPO)
    monkeypatch.setenv("SCION_ALLOW_DIRTY", "1")
    spec = importlib.util.spec_from_file_location("worker", REPO / "scripts" / "train_eval_render_one.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    level = [] if shutil.which("tmc3") else ["compression=quantized"]
    argv = ["--variant", "ours_small", "--dataset", "mipnerf360", "--scene", "garden", "--seed", "0",
            "--runs-root", str(tmp_path)] + [a for o in SHORT + level for a in ("--override", o)]
    assert worker.main(argv) == 0
    rec = json.loads((tmp_path / "ours_small" / "mipnerf360" / "garden" / "seed0" / "run.json").read_text())
    assert rec["status"] == "DONE" and rec["backend_tiers"]
    assert rec["split"]["n_eval"] == 24 and rec["split"]["downscale_factor"] == 4
    assert rec["model_stats"]["num_instances"] > 0 and rec["median_step_s"] > 0
    run_dir = tmp_path / "ours_small" / "mipnerf360" / "garden" / "seed0"
    for path in rec["artifacts"]["checkpoints"].values():
        assert (run_dir / path).is_file()
    assert any((run_dir / "candidates").glob("isolation_*.jpg"))
