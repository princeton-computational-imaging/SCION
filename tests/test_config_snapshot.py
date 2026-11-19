"""Resolved config against a frozen snapshot: scion.engine.run builds the
recorded optimizer groups / learning rates (scene_lr_scale included), model
config, split and trainer cadence for every paper variant / dataset / scene.
Reference: tests/fixtures/snapshot/config.json. Only the learning rates need
the scene on disk (camera radius); Adam eps is checked on a real Trainer in
tests/test_trainer.py.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from scion.data import load_scene
from scion.engine import run
from scion.util.provenance import compose

REPO = Path(__file__).resolve().parents[1]
REF = json.loads((REPO / "tests" / "fixtures" / "snapshot" / "config.json").read_text())


def _cfg(key):
    variant, dataset, scene = key.split("/")
    return compose(["method=codebook", f"variant={variant}", f"dataset={dataset}",
                    f"scene={scene}", "machine.seed=0"])


@pytest.mark.parametrize("key", sorted(REF))
def test_model_config_cadence_and_split_match_snapshot(key, tmp_path):
    ref, cfg = REF[key], _cfg(key)
    model = dataclasses.asdict(run.model_config_from_hydra(cfg))
    assert json.loads(json.dumps(model, default=list)) == ref["model"]

    tc = run.trainer_config(cfg, tmp_path)
    assert [tc.max_steps, tc.save_every, tc.eval_every, tc.seed] == ref["cadence"]
    if ref["split"] is not None:
        kw = run.scene_kwargs(cfg)
        assert {k: kw.get(k) for k in ref["split"] if kw.get(k) is not None} == {
            k: v for k, v in ref["split"].items() if v is not None}


@pytest.mark.parametrize("key", sorted(REF))
def test_learning_rates_match_snapshot(key):
    scene = key.split("/")[2]
    if not (REPO / "data" / scene).is_dir():
        pytest.skip(f"data/{scene} missing (scene_lr_scale needs its cameras)")
    ref, cfg = REF[key], _cfg(key)
    # load_scene, not scene_from_config: only the cameras matter, so no seed is triangulated
    lrs = run.learning_rates(cfg, load_scene(Path(cfg.data), **run.scene_kwargs(cfg)))
    assert lrs.keys() == ref["lrs"].keys()
    for name, lr in ref["lrs"].items():
        assert lrs[name] == pytest.approx(lr, rel=1e-6), name
