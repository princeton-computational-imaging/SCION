"""Worker v2 (scripts/train_eval_render_one.py): the run.json it builds has the
schema-v2 shape of tests/fixtures/run_json_v2/example_run.json, and the
overrides it composes training from hash like the registry's (the
idempotency key the reproduce runner compares)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.paper.targets import load_registry
from scion.util import provenance
from scion.util.provenance import compose, config_sha256
from scion.util.run_index import JobKey

REPO = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((REPO / "tests" / "fixtures" / "run_json_v2" / "example_run.json").read_text())


@pytest.fixture(scope="module")
def worker():
    spec = importlib.util.spec_from_file_location(
        "train_eval_render_one", REPO / "scripts" / "train_eval_render_one.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shape(obj, depth=2):
    """Key structure of nested dicts down to ``depth``."""
    if not isinstance(obj, dict) or depth == 0:
        return None
    return {k: _shape(v, depth - 1) for k, v in obj.items()}


@pytest.fixture(autouse=True)
def _no_scene_data(monkeypatch):
    """split.data_sha256 without the scene on disk (tests/test_provenance.py covers it)."""
    monkeypatch.setattr(provenance, "data_sha256", lambda scene_dir: "d" * 64)


def test_done_record_matches_v2_fixture(worker):
    key = JobKey("ours_small", None, "mipnerf360", "garden", 0)
    overrides = ["method=codebook", "variant=ours_small", "dataset=mipnerf360",
                 "scene=garden", "machine.seed=0"]
    cfg = compose(overrides)
    names = [f"frame_{i:05d}.JPG" for i in range(1, 186)]
    scene = SimpleNamespace(downscale_factor=4, image_paths=[Path("images_4") / n for n in names],
                            train_indices=[i for i in range(185) if i % 8],
                            eval_indices=[i for i in range(185) if i % 8 == 0])
    trainer = SimpleNamespace(train_time_s=1066.0, median_step_s=0.031,
                              peak_vram_bytes=15032385536, scene=scene)
    levels = {k: FIXTURE[k] for k in ("metrics", "codec")}
    levels["artifacts"] = {k: FIXTURE["artifacts"][k] for k in ("checkpoints", "eval_views")}

    rec = worker.done_record(worker.running_record(key, overrides, cfg, dirty=False), cfg=cfg,
                             trainer=trainer, model_stats=FIXTURE["model_stats"], levels=levels)

    # backend_tiers lists the ops dispatched in this process (none on CPU).
    assert set(rec["backend_tiers"]) <= set(FIXTURE["backend_tiers"])
    assert _shape({**rec, "backend_tiers": None}) == _shape({**FIXTURE, "backend_tiers": None})
    assert rec["status"] == "DONE" and rec["schema_version"] == 2
    assert rec["config_sha256"] == config_sha256(overrides) and len(rec["git_sha"]) == 40
    assert rec["split"] == {**rec["split"], "eval_mode": "interval", "eval_interval": 8,
                            "train_split_fraction": None, "n_train": 161, "n_eval": 24,
                            "data_sha256": "d" * 64}
    assert rec["slot"] == key.slot and rec["n_gaussians"] == FIXTURE["metrics"]["raw"]["n_gaussians"]
    assert "git_dirty" not in rec
    assert worker.running_record(key, overrides, cfg, dirty=True)["git_dirty"] is True


ARGV = ["--variant", "ours_small", "--dataset", "mipnerf360", "--scene", "garden", "--seed", "0"]


def _record(runs):
    return json.loads((runs / "ours_small" / "mipnerf360" / "garden" / "seed0" / "run.json").read_text())


def _fake_trainer():
    names = [f"frame_{i:05d}.JPG" for i in range(1, 17)]
    scene = SimpleNamespace(downscale_factor=4, image_paths=[Path(n) for n in names],
                            train_indices=list(range(1, 16)), eval_indices=[0],
                            eval_cameras=lambda: [SimpleNamespace(to=lambda d: None)])
    return SimpleNamespace(model=SimpleNamespace(device="cpu"), scene=scene, train_time_s=1.0,
                           median_step_s=0.01, peak_vram_bytes=None)


@pytest.fixture
def job_env(worker, tmp_path, monkeypatch):
    from scion.engine import run

    monkeypatch.setenv("SCION_ALLOW_DIRTY", "1")
    levels = {k: FIXTURE[k] for k in ("metrics", "codec")}
    levels["artifacts"] = {k: FIXTURE["artifacts"][k] for k in ("checkpoints", "eval_views")}
    monkeypatch.setattr(run, "fit", lambda cfg, run_dir: _fake_trainer())
    monkeypatch.setattr(run, "score", lambda cfg, trainer, run_dir: levels)
    monkeypatch.setattr(worker, "_model_stats", lambda model: FIXTURE["model_stats"])
    monkeypatch.setattr(worker, "_render_isolations", lambda *a, **k: [])
    return ARGV + ["--runs-root", str(tmp_path / "runs")], tmp_path / "runs", run


@pytest.mark.parametrize("exc", [SystemExit(143), KeyboardInterrupt()])
def test_interrupt_records_failed_train_and_reraises(worker, job_env, monkeypatch, exc):
    argv, runs, run = job_env

    def fit(cfg, run_dir):
        raise exc

    monkeypatch.setattr(run, "fit", fit)
    with pytest.raises(type(exc)):
        worker.main(argv)
    rec = _record(runs)
    assert rec["status"] == "FAILED_TRAIN" and rec["ended_at"] and type(exc).__name__ in rec["error"]


def test_early_refusal_removes_the_previous_record(worker, job_env, monkeypatch):
    argv, runs, _run = job_env
    path = runs / "ours_small" / "mipnerf360" / "garden" / "seed0" / "run.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**FIXTURE, "status": "DONE"}))
    monkeypatch.delenv("SCION_ALLOW_DIRTY")
    monkeypatch.setattr(provenance, "tree_is_dirty", lambda: True)
    assert worker.main(argv) == 2
    assert not path.exists()     # no stale DONE outlives the refused rerun


def test_render_failure_keeps_done_and_metrics(worker, job_env, monkeypatch):
    argv, runs, _run = job_env

    def boom(*a, **k):
        raise RuntimeError("render broke")

    monkeypatch.setattr(worker, "_render_templates", boom)
    assert worker.main(argv) == 0
    rec = _record(runs)
    assert rec["status"] == "DONE" and rec["metrics"] == FIXTURE["metrics"]
    assert "templates" not in rec["artifacts"]


def _worker_args(worker, job):
    """The worker's parse of the argv the reproduce runner launches it with."""
    from experiments.paper.reproduce import worker_argv

    return worker._parse_args(worker_argv(job, Path("runs"))[2:])


def test_worker_overrides_hash_like_the_registry(worker):
    jobs = load_registry().jobs()
    picks = [next(j for j in jobs if j.key.dataset == "synthetic"),
             next(j for j in jobs if j.key.ablation)]
    for job in picks:
        ov = worker.job_overrides(_worker_args(worker, job))
        assert ov == list(job.overrides)
        assert config_sha256(ov) == config_sha256(list(job.overrides))
    # An extra registry override (e.g. a per-job K) rides along via --override.
    job = picks[0]
    extra = SimpleNamespace(key=job.key, overrides=job.overrides + ("method.model.num_candidates=32",))
    ov = worker.job_overrides(_worker_args(worker, extra))
    assert config_sha256(ov) == config_sha256(list(extra.overrides)) != config_sha256(list(job.overrides))
    assert compose(ov).method.model.num_candidates == 32
