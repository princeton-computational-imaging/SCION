"""Paper driver (experiments/paper/reproduce.py, run by `make reproduce`). CPU-only: the worker,
provenance, setup steps and sbatch are faked."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

from experiments.paper import reproduce, setup_data
from experiments.paper.targets import load_registry

REPO = Path(__file__).resolve().parents[1]

FIX = Path(__file__).parent / "fixtures" / "paper"

MINI = """
seeds: 3
variants: {ours_small: S, ours_large: L}
datasets:
  mipnerf360: {label: M, scenes: [garden, bicycle]}
targets:
  tab:
    renderer: run_main_results
    jobs: [{variants: [ours_small], datasets: [mipnerf360], scenes: [garden], env: {FOO: bar}}]
"""

FAKE_WORKER = r"""
import argparse, json, os, signal, sys
from pathlib import Path
p = argparse.ArgumentParser()
for a in ("--variant", "--dataset", "--scene", "--seed", "--runs-root", "--ablation", "--tag"):
    p.add_argument(a)
p.add_argument("--override", action="append", default=[])
a = p.parse_args()
assert os.environ["FOO"] == "bar" and os.environ["CUDA_VISIBLE_DEVICES"] in ("0", "1")
d = Path(a.runs_root) / a.variant / a.dataset / a.scene / f"seed{a.seed}"
d.mkdir(parents=True, exist_ok=True)
n = len(list(d.glob("attempt*"))) + 1
(d / f"attempt{n}").touch()
if os.environ.get("FAKE_SIGINT_SEED") == a.seed:                 # as if Ctrl-C hit the worker
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)
overrides = ["method=codebook", f"variant={a.variant}", f"dataset={a.dataset}", f"scene={a.scene}",
             f"machine.seed={a.seed}"] + a.override
rec = {"status": "DONE", "config_sha256": "cfg:" + "|".join(overrides), "git_sha": "a" * 40}
if os.environ.get("SCION_ALLOW_DIRTY") == "1":
    rec["git_dirty"] = True
if a.seed == "1" and n == 1:
    rec["git_sha"] = "b" * 40          # exits 0 with DONE, but for another commit: not done
if a.seed == "2":
    rec["status"] = "FAILED_TRAIN"     # always fails
(d / "run.json").write_text(json.dumps(rec))
sys.exit(2 if rec["status"] != "DONE" else 0)
"""


def _cfg_sha(overrides):
    return "cfg:" + "|".join(overrides)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Mini registry, fake provenance/setup/worker, empty data root, tmp runs root."""
    reg = tmp_path / "targets.yaml"
    reg.write_text(MINI)
    monkeypatch.setattr(reproduce, "load_registry", lambda: load_registry(reg))
    prov = types.SimpleNamespace(dirty=False)
    fake = types.ModuleType("scion.util.provenance")
    fake.config_sha256 = _cfg_sha
    fake.git_sha = lambda: "a" * 40
    fake.tree_is_dirty = lambda: prov.dirty
    monkeypatch.setitem(sys.modules, "scion.util.provenance", fake)
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    monkeypatch.setattr(reproduce, "WORKER", worker)
    monkeypatch.setattr(setup_data, "prepare_data", lambda scenes, root: [])
    monkeypatch.setattr(setup_data, "tmc3", lambda: "/opt/tmc3")
    monkeypatch.setattr(setup_data, "dino_argv", lambda s, root: [sys.executable, "-c", "pass"])
    rendered = []
    monkeypatch.setattr(reproduce, "render_phase", lambda reg, names, runs, out: rendered.append(names) or 0)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    prov.runs, prov.rendered, prov.reg = tmp_path / "runs", rendered, load_registry(reg)
    return prov


def _main(env, *args):
    return reproduce.main([*args, "--runs-root", str(env.runs)])


def _record(env, job, **fields):
    path = job.key.path(env.runs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields))


# ---------------------------------------------------------------- resolve

def test_worker_argv_passes_only_non_flag_overrides():
    reg = load_registry()
    job = next(j for j in reg.jobs() if j.key.dataset == "synthetic" and j.key.variant == "ours_small")
    # Synthetic K lives in config (config/dataset_variant/); add an extra override by hand.
    job = dataclasses.replace(job, overrides=job.overrides + ("method.model.num_candidates=64",))
    argv = reproduce.worker_argv(job, Path("/r"))
    assert argv[1] == str(reproduce.WORKER)
    assert argv[2:] == ["--variant", "ours_small", "--dataset", "synthetic", "--scene", job.key.scene,
                        "--seed", str(job.key.seed), "--runs-root", "/r",
                        "--override", "method.model.num_candidates=64"]
    tagged = next(j for j in reg.jobs() if j.key.tag == "python")
    assert reproduce.worker_argv(tagged, Path("/r"))[-2:] == ["--tag", "python"]


# ---------------------------------------------------------------- idempotency

def test_done_needs_same_config_and_git_sha(env):
    job = env.reg.targets["tab"].jobs[0]
    good = dict(status="DONE", config_sha256=_cfg_sha(list(job.overrides)), git_sha="a" * 40)
    _record(env, job, **good)
    assert reproduce.is_done(job, env.runs)
    for change in ({"git_sha": "b" * 40}, {"config_sha256": "other"}, {"status": "FAILED_EVAL"},
                   {"git_dirty": True}):
        _record(env, job, **{**good, **change})
        assert not reproduce.is_done(job, env.runs), change
    assert reproduce.is_done(job, env.runs, accept_dirty=True)   # an --allow-dirty run's own check


def test_done_jobs_are_skipped(env, capsys):
    jobs = env.reg.targets["tab"].jobs
    _record(env, jobs[0], status="DONE", config_sha256=_cfg_sha(list(jobs[0].overrides)), git_sha="a" * 40)
    _record(env, jobs[1], status="DONE", config_sha256=_cfg_sha(list(jobs[1].overrides)), git_sha="b" * 40)
    assert _main(env, "tab", "--dry-run") == 0
    out = capsys.readouterr().out
    assert re.search(r"^tab\s+3\s+1\s+2$", out, re.M), out
    assert "nothing was run" in out and not list(env.runs.rglob("attempt*"))


def test_dirty_tree_refused_unless_allowed(env, capsys):
    env.dirty = True
    with pytest.raises(SystemExit, match="the code has uncommitted changes"):
        _main(env, "tab", "--gpus", "0")
    assert not list(env.runs.rglob("attempt*"))
    assert _main(env, "tab", "--gpus", "0,1", "--allow-dirty") == 1    # runs (seed 2 always fails)
    out = capsys.readouterr().out
    assert "--allow-dirty: running uncommitted changes" in out
    assert "[summary] done 2, skipped 0 (already DONE), failed 1" in out


# ---------------------------------------------------------------- backends

def test_local_pool_retries_once_and_reports_failures(env, capsys):
    assert _main(env, "tab", "--gpus", "0,1", "--jobs-per-gpu", "2") == 1
    out = capsys.readouterr().out
    seed = {n: env.runs / f"ours_small/mipnerf360/garden/seed{n}" for n in range(3)}
    assert [len(list(seed[n].glob("attempt*"))) for n in range(3)] == [1, 2, 2]
    # seed 1 exited 0 with DONE for another git_sha: not done, so retried
    assert "FAILED  ours_small/mipnerf360/garden/seed1 (exit 0" in out and "retrying" in out
    assert "DONE    ours_small/mipnerf360/garden/seed1" in out
    assert "giving up" in out and "[summary] done 2, skipped 0 (already DONE), failed 1" in out
    assert "[summary] FAILED ours_small/mipnerf360/garden/seed2" in out
    assert (env.runs.parent / "logs/ours_small/mipnerf360/garden/seed2.log").is_file()
    assert env.rendered == [["tab"]]                 # render phase still runs after the jobs


def test_local_ctrl_c_stops_scheduling_without_retry(env, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_SIGINT_SEED", "0")
    assert _main(env, "tab", "--gpus", "0") == 130          # one slot: seeds run in order
    seeds = [env.runs / f"ours_small/mipnerf360/garden/seed{n}" for n in range(3)]
    assert [len(list(d.glob("attempt*"))) for d in seeds] == [1, 0, 0]
    assert "INTERRUPTED ours_small/mipnerf360/garden/seed0" in capsys.readouterr().out
    assert env.rendered == []


def test_failed_dino_step_blocks_its_jobs_with_its_log(env, monkeypatch, capsys):
    monkeypatch.setattr(setup_data, "dino_argv", lambda s, root: [sys.executable, "-c", "raise SystemExit(1)"])
    assert _main(env, "tab", "--gpus", "0") == 1
    out = capsys.readouterr().out
    assert "failed 3" in out and not list(env.runs.rglob("attempt*"))
    assert f"[summary] FAILED ours_small/mipnerf360/garden/seed0  log: {env.runs.parent}/logs/dino/garden.log" in out


def _submit_slurm(env, tmp_path, monkeypatch):
    """Run the slurm backend against a fake sbatch; returns (script dir, sbatch call log)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch.log"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(f'#!/bin/bash\necho "$@" >> {log}\necho "$((1000 + $(wc -l < {log})));cl"\n')
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("SLURM_GRES", "gpu:1")
    monkeypatch.setenv("SLURM_TIME", "01:00:00")
    monkeypatch.setenv("SLURM_ACCOUNT", "acct")
    monkeypatch.setenv("SLURM_MEM", "48G")
    monkeypatch.delenv("SLURM_PARTITION", raising=False)
    monkeypatch.delenv("SLURM_CPUS", raising=False)
    assert _main(env, "tab", "--backend", "slurm") == 0
    (sub,) = (env.runs.parent / "slurm").iterdir()
    return sub, log


def test_slurm_writes_job_list_and_chained_arrays(env, tmp_path, monkeypatch, capsys):
    sub, log = _submit_slurm(env, tmp_path, monkeypatch)
    out = capsys.readouterr().out
    lines = (sub / "jobs.txt").read_text().splitlines()
    assert len(lines) == 3
    for n, line in enumerate(lines):
        argv = shlex.split(line)
        assert argv[:3] == ["env", "FOO=bar", "TMC13_BINARY=/opt/tmc3"]
        assert argv[3:5] == [sys.executable, str(reproduce.WORKER)]
        assert argv[argv.index("--seed") + 1] == str(n)
    assert len((sub / "dino.txt").read_text().splitlines()) == 1
    jobs_sbatch = (sub / "jobs.sbatch").read_text()
    for s in ("--array=0-2%16", "--gres=gpu:1", "--time=01:00:00", "--account=acct", "--mem=48G"):
        assert f"#SBATCH {s}" in jobs_sbatch
    assert "partition" not in jobs_sbatch and "cpus-per-task" not in jobs_sbatch
    render = (sub / "render.sbatch").read_text()
    assert "--render-only" in render and "gres" not in render
    calls = log.read_text().splitlines()
    assert calls[0] == f"--parsable {sub / 'dino.sbatch'}"
    assert calls[1] == f"--parsable --dependency=afterany:1001 {sub / 'jobs.sbatch'}"
    assert calls[2] == f"--parsable --dependency=afterany:1002 {sub / 'render.sbatch'}"
    assert "jobs: job 1002" in out and not list(env.runs.rglob("attempt*")) and env.rendered == []


def test_slurm_array_task_fails_loudly(env, tmp_path, monkeypatch):
    sub, _ = _submit_slurm(env, tmp_path, monkeypatch)
    listing, script = sub / "jobs.txt", sub / "jobs.sbatch"

    def task(task_id="0"):
        return subprocess.run(["bash", str(script)], env={**os.environ, "SLURM_ARRAY_TASK_ID": task_id},
                              capture_output=True, text=True).returncode

    listing.write_text("true\n")
    assert task() == 0
    listing.write_text("exit 3\n")                  # fails, and fails again on the retry
    assert task() != 0
    assert task("5") != 0                            # no such line
    listing.unlink()
    assert task() != 0                               # missing listing


def test_slurm_needs_gres_and_time(env, monkeypatch):
    monkeypatch.delenv("SLURM_GRES", raising=False)
    with pytest.raises(SystemExit, match="SLURM_GRES and SLURM_TIME"):
        _main(env, "tab", "--backend", "slurm")


# ---------------------------------------------------------------- dry run / render

def test_dry_run_all_on_real_registry_without_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(reproduce, "WORKER", tmp_path / "must-not-run.py")
    assert reproduce.main(["--dry-run", "all", "--runs-root", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^main_results\s+126\s+0\s+126$", out, re.M), out
    assert "jobs compose; nothing was run" in out and not (tmp_path / "data").exists()


def test_override_needs_a_scratch_runs_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    with pytest.raises(SystemExit):
        reproduce.main(["--dry-run", "compute", "--override", "trainer.max_num_iterations=10"])
    assert "scratch --runs-root" in capsys.readouterr().err
    assert reproduce.main(["--dry-run", "compute", "--override", "trainer.max_num_iterations=10",
                           "--runs-root", str(tmp_path / "runs")]) == 0


def test_render_only_writes_tables_and_skips_figures_without_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(FIX))
    import generate_pool

    generate_pool.write_pool(tmp_path / "pool")
    out = tmp_path / "paper"
    runs = str(tmp_path / "pool" / "runs")
    assert reproduce.main(["--render-only", "component_ablation", "--runs-root", runs,
                           "--paper-dir", str(out)]) == 0
    assert list((out / "tables").glob("*.tex"))
    assert reproduce.main(["--render-only", "primitives", "--runs-root", str(tmp_path / "none"),
                           "--paper-dir", str(out)]) == 0
    assert "primitives: 1/1 input runs not done, generator not run" in capsys.readouterr().out
    # Like the tables, figures accept a clean DONE record from an earlier commit
    # (renderer-only commits must not block them) ...
    job = load_registry().targets["primitives"].jobs[0]
    path = job.key.path(tmp_path / "none")
    path.parent.mkdir(parents=True)
    rec = {"status": "DONE", "git_sha": "f" * 40,
           "config_sha256": reproduce._prov().config_sha256(list(job.overrides))}
    path.write_text(json.dumps(rec))
    monkeypatch.setattr(reproduce, "visible_gpus", lambda: [])
    assert reproduce.main(["--render-only", "primitives", "--runs-root", str(tmp_path / "none"),
                           "--paper-dir", str(out)]) == 0
    assert "primitives: needs a GPU" in capsys.readouterr().out
    # ... but a record made from a dirty tree still blocks the generator, and is named.
    path.write_text(json.dumps({**rec, "git_dirty": True}))
    assert reproduce.main(["--render-only", "primitives", "--runs-root", str(tmp_path / "none"),
                           "--paper-dir", str(out)]) == 0
    assert f"{path}: made from a dirty tree" in capsys.readouterr().out


# ---------------------------------------------------------------- setup

def test_download_script_fails_closed_on_hash_and_manual_archive(tmp_path):
    script = REPO / "scripts" / "download_data.sh"
    (tmp_path / "_archives").mkdir()
    (tmp_path / "_archives" / "tandt_db.zip").write_bytes(b"not the archive")
    env = {**os.environ, "DATA_ROOT": str(tmp_path)}
    run = lambda name: subprocess.run(["bash", str(script), name], env=env, capture_output=True, text=True)
    r = run("tandt")
    assert r.returncode == 1 and "does not match its SHA256" in r.stderr
    r = run("synthetic")                                # Google Drive only: never fetched by the script
    assert r.returncode == 1 and "download nerf_synthetic.zip from https://drive.google.com" in r.stderr
    assert run("garden").returncode == 2                # datasets only, no per-scene option


def test_setup_names_private_data_and_links_data_root(tmp_path, monkeypatch):
    (tmp_path / "repo").mkdir()
    monkeypatch.setattr(setup_data, "REPO", tmp_path / "repo")
    data = tmp_path / "data"
    assert setup_data.prepare_data({("private", "scene")}, data) == [
        f"{data / 'scene'} is missing: no source known for private"]
    assert (tmp_path / "repo" / "data").resolve() == data.resolve()   # configs read data/<scene>
