"""Regenerate every SCION number and figure in the paper.

    python -m experiments.paper.reproduce [TARGETS ...] [--backend local|slurm] [--gpus 0,1]
        [--jobs-per-gpu N] [--render-only] [--dry-run] [--paper-dir DIR] [--allow-dirty]

Usually run through the Makefile (`make reproduce`, `make dry-run`, `make render`).

TARGETS are names from experiments/paper/targets.yaml (default `all`). Each
job is one run of scripts/train_eval_render_one.py; a job whose run.json is
DONE with the same config_sha256 and git_sha is skipped. The render phase then
writes <paper-dir>/tables/*.tex and runs the figure generators. Guide:
docs/reproducing_results.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from experiments.paper import setup_data
from experiments.paper.targets import Job, Registry, load_registry

REPO = setup_data.REPO
WORKER = REPO / "scripts" / "train_eval_render_one.py"
BASE_KEYS = {"method", "variant", "dataset", "scene", "machine.seed", "ablation"}  # set by the worker's flags


def _prov():
    return importlib.import_module("scion.util.provenance")


def label(job: Job) -> str:
    return str(job.key.path("").parent)


def record(job: Job, runs_root: Path) -> dict:
    """The job's run.json, or {} when absent or unreadable."""
    try:
        rec = json.loads(job.key.path(runs_root).read_text())
    except (OSError, ValueError):
        return {}
    return rec if isinstance(rec, dict) else {}


def not_done(job: Job, runs_root: Path, accept_dirty: bool = False,
             same_code: bool = True) -> Optional[str]:
    """Why the job's run.json is not DONE with this checkout's git_sha and the
    job's config_sha256 (None if it is). A --allow-dirty record (git_dirty)
    never counts as done for skipping. ``same_code=False`` drops the git_sha
    check: figures, like tables, render finished records after later
    renderer-only commits; skipping a training job still needs the same code."""
    rec = record(job, runs_root)
    if rec.get("status") != "DONE":
        return f"status {rec.get('status', 'missing')}"
    if rec.get("git_dirty") and not accept_dirty:
        return "made from a dirty tree"
    if same_code and rec.get("git_sha") != _prov().git_sha():
        return f"git_sha {rec.get('git_sha')} is not this checkout's"
    if rec.get("config_sha256") != _prov().config_sha256(list(job.overrides)):
        return "config_sha256 is not the job's"
    return None


def is_done(job: Job, runs_root: Path, accept_dirty: bool = False) -> bool:
    return not_done(job, runs_root, accept_dirty) is None


def visible_gpus() -> List[str]:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [g for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return out.split()


# ------------------------------------------------------------------ resolve

def compose(jobs: Sequence[Job]) -> Tuple[Dict[Job, str], List[str]]:
    """Compose every job's Hydra overrides: (job -> compression level, errors)."""
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from scion.util.hydra import CONF_DIR

    levels, errors = {}, []
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        for j in jobs:
            try:
                levels[j] = hydra_compose("config", overrides=list(j.overrides)).compression.level
            except Exception as e:  # any composition failure is reported per job
                errors.append(f"  {label(j)}: {type(e).__name__}: {str(e).splitlines()[0]}")
    GlobalHydra.instance().clear()
    return levels, errors


def worker_argv(job: Job, runs_root: Path) -> List[str]:
    k = job.key
    argv = [sys.executable, str(WORKER), "--variant", k.variant, "--dataset", k.dataset,
            "--scene", k.scene, "--seed", str(k.seed), "--runs-root", str(runs_root)]
    if k.ablation:
        argv += ["--ablation", k.ablation]
    if k.tag:
        argv += ["--tag", k.tag]
    for o in job.overrides:
        if o.split("=", 1)[0] not in BASE_KEYS:
            argv += ["--override", o]
    return argv


# ------------------------------------------------------------------ backends

def run_local(todo: List[Job], runs_root: Path, data_root: Path, env: Dict[str, str],
              gpus: List[str], jobs_per_gpu: int, allow_dirty: bool = False) -> List[Tuple[Job, Path]]:
    """GPU pool: DINO cache per scene first, then the jobs; one retry each.
    Returns (failed job, its log). Ctrl-C stops scheduling and raises KeyboardInterrupt."""
    free: Queue = Queue()
    for g in gpus * jobs_per_gpu:
        free.put(g)
    stop = threading.Event()
    logs = runs_root.parent / "logs"

    def run(name: str, argv: List[str], extra_env: Dict[str, str], log: Path,
            ok: Callable[[], bool] = lambda: True) -> bool:
        log.parent.mkdir(parents=True, exist_ok=True)
        for attempt in (1, 2):
            gpu = free.get()
            try:
                if stop.is_set():
                    return False
                print(f"[run] gpu {gpu}  start   {name} (attempt {attempt}/2, log {log})")
                t0 = time.time()
                with open(log, "a") as f:
                    rc = subprocess.run(argv, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                                        env={**os.environ, **extra_env, "CUDA_VISIBLE_DEVICES": gpu}).returncode
            finally:
                free.put(gpu)
            took = f"{(time.time() - t0) / 60:.1f} min"
            if rc == -signal.SIGINT:
                stop.set()
                print(f"[run] gpu {gpu}  INTERRUPTED {name}; scheduling no more jobs")
                return False
            if rc == 0 and ok():
                print(f"[run] gpu {gpu}  DONE    {name} ({took})")
                return True
            then = "retrying" if attempt == 1 else "giving up"
            print(f"[run] gpu {gpu}  FAILED  {name} (exit {rc}, {took}), {then}")
        return False

    def job_log(j: Job) -> Path:
        return (logs / label(j)).with_suffix(".log")

    def dino_log(scene: str) -> Path:
        return logs / "dino" / f"{scene}.log"

    scenes = sorted({j.key.scene for j in todo})
    with ThreadPoolExecutor(len(gpus) * jobs_per_gpu) as pool:
        try:
            dino = dict(zip(scenes, pool.map(
                lambda s: run(f"dino cache {s}", setup_data.dino_argv(s, data_root), {}, dino_log(s)),
                scenes)))
            runnable = [j for j in todo if dino[j.key.scene]]
            results = list(pool.map(
                lambda j: run(label(j), worker_argv(j, runs_root), {**dict(j.env), **env}, job_log(j),
                              ok=lambda: is_done(j, runs_root, accept_dirty=allow_dirty)),
                runnable))
        except KeyboardInterrupt:
            stop.set()      # before the pool drains: queued jobs return without starting
            raise
    if stop.is_set():
        raise KeyboardInterrupt
    return ([(j, dino_log(j.key.scene)) for j in todo if not dino[j.key.scene]]
            + [(j, job_log(j)) for j, ok in zip(runnable, results) if not ok])


ARRAY_SBATCH = """#!/bin/bash
#SBATCH --job-name=scion-{name}
#SBATCH --array=0-{last}%{maxpar}
#SBATCH --gres={gres}
#SBATCH --time={time}
#SBATCH --output={out}/logs/{name}_%A_%a.log
{extra}
# One line of {name}.txt per array task, retried once.
set -euo pipefail
cd {repo}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {listing})
[[ -n "$line" ]] || {{ echo "[slurm] no line $SLURM_ARRAY_TASK_ID in {listing}" >&2; exit 1; }}
echo "$line"
bash -c "$line" || {{ echo "[slurm] failed, retrying once"; bash -c "$line"; }}
"""

RENDER_SBATCH = """#!/bin/bash
#SBATCH --job-name=scion-render
#SBATCH --time={time}
#SBATCH --output={out}/logs/render_%j.log
{extra}
set -euo pipefail
cd {repo}
export CUDA_VISIBLE_DEVICES=   # CPU job: figure generators are printed, not run
{cmd}
"""


def submit_slurm(todo: List[Job], runs_root: Path, data_root: Path, env: Dict[str, str],
                 render_argv: List[str]) -> List[str]:
    """DINO array (one task per scene) -> job array (one task per job) -> CPU
    render job, chained with afterany. Returns the three Slurm job ids."""
    gres, wall = os.environ.get("SLURM_GRES"), os.environ.get("SLURM_TIME")
    if not (gres and wall):
        raise SystemExit("--backend slurm needs SLURM_GRES and SLURM_TIME (and usually SLURM_ACCOUNT / "
                         "SLURM_PARTITION); set them in reproduce.env, see reproduce.env.example")
    out = runs_root.parent / "slurm" / datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / "logs").mkdir(parents=True)
    extra = "\n".join(f"#SBATCH --{k}={os.environ[v]}" for k, v in
                      (("account", "SLURM_ACCOUNT"), ("partition", "SLURM_PARTITION"),
                       ("cpus-per-task", "SLURM_CPUS"), ("mem", "SLURM_MEM")) if os.environ.get(v))
    lists = {
        "dino": [shlex.join(setup_data.dino_argv(s, data_root)) for s in sorted({j.key.scene for j in todo})],
        "jobs": [shlex.join(["env", *(f"{k}={v}" for k, v in sorted({**dict(j.env), **env}.items())),
                             *worker_argv(j, runs_root)]) for j in todo],
    }
    ids: List[str] = []
    for name, lines in lists.items():
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n")
        (out / f"{name}.sbatch").write_text(ARRAY_SBATCH.format(
            name=name, last=len(lines) - 1, maxpar=os.environ.get("SLURM_MAXPAR", "16"), gres=gres,
            time=wall, out=out, extra=extra, repo=shlex.quote(str(REPO)),
            listing=shlex.quote(str(out / f"{name}.txt"))))
    (out / "render.sbatch").write_text(RENDER_SBATCH.format(
        time=wall, out=out, extra=extra, repo=shlex.quote(str(REPO)), cmd=shlex.join(render_argv)))
    for name in ("dino", "jobs", "render"):
        dep = [f"--dependency=afterany:{ids[-1]}"] if ids else []
        res = subprocess.run(["sbatch", "--parsable", *dep, str(out / f"{name}.sbatch")],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(f"sbatch {name}.sbatch failed: {res.stderr.strip()} (submitted so far: {ids})")
        ids.append(res.stdout.strip().split(";")[0])
        print(f"[slurm] {name}: job {ids[-1]}  ({out / (name + '.sbatch')})")
    return ids


# ------------------------------------------------------------------ render

def render_phase(reg: Registry, names: Sequence[str], runs_root: Path, paper_dir: Path) -> int:
    """Tables via experiments.paper.render, then each figure generator whose
    input runs are all DONE, clean, with the job's config and one git_sha. Returns non-zero if anything was
    refused or failed."""
    names = ["all"] if "all" in names else list(names)
    rc = 0
    for n in names:
        rc |= subprocess.run([sys.executable, "-m", "experiments.paper.render", n, "--runs-root",
                              str(runs_root), "--out", str(paper_dir)], cwd=REPO).returncode
    roots = (("figures/", paper_dir / "figures"), ("renders/", paper_dir / "renders"),
             ("evals/runs/", runs_root))
    gpu = bool(visible_gpus())
    for t in (reg.targets[n] for n in (reg.targets if names == ["all"] else names)):
        if not t.command:
            continue
        blocked = [(j, why) for j in t.jobs if (why := not_done(j, runs_root, same_code=False))]
        shas = {record(j, runs_root).get("git_sha") for j in t.jobs} if not blocked else set()
        if len(shas) > 1:  # one figure, one code version (as the tables require)
            blocked = [(t.jobs[0], f"input runs come from several commits: {sorted(shas)}")]
        if blocked:
            print(f"[figures] {t.name}: {len(blocked)}/{len(t.jobs)} input runs not done, generator not run:")
            for j, why in blocked:
                print(f"[figures]   {j.key.path(runs_root)}: {why}")
            continue
        if not gpu:
            print(f"[figures] {t.name}: needs a GPU, run on a GPU host: {t.command}")
            continue
        argv = [next((str(r / a[len(p):]) for p, r in roots if a.startswith(p)), a)
                for a in shlex.split(t.command)]  # generator outputs land next to the tables
        argv[0] = sys.executable if argv[0] == "python" else argv[0]
        for a in argv:
            if a.startswith(str(paper_dir)):
                Path(a).parent.mkdir(parents=True, exist_ok=True)
        ok = subprocess.run(argv, cwd=REPO).returncode == 0
        print(f"[figures] {t.name}: {'wrote panels' if ok else 'generator FAILED'}; manual: {t.manual}")
        rc |= not ok
    return int(bool(rc))


# ------------------------------------------------------------------ main

def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # keep our lines ordered with subprocess output
    p = argparse.ArgumentParser(prog="python -m experiments.paper.reproduce", description=__doc__.splitlines()[0])
    p.add_argument("targets", nargs="*", default=["all"], help="targets.yaml names, or `all` (default)")
    p.add_argument("--backend", choices=["local", "slurm"], default="local")
    p.add_argument("--gpus", help="comma-separated GPU ids for the local backend (default: all visible)")
    p.add_argument("--jobs-per-gpu", type=int, default=1, help="concurrent jobs per GPU (default 1)")
    p.add_argument("--render-only", action="store_true", help="skip training, only render")
    p.add_argument("--dry-run", action="store_true",
                   help="print the job matrix and check every job's Hydra overrides compose; run nothing")
    p.add_argument("--paper-dir", type=Path, default=REPO / "paper_artifacts",
                   help="gets tables/ and figures/ (default: paper_artifacts/)")
    p.add_argument("--allow-dirty", action="store_true", help="run with uncommitted code changes (records get git_dirty: true)")
    p.add_argument("--runs-root", type=Path, default=REPO / "evals" / "runs", help=argparse.SUPPRESS)
    # Smoke runs (e.g. trainer.max_num_iterations=200, with a scratch --runs-root):
    # appended to every job's overrides, so the records' config_sha256 says so.
    p.add_argument("--override", action="append", default=[], help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    runs_root, paper_dir = args.runs_root.resolve(), args.paper_dir.resolve()
    if args.override and runs_root == (REPO / "evals" / "runs").resolve():
        p.error("--override is for smoke runs: pass a scratch --runs-root (not evals/runs)")
    data_root = (REPO / (os.environ.get("DATA_ROOT") or "data")).resolve()
    reg = load_registry()
    unknown = sorted(set(args.targets) - set(reg.targets) - {"all"})
    if unknown:
        p.error(f"unknown target(s) {', '.join(unknown)}; choose from: all, {', '.join(reg.targets)}")

    if args.render_only:
        return render_phase(reg, args.targets, runs_root, paper_dir)

    targets = [reg.targets[n] for n in (reg.targets if "all" in args.targets else args.targets)]
    jobs = [dataclasses.replace(j, overrides=j.overrides + tuple(args.override))
            for j in reg.jobs([t.name for t in targets])]
    levels, errors = compose(jobs)
    done = {j for j in jobs if j in levels and is_done(j, runs_root)}
    todo = [j for j in jobs if j not in done]

    print(f"{'target':<24}{'jobs':>6}{'done':>6}{'to run':>8}")
    done_keys = {j.key for j in done}
    for t in targets:
        n_done = sum(j.key in done_keys for j in t.jobs)
        print(f"{t.name:<24}{len(t.jobs):>6}{n_done:>6}{len(t.jobs) - n_done:>8}")
    print(f"{'total (deduplicated)':<24}{len(jobs):>6}{len(done):>6}{len(todo):>8}")
    if errors:
        print(f"[compose] {len(errors)} job(s) do not compose:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    absent = sorted({j.key.scene for j in todo if not setup_data.ready(data_root / j.key.scene)})
    if absent:
        print(f"[setup] {len(absent)} scene(s) not under {data_root} yet, fetched on first need: "
              f"{' '.join(absent)}")
    if args.dry_run:
        print(f"[dry-run] all {len(jobs)} jobs compose; nothing was run")
        return 0

    failed: List[Tuple[Job, Path]] = []
    if todo:
        env: Dict[str, str] = {}
        if _prov().tree_is_dirty():
            if not args.allow_dirty:
                raise SystemExit("refusing to run: the code has uncommitted changes (modified or untracked "
                                 "files under the code paths, see scion/util/provenance.py), so "
                                 "run.json git_sha would not describe it. Commit them, or pass --allow-dirty.")
            print(f"[WARNING] --allow-dirty: running uncommitted changes on top of {_prov().git_sha()}; "
                  "these records' git_sha does NOT describe the code that produced them")
            env["SCION_ALLOW_DIRTY"] = "1"
        setup_errors = setup_data.prepare_data({(j.key.dataset, j.key.scene) for j in todo}, data_root)
        if any(levels[j] == "codec" for j in todo):
            try:
                env["TMC13_BINARY"] = setup_data.tmc3()
            except setup_data.SetupError as e:
                setup_errors.append(str(e))
        if setup_errors:
            raise SystemExit("[setup] failed:\n  " + "\n  ".join(setup_errors))

        if args.backend == "slurm":
            submit_slurm(todo, runs_root, data_root, env,
                         [sys.executable, "-m", "experiments.paper.reproduce", *args.targets, "--render-only",
                          "--runs-root", str(runs_root), "--paper-dir", str(paper_dir)])
            print("[slurm] submitted; the render job runs after the array finishes")
            return 0
        gpus = args.gpus.split(",") if args.gpus else visible_gpus()
        if not gpus:
            raise SystemExit("no GPUs visible; pass --gpus 0,1 (or use --backend slurm)")
        try:
            failed = run_local(todo, runs_root, data_root, env, gpus, args.jobs_per_gpu, args.allow_dirty)
        except KeyboardInterrupt:
            print("[run] interrupted; re-run the same command to continue (DONE jobs are skipped)")
            return 130

    print(f"[summary] done {len(todo) - len(failed)}, skipped {len(done)} (already DONE), failed {len(failed)}")
    for j, log in failed:
        print(f"[summary] FAILED {label(j)}  log: {log}")
    rc = render_phase(reg, args.targets, runs_root, paper_dir)
    return 1 if failed else rc


if __name__ == "__main__":
    sys.exit(main())
