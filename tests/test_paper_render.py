"""Paper registry + renderers (experiments/paper/). CPU-only.

The golden test renders every table target from a synthetic pool generated
into a temp dir (tests/fixtures/paper/generate_pool.py) and diffs against the
committed tests/fixtures/paper/expected/ byte-for-byte. After an intended
layout change: ``python tests/fixtures/paper/generate_pool.py``.
"""
from __future__ import annotations

import importlib
import json
import shutil
import statistics
import sys
from pathlib import Path

import pytest

from experiments.paper import render
from experiments.paper.common import REPO, Baselines, RenderError
from experiments.paper.run_capacity_sweep import frontier
from experiments.paper.targets import REGISTRY, load_registry

FIX = Path(__file__).parent / "fixtures" / "paper"
sys.path.insert(0, str(FIX))
import generate_pool  # noqa: E402

MIP = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"]


@pytest.fixture(scope="module")
def pool(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pool")
    generate_pool.write_pool(root)
    return root


def _render(pool: Path, target: str, out: Path, runs: Path = None) -> int:
    return render.main([target, "--runs-root", str(runs or pool / "runs"),
                        "--out", str(out)])


@pytest.fixture(scope="module")
def rendered(pool, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("out")
    assert _render(pool, "all", out) == 0
    return out / "tables"


# ---------------------------------------------------------------- registry

def test_every_target_resolves():
    reg = load_registry()
    for name, t in reg.targets.items():
        if t.renderer:
            assert hasattr(importlib.import_module(f"experiments.paper.{t.renderer}"), "render"), name
            assert t.jobs, name
        else:
            assert t.jobs and "manual" in t.spec, f"figure target {name} needs jobs and a manual note"
    paths = [j.key.path("") for j in reg.jobs()]
    assert len(paths) == len(set(paths))


@pytest.mark.parametrize("extra, error", [
    # `<v>_abl_x_y` and `<v>_abl_x` + tag `y` would share a run path.
    ("  a:\n    jobs: [{variants: [ours_small], ablations: [x_y], datasets: [tandt]}]\n"
     "  b:\n    jobs: [{variants: [ours_small], ablations: [x], datasets: [tandt], tag: y}]\n",
     "share run path"),
    ("  a:\n    jobs: [{datasets: [tandt], seeds: [0, 0, 1]}]\n", "duplicate seeds"),
    ("  a:\n    jobs: [{target: compute, scenes: [garden]}]\n", "reuse entry"),
    ("  a:\n    jobs: [{target: no_such_target}]\n", "reuse entry"),
])
def test_registry_refuses(tmp_path, extra, error):
    (tmp_path / "t.yaml").write_text(REGISTRY.read_text().replace("targets:\n", "targets:\n" + extra, 1))
    with pytest.raises(ValueError, match=error):
        load_registry(tmp_path / "t.yaml")


def test_every_registry_job_composes():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        for overrides in sorted({j.overrides for j in load_registry().jobs()}):
            OmegaConf.to_yaml(compose("config", overrides=list(overrides)), resolve=True)


# ---------------------------------------------------------------- golden

def test_golden_tables(rendered):
    got = {p.name: p.read_bytes() for p in rendered.iterdir()}
    want = {p.name: p.read_bytes() for p in (FIX / "expected").iterdir()}
    assert sorted(got) == sorted(want)
    for name in want:
        assert got[name] == want[name], f"{name}: re-render with generate_pool.py if intended"


def test_oracle_cells(pool, rendered):
    """Recompute a few table_1 cells straight from the fixture JSON."""
    table = (rendered / "table_1_main_results.tex").read_text()

    def cell(level, field, scale, decimals):
        per_seed = []
        for seed in range(3):
            recs = [json.loads((pool / "runs" / "ours_small" / "mipnerf360" / s / f"seed{seed}"
                                / "run.json").read_text()) for s in MIP]
            per_seed.append(statistics.fmean(r["metrics"][level][field] / scale for r in recs))
        m, s = statistics.fmean(per_seed), statistics.stdev(per_seed)
        return f"${m:.{decimals}f}_{{\\pm {s:.{decimals}f}}}$"

    for level, field, scale, decimals in [("quantized", "size_bytes", 2 ** 20, 2),
                                          ("codec", "size_bytes", 2 ** 20, 3),  # < 1 MB
                                          ("quantized", "psnr", 1, 2)]:
        assert cell(level, field, scale, decimals) in table, (level, field)


# ---------------------------------------------------------------- refusals

def _edit(path: Path, fn) -> None:
    rec = json.loads(path.read_text())
    fn(rec)
    path.write_text(json.dumps(rec))


A1 = "ours_small_abl_a/mipnerf360/garden/seed1"
CASES = {
    "mixed git_sha": lambda r: _edit(r / A1 / "run.json", lambda d: d.update(git_sha="f" * 40)),
    "mixed split": lambda r: _edit(r / A1 / "run.json", lambda d: d["split"].update(eval_images_sha256="0" * 64)),
    "mixed split.data_sha256": lambda r: _edit(r / A1 / "run.json", lambda d: d["split"].update(data_sha256="0" * 64)),
    "split.data_sha256": lambda r: _edit(r / A1 / "run.json", lambda d: d["split"].pop("data_sha256")),
    "seed3": lambda r: shutil.copytree(r / A1, r / A1.replace("seed1", "seed3")),
    "claims job": lambda r: _edit(r / A1 / "run.json", lambda d: d["job"].update(seed=2)),
    "schema_version": lambda r: _edit(r / A1 / "run.json", lambda d: d.update(schema_version=1)),
    "no metrics.codec": lambda r: _edit(r / A1 / "run.json", lambda d: d["metrics"].pop("codec")),
    "status RUNNING": lambda r: _edit(r / A1 / "run.json", lambda d: d.update(status="RUNNING")),
    "dirty tree": lambda r: _edit(r / A1 / "run.json", lambda d: d.update(git_dirty=True)),
    "slot 'ours_small'": lambda r: _edit(r / A1 / "run.json", lambda d: d.update(slot="ours_small")),
    "trained without": lambda r: _edit(r / A1 / "run.json", lambda d: d["overrides"].remove("ablation=a")),
    # A reproduce --override smoke record: the job's overrides plus extras.
    "config_sha256 is not": lambda r: _edit(r / A1 / "run.json", lambda d: d.update(
        overrides=d["overrides"] + ["trainer.max_num_iterations=300"], config_sha256="0" * 64)),
    "malformed record": lambda r: _edit(r / A1 / "run.json", lambda d: d["metrics"]["quantized"].update(psnr="x")),
    "missing": lambda r: (r / A1 / "run.json").unlink(),
}


@pytest.mark.parametrize("error", sorted(CASES))
def test_refusal_names_the_record(pool, tmp_path, capsys, error):
    runs = tmp_path / "runs"
    shutil.copytree(pool / "runs", runs)
    CASES[error](runs)
    assert _render(pool, "component_ablation", tmp_path / "out", runs) == 1
    err = capsys.readouterr().err
    assert error in err and ("mixed" in error or A1.split("/")[0] in err), err
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------- pieces

def test_baseline_gaps(tmp_path):
    y = tmp_path / "b.yaml"
    y.write_text("A:\n  source: 'author rerun; run settings: TODO(author)'\n  size_unit: MB_1e6\n"
                 "  datasets:\n    d: {_dataset_mean: {psnr: 1.0, mb: 1.760, ssim: TODO, lpips: ~}}\n"
                 "B:\n  required: false\n  size_unit: MiB\n"
                 "  datasets: {d: {_dataset_mean: {psnr: 2.0}}}\n")
    b = Baselines(y)
    assert round(b.cell("A", "d", "_dataset_mean", "mb"), 3) == 1.678   # 1e6 B -> MiB
    assert b.cell("A", "d", "_dataset_mean", "psnr") == 1.0             # TODO in prose is fine
    assert b.cell("A", "d", "_dataset_mean", "lpips") is None          # explicitly not reported
    for metric in ("ssim", "missing_key"):
        with pytest.raises(RenderError):
            b.cell("A", "d", "_dataset_mean", metric)
    assert b.cell("B", "d", "_dataset_mean", "psnr") is None and b.warnings  # no source


def test_frontier():
    pts = [(1.0, 20.0), (2.0, 22.0), (2.5, 21.0), (3.0, 22.0), (0.5, 19.0)]
    assert frontier(pts, +1) == [4, 0, 1]
    assert frontier([(1.0, 0.4), (2.0, 0.3), (2.0, 0.5)], -1) == [0, 1]
