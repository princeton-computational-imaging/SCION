"""``scion.util.run_index`` on run.json v2. CPU-only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scion.util.run_index import JobKey, flatten, read
from scion.util.stats import mean_std

EXAMPLE = Path(__file__).parent / "fixtures" / "run_json_v2" / "example_run.json"


def test_slot_and_path():
    key = JobKey("ours_small", None, "mipnerf360", "garden", 0)
    assert key.slot == "ours_small"
    assert key.path("r") == Path("r/ours_small/mipnerf360/garden/seed0/run.json")
    assert JobKey("ours_large", "no_inst_dc", "tandt", "truck", 2).slot == "ours_large_abl_no_inst_dc"
    assert JobKey("ours_small", None, "mipnerf360", "garden", 1, "python").slot == "ours_small_python"
    assert JobKey("ours_small", "a", "mipnerf360", "garden", 1, "cuda").slot == "ours_small_abl_a_cuda"


def test_read_example_fixture():
    run = read(EXAMPLE)
    assert run.key == JobKey("ours_small", None, "mipnerf360", "garden", 0)
    assert run.status == "DONE" and len(run.git_sha) == 40
    # Levels flatten to <level>.<metric>: no collision between levels.
    assert run.metrics["raw.psnr"] == 21.862
    assert run.metrics["quantized.psnr"] == 21.848
    assert run.metrics["codec.size_bytes"] == 582853
    assert run.metrics["model_stats.effective_candidates"] == 115.84
    assert run.metrics["train_time_s"] == 1066.0
    assert run.metrics["model_stats.num_splats"] == 4006 and run.metrics["median_step_s"] == 0.031


def test_flatten_drops_non_finite():
    out = flatten({"metrics": {"codec": {"psnr": float("nan"), "ssim": 0.5,
                                         "by_subset": {"id": {"psnr": 18.1, "n_views": 10}}}}})
    assert "codec.psnr" not in out and out["codec.ssim"] == 0.5
    assert out["codec.by_subset.id.psnr"] == 18.1 and out["codec.by_subset.id.n_views"] == 10


def test_read_rejects_v1_and_malformed(tmp_path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"schema_version": 1, "status": "DONE"}))
    with pytest.raises(ValueError, match="schema_version"):
        read(p)
    good = json.loads(EXAMPLE.read_text())
    for bad in ({**good, "job": "garden"}, {**good, "metrics": {"raw": 3}},
                {**good, "metrics": {"raw": {"psnr": "high"}}}, [1, 2]):
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=str(p)):
            read(p)


def test_mean_std_is_sample_std():
    assert mean_std([1.0, 2.0, 3.0]) == (2.0, 1.0)
    with pytest.raises(ValueError):
        mean_std([1.0])
