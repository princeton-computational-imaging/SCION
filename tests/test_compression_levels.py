"""Compression levels raw < quantized < codec (scion.compression.levels).

CPU tests use ``position_format="fp16"`` unless a ``tmc3`` binary is found;
the GPU test at the bottom scores real renders against degradation budgets.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scion.compression import levels, omg_posthoc as omg, tmc13
from scion.compression.omg_posthoc import EncodeOptions
from scion.util.units import MiB, bits_per_gaussian
from test_compact_checkpoint import _make_tiny_codebook_model

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "run_json_v2" / "example_run.json"
CPU_CODEC = EncodeOptions(position_format="fp16")
POSITION_FORMATS = [
    "fp16",
    pytest.param("tmc13", marks=pytest.mark.skipif(
        not tmc13.is_tmc13_available(), reason="tmc3 binary not found")),
]


def _model():
    # T = 8*16 = 128 splat rows and N = 200 instances: both >= min_rows_for_svq,
    # so SVQ (splats) and quat-SVQ (instances) are exercised.
    return _make_tiny_codebook_model(K=8, S=16, N=200, T_max=160, N_max=256)


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def test_units():
    assert MiB == 1_048_576
    assert bits_per_gaussian(1000, 100) == 80.0
    assert bits_per_gaussian(3 * MiB, 3) == 8 * 2**20  # bytes, not MiB x 10^6


# ---------------------------------------------------------------------------
# config group
# ---------------------------------------------------------------------------

def test_codec_yaml_matches_encode_options_defaults():
    cfg = yaml.safe_load((ROOT / "config/compression/codec.yaml").read_text())
    level, opts = levels.from_config(cfg)
    assert level == "codec"
    assert opts == EncodeOptions()


def test_default_level_is_codec_and_others_select():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        assert compose(config_name="config").compression.level == "codec"
        for lvl in ("raw", "quantized"):
            cfg = compose(config_name="config", overrides=[f"compression={lvl}"])
            assert levels.from_config(cfg.compression)[0] == lvl


def test_unknown_level_rejected():
    with pytest.raises(ValueError, match="compression level"):
        levels.from_config({"level": "zip"})


# ---------------------------------------------------------------------------
# fail loudly at level codec
# ---------------------------------------------------------------------------

def test_missing_tmc3_raises_with_quantized_hint(monkeypatch):
    monkeypatch.setattr(tmc13, "_find_tmc3_binary", lambda: None)
    with pytest.raises(tmc13.TMC13UnavailableError, match="compression=quantized"):
        levels.encode(_model().state_dict(), "codec", EncodeOptions())
    # Each entry point names its own knob; levels below codec need no tmc3.
    with pytest.raises(tmc13.TMC13UnavailableError, match="--compression quantized"):
        levels.preflight("codec", skip_hint="--compression quantized")
    levels.preflight("quantized")
    levels.preflight("codec", CPU_CODEC)  # fp16 positions: no tmc3 needed


def test_tmc13_matcher_refuses_a_mismatched_reconstruction():
    grid = np.array([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(RuntimeError, match=r"does not match the input grid \(1 of 2"):
        tmc13._bipartite_match_by_int_key(grid, np.array([[0, 0, 0], [2, 2, 2]]))


def test_bad_tmc13_binary_env_is_not_silently_replaced(monkeypatch, tmp_path):
    monkeypatch.setenv("TMC13_BINARY", str(tmp_path / "nope"))
    assert tmc13._find_tmc3_binary() is None
    with pytest.raises(tmc13.TMC13UnavailableError, match="TMC13_BINARY="):
        tmc13.require_tmc3()


def test_missing_constriction_raises(monkeypatch):
    monkeypatch.setattr(omg, "constriction", None)
    with pytest.raises(ImportError, match="compression=quantized"):
        levels.encode(_model().state_dict(), "codec", CPU_CODEC)


def test_missing_sklearn_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    with pytest.raises(ImportError, match="compression=quantized"):
        levels.encode(_model().state_dict(), "codec", CPU_CODEC)


# ---------------------------------------------------------------------------
# encode -> decode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("position_format", POSITION_FORMATS)
def test_codec_decode_is_deterministic_and_valid(position_format):
    """SVQ / G-PCC are lossy, so check structure + determinism, not values."""
    model = _model()
    sd = model.state_dict()
    opts = replace(EncodeOptions(), position_format=position_format)
    blob, info = levels.encode(sd, "codec", opts)
    assert levels.encode(sd, "codec", opts)[0] == blob  # fixed seed -> same bytes
    assert info["position_format"] == position_format

    codec = levels.decode(blob, "codec")
    quant = levels.decode(levels.encode(sd, "quantized")[0], "quantized")
    assert codec.keys() == quant.keys()
    for k, q in quant.items():
        c = codec[k]
        assert c.shape == q.shape and c.dtype == q.dtype, k
        if c.is_floating_point():
            assert torch.isfinite(c).all(), k
    # Integer streams are lossless up to the per-instance permutation.
    assert torch.equal(codec["splat_candidate_id"], quant["splat_candidate_id"])
    assert torch.equal(codec["instance_assignments"].sort().values,
                       quant["instance_assignments"].sort().values)

    dst = _model()
    dst.load_state_dict(codec, strict=False)
    assert dst.num_points == model.num_points
    assert torch.isfinite(dst._assemble_gaussians().means).all()


def test_raw_level_is_lossless():
    model = _model()
    sd = levels.decode(levels.encode(model.state_dict(), "raw")[0], "raw")
    for k, v in sd.items():
        assert torch.equal(v, model.state_dict()[k]), k


# ---------------------------------------------------------------------------
# evaluate_levels -> run.json v2 blocks
# ---------------------------------------------------------------------------

def _fake_eval_fn(views: Path):
    # Distinct per-view / per-level rows so the means are checkable.
    off = {"raw": 0.0, "quantized": 0.1, "codec": 0.2}[views.name]
    rows = [{"psnr": 20.0 + i + off, "ssim": 0.5, "lpips": 0.3 + off} for i in range(4)]
    return rows, 2.0


def test_evaluate_levels_matches_run_json_v2(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    res = levels.evaluate_levels(
        model, level="codec", run_dir=tmp_path, eval_fn=_fake_eval_fn,
        options=CPU_CODEC,
    )

    assert res["metrics"].keys() == fixture["metrics"].keys()
    for lvl, m in res["metrics"].items():
        assert m.keys() == fixture["metrics"][lvl].keys()
        size = (tmp_path / res["artifacts"]["checkpoints"][lvl]).stat().st_size
        assert m["size_bytes"] == size
        assert m["bpg"] == bits_per_gaussian(size, m["n_gaussians"])
        assert m["n_gaussians"] == model.num_points
        assert (tmp_path / res["artifacts"]["eval_views"][lvl]).is_dir()
    assert res["metrics"]["raw"]["size_bytes"] > res["metrics"]["quantized"]["size_bytes"]
    assert res["metrics"]["codec"]["psnr"] == pytest.approx(21.7)

    assert res["codec"].keys() == fixture["codec"].keys()
    assert res["codec"]["options"] == asdict(CPU_CODEC)
    assert res["codec"]["position_format_used"] == "fp16"
    assert res["codec"]["tmc13_version"] is None
    for block in ("checkpoints", "eval_views"):
        assert res["artifacts"][block] == fixture["artifacts"][block]

    after = model.state_dict()  # restored to the raw (lossless) state
    for k, v in before.items():
        assert torch.equal(after[k], v), k


def test_each_level_is_scored_on_its_own_decoded_model(tmp_path):
    model = _model()
    with torch.no_grad():  # nonzero quantizer offset, so a dropped `+ zero` shows
        model.splat_features_dc += 2.0
    orig = model.state_dict()["splat_features_dc"].clone()
    seen = {}

    def eval_fn(views):
        seen[views.name] = model.splat_features_dc.detach().clone()
        return [{"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}], 1.0

    levels.evaluate_levels(
        model, level="codec", run_dir=tmp_path, eval_fn=eval_fn, options=CPU_CODEC,
    )
    assert not model.training
    assert torch.equal(seen["raw"], orig)

    q_ckpt = torch.load(tmp_path / "checkpoints/quantized.ckpt", weights_only=True)
    scale = float(q_ckpt["__quantized_meta__.splat_features_dc.scale"])
    err = (seen["quantized"] - orig).abs().max().item()
    assert 0 < err <= scale / 2 + 1e-5

    codec = levels.decode((tmp_path / "codec.omg").read_bytes(), "codec")
    assert torch.equal(seen["codec"], codec["splat_features_dc"])


def test_evaluate_levels_scores_eval_groups(tmp_path):
    res = levels.evaluate_levels(
        _model(), level="quantized", run_dir=tmp_path, eval_fn=_fake_eval_fn,
        subsets={"id": [0, 1], "ood": [3]},
    )
    assert res["metrics"]["quantized"]["psnr"] == pytest.approx(21.6)  # overall mean unchanged
    assert res["metrics"]["quantized"]["by_subset"] == {
        "id": pytest.approx({"psnr": 20.6, "ssim": 0.5, "lpips": 0.4, "n_views": 2}),
        "ood": pytest.approx({"psnr": 23.1, "ssim": 0.5, "lpips": 0.4, "n_views": 1})}
    assert res["metrics"]["raw"]["by_subset"]["ood"]["psnr"] == pytest.approx(23.0)


def test_evaluate_levels_stops_at_configured_level(tmp_path):
    res = levels.evaluate_levels(
        _model(), level="quantized", run_dir=tmp_path, eval_fn=_fake_eval_fn,
    )
    assert list(res["metrics"]) == ["raw", "quantized"]
    assert res["codec"] is None
    assert not (tmp_path / "codec.omg").exists()


# ---------------------------------------------------------------------------
# GPU: degradation budget of codec vs quantized on a trained checkpoint
# ---------------------------------------------------------------------------

# Measured on garden: SVQ M=3 costs -0.11 dB / +0.003
# LPIPS vs quantized, G-PCC q=1e-3 -0.04 dB, quaternion SVQ -0.05 dB, so the
# codec level sits ~0.2 dB / ~0.005 LPIPS below quantized. Budgets are ~2.5x.
MAX_PSNR_DROP_DB = 0.5
MAX_LPIPS_INCREASE = 0.01


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not tmc13.is_tmc13_available(), reason="tmc3 binary not found")
def test_codec_within_degradation_budget_of_quantized(tmp_path):
    fixture = os.environ.get("SCION_CODEC_FIXTURE")
    if not fixture:
        pytest.skip("set SCION_CODEC_FIXTURE to a trained garden run dir "
                    "(evals/runs/<slot>/mipnerf360/garden/seed<N>)")

    from scion.engine.checkpoint import load
    from scion.engine.eval import eval_views

    loaded = load(Path(fixture), level="raw")
    res = levels.evaluate_levels(
        loaded.model, level="codec", run_dir=tmp_path,
        eval_fn=lambda d: eval_views(loaded.model, loaded.scene, d),
    )

    q, c = res["metrics"]["quantized"], res["metrics"]["codec"]
    assert res["codec"]["position_format_used"] == "tmc13"
    assert c["size_bytes"] < q["size_bytes"]
    assert q["psnr"] - c["psnr"] <= MAX_PSNR_DROP_DB, (q, c)
    assert c["lpips"] - q["lpips"] <= MAX_LPIPS_INCREASE, (q, c)
