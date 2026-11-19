"""Tests for compact CodebookModel checkpoint save/load.

``CodebookModel`` keeps flat ``[T_max, dims]`` splat buffers pre-allocated
to ``max_candidates * max_splats_per_candidate`` (often >100k rows) even when
only a small fraction are active, so the raw ``state_dict`` is large
regardless of usage.  The compact hooks strip inactive rows
on save; load resizes the buffers to the saved shapes.

These tests build a hand-rolled CodebookModel (skipping the heavy
``populate_modules`` path) and exercise the hooks directly.

They only touch ``CodebookModel.state_dict`` / ``load_state_dict`` and
the ``scion.engine.checkpoint`` save path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from scion.models.codebook import CodebookModel, CodebookModelConfig
from scion.models.codebook_base import SPLAT_FREE


# ---------------------------------------------------------------------------
# Minimal CodebookModel fixture — bypasses populate_modules so we do not pay
# for LPIPS/DINO loading or a point-cloud init.
# ---------------------------------------------------------------------------


def _make_tiny_codebook_model(
    *,
    K: int = 4,
    S: int = 3,
    N: int = 6,
    T_max: int = 32,
    N_max: int = 64,
) -> CodebookModel:
    """Return a minimally initialized CodebookModel with K·S active splats.

    Constructs the buffers manually to match what ``populate_modules``
    would produce, then wires up just enough state for
    ``_assemble_gaussians`` to run end-to-end.  LPIPS / DINO / discriminator
    are intentionally skipped.
    """
    T_active = K * S
    assert T_active <= T_max and N <= N_max

    # Bypass CodebookModel.__init__ — it wants a point cloud, LPIPS and DINO.
    model = CodebookModel.__new__(CodebookModel)
    nn.Module.__init__(model)

    # Config: a CodebookModelConfig default is sufficient; we override fields
    # we care about so ``load_state_dict`` can read ``max_num_iterations`` etc.
    cfg = CodebookModelConfig()
    cfg.num_candidates = K
    cfg.initial_splats_per_candidate = S
    cfg.max_candidates = K
    cfg.max_splats_per_candidate = S
    cfg.num_instances = N
    cfg.max_num_instances = N_max
    cfg.max_num_iterations = 100  # attribute read by load_state_dict
    model.config = cfg
    model.step = 0

    torch.manual_seed(0)
    # --- Splat-level Parameters + buffer ---
    splat_means = torch.randn(T_max, 3) * 0.05
    splat_scales = torch.randn(T_max, 3) * 0.1 - 2.0
    splat_quats = F.normalize(torch.randn(T_max, 4), dim=-1)
    # Identity quaternion on free rows to match populate_modules convention.
    splat_opacities = torch.logit(torch.full((T_max, 1), 0.5))
    splat_features_dc = torch.rand(T_max, 3)

    model.register_parameter("splat_means", Parameter(splat_means))
    model.register_parameter("splat_scales", Parameter(splat_scales))
    model.register_parameter("splat_quats", Parameter(splat_quats))
    model.register_parameter("splat_opacities", Parameter(splat_opacities))
    model.register_parameter("splat_features_dc", Parameter(splat_features_dc))

    cid = torch.full((T_max,), SPLAT_FREE, dtype=torch.long)
    for k in range(K):
        cid[k * S : (k + 1) * S] = k
    model.register_buffer("splat_candidate_id", cid)

    free_list = torch.zeros(T_max, dtype=torch.long)
    n_free = T_max - T_active
    free_list[:n_free] = torch.arange(T_active, T_max)
    model.register_buffer("free_list", free_list)
    model.free_count = n_free

    # --- Instance-level Parameters + buffers ---
    model.register_parameter("instance_positions", Parameter(torch.randn(N, 3)))
    model.register_parameter(
        "instance_rotations",
        Parameter(F.normalize(torch.randn(N, 4), dim=-1).contiguous()),
    )
    model.register_parameter("instance_scales", Parameter(torch.zeros(N, 1)))
    model.register_parameter(
        "instance_opacities", Parameter(torch.zeros(N, 1))
    )
    assignments = torch.arange(N, dtype=torch.long) % K
    model.register_buffer("instance_assignments", assignments)

    # --- CodebookMixin init (register_buffers=False: already registered) ---
    model.init_codebook(
        spatial_dim=3, rot_dim=4, volume_exp=1.0 / 3.0,
        p_splat=14, p_inst=9,
        K=K, N=N, T_max=T_max,
        register_buffers=False,
    )

    # Hook state expected by CodebookModel downstream paths.
    model._pending_compact_load = None
    model._assembly_global_idx = None
    model._assembly_inst_idx = None
    model._last_assignment_per_gaussian = None
    model.dino_oracle = None
    model.discriminator = None
    model._d_optimizer = None
    model._adv_metrics = {}
    model._d_last_metrics = {}
    model._d_forward = None
    model.free_count = n_free  # init_codebook clobbers this
    model.num_candidates = K
    model.num_instances = N

    # Populate forward + assembly caches.
    model.recompute_assembly_indices()
    return model


def _assemble_snapshot(model: CodebookModel) -> dict:
    """Run ``_assemble_gaussians`` and capture all world-space tensors."""
    bundle = model._assemble_gaussians()
    return {
        "means": bundle.means.detach().clone(),
        "scales": bundle.scales.detach().clone(),
        "quats": bundle.quats.detach().clone(),
        "opacities": bundle.opacities.detach().clone(),
        "features_dc": bundle.features_dc.detach().clone(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_state_dict_has_compact_metadata():
    """Saved state_dict carries ``__codebook_meta__.*`` header keys."""
    model = _make_tiny_codebook_model()
    sd = model.state_dict()
    meta_keys = [k for k in sd if CodebookModel._COMPACT_META_PREFIX in k]
    assert any(k.endswith("T_max") for k in meta_keys)
    assert any(k.endswith("N_max") for k in meta_keys)
    assert any(k.endswith("num_candidates") for k in meta_keys)
    assert any(k.endswith("num_instances") for k in meta_keys)
    # T_max collapses to T_active, so there is no free-row state to store.
    assert not any(k.endswith("free_count") for k in meta_keys)


def test_compact_save_is_smaller_than_naive():
    """Compact save must strip inactive rows — size strictly smaller than raw."""
    # K·S = 12 active rows in T_max = 32 -> ~12/32 of splat-row bytes.
    model = _make_tiny_codebook_model(K=4, S=3, N=6, T_max=64)

    compact_sd = model.state_dict()
    # Simulate a "naive" full-size state_dict by bypassing our override.
    naive_sd = nn.Module.state_dict(model)

    def _bytes(sd):
        return sum(
            v.numel() * v.element_size()
            for v in sd.values()
            if isinstance(v, torch.Tensor)
        )

    compact_bytes = _bytes(compact_sd)
    naive_bytes = _bytes(naive_sd)
    assert compact_bytes < naive_bytes * 0.5, (
        f"Compact ({compact_bytes}) should be <50% of naive ({naive_bytes})"
    )


def test_save_load_roundtrip_preserves_forward_pass():
    """Full roundtrip: save -> torch.save -> load -> fresh model matches."""
    src = _make_tiny_codebook_model(K=4, S=3, N=6, T_max=64)

    # Snapshot assembly *before* any mutation.
    pre = _assemble_snapshot(src)

    # Save via compact path.
    sd = src.state_dict()
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        torch.save(sd, f.name)
        ckpt_path = f.name

    try:
        # Confirm the file is smaller than a naive save.
        naive_sd = nn.Module.state_dict(src)
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f2:
            torch.save(naive_sd, f2.name)
            naive_path = f2.name
        try:
            size_compact = os.path.getsize(ckpt_path)
            size_naive = os.path.getsize(naive_path)
            assert size_compact < size_naive, (
                f"compact={size_compact} naive={size_naive}"
            )
        finally:
            os.unlink(naive_path)

        # --- Load into a FRESH model with different random init. ---
        torch.manual_seed(1234)
        dst = _make_tiny_codebook_model(K=4, S=3, N=6, T_max=64)
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        dst.load_state_dict(loaded, strict=True)

        # 1. K / N restored. T_max in the new format collapses to T_active.
        src_active = int((src.splat_candidate_id != SPLAT_FREE).sum().item())
        assert dst.T_max == src_active
        assert dst.num_candidates == src.num_candidates
        assert dst.num_instances == src.num_instances

        # 2. splat_candidate_id active entries match (free entries are
        #    dropped entirely in the new format, so the dst tensor is
        #    shorter and every entry is != SPLAT_FREE).
        src_active_mask = src.splat_candidate_id != SPLAT_FREE
        assert torch.equal(
            torch.sort(src.splat_candidate_id[src_active_mask]).values,
            torch.sort(dst.splat_candidate_id).values,
        )
        assert (dst.splat_candidate_id != SPLAT_FREE).all()

        # 3. instance_assignments byte-equal.
        assert torch.equal(dst.instance_assignments, src.instance_assignments)

        # 4. free_list is empty in the new format; free_count == 0.
        assert dst.free_count == 0
        assert dst.free_list.numel() == 0

        # 5. Forward-pass snapshot matches exactly (active rows preserved;
        #    inactive rows never read in either version).
        post = _assemble_snapshot(dst)
        for key in pre:
            assert torch.allclose(pre[key], post[key], atol=1e-6), (
                f"Mismatch on {key}: max diff {(pre[key] - post[key]).abs().max()}"
            )

        # 6. splat-Parameter active rows match (after gathering active rows
        #    on both sides; rows are repacked in the new format).
        src_perm = src_active_mask.nonzero(as_tuple=True)[0]
        for name in ("splat_means", "splat_scales", "splat_quats",
                     "splat_opacities", "splat_features_dc"):
            s = getattr(src, name)[src_perm]
            d = getattr(dst, name)
            assert torch.allclose(s, d, atol=1e-6), f"{name} active rows differ"
    finally:
        os.unlink(ckpt_path)


def test_load_accepts_uncompacted_state_dict():
    """An uncompacted state_dict (training-state resume) loads cleanly."""
    src = _make_tiny_codebook_model(K=4, S=3, N=6, T_max=64)
    pre = _assemble_snapshot(src)

    full_sd = nn.Module.state_dict(src)  # full-size, no __codebook_meta__

    torch.manual_seed(42)
    dst = _make_tiny_codebook_model(K=4, S=3, N=6, T_max=64)
    dst.load_state_dict(full_sd, strict=True)

    post = _assemble_snapshot(dst)
    for key in pre:
        assert torch.allclose(pre[key], post[key], atol=1e-6)


def test_size_bytes_is_plausible():
    """``size_bytes`` returns a small positive number."""
    model = _make_tiny_codebook_model()
    raw = model.size_bytes("raw")
    assert 0 < raw < 1_000_000, f"raw={raw} bytes is implausibly large for tiny model"


def test_engine_checkpoint_save_roundtrip(tmp_path: Path):
    """``scion.engine.checkpoint.save`` writes the same compact bytes.

    This test drives the checkpoint envelope end-to-end at the file layer:
    we save a tiny model
    via ``engine.checkpoint.save``, then re-read the resulting
    ``.ckpt`` + ``meta.json`` by hand and verify

    1. the on-disk state-dict matches the model's compact ``state_dict()``
       (so the compact format flows through the envelope);
    2. ``meta.json`` has the expected schema and points at the primary
       ckpt;
    3. loading the on-disk state-dict back into a fresh tiny model with
       ``load_state_dict`` reproduces the forward pass.

    The full ``engine.checkpoint.load(...)`` entry point requires a
    config snapshot that can rebuild a CodebookModel via
    ``populate_modules`` — out of scope for this CPU-only fixture, which
    explicitly bypasses populate_modules. The state-dict equivalence
    check exercises everything load needs to do that doesn't require
    standing up gsplat / LPIPS.
    """
    import json

    from scion.engine.checkpoint import (
        CHECKPOINTS_SUBDIR,
        CKPT_FILENAME_TMPL,
        META_FILENAME,
        save,
    )

    src = _make_tiny_codebook_model(K=4, S=3, N=6, T_max=64)
    pre = _assemble_snapshot(src)

    primary = save(
        model=src,
        step=42,
        output_dir=tmp_path,
        scene_path=tmp_path / "fake_scene",
        config_snapshot={"num_train_data": 0},  # not consumed in this test
    )

    # Layout assertions.
    assert primary == tmp_path / CHECKPOINTS_SUBDIR / CKPT_FILENAME_TMPL.format(step=42)
    assert primary.is_file()
    meta_path = tmp_path / CHECKPOINTS_SUBDIR / META_FILENAME
    assert meta_path.is_file()

    # meta.json shape.
    meta = json.loads(meta_path.read_text())
    assert meta["step"] == 42
    assert meta["format_version"] >= 1
    assert meta["primary_ckpt"].endswith(CKPT_FILENAME_TMPL.format(step=42))
    assert meta["scene_path"] == str(tmp_path / "fake_scene")

    # On-disk state-dict equivalence (modulo the ``_model.`` key prefix
    # that save adds).
    payload = torch.load(primary, map_location="cpu", weights_only=True)
    on_disk = payload["pipeline"]
    assert all(k.startswith("_model.") for k in on_disk)
    stripped = {k[len("_model."):]: v for k, v in on_disk.items()}

    compact = src.state_dict()
    assert set(stripped) == set(compact)
    for k in compact:
        a, b = compact[k], stripped[k]
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), f"{k} mismatches on disk"
        else:
            assert a == b

    # Re-hydrate a fresh tiny model from the on-disk state_dict —
    # exercises the load_state_dict half of the round-trip without
    # needing populate_modules.
    torch.manual_seed(1234)
    dst = _make_tiny_codebook_model(K=4, S=3, N=6, T_max=64)
    dst.load_state_dict(stripped, strict=True)
    post = _assemble_snapshot(dst)
    for key in pre:
        assert torch.allclose(pre[key], post[key], atol=1e-6)
