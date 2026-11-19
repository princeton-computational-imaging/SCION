"""Regression tests for the codec (OMG-style) bundle's encode / decode
round trip, independent of the model. Rendering the decoded model is what
``levels.evaluate_levels`` scores on every codec run.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from scion.compression import omg_posthoc as omg


def _synthetic_quantized():
    """Tiny state_dict shaped like a real ``quantized.ckpt``."""
    torch.manual_seed(0)
    np.random.seed(0)
    N, K, T = 256, 32, 96
    return {
        "_model.instance_positions": torch.randn(N, 3).half(),
        "_model.instance_rotations": torch.randn(N, 4).half(),
        "_model.instance_scales": torch.randn(N, 1).half(),
        "_model.instance_opacities": torch.randint(0, 256, (N, 1), dtype=torch.uint8),
        "_model.instance_assignments": torch.randint(0, K, (N,), dtype=torch.int64),
        "_model.splat_means": torch.randn(T, 3).half(),
        "_model.splat_scales": torch.randn(T, 3).half(),
        "_model.splat_quats": torch.randn(T, 4).half(),
        "_model.splat_features_dc": torch.randint(0, 256, (T, 3), dtype=torch.uint8),
        "_model.splat_opacities": torch.randint(0, 256, (T, 1), dtype=torch.uint8),
        "_model.splat_candidate_id": torch.randint(0, K, (T,), dtype=torch.int64),
        "_model.__codebook_meta__.num_candidates": torch.tensor(K, dtype=torch.int64),
        "_model.__codebook_meta__.num_instances": torch.tensor(N, dtype=torch.int64),
        "_model.__codebook_meta__.T_max": torch.tensor(T, dtype=torch.int64),
        "_model.__codebook_meta__.N_max": torch.tensor(N, dtype=torch.int64),
        "__quantized_meta__._model.splat_features_dc.scale": torch.tensor(1.0),
        "__quantized_meta__._model.splat_features_dc.zero": torch.tensor(0.0),
        "__quantized_meta__._model.instance_opacities.scale": torch.tensor(1.0 / 255),
        "__quantized_meta__._model.instance_opacities.zero": torch.tensor(0.0),
        "__quantized_meta__._model.splat_opacities.scale": torch.tensor(1.0 / 255),
        "__quantized_meta__._model.splat_opacities.zero": torch.tensor(0.0),
    }


def _encode_and_wrap_for_torch_save(state):
    """Mimic the encoder side of the codec encoder."""
    options = omg.EncodeOptions(
        M=3, B=64, position_format="fp16",
        svq_instance_attrs=True, instance_svq_M=4, instance_svq_B=64,
    )
    blob, _ = omg.encode_quantized_to_omg(state, options)
    blob_t = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
    return {"step": 123, "pipeline": {"__omg_blob__": blob_t}}


def test_save_load_roundtrip_via_torch_save():
    """On-disk uint8 tensor format round-trips via torch.save / torch.load."""
    state = _synthetic_quantized()
    wrapped = _encode_and_wrap_for_torch_save(state)

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        torch.save(wrapped, f.name)
        path = Path(f.name)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["step"] == 123
    pipe_state = loaded["pipeline"]
    assert "__omg_blob__" in pipe_state
    blob_t = pipe_state["__omg_blob__"]
    assert isinstance(blob_t, torch.Tensor)
    assert blob_t.dtype == torch.uint8
    # Convert back to bytes (matches the load_pipeline branch).
    blob = bytes(blob_t.cpu().numpy().tobytes())
    decoded = omg.decode_omg_to_state_dict(blob)
    # Should carry both kinds of meta keys after decoding.
    assert any("__quantized_meta__" in k for k in decoded)
    assert any("__codebook_meta__" in k for k in decoded)
    # Some recognizable tensors should be back.
    assert "_model.instance_positions" in decoded
    assert "_model.splat_candidate_id" in decoded


def test_uint8_tensor_format_has_low_overhead():
    """Wrapping the OMG blob as a uint8 tensor must NOT inflate by >5%.

    torch.save serializes bytes via base64 (1.5× inflation) but stores
    uint8 tensors natively. The wrapping choice in
    the codec encoder matters because the blob is
    already LZMA-compressed — re-encoding it costs real bytes.
    """
    state = _synthetic_quantized()
    wrapped = _encode_and_wrap_for_torch_save(state)
    blob_size = wrapped["pipeline"]["__omg_blob__"].numel()

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        torch.save(wrapped, f.name)
        on_disk = Path(f.name).stat().st_size

    overhead = on_disk / blob_size
    # The wire format adds the torch dict header + a small zip-file
    # bookkeeping section; for a 1+ KB blob this should be well under
    # 50% relative. (Empirical: 0.3% on a 567 KB blob.)
    assert overhead < 1.5, (
        f"on-disk {on_disk} / blob {blob_size} = {overhead:.3f}; "
        f"should be < 1.5 (uint8 tensor format)."
    )
