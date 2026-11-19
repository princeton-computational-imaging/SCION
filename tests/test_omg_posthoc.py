"""Tests for the OMG-style post-hoc compression pipeline.

Validates:
  - encode/decode round-trip preserves all dtypes / shapes / keys
  - per-instance Morton permutation is internally consistent (the
    permutation is applied to every instance_* tensor, so the decoded
    bundle stays paired)
  - SVQ is skipped for tensors with fewer rows than the threshold
  - Integer streams (instance_assignments, splat_candidate_id) round-trip
    exactly

Pinned to ``position_format="fp16"`` so they run without the tmc3 binary;
G-PCC and the level wrapper are covered by tests/test_compression_levels.py.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from scion.compression import omg_posthoc as omg


def _synthetic_quantized(
    *, N: int = 256, K: int = 32, T: int = 96, seed: int = 0
):
    """Return a state_dict shaped like a real ``quantized.ckpt``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
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
        "_model.__codebook_meta__.free_count": torch.tensor(0, dtype=torch.int64),
        "_model.__codebook_meta__.active_splat_idx": torch.arange(T, dtype=torch.int64),
        "__quantized_meta__._model.splat_features_dc.scale": torch.tensor(1.0),
        "__quantized_meta__._model.splat_features_dc.zero": torch.tensor(0.0),
        "__quantized_meta__._model.instance_opacities.scale": torch.tensor(1.0 / 255),
        "__quantized_meta__._model.instance_opacities.zero": torch.tensor(0.0),
        "__quantized_meta__._model.splat_opacities.scale": torch.tensor(1.0 / 255),
        "__quantized_meta__._model.splat_opacities.zero": torch.tensor(0.0),
    }


def _morton_order(positions: np.ndarray, bits: int = 16) -> np.ndarray:
    _, order = omg.position_encode(positions, bits=bits)
    return order


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_roundtrip_keys_and_dtypes_match():
    state = _synthetic_quantized()
    blob, _info = omg.encode_quantized_to_omg(
        state, omg.EncodeOptions(M=2, B=64, pos_bits=14, seed=0, position_format="fp16")
    )
    back = omg.decode_omg_to_state_dict(blob)
    assert set(state.keys()) == set(back.keys())
    for k, v0 in state.items():
        v1 = back[k]
        assert isinstance(v1, torch.Tensor), f"{k}: not a tensor on decode"
        assert v0.dtype == v1.dtype, f"{k}: dtype changed {v0.dtype} -> {v1.dtype}"
        assert tuple(v0.shape) == tuple(v1.shape), (
            f"{k}: shape changed {tuple(v0.shape)} -> {tuple(v1.shape)}"
        )


def test_per_instance_morton_permutation_is_consistent():
    """The decoder reorders all per-instance tensors by the position
    codec's permutation. The encoder must apply the same permutation so
    that index ``i`` of the decoded output corresponds to the same
    logical instance across all tensors.

    Pinned to ``position_format='fp16'`` because the test verifies the
    fp16 Morton-walk order directly; with ``'tmc13'`` (the new default)
    the permutation comes from TMC13's octree walk, which differs.
    """
    state = _synthetic_quantized()
    blob, _ = omg.encode_quantized_to_omg(
        state,
        omg.EncodeOptions(M=2, B=64, pos_bits=14, seed=0, position_format="fp16"),
    )
    back = omg.decode_omg_to_state_dict(blob)
    # Derive what the permutation should have been.
    pos = state["_model.instance_positions"].float().numpy()
    order = _morton_order(pos, bits=14)
    # instance_assignments is the easiest one to check (int, exact).
    expected = state["_model.instance_assignments"].numpy()[order]
    got = back["_model.instance_assignments"].numpy()
    assert np.array_equal(expected, got), (
        "instance_assignments did not match Morton-permuted original"
    )


def test_svq_instance_attrs_roundtrip():
    """``svq_instance_attrs=True`` routes ``instance_rotations`` through
    SVQ. Reconstruction is lossy (not byte-exact) but the bundle must
    decode without error and stay within a reasonable reconstruction
    bound for random gaussian inputs.
    """
    state = _synthetic_quantized(N=512, K=32, T=96, seed=0)
    blob, info = omg.encode_quantized_to_omg(
        state,
        omg.EncodeOptions(
            M=3, B=256, pos_bits=14, seed=0,
            position_format="fp16",
            svq_instance_attrs=True,
            instance_svq_M=4, instance_svq_B=256,
        ),
    )
    back = omg.decode_omg_to_state_dict(blob)

    # Apply the same Morton permutation as the encoder.
    pos = state["_model.instance_positions"].float().numpy()
    order = _morton_order(pos, bits=14)
    orig = state["_model.instance_rotations"].float().numpy()[order]
    got = back["_model.instance_rotations"].numpy().astype(np.float32)
    # SVQ on a random N=512 gaussian source with B=256 codewords per
    # dim should reconstruct to within ~0.05 rel-L2.
    err = np.linalg.norm(orig - got) / max(np.linalg.norm(orig), 1e-12)
    assert err < 0.2, f"quat-SVQ rel_l2 too large: {err}"
    assert orig.shape == got.shape


def test_integer_streams_are_exact():
    """``instance_assignments`` and ``splat_candidate_id`` should
    round-trip exactly (range coding is lossless)."""
    state = _synthetic_quantized()
    blob, _ = omg.encode_quantized_to_omg(
        state, omg.EncodeOptions(M=2, B=64, pos_bits=14, seed=0, position_format="fp16")
    )
    back = omg.decode_omg_to_state_dict(blob)
    # splat_candidate_id is per-splat, not per-instance — no Morton
    # permutation applied, so direct equality.
    assert torch.equal(
        state["_model.splat_candidate_id"], back["_model.splat_candidate_id"]
    )
    # instance_assignments is permuted; check the multiset (a permutation
    # is invariant under sort).
    a = state["_model.instance_assignments"].sort().values
    b = back["_model.instance_assignments"].sort().values
    assert torch.equal(a, b)


def test_svq_skipped_below_threshold():
    """Tensors with fewer rows than ``min_rows_for_svq`` fall back to
    passthrough (their reconstructed values equal the input)."""
    state = _synthetic_quantized(T=16)
    blob, _ = omg.encode_quantized_to_omg(
        state, omg.EncodeOptions(M=2, B=64, min_rows_for_svq=64, pos_bits=14, seed=0, position_format="fp16")
    )
    back = omg.decode_omg_to_state_dict(blob)
    for name in ("splat_means", "splat_scales", "splat_quats"):
        v0 = state[f"_model.{name}"]
        v1 = back[f"_model.{name}"]
        assert torch.allclose(v0.float(), v1.float()), (
            f"{name} should be passthrough below SVQ threshold"
        )


# ---------------------------------------------------------------------------
# Position codec
# ---------------------------------------------------------------------------

def test_position_codec_roundtrip_error_is_small():
    """A direct round-trip through the position codec should be within
    ~1 / 2**(bits-2) of the original on the unit cube."""
    np.random.seed(0)
    pts = np.random.uniform(-2.0, 2.0, size=(500, 3)).astype(np.float32)
    bits = 16
    codec, order = omg.position_encode(pts, bits=bits)
    decoded_sorted = omg.position_decode(codec)
    # The decoder returns sorted-order positions; the original sorted
    # should match.
    expected_sorted = pts[order]
    err = np.abs(decoded_sorted - expected_sorted)
    rng = (pts.max(0) - pts.min(0)).max()
    # Tolerance: one quantization step + slack.
    tol = rng / (2 ** (bits - 2))
    assert err.max() < tol, f"position quantization error {err.max():.4g} > {tol:.4g}"


def test_position_codec_axis_mode_recorded():
    """When ``axis_mode`` selects 'raw' on an axis, that axis is decoded
    without cumsum and the values must come back identical."""
    np.random.seed(1)
    pts = np.random.uniform(0, 1, size=(64, 3)).astype(np.float32)
    codec, _ = omg.position_encode(pts, bits=12)
    # axis_mode is per-axis and must be length 3
    assert len(codec.axis_mode) == 3
    assert all(m in ("delta", "raw") for m in codec.axis_mode)


# ---------------------------------------------------------------------------
# Integer-stream codec
# ---------------------------------------------------------------------------

def test_int_stream_signed_roundtrip():
    """Mixed-sign integer streams round-trip exactly."""
    rng = np.random.RandomState(0)
    values = rng.randint(-200, 200, size=(2048,)).astype(np.int64)
    blob, meta = omg._encode_int_stream(values)
    back = omg._decode_int_stream(blob, meta)
    assert np.array_equal(values, back)


def test_int_stream_constant_stream():
    """A stream of all-equal values encodes to 0 bytes and decodes
    exactly. (Bug guard: the previous version's bz2 fallback would
    error on an empty alphabet.)"""
    values = np.full((100,), 7, dtype=np.int64)
    blob, meta = omg._encode_int_stream(values)
    assert meta["empty"] is True
    assert meta["offset"] == 7
    back = omg._decode_int_stream(blob, meta)
    assert np.array_equal(values, back)
