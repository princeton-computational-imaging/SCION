"""The OMG bundle container must not execute code on load.

It is a JSON-header + raw-buffer container; anything else is refused.
"""

import struct

import numpy as np
import pytest
import torch

from scion.compression import omg_posthoc as omg


def test_unknown_magic_rejected():
    with pytest.raises(ValueError, match="magic mismatch"):
        omg.decode_omg_to_state_dict(b"NOTOMG\x00\x01whatever")


def test_truncated_v2_blob_rejected():
    with pytest.raises(ValueError):
        omg.decode_omg_to_state_dict(omg.MAGIC + struct.pack("<B", 0) + b"\x04")


def test_v2_header_cannot_exceed_blob():
    body = struct.pack("<Q", 1 << 40) + b"{}"
    with pytest.raises(ValueError, match="header length"):
        omg.decode_omg_to_state_dict(omg.MAGIC + struct.pack("<B", 0) + body)


def test_container_roundtrip_preserves_types():
    bundle = {
        "version": 2,
        "ints": 7,
        "floats": 1.5,
        "none": None,
        "flag": True,
        "text": "hello",
        "tup": (1, "two", 3.0),
        "blob": b"\x00\x01\x02",
        "arr0d": np.array(4.0, dtype=np.float16),
        "arr2d": np.arange(6, dtype=np.int32).reshape(2, 3),
        "nested": {"list": [1, [2, {"deep": np.ones(2, dtype=np.uint8)}]]},
        "buffers": "not-a-reserved-key-collision",
    }
    out = omg._deserialize_bundle(omg._serialize_bundle(bundle))
    assert out["ints"] == 7 and out["floats"] == 1.5 and out["none"] is None
    assert out["flag"] is True and out["text"] == "hello"
    assert out["tup"] == (1, "two", 3.0) and isinstance(out["tup"], tuple)
    assert out["blob"] == b"\x00\x01\x02"
    assert out["arr0d"].shape == () and out["arr0d"].dtype == np.float16
    assert np.array_equal(out["arr2d"], bundle["arr2d"])
    assert np.array_equal(out["nested"]["list"][1][1]["deep"], np.ones(2, np.uint8))


def test_encoder_rejects_arbitrary_objects():
    class Weird:
        pass

    with pytest.raises(TypeError, match="unsupported value type"):
        omg._serialize_bundle({"x": Weird()})


def test_decoded_arrays_are_torch_convertible():
    bundle = {"version": 2, "a": np.arange(3, dtype=np.float32)}
    out = omg._deserialize_bundle(omg._serialize_bundle(bundle))
    t = torch.from_numpy(out["a"])  # requires a writable array
    t += 1
    assert t.sum().item() == 6.0
