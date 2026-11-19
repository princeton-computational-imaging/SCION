"""Compression level ``codec``: OMG-style post-hoc coding of a ``quantized`` state_dict.

Input is the ``quantized`` level from :mod:`scion.compression.quantize`
(fp16 / uint8 tensors + ``__quantized_meta__`` affine params); output is
a self-contained ``.omg`` blob. Nothing here runs during training. The
decoder is the exact inverse of the container, so its output goes straight
through :func:`scion.compression.quantize.dequantize_state_dict`.

Stages (knobs: :class:`EncodeOptions`, set by ``config/compression/codec.yaml``):

1. **Sub-vector quantization (SVQ)** of the splat-codebook buffers
   (``splat_means``, ``splat_scales``, ``splat_quats``,
   ``splat_features_dc``, ``splat_opacities``): each row is split into
   ``M`` sub-vectors, each VQ'd against ``B`` k-means codewords
   (scikit-learn ``MiniBatchKMeans``). With ``svq_instance_attrs`` the
   per-instance quaternions get the same treatment (``instance_svq_M``,
   ``instance_svq_B``). Tensors with fewer than ``min_rows_for_svq`` rows
   pass through.
2. **Positions** (``instance_positions``): ``position_format="tmc13"``
   codes them with the MPEG G-PCC reference encoder ``tmc3``
   (:mod:`scion.compression.tmc13`) at ``tmc13_quant_step``. Its
   octree walk reorders points, so every per-instance tensor is permuted
   to the decoded order at encode time and no permutation is stored.
   ``position_format="fp16"`` is a pure-Python Morton-sort + per-axis
   delta path, used only when asked for explicitly.
3. **Range coding** (``constriction``) of every integer stream: SVQ
   indices, position deltas, ``instance_assignments``,
   ``splat_candidate_id``.
4. **LZMA** over the whole container.

No silent substitutes: a missing ``tmc3`` binary, ``constriction`` or
scikit-learn raises (the error names ``compression=quantized`` as the way
to skip this level) instead of swapping in a different algorithm.

Credits. The design follows OMG (Lee, Ko & Park, "Optimized Minimal 3D
Gaussian Splatting", 2025, https://github.com/maincold2/OMG): SVQ of
attributes and G-PCC for positions. This module is a re-implementation
for SCION's codebook layout, not a copy of OMG's code. The ``tmc3``
subprocess pattern follows the gpcc utils of HAC++
(https://github.com/YihangChen-ee/HAC-plus), which OMG also uses (approach
only; no HAC++ code is included). G-PCC is MPEG-PCC-TMC13
(https://github.com/MPEGGroup/mpeg-pcc-tmc13, pinned ``release-v23.0-rc2``
/ ``a3d15c5``, built by ``scripts/setup/build_tmc13.sh``). Range coding:
constriction (https://github.com/bamler-lab/constriction). k-means:
scikit-learn. See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import json
import lzma
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from scion.compression import tmc13 as _tmc13


# --------------------------------------------------------------------------
# Magic / version bytes
# --------------------------------------------------------------------------

MAGIC = b"OMGv2\x00"  # 6 bytes; bumped if encoded layout changes


# --------------------------------------------------------------------------
# Container: JSON header + raw buffers (no pickle, not executable)
# --------------------------------------------------------------------------
#
# Model files are meant to be shared, so loading one must not execute code
# (as ``pickle.loads`` would). The wire format is:
#
#   MAGIC ‖ 1-byte lzma flag ‖ [lzma](u64 header_len ‖ header_json ‖ buffers)
#
# ``header_json`` mirrors the bundle tree with binary leaves swapped for
# sentinel descriptors referencing the concatenated raw buffers:
#   ndarray -> {"__nd__": [dtype_str, shape, buf_idx]}
#   bytes   -> {"__by__": buf_idx}
#   tuple   -> {"__tu__": [packed items]}
# Everything else must be JSON-native (dict/list/str/int/float/bool/None);
# the encoder raises TypeError otherwise, so nothing unpicklable can leak
# into the format by accident.

_SENTINELS = ("__nd__", "__by__", "__tu__")


def _pack_tree(obj: Any, buffers: List[bytes], path: str = "$") -> Any:
    if isinstance(obj, np.generic):
        obj = np.asarray(obj)
    if isinstance(obj, np.ndarray):
        # NB: ascontiguousarray promotes 0-d to 1-d, so record the
        # *original* shape — the bytes are identical either way.
        arr = np.ascontiguousarray(obj)
        buffers.append(arr.tobytes())
        return {"__nd__": [arr.dtype.str, list(obj.shape), len(buffers) - 1]}
    if isinstance(obj, (bytes, bytearray)):
        buffers.append(bytes(obj))
        return {"__by__": len(buffers) - 1}
    if isinstance(obj, tuple):
        return {"__tu__": [
            _pack_tree(v, buffers, f"{path}[{i}]") for i, v in enumerate(obj)
        ]}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"OMG bundle: non-string dict key {k!r} at {path}"
                )
            if k in _SENTINELS:
                raise TypeError(
                    f"OMG bundle: reserved key {k!r} at {path}"
                )
            out[k] = _pack_tree(v, buffers, f"{path}.{k}")
        return out
    if isinstance(obj, list):
        return [_pack_tree(v, buffers, f"{path}[{i}]") for i, v in enumerate(obj)]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(
        f"OMG bundle: unsupported value type {type(obj).__name__} at {path} "
        "(the container only carries JSON-native values, numpy arrays, "
        "bytes, and tuples)"
    )


def _unpack_tree(node: Any, buffers: List[memoryview]) -> Any:
    if isinstance(node, dict):
        if "__nd__" in node:
            dtype_str, shape, idx = node["__nd__"]
            arr = np.frombuffer(buffers[idx], dtype=np.dtype(dtype_str))
            # .copy() -> writable, so torch.from_numpy downstream is safe.
            return arr.reshape(shape).copy()
        if "__by__" in node:
            return bytes(buffers[node["__by__"]])
        if "__tu__" in node:
            return tuple(_unpack_tree(v, buffers) for v in node["__tu__"])
        return {k: _unpack_tree(v, buffers) for k, v in node.items()}
    if isinstance(node, list):
        return [_unpack_tree(v, buffers) for v in node]
    return node


def _serialize_bundle(bundle: Dict[str, Any]) -> bytes:
    buffers: List[bytes] = []
    tree = _pack_tree(bundle, buffers)
    header = json.dumps(
        {"tree": tree, "buffers": [len(b) for b in buffers]},
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join(
        [struct.pack("<Q", len(header)), header] + buffers
    )


def _deserialize_bundle(raw: bytes) -> Dict[str, Any]:
    if len(raw) < 8:
        raise ValueError("OMG bundle: truncated container")
    (header_len,) = struct.unpack("<Q", raw[:8])
    if 8 + header_len > len(raw):
        raise ValueError("OMG bundle: header length exceeds blob size")
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    lens = header["buffers"]
    view = memoryview(raw)
    buffers: List[memoryview] = []
    off = 8 + header_len
    for n in lens:
        if off + n > len(raw):
            raise ValueError("OMG bundle: buffer table exceeds blob size")
        buffers.append(view[off : off + n])
        off += n
    return _unpack_tree(header["tree"], buffers)


# --------------------------------------------------------------------------
# Required libraries (raise, never substitute)
# --------------------------------------------------------------------------

try:
    import constriction  # type: ignore
except ImportError:  # pragma: no cover
    constriction = None  # type: ignore[assignment]

def _require_constriction(skip_hint: str = "compression=quantized") -> None:
    if constriction is None:
        raise ImportError(
            "compression level `codec` range-codes with `constriction`, "
            "which is not installed (pip install constriction), or run with "
            f"{skip_hint} to skip the codec level."
        )


def require_dependencies(
    options: "EncodeOptions", skip_hint: str = "compression=quantized",
) -> None:
    """Raise unless this machine can encode with ``options``: the tmc3
    binary (``position_format="tmc13"``), constriction and scikit-learn.
    ``skip_hint`` names the caller's knob for skipping the codec level."""
    if options.position_format == "tmc13":
        _tmc13.require_tmc3(skip_hint)
    _require_constriction(skip_hint)
    try:
        import sklearn.cluster  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "compression level `codec` runs SVQ k-means with scikit-learn, "
            "which is not installed (pip install scikit-learn), or run with "
            f"{skip_hint} to skip the codec level."
        ) from exc


def _frequency_table(symbols: np.ndarray, alphabet_size: int) -> np.ndarray:
    """Categorical probability vector with Laplace smoothing.

    ``symbols`` is a 1-D int array in ``[0, alphabet_size)``. We add 1 to
    every bin so the entropy coder never sees an exact-zero probability
    (would mean "this symbol cannot appear" → decoder failure on noise).
    """
    if alphabet_size <= 0:
        raise ValueError(f"alphabet_size must be > 0, got {alphabet_size}")
    counts = np.bincount(symbols.astype(np.int64), minlength=alphabet_size).astype(
        np.float64
    )
    counts += 1.0
    probs = counts / counts.sum()
    return probs.astype(np.float32)


def _range_encode_categorical(
    symbols: np.ndarray, alphabet_size: int
) -> Tuple[bytes, np.ndarray]:
    """Range-code ``symbols`` with a fitted categorical model.

    Returns ``(compressed_bytes, probability_table)``. The probability
    table is small (≤ alphabet_size × 4 B) and stored as part of the
    bundle so the decoder can rebuild the exact same model.
    """
    _require_constriction()
    symbols = np.ascontiguousarray(symbols, dtype=np.int32)
    probs = _frequency_table(symbols, alphabet_size)
    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols, model)
    return encoder.get_compressed().tobytes(), probs


def _range_decode_categorical(
    blob: bytes, probs: np.ndarray, n_symbols: int
) -> np.ndarray:
    """Inverse of :func:`_range_encode_categorical`."""
    _require_constriction()
    arr_u32 = np.frombuffer(blob, dtype=np.uint32).copy()
    model = constriction.stream.model.Categorical(probs.astype(np.float32), perfect=False)
    decoder = constriction.stream.queue.RangeDecoder(arr_u32)
    decoded = decoder.decode(model, n_symbols)
    return np.asarray(decoded, dtype=np.int32)


# --------------------------------------------------------------------------
# SVQ (Sub-Vector Quantization) — scikit-learn k-means
# --------------------------------------------------------------------------

def _kmeans_codebook(
    X: np.ndarray, B: int, n_iter: int = 20, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means on ``X[N, D]``. Returns ``(codebook[K_eff, D], indices[N])``.

    ``K_eff = min(B, N)`` so we never request more clusters than samples.
    Indices are uint8 because B ≤ 256.
    """
    N, D = X.shape
    K_eff = int(min(B, N))
    if K_eff <= 1:
        # Degenerate case: a single codeword, every row maps to 0.
        codebook = X.mean(axis=0, keepdims=True) if N else np.zeros((1, D), dtype=X.dtype)
        idx = np.zeros((N,), dtype=np.uint8)
        return codebook.astype(np.float32, copy=False), idx

    from sklearn.cluster import MiniBatchKMeans  # checked by require_dependencies

    # ``random_state`` pinned so encoder runs are reproducible.
    km = MiniBatchKMeans(
        n_clusters=K_eff,
        random_state=seed,
        n_init=3,
        max_iter=n_iter,
        batch_size=min(4096, max(256, N // 4)),
        reassignment_ratio=0.01,
    )
    idx = km.fit_predict(X)
    codebook = km.cluster_centers_.astype(np.float32, copy=False)
    return codebook, idx.astype(np.uint8, copy=False)


@dataclass
class SvqResult:
    """SVQ codebooks + indices for one tensor."""

    codebooks: List[np.ndarray]   # M codebooks, each [<=B, sub_dim]
    indices: List[np.ndarray]     # M index arrays, each [N] uint8
    sub_dims: List[int]            # length of each sub-vector
    M: int
    B: int
    N: int
    L: int                          # original row dim
    orig_dtype: str                 # e.g. "float16", "uint8" — for round-trip

    def reconstruct(self) -> np.ndarray:
        """Rebuild the [N, L] reconstruction by gathering+concatenating codewords."""
        cols: List[np.ndarray] = []
        for cb, idx in zip(self.codebooks, self.indices):
            cols.append(cb[idx])
        out = np.concatenate(cols, axis=1)
        if out.shape != (self.N, self.L):
            raise RuntimeError(
                f"SVQ reconstruction shape mismatch: got {out.shape}, want {(self.N, self.L)}"
            )
        return out


def _split_dims(L: int, M: int) -> List[int]:
    """Distribute ``L`` columns into ``M`` sub-vectors as evenly as possible."""
    base = L // M
    rem = L % M
    sizes = [base + (1 if i < rem else 0) for i in range(M)]
    assert sum(sizes) == L
    return sizes


def svq_encode(
    rows: np.ndarray, M: int, B: int, *, seed: int = 0
) -> SvqResult:
    """Sub-vector quantize ``rows[N, L]`` with ``M`` codebooks of size ≤ B.

    ``M`` is auto-capped at ``L`` (no point in more codebooks than columns).
    """
    N, L = rows.shape
    M_eff = max(1, min(int(M), L))
    sizes = _split_dims(L, M_eff)
    codebooks: List[np.ndarray] = []
    indices: List[np.ndarray] = []
    cursor = 0
    rows_f32 = rows.astype(np.float32, copy=False)
    for m, sub in enumerate(sizes):
        sub_rows = rows_f32[:, cursor : cursor + sub]
        cb, idx = _kmeans_codebook(sub_rows, B, seed=seed + m)
        codebooks.append(cb.astype(np.float32, copy=False))
        indices.append(idx.astype(np.uint8, copy=False))
        cursor += sub
    return SvqResult(
        codebooks=codebooks,
        indices=indices,
        sub_dims=sizes,
        M=M_eff,
        B=int(B),
        N=int(N),
        L=int(L),
        orig_dtype=str(rows.dtype),
    )


# --------------------------------------------------------------------------
# Morton (Z-order) sort + delta coding for positions
# --------------------------------------------------------------------------

def _interleave_bits_3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Interleave the low 16 bits of three uint32 arrays into a 48-bit Morton code.

    Uses ``np.uint64`` so we have room for 48-bit codes; only the lower
    48 bits are populated. Returned as ``np.uint64``.
    """
    def _spread16(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint64) & np.uint64(0xFFFF)
        v = (v | (v << np.uint64(24))) & np.uint64(0x0000FF000000FF)
        v = (v | (v << np.uint64(12))) & np.uint64(0x000F00F00F00F00F)
        v = (v | (v << np.uint64(6))) & np.uint64(0x0303030303030303)
        v = (v | (v << np.uint64(2))) & np.uint64(0x9249249249249249)
        return v

    return _spread16(x) | (_spread16(y) << np.uint64(1)) | (_spread16(z) << np.uint64(2))


@dataclass
class PositionCodec:
    """Quantized + Morton-sorted + delta-coded position stream.

    The decoder re-emits positions **in Morton-sorted order**, NOT in the
    original per-instance order. The caller (encoder side) is responsible
    for applying the same permutation to every other per-instance tensor
    so that index ``i`` in the decoded output corresponds to the same
    logical instance across all tensors. This drops the permutation from
    the bitstream entirely — a uniform random permutation of N
    instances is incompressible (~N log2 N bits) and dominates the
    budget; sorting the *other* tensors along with positions is free.

    Stored fields:

    - ``lo``, ``hi``: per-axis min/max (float32, length 3 each)
    - ``deltas``: per-axis int32 first-difference of the Morton-sorted
      uint16 quantized positions
    """

    lo: np.ndarray
    hi: np.ndarray
    deltas: List[np.ndarray]   # 3 arrays of length N (int32)
    bits: int = 16
    # Per-axis encoding mode: "delta" (cumsum to recover) or "raw"
    # (already the quantized values). Picked at encode time per axis.
    axis_mode: List[str] = field(default_factory=lambda: ["delta", "delta", "delta"])


def position_encode(
    positions: np.ndarray, bits: int = 16
) -> Tuple[PositionCodec, np.ndarray]:
    """Quantize, Morton-sort, delta-code ``positions[N, 3]``.

    Returns ``(codec, order)`` — the ``order`` permutation is the one the
    caller must apply to every per-instance buffer so the decoder's
    sorted output stays consistent.

    For each axis we try both raw quantized values and first-difference
    deltas and pick whichever has the smaller post-entropy-coding
    alphabet (cheap upper-bound proxy for the actual coded size; the
    range coder fits to the chosen stream's frequency table). The choice
    per axis is recorded in ``axis_mode`` so the decoder knows whether
    to ``cumsum`` or not.
    """
    if positions.shape[1] != 3:
        raise ValueError(f"expected [N, 3], got {positions.shape}")
    p = np.ascontiguousarray(positions, dtype=np.float32)
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    rng = np.maximum(hi - lo, 1e-12)
    max_q = (1 << bits) - 1
    q = np.round((p - lo) / rng * max_q).astype(np.int64)
    q = np.clip(q, 0, max_q).astype(np.uint32)

    # Morton-sort.
    morton = _interleave_bits_3(q[:, 0], q[:, 1], q[:, 2])
    order = np.argsort(morton, kind="stable").astype(np.int64)
    q_sorted = q[order]

    streams: List[np.ndarray] = []
    axis_mode: List[str] = []
    for axis in range(3):
        s = q_sorted[:, axis].astype(np.int32)
        d = np.empty_like(s)
        d[0] = s[0]
        d[1:] = s[1:] - s[:-1]
        # Heuristic: prefer whichever stream has the smaller dynamic
        # range (smaller alphabet → cheaper entropy coding). Tied or
        # almost-tied → delta loses (raw decode is simpler).
        s_alpha = int(s.max()) - int(s.min()) + 1
        d_alpha = int(d.max()) - int(d.min()) + 1
        if d_alpha + 64 < s_alpha:
            streams.append(d)
            axis_mode.append("delta")
        else:
            streams.append(s)
            axis_mode.append("raw")
    return PositionCodec(
        lo=lo.astype(np.float32),
        hi=hi.astype(np.float32),
        deltas=streams,
        bits=bits,
        axis_mode=axis_mode,
    ), order


def position_decode(codec: PositionCodec) -> np.ndarray:
    """Decode positions in **Morton-sorted** order. Caller un-permutes if needed."""
    deltas = codec.deltas
    N = len(deltas[0])
    max_q = (1 << codec.bits) - 1
    q_sorted = np.empty((N, 3), dtype=np.int64)
    for axis in range(3):
        s = deltas[axis].astype(np.int64)
        if codec.axis_mode[axis] == "delta":
            q_sorted[:, axis] = np.cumsum(s)
        else:
            q_sorted[:, axis] = s
    q_sorted = np.clip(q_sorted, 0, max_q)
    rng = np.maximum(codec.hi - codec.lo, 1e-12)
    p = q_sorted.astype(np.float32) / float(max_q) * rng + codec.lo
    return p


# --------------------------------------------------------------------------
# Small int stream codec (with shifted alphabet for signed values)
# --------------------------------------------------------------------------

def _encode_int_stream(values: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
    """Range-encode a 1-D integer stream of arbitrary sign.

    Returns ``(blob, meta)`` where ``meta`` carries the offset and
    alphabet size needed to round-trip. The encoder shifts symbols to be
    non-negative and clips the alphabet to ``[0, max-min+1)``.
    """
    if values.ndim != 1:
        raise ValueError(f"expected 1-D int stream, got shape {values.shape}")
    if values.size == 0:
        return b"", {"empty": True, "offset": 0, "alphabet": 1, "n": 0}
    v = values.astype(np.int64)
    lo = int(v.min())
    hi = int(v.max())
    shifted = (v - lo).astype(np.int32)
    alphabet = hi - lo + 1
    if alphabet <= 1:
        # All-equal stream: degenerate, encode nothing.
        return b"", {"empty": True, "offset": lo, "alphabet": alphabet, "n": int(len(v))}
    blob, probs = _range_encode_categorical(shifted, alphabet)
    return blob, {
        "offset": lo,
        "alphabet": int(alphabet),
        "n": int(len(v)),
        "probs": probs,
        "empty": False,
    }


def _decode_int_stream(blob: bytes, meta: Dict[str, Any]) -> np.ndarray:
    if meta.get("empty", False):
        return np.full((meta["n"],), meta["offset"], dtype=np.int64)
    shifted = _range_decode_categorical(blob, meta["probs"], meta["n"])
    return shifted.astype(np.int64) + meta["offset"]


# --------------------------------------------------------------------------
# Bundle = the dict we serialise (container above) + LZMA-compress.
# --------------------------------------------------------------------------
#
# Layout sketch:
#
#   {
#     "version": 2,
#     "tensors": {
#         <key>: {"kind": "passthrough" | "svq" | "positions" | "intstream",
#                 <payload depending on kind>,
#                 "shape": (...), "dtype": "..."},
#     },
#     "meta_keys": {<meta key>: <small tensor or scalar>},  # quantized-level meta
#     "scalar_meta": {<key>: scalar},
#   }
#
# All numpy arrays inside payload are LEFT as numpy (so torch is not
# required to decode); strings are plain Python; entropy-coded blobs are
# raw ``bytes``.

# Tensor key classification
_SVQ_TENSORS_SPLAT = (
    "splat_means", "splat_scales", "splat_quats",
    "splat_features_dc", "splat_opacities",
)
_POS_TENSORS = ("instance_positions",)
_INT_STREAM_TENSORS = (
    "instance_assignments", "splat_candidate_id",
)
# Per-instance attribute tensors that the encoder optionally SVQ's
# (``EncodeOptions.svq_instance_attrs``). The 4-D quat tensor is the
# biggest residual line item after TMC13 positions, so this is where the
# next compression win lives.
_SVQ_TENSORS_INSTANCE = ("instance_rotations",)


def _classify(key: str, options: Optional["EncodeOptions"] = None) -> str:
    """Decide how to encode ``key``.

    ``options.svq_instance_attrs`` controls whether ``instance_rotations``
    is routed through the SVQ path or kept as fp16.
    """
    base = key.rsplit(".", 1)[-1]
    if base in _POS_TENSORS:
        return "positions"
    if base in _SVQ_TENSORS_SPLAT:
        return "svq"
    if (
        options is not None
        and options.svq_instance_attrs
        and base in _SVQ_TENSORS_INSTANCE
    ):
        return "svq"
    if base in _INT_STREAM_TENSORS:
        return "intstream"
    return "passthrough"


def _svq_params_for_key(key: str, options: "EncodeOptions") -> Tuple[int, int]:
    """Return (M, B) for SVQ encoding ``key``.

    Per-instance attributes use ``instance_svq_M`` / ``instance_svq_B``
    so callers can pick M=2 for 4-D quats and M=3 for 3-D splat means
    independently.
    """
    base = key.rsplit(".", 1)[-1]
    if base in _SVQ_TENSORS_INSTANCE:
        return (int(options.instance_svq_M), int(options.instance_svq_B))
    return (int(options.M), int(options.B))


# --------------------------------------------------------------------------
# Encode / decode entry points
# --------------------------------------------------------------------------

@dataclass
class EncodeOptions:
    """Codec knobs. Defaults are the shipped ``codec`` level and must match
    ``config/compression/codec.yaml`` (enforced by a test)."""

    M: int = 3
    B: int = 256
    min_rows_for_svq: int = 64
    pos_bits: int = 16  # fp16 position path only
    seed: int = 0
    lzma_wrap: bool = True
    # "tmc13" = MPEG G-PCC via the tmc3 binary (raises if missing);
    # "fp16" = pure-Python Morton + delta path.
    position_format: str = "tmc13"
    # TMC13 grid spacing in world units (metres for typical 3DGS scenes).
    tmc13_quant_step: float = 1e-3
    # SVQ the per-instance quaternions with (instance_svq_M, instance_svq_B).
    # M=4 (scalar VQ per component): 23% smaller bundle at -0.05 dB on
    # garden; M=2 costs ~1 dB because codeword pairs leave the unit sphere.
    svq_instance_attrs: bool = True
    instance_svq_M: int = 4
    instance_svq_B: int = 256


@dataclass
class ByteBudget:
    """Per-tensor byte accounting for the report."""

    per_tensor: Dict[str, int] = field(default_factory=dict)
    total: int = 0

    def add(self, key: str, n: int) -> None:
        self.per_tensor[key] = self.per_tensor.get(key, 0) + int(n)
        self.total += int(n)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().cpu().numpy()


def _find_instance_position_key(state_dict: Dict[str, Any]) -> Optional[str]:
    """Locate the ``instance_positions`` tensor regardless of prefix."""
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and k.rsplit(".", 1)[-1] == "instance_positions":
            return k
    return None


def _is_per_instance_tensor(
    key: str, tensor: torch.Tensor, n_instances: int
) -> bool:
    """A tensor is per-instance if its first dim matches ``n_instances``
    *and* its semantic key starts with ``instance_``.

    Returning True means: encoder is allowed to permute this tensor along
    dim 0 by the Morton order without breaking semantics. We use both the
    key-prefix and the shape check to be conservative — other tensors
    happen to have the same first dim but aren't per-instance (e.g.,
    ``free_list`` would clash here in pathological configs).
    """
    base = key.rsplit(".", 1)[-1]
    if not base.startswith("instance_"):
        return False
    if tensor.shape[0] != n_instances:
        return False
    return True


def encode_quantized_to_omg(
    state_dict: Dict[str, Any],
    options: Optional[EncodeOptions] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Encode a ``quantized``-level state_dict (``quantize_state_dict``) to an OMG-style blob.

    Returns ``(payload_bytes, info)`` where ``info`` carries the
    per-tensor byte budget so the caller can print a Pareto-friendly
    breakdown.
    """
    if options is None:
        options = EncodeOptions()

    pos_format = options.position_format
    if pos_format not in ("tmc13", "fp16"):
        raise ValueError(f"unknown position_format: {pos_format!r}")
    require_dependencies(options)

    # --- Pass 1: locate instance_positions and derive the canonical order. ---
    # We apply the same permutation to every per-instance tensor so the
    # decoded bundle stays internally consistent without storing the
    # permutation (which would be incompressible).
    pos_key = _find_instance_position_key(state_dict)
    morton_order: Optional[np.ndarray] = None
    pos_codec: Optional[PositionCodec] = None
    tmc13_blob: Optional[bytes] = None
    tmc13_orig_dtype: Optional[str] = None
    tmc13_orig_shape: Optional[list] = None
    if pos_key is not None:
        pos_tensor = state_dict[pos_key]
        pos_arr = _to_numpy(pos_tensor).astype(np.float32, copy=False)
        n_instances = pos_arr.shape[0]
        if pos_format == "tmc13":
            # TMC13 encodes in its own Morton-walk order. Compute the
            # permutation by encode-then-decode; use it to rewrite all
            # per-instance tensors so the decoder can read everything
            # back in lock-step.
            pos_t = torch.from_numpy(pos_arr)
            tmc13_blob, perm_t = _tmc13.encode_positions_with_perm(
                pos_t, quant_step=float(options.tmc13_quant_step)
            )
            # perm is "decoded[k] = original[perm[k]]" — i.e. apply
            # arr[perm] to land any auxiliary tensor in decode order.
            morton_order = perm_t.cpu().numpy().astype(np.int64)
            tmc13_orig_dtype = str(pos_tensor.dtype)
            tmc13_orig_shape = list(pos_tensor.shape)
        else:
            pos_codec, morton_order = position_encode(pos_arr, bits=options.pos_bits)
    else:
        n_instances = 0

    tensors: Dict[str, Any] = {}
    meta_keys: Dict[str, Any] = {}
    scalar_meta: Dict[str, Any] = {}
    budget = ByteBudget()

    for key, val in state_dict.items():
        # quantized-level meta passthroughs (small float scalars per-tensor).
        if "__quantized_meta__" in key:
            if isinstance(val, torch.Tensor):
                meta_keys[key] = _to_numpy(val)
            else:
                scalar_meta[key] = val
            continue
        if "__codebook_meta__" in key:
            if isinstance(val, torch.Tensor):
                meta_keys[key] = _to_numpy(val)
            else:
                scalar_meta[key] = val
            continue
        if not isinstance(val, torch.Tensor):
            scalar_meta[key] = val
            continue

        arr = _to_numpy(val)
        kind = _classify(key, options)

        # Permute per-instance tensors into Morton order so the decoder
        # can read them back in lock-step with the sorted positions.
        if (
            morton_order is not None
            and _is_per_instance_tensor(key, val, n_instances)
            and kind != "positions"
        ):
            arr = arr[morton_order]

        if kind == "svq":
            rows = arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)
            N = rows.shape[0]
            if N < options.min_rows_for_svq or rows.shape[1] <= 1:
                # Too few rows or 1-D: store as fp16 passthrough.
                kind = "passthrough"
            else:
                svq_M, svq_B = _svq_params_for_key(key, options)
                svq = svq_encode(rows, M=svq_M, B=svq_B, seed=options.seed)
                # Entropy-code each per-codebook index stream (uint8 categorical).
                index_blobs: List[bytes] = []
                index_probs: List[np.ndarray] = []
                for idx_arr in svq.indices:
                    # Alphabet is the actual number of codewords for this
                    # sub-codebook (may be < B if N < B was used).
                    alphabet = int(idx_arr.max()) + 1 if idx_arr.size else 1
                    alphabet = max(alphabet, 1)
                    blob, _ = _range_encode_categorical(idx_arr.astype(np.int32), alphabet)
                    index_blobs.append(blob)
                    index_probs.append(_frequency_table(idx_arr.astype(np.int32), alphabet))
                payload = {
                    "kind": "svq",
                    "codebooks": [cb.astype(np.float16, copy=False) for cb in svq.codebooks],
                    "index_blobs": index_blobs,
                    "index_probs": index_probs,
                    "sub_dims": list(svq.sub_dims),
                    "M": svq.M,
                    "B": svq.B,
                    "N": svq.N,
                    "L": svq.L,
                    "orig_dtype": str(val.dtype),
                    "orig_shape": list(val.shape),
                }
                tensors[key] = payload
                # Byte budget: codebooks + entropy blobs.
                b = sum(cb.nbytes for cb in payload["codebooks"])
                b += sum(len(blob) for blob in index_blobs)
                b += sum(p.nbytes for p in index_probs)
                budget.add(key, b)
                continue

        if kind == "positions":
            # TMC13 path takes precedence when configured + available.
            if pos_format == "tmc13":
                assert tmc13_blob is not None, (
                    "tmc13_blob should be set when pos_format=='tmc13'"
                )
                payload = {
                    "kind": "positions_tmc13",
                    "blob": tmc13_blob,
                    "quant_step": float(options.tmc13_quant_step),
                    "n": int(n_instances),
                    "orig_dtype": tmc13_orig_dtype or str(val.dtype),
                    "orig_shape": tmc13_orig_shape or list(val.shape),
                }
                tensors[key] = payload
                budget.add(key, len(tmc13_blob))
                continue

            # Otherwise emit the fp16 (Morton+delta+range) bundle.
            assert pos_codec is not None, "pos_codec should be set when kind=='positions'"
            codec = pos_codec
            delta_blobs: List[bytes] = []
            delta_metas: List[Dict[str, Any]] = []
            for d in codec.deltas:
                blob, meta = _encode_int_stream(d.astype(np.int32))
                delta_blobs.append(blob)
                delta_metas.append(meta)
            payload = {
                "kind": "positions",
                "lo": codec.lo,
                "hi": codec.hi,
                "bits": codec.bits,
                "n": int(len(codec.deltas[0])),
                "delta_blobs": delta_blobs,
                "delta_metas": delta_metas,
                "axis_mode": list(codec.axis_mode),
                "orig_dtype": str(val.dtype),
                "orig_shape": list(val.shape),
            }
            tensors[key] = payload
            b = sum(len(blob) for blob in delta_blobs)
            b += codec.lo.nbytes + codec.hi.nbytes
            for m in delta_metas:
                if not m.get("empty", False):
                    b += m["probs"].nbytes
            budget.add(key, b)
            continue

        if kind == "intstream":
            flat = arr.reshape(-1)
            blob, meta = _encode_int_stream(flat.astype(np.int64))
            payload = {
                "kind": "intstream",
                "blob": blob,
                "meta": meta,
                "orig_dtype": str(val.dtype),
                "orig_shape": list(val.shape),
            }
            tensors[key] = payload
            b = len(blob)
            if not meta.get("empty", False):
                b += meta["probs"].nbytes
            budget.add(key, b)
            continue

        # passthrough (fp16 / uint8). Stored as raw numpy.
        payload = {
            "kind": "passthrough",
            "data": arr,
            "orig_dtype": str(val.dtype),
            "orig_shape": list(val.shape),
        }
        tensors[key] = payload
        budget.add(key, int(arr.nbytes))

    bundle = {
        "version": 2,
        "tensors": tensors,
        "meta_keys": meta_keys,
        "scalar_meta": scalar_meta,
        "encode_options": asdict(options),
    }

    serialized = _serialize_bundle(bundle)
    if options.lzma_wrap:
        body = lzma.compress(serialized, preset=lzma.PRESET_EXTREME)
    else:
        body = serialized

    blob = MAGIC + struct.pack("<B", int(options.lzma_wrap)) + body

    info = {
        "tensor_budget": dict(budget.per_tensor),
        "tensor_budget_total": budget.total,
        "serialized_size": len(serialized),
        "final_size": len(blob),
        "container": "v2-json-buffers",
        "position_format": pos_format,
    }
    return blob, info


def decode_omg_to_state_dict(blob: bytes) -> Dict[str, Any]:
    """Inverse of :func:`encode_quantized_to_omg`.

    Returns a plain dict with the same keys as the original
    ``quantized``-level state_dict (tensors are :mod:`torch` tensors in
    their original dtypes). The container is non-executable (JSON header +
    raw buffers); a blob without :data:`MAGIC` raises ``ValueError``.
    """
    if not blob.startswith(MAGIC):
        raise ValueError("blob magic mismatch; not an OMG bundle")
    flag = blob[len(MAGIC)]
    body = blob[len(MAGIC) + 1 :]
    raw = lzma.decompress(body) if flag else body
    bundle = _deserialize_bundle(raw)
    if bundle.get("version", 0) != 2:
        raise ValueError(f"unknown bundle version {bundle.get('version')}")

    out: Dict[str, Any] = {}
    # meta + scalars
    for k, v in bundle.get("meta_keys", {}).items():
        arr = np.asarray(v)
        # ``np.ascontiguousarray`` promotes a 0-d array to 1-d; we want
        # to preserve the original 0-d / N-d shape exactly so the live
        # CodebookModel's load_state_dict path doesn't see surprise
        # singleton dimensions on ``__codebook_meta__.*`` scalars.
        if arr.ndim == 0:
            out[k] = torch.from_numpy(np.array(arr.item(), dtype=arr.dtype))
            # ``torch.from_numpy(np.array(scalar))`` yields a 0-d tensor.
        else:
            out[k] = torch.from_numpy(np.ascontiguousarray(arr))
    for k, v in bundle.get("scalar_meta", {}).items():
        out[k] = v

    for key, payload in bundle["tensors"].items():
        kind = payload["kind"]
        orig_dtype = payload["orig_dtype"]
        orig_shape = payload["orig_shape"]

        if kind == "passthrough":
            data = np.ascontiguousarray(payload["data"])
            t = torch.from_numpy(data)
            out[key] = t.view(*orig_shape) if t.shape != tuple(orig_shape) else t
            continue

        if kind == "svq":
            cols = []
            for cb, blob_i, probs_i in zip(
                payload["codebooks"], payload["index_blobs"], payload["index_probs"]
            ):
                alphabet = max(int(cb.shape[0]), 1)
                if alphabet == 1:
                    idx = np.zeros((payload["N"],), dtype=np.int32)
                else:
                    idx = _range_decode_categorical(blob_i, probs_i, payload["N"])
                cols.append(np.asarray(cb, dtype=np.float32)[idx])
            mat = np.concatenate(cols, axis=1)
            # Restore original dtype (fp16/uint8); for uint8
            # we need to round and clamp into [0,255].
            target_dtype = _torch_dtype_from_str(orig_dtype)
            if target_dtype == torch.uint8:
                # SVQ on uint8 means the codebook holds float means; round+clip.
                tens = torch.from_numpy(np.clip(np.round(mat), 0, 255).astype(np.uint8))
            elif target_dtype == torch.float16:
                tens = torch.from_numpy(mat.astype(np.float16))
            elif target_dtype == torch.float32:
                tens = torch.from_numpy(mat.astype(np.float32))
            else:
                tens = torch.from_numpy(mat).to(target_dtype)
            out[key] = tens.reshape(*orig_shape)
            continue

        if kind == "positions_tmc13":
            pos_t = _tmc13.decode_positions(payload["blob"])
            pos = pos_t.detach().cpu().numpy()
            target_dtype = _torch_dtype_from_str(orig_dtype)
            if target_dtype == torch.float16:
                tens = torch.from_numpy(pos.astype(np.float16))
            else:
                tens = torch.from_numpy(pos.astype(np.float32)).to(target_dtype)
            out[key] = tens.reshape(*orig_shape)
            continue

        if kind == "positions":
            codec = PositionCodec(
                lo=payload["lo"],
                hi=payload["hi"],
                deltas=[
                    _decode_int_stream(b, m).astype(np.int32)
                    for b, m in zip(payload["delta_blobs"], payload["delta_metas"])
                ],
                bits=payload["bits"],
                axis_mode=list(payload.get("axis_mode", ["delta"] * 3)),
            )
            pos = position_decode(codec)
            target_dtype = _torch_dtype_from_str(orig_dtype)
            if target_dtype == torch.float16:
                tens = torch.from_numpy(pos.astype(np.float16))
            else:
                tens = torch.from_numpy(pos.astype(np.float32)).to(target_dtype)
            out[key] = tens.reshape(*orig_shape)
            continue

        if kind == "intstream":
            flat = _decode_int_stream(payload["blob"], payload["meta"])
            target_dtype = _torch_dtype_from_str(orig_dtype)
            if target_dtype == torch.int64:
                tens = torch.from_numpy(flat.astype(np.int64))
            elif target_dtype == torch.int32:
                tens = torch.from_numpy(flat.astype(np.int32))
            else:
                tens = torch.from_numpy(flat).to(target_dtype)
            out[key] = tens.reshape(*orig_shape)
            continue

        raise ValueError(f"unknown tensor kind: {kind!r}")

    return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_TORCH_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
    "torch.bfloat16": torch.bfloat16,
}


def _torch_dtype_from_str(s: str) -> torch.dtype:
    if s in _TORCH_DTYPE_MAP:
        return _TORCH_DTYPE_MAP[s]
    raise KeyError(f"unknown torch dtype string: {s!r}")
