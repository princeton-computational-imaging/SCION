"""MPEG G-PCC (TMC13) position codec wrapper.

Shells out to the reference ``tmc3`` binary from
https://github.com/MPEGGroup/mpeg-pcc-tmc13.  Encodes ``[N, 3]`` float32
world-space positions and returns a self-contained byte blob; the
inverse decode returns the positions in **TMC13's canonical output
order** (which is *not* the same as the input order — see below).

Why TMC13 reorders
==================

TMC13 builds an octree over an integer-quantised grid and serialises
nodes in Morton order.  The decoder walks the octree the same way, so
its output is in *Morton order*, not input order.  This is intrinsic
to G-PCC and not something we can disable.

For our compression slice this means: callers that want to keep
per-point auxiliary tensors aligned (rotations, scales, opacities,
assignments) must **reorder them to match the decoded order at encode
time**, then they need no permutation at decode time.  See
``encode_positions_with_perm`` for the helper that does this.

Binary
======

``bash scripts/setup/build_tmc13.sh`` clones MPEG-PCC-TMC13 at the pinned
``release-v23.0-rc2`` (``a3d15c5``) and builds ``tmc3`` into
``$XDG_CACHE_HOME/scion/mpeg-pcc-tmc13`` (default ``~/.cache``).
Lookup order: ``$TMC13_BINARY`` (if set, nothing else is tried), the build
script's output, ``tmc3`` on ``PATH``. A missing binary raises
:class:`TMC13UnavailableError`; there is no fallback codec.

The PLY-in / ``tmc3`` subprocess / PLY-out pattern follows the gpcc utils
of HAC++ (https://github.com/YihangChen-ee/HAC-plus), also used by OMG
(approach only; no HAC++ code is included).

Bit-rate / accuracy
===================

Empirical numbers on the MipNeRF-360 ``garden`` checkpoint
(N=53,785 instances, position range ~21 m):

================  =============  ==================
quant_step (m)    blob size (KB)  worst-case error
================  =============  ==================
1e-2 (1 cm)       46             ~5 mm (q/2)
1e-3 (1 mm)       111            ~0.5 mm
1e-4 (0.1 mm)     178            ~0.05 mm
fp16 baseline     315            ~0.001 m
================  =============  ==================

For 3DGS-style rendering the visual impact at 1mm is below LPIPS noise.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def _default_build_path() -> str:
    """Where ``scripts/setup/build_tmc13.sh`` puts the binary."""
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "scion", "mpeg-pcc-tmc13", "build", "tmc3", "tmc3")


def _is_exe(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _find_tmc3_binary() -> Optional[str]:
    """Return the tmc3 binary to use, or None (see module docstring)."""
    env = os.environ.get("TMC13_BINARY")
    if env:  # explicit override: never silently use a different binary
        return env if _is_exe(env) else None
    built = _default_build_path()
    return built if _is_exe(built) else shutil.which("tmc3")


class TMC13UnavailableError(FileNotFoundError):
    """Raised when the tmc3 binary cannot be located."""


def require_tmc3(skip_hint: str = "compression=quantized") -> str:
    """Path of the tmc3 binary; ``skip_hint`` names the caller's opt-out knob."""
    binary = _find_tmc3_binary()
    if binary is None:
        env = os.environ.get("TMC13_BINARY")
        bad_env = f"TMC13_BINARY={env} is not an executable file. " if env else ""
        raise TMC13UnavailableError(
            f"{bad_env}tmc3 (MPEG G-PCC, needed by compression level `codec`) "
            "not found. Build it with `bash scripts/setup/build_tmc13.sh` or "
            f"set TMC13_BINARY=/path/to/tmc3, or run with {skip_hint} "
            "to skip the codec level."
        )
    return binary


# ---------------------------------------------------------------------------
# PLY I/O (minimal — we control both encode + decode)
# ---------------------------------------------------------------------------


def _write_int_ply(integers_xyz: np.ndarray, path: str) -> None:
    """Write a 3-column int array as a binary float-PLY.

    TMC13 reads ``float`` PLYs but interprets numeric values as raw
    grid coordinates after the ``--inputScale`` multiplier.  Storing
    integers as float32 with ``inputScale=1`` is the simplest way to
    feed pre-quantised positions in.
    """
    N = integers_xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(integers_xyz.astype(np.float32).tobytes())


def _read_int_ply(path: str) -> np.ndarray:
    """Parse a TMC13-emitted PLY into an ``[N, 3]`` int64 array."""
    with open(path, "rb") as f:
        header_bytes = bytearray()
        while True:
            line = f.readline()
            header_bytes.extend(line)
            if line.startswith(b"end_header"):
                break
        rest = f.read()

    header = header_bytes.decode("ascii", errors="replace")
    fmt = None
    n_vertex = 0
    vertex_props = []  # list of (dtype, name) tuples in order
    in_vertex = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("format"):
            fmt = line.split()[1]
        elif line.startswith("element vertex"):
            n_vertex = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element"):
            in_vertex = False
        elif line.startswith("property") and in_vertex:
            parts = line.split()
            # property <dtype> <name>  or  property list <count_dtype> <list_dtype> <name>
            if parts[1] != "list":
                vertex_props.append((parts[1], parts[2]))

    if fmt != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format: {fmt}")

    # Determine field offsets for x, y, z.
    type_size = {
        "float": 4, "float32": 4,
        "double": 8, "float64": 8,
        "int": 4, "int32": 4, "uint": 4, "uint32": 4,
        "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
        "char": 1, "int8": 1, "uchar": 1, "uint8": 1,
    }
    rec_sz = sum(type_size[p[0]] for p in vertex_props)
    field_offsets = {}
    o = 0
    for dt, name in vertex_props:
        field_offsets[name] = (o, dt)
        o += type_size[dt]

    xyz = np.empty((n_vertex, 3), dtype=np.int64)
    np_dtype = {
        "float": np.float32, "float32": np.float32,
        "double": np.float64, "float64": np.float64,
        "int": np.int32, "int32": np.int32,
        "uint": np.uint32, "uint32": np.uint32,
    }
    expected_total = n_vertex * rec_sz
    raw = rest[:expected_total]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_vertex, rec_sz)
    for i, name in enumerate(("x", "y", "z")):
        if name not in field_offsets:
            raise ValueError(f"PLY missing '{name}' field")
        off, dt = field_offsets[name]
        sz = type_size[dt]
        col = arr[:, off:off+sz].copy()
        # Cast via numpy
        col_np = np.frombuffer(col.tobytes(), dtype=np_dtype[dt])
        xyz[:, i] = col_np.astype(np.int64)
    return xyz


# ---------------------------------------------------------------------------
# Public encode / decode API
# ---------------------------------------------------------------------------


# Bytes layout of the encoded blob:
#   magic ("TMC1")
#   uint8 version
#   float64 quant_step
#   3 x float64 shift  (subtract before quantise; add back at decode)
#   uint32 N
#   uint32 stream_len
#   stream_len bytes of raw TMC13 bitstream


_MAGIC = b"TMC1"
_VERSION = 1


def _pack_blob(stream: bytes, quant_step: float, shift: np.ndarray, N: int) -> bytes:
    buf = io.BytesIO()
    buf.write(_MAGIC)
    buf.write(struct.pack("<B", _VERSION))
    buf.write(struct.pack("<d", float(quant_step)))
    buf.write(struct.pack("<3d", float(shift[0]), float(shift[1]), float(shift[2])))
    buf.write(struct.pack("<I", int(N)))
    buf.write(struct.pack("<I", len(stream)))
    buf.write(stream)
    return buf.getvalue()


def _unpack_blob(blob: bytes) -> Tuple[bytes, float, np.ndarray, int]:
    if blob[:4] != _MAGIC:
        raise ValueError(f"Not a TMC1 blob (magic: {blob[:4]!r})")
    o = 4
    version = struct.unpack_from("<B", blob, o)[0]; o += 1
    if version != _VERSION:
        raise ValueError(f"Unsupported TMC1 version {version}")
    quant_step = struct.unpack_from("<d", blob, o)[0]; o += 8
    shift = np.array(struct.unpack_from("<3d", blob, o), dtype=np.float64); o += 24
    N = struct.unpack_from("<I", blob, o)[0]; o += 4
    stream_len = struct.unpack_from("<I", blob, o)[0]; o += 4
    stream = blob[o:o+stream_len]
    if len(stream) != stream_len:
        raise ValueError(
            f"Truncated TMC1 blob: stream_len={stream_len} actual={len(stream)}"
        )
    return stream, quant_step, shift, N


def encode_positions(
    positions: torch.Tensor,
    quant_step: float = 1e-3,
    *,
    workdir: Optional[str] = None,
) -> bytes:
    """Encode positions to a G-PCC bitstream + header.

    Parameters
    ----------
    positions : ``[N, 3]`` float32 tensor
        World-space positions in arbitrary units.  Encoded losslessly
        on an integer grid with spacing ``quant_step``.
    quant_step : float
        Quantisation granularity in the same units as ``positions``.
        Smaller → bigger stream + finer fidelity.  1e-3 is a good
        operating point for metre-scale scenes.

    Returns
    -------
    bytes
        Self-contained blob: ``magic + header + tmc13_bitstream``.

    Notes
    -----
    The decoded positions will be in **G-PCC Morton order**, not the
    input order.  Use :func:`encode_positions_with_perm` if you need
    to keep auxiliary tensors aligned.
    """
    if positions.dim() != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be [N, 3]; got {tuple(positions.shape)}")

    binary = require_tmc3()

    pos32 = positions.detach().to(torch.float32).cpu().numpy()
    shift = pos32.min(axis=0)  # uint coords -> non-negative
    scaled = np.round((pos32 - shift) / quant_step).astype(np.int64)
    if (scaled < 0).any():
        scaled = scaled - scaled.min(axis=0)
        # if rounding underflows the shift, re-bias
    # Range check — TMC13 supports up to 21-bit unsigned per axis at safe defaults
    if scaled.max() >= (1 << 30):
        raise ValueError(
            f"Quantised positions exceed 30-bit range "
            f"(max={scaled.max()}). Increase quant_step."
        )

    with tempfile.TemporaryDirectory(prefix="tmc13_", dir=workdir) as tmp:
        ply_in = os.path.join(tmp, "in.ply")
        bin_out = os.path.join(tmp, "out.bin")

        _write_int_ply(scaled, ply_in)

        cmd = [
            binary,
            "--mode=0",
            f"--uncompressedDataPath={ply_in}",
            f"--compressedStreamPath={bin_out}",
            "--inputScale=1",
            "--codingScale=1",
            # Crucial: keep all points even when they coincide on the grid.
            "--mergeDuplicatedPoints=0",
            # Silence chatty output
            # (tmc3 does not have a --quiet flag; we capture stdout)
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"tmc3 encode failed (exit {proc.returncode}):\n"
                f"STDOUT: {proc.stdout[-2000:]}\n"
                f"STDERR: {proc.stderr[-2000:]}"
            )

        with open(bin_out, "rb") as f:
            stream = f.read()

    return _pack_blob(stream, quant_step, shift, scaled.shape[0])


def decode_positions(
    blob: bytes,
    *,
    workdir: Optional[str] = None,
) -> torch.Tensor:
    """Decode a TMC1 blob back to positions.

    Returns a ``[N, 3]`` float32 CPU tensor.

    The returned points are in **G-PCC Morton order**, which differs
    from the order they were encoded in.  See module docstring for the
    workflow that keeps auxiliary tensors aligned.
    """
    stream, quant_step, shift, _N = _unpack_blob(blob)
    binary = require_tmc3()

    with tempfile.TemporaryDirectory(prefix="tmc13_", dir=workdir) as tmp:
        bin_in = os.path.join(tmp, "in.bin")
        ply_out = os.path.join(tmp, "out.ply")
        with open(bin_in, "wb") as f:
            f.write(stream)

        cmd = [
            binary,
            "--mode=1",
            f"--compressedStreamPath={bin_in}",
            f"--reconstructedDataPath={ply_out}",
            # outputScaling=0 -> conformance (no rescale; integers come out raw)
            "--outputScaling=0",
            "--outputBinaryPly=1",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"tmc3 decode failed (exit {proc.returncode}):\n"
                f"STDOUT: {proc.stdout[-2000:]}\n"
                f"STDERR: {proc.stderr[-2000:]}"
            )

        ints = _read_int_ply(ply_out)

    recon = ints.astype(np.float64) * quant_step + shift
    return torch.from_numpy(recon.astype(np.float32))


def encode_positions_with_perm(
    positions: torch.Tensor,
    quant_step: float = 1e-3,
    *,
    workdir: Optional[str] = None,
) -> Tuple[bytes, torch.Tensor]:
    """Encode positions and return the *encode-order → decode-order* permutation.

    Returns ``(blob, perm)`` where ``decoded_positions ≈ positions[perm]``
    (modulo quantisation).  Use this to reorder auxiliary per-point
    tensors at encode time so they line up with the decoded position
    order at decode time::

        blob, perm = encode_positions_with_perm(positions, quant_step)
        rotations = rotations[perm]
        scales = scales[perm]
        # save blob, rotations, scales — they're all in decode order now

    The permutation is computed on the **quantised integer grid**, so it
    works even when multiple input points collapse onto the same cell
    (in that case the assignment is arbitrary among duplicates — but the
    decoded position is the same, so any choice keeps decoded[k] aligned
    with positions[perm[k]] in world-space).
    """
    blob = encode_positions(positions, quant_step, workdir=workdir)
    decoded = decode_positions(blob, workdir=workdir)

    pos_np = positions.detach().to(torch.float32).cpu().numpy()
    dec_np = decoded.detach().to(torch.float32).cpu().numpy()

    # Match on the quantised grid: both sides live on the same integer
    # lattice with spacing `quant_step`. Compute keys with rounded integer
    # coords + bucket sort. Duplicates are resolved in original/decoded
    # insertion order which is deterministic.
    N = pos_np.shape[0]
    shift = pos_np.min(axis=0)
    pos_int = np.round((pos_np - shift) / quant_step).astype(np.int64)
    # Decoded was emitted with the same shift+quant; re-quantise too so
    # any float64-to-float32 roundoff cancels.
    dec_int = np.round((dec_np - shift) / quant_step).astype(np.int64)

    perm = _bipartite_match_by_int_key(pos_int, dec_int)
    return blob, torch.from_numpy(perm)


def _bipartite_match_by_int_key(
    original_int: np.ndarray, decoded_int: np.ndarray
) -> np.ndarray:
    """Match decoded[k] -> some original[perm[k]] with the same integer key.

    Both inputs are ``[N, 3]`` int arrays.  A matching is "valid" iff
    every original row is used exactly once and ``decoded[k] ==
    original[perm[k]]`` for each k (so the matching is identity in
    world-space, up to grid resolution).

    Algorithm: bucket original indices by their (x, y, z) tuple, then
    walk decoded rows in order and pop one matching original index from
    each bucket.  O(N) with hashable tuples.

    With ``mergeDuplicatedPoints=0`` every decoded row has a matching
    original; if any does not (or the counts differ), the reconstruction
    is not the input grid and this raises instead of guessing a pairing.
    """
    N = original_int.shape[0]
    if decoded_int.shape[0] != N:
        raise RuntimeError(
            f"tmc3 reconstruction does not match the input grid "
            f"({decoded_int.shape[0]} decoded vs {N} input points)"
        )

    # Bucket originals by integer key (uses 3-tuple as dict key).
    from collections import defaultdict
    buckets = defaultdict(list)
    for i in range(N):
        buckets[(int(original_int[i, 0]), int(original_int[i, 1]),
                 int(original_int[i, 2]))].append(i)

    perm = np.empty(N, dtype=np.int64)
    misses = 0
    for k in range(N):
        key = (int(decoded_int[k, 0]), int(decoded_int[k, 1]),
               int(decoded_int[k, 2]))
        if buckets[key]:
            perm[k] = buckets[key].pop()
        else:
            misses += 1
    if misses:
        raise RuntimeError(
            f"tmc3 reconstruction does not match the input grid "
            f"({misses} of {N} points)"
        )
    return perm


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def is_tmc13_available() -> bool:
    return _find_tmc3_binary() is not None


def tmc13_version_string() -> Optional[str]:
    """Version the binary reports, as ``"<tag> <short sha>"``.

    ``tmc3 --help`` prints e.g. ``MPEG PCC tmc3 version
    release-v23.0-rc2-0-ga3d15c5`` (git-describe), recorded as
    ``"release-v23.0-rc2 a3d15c5"``. Unparseable output is returned as is;
    None if the binary is missing or does not run.
    """
    binary = _find_tmc3_binary()
    if binary is None:
        return None
    try:
        out = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=10
        )
        first = (out.stdout + out.stderr).splitlines()[0].strip()
    except Exception:
        return None
    m = re.search(r"version\s+(\S+?)-\d+-g([0-9a-f]+)", first)
    return f"{m.group(1)} {m.group(2)}" if m else first
