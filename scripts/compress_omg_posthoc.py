#!/usr/bin/env python3
"""Apply compression level ``codec`` to a ``quantized`` checkpoint by hand.

Usage:
    python scripts/compress_omg_posthoc.py \
        --input  <run_dir>/checkpoints/quantized.ckpt \
        --output codec.omg \
        --decode-verify

Defaults are the shipped codec level (``EncodeOptions()`` =
``config/compression/codec.yaml``): SVQ M=3 B=256 on the splat attributes,
MPEG G-PCC (``tmc3``) positions, quaternion SVQ M=4 B=256, range coding,
LZMA. See ``scion/compression/omg_posthoc.py``. Training runs do
this automatically (``scion.compression.levels``); this script is
for sweeping codec options on an existing checkpoint.

``--decode-verify`` decodes the bundle again and prints per-tensor deltas;
``python -m scion.eval --ckpt <run_dir> --level codec --artifact <bundle>``
renders and scores it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Ensure the repo root is importable when running from outside the worktree.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scion.compression import omg_posthoc as omg  # noqa: E402
from scion.compression import tmc13  # noqa: E402
from scion.compression.quantize import is_quantized_state_dict  # noqa: E402

_D = omg.EncodeOptions()  # CLI defaults = the codec level


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", type=Path, required=True,
                   help="a quantized checkpoint (<run_dir>/checkpoints/quantized.ckpt)")
    p.add_argument("--output", "-o", type=Path, required=True,
                   help="where to write the .omg bundle")
    p.add_argument("--M", type=int, default=_D.M, help="SVQ sub-vector count")
    p.add_argument("--B", type=int, default=_D.B, help="SVQ codebook size")
    p.add_argument("--min-rows-for-svq", type=int, default=_D.min_rows_for_svq,
                   help="skip SVQ for tensors with fewer rows")
    p.add_argument("--pos-bits", type=int, default=_D.pos_bits,
                   help="bit depth for instance_positions per-axis quant "
                        "(only used when --position-format=fp16)")
    p.add_argument("--seed", type=int, default=_D.seed)
    p.add_argument("--no-lzma", action="store_true",
                   help="emit the bundle without the outer LZMA wrap")
    p.add_argument("--position-format", type=str, default=_D.position_format,
                   choices=("tmc13", "fp16"),
                   help="how to encode instance_positions (tmc13 = MPEG G-PCC)")
    p.add_argument("--tmc13-quant-step", type=float, default=_D.tmc13_quant_step,
                   help="TMC13 quantisation step in world-space units "
                        "(only used when --position-format=tmc13)")
    p.add_argument("--svq-instance-attrs", action=argparse.BooleanOptionalAction,
                   default=_D.svq_instance_attrs,
                   help="SVQ-encode instance_rotations (4-D quats) with "
                        "--instance-svq-M / --instance-svq-B codebooks.")
    p.add_argument("--instance-svq-M", type=int, default=_D.instance_svq_M,
                   help="M for per-instance SVQ (4 = scalar VQ per quat "
                        "component; M=2 costs ~1 dB because pairs of "
                        "codewords don't combine onto the unit sphere).")
    p.add_argument("--instance-svq-B", type=int, default=_D.instance_svq_B,
                   help="B for per-instance SVQ.")
    p.add_argument("--decode-verify", action="store_true",
                   help="also decode and report PSNR delta per tensor")
    p.add_argument("--report-json", type=Path, default=None,
                   help="write a per-tensor byte-budget report here")
    return p.parse_args()


def _load_quantized(path: Path) -> tuple[Dict[str, Any], int]:
    """Load a quantized checkpoint and return ``(state_dict, step)``.

    Accepts a bare state_dict (``checkpoints/quantized.ckpt``) or a
    ``{"step", "pipeline"}`` wrapper.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"not a checkpoint dict: {path}")
    state = dict(ckpt["pipeline"]) if "pipeline" in ckpt else dict(ckpt)
    if not is_quantized_state_dict(state):
        raise ValueError(
            f"{path} is not a quantized checkpoint (no __quantized_meta__ "
            "keys); pass checkpoints/quantized.ckpt, not raw.ckpt"
        )
    return state, int(ckpt.get("step", -1))


def _tensor_l2_delta(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute mean / max absolute difference between two tensors."""
    af = a.detach().cpu().float()
    bf = b.detach().cpu().float()
    if af.shape != bf.shape:
        return {"shape_mismatch": True,
                "shape_a": tuple(af.shape), "shape_b": tuple(bf.shape)}
    diff = (af - bf).abs()
    return {
        "shape": tuple(af.shape),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "rel_l2": float(
            (diff.flatten().norm() / max(af.flatten().norm().item(), 1e-12)).item()
        ) if diff.numel() else 0.0,
    }


def main() -> int:
    args = _parse_args()

    if not args.input.exists():
        print(f"FATAL: input not found: {args.input}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[omg] reading {args.input}")
    state, step = _load_quantized(args.input)
    n_tensors = sum(1 for v in state.values() if isinstance(v, torch.Tensor))
    print(f"[omg] step={step}, {n_tensors} tensor entries in state_dict")

    options = omg.EncodeOptions(
        M=args.M, B=args.B,
        min_rows_for_svq=args.min_rows_for_svq,
        pos_bits=args.pos_bits, seed=args.seed,
        lzma_wrap=not args.no_lzma,
        position_format=args.position_format,
        tmc13_quant_step=args.tmc13_quant_step,
        svq_instance_attrs=args.svq_instance_attrs,
        instance_svq_M=args.instance_svq_M,
        instance_svq_B=args.instance_svq_B,
    )
    print(f"[omg] encoding: M={options.M}, B={options.B}, "
          f"min_rows_for_svq={options.min_rows_for_svq}, pos_bits={options.pos_bits}, "
          f"lzma={options.lzma_wrap}, "
          f"position_format={options.position_format}, "
          f"tmc13_quant_step={options.tmc13_quant_step}, "
          f"tmc3={tmc13._find_tmc3_binary()}")

    blob, info = omg.encode_quantized_to_omg(state, options)
    with open(args.output, "wb") as f:
        f.write(blob)

    in_size = args.input.stat().st_size
    out_size = args.output.stat().st_size
    print(f"[omg] wrote {args.output} ({out_size/1024/1024:.3f} MB)")
    print(f"[omg] input ckpt: {in_size/1024/1024:.3f} MB; "
          f"compression ratio: {in_size/max(out_size,1):.2f}x")

    # Budget summary
    print("[omg] per-tensor budget (encoded contribution, pre-LZMA):")
    budget = info.get("tensor_budget", {})
    for k in sorted(budget, key=lambda kk: -budget[kk]):
        print(f"        {k:65s} {budget[k]/1024:>8.1f} KB")
    print(f"        TOTAL pre-LZMA tensor bytes: "
          f"{info['tensor_budget_total']/1024/1024:.3f} MB")
    print(f"        container size (pre-LZMA):   "
          f"{info['serialized_size']/1024/1024:.3f} MB")
    print(f"        final blob (LZMA-wrapped):   "
          f"{info['final_size']/1024/1024:.3f} MB")

    # Decode + verify
    decoded: Dict[str, Any] = {}
    if args.decode_verify:
        with open(args.output, "rb") as f:
            wire = f.read()
        decoded = omg.decode_omg_to_state_dict(wire)

    if args.decode_verify:
        print("[omg] decode-verify: per-tensor delta (round-trip)")
        # The encoder reorders all per-instance tensors into the position
        # codec's canonical order (Morton for fp16, TMC13's octree walk
        # for tmc13); the decoder emits them in that same order. To
        # compare pre/post round-trip we apply the same permutation to
        # the original tensors before diffing. The permutation depends on
        # the position format used.
        from scion.compression.omg_posthoc import position_encode
        pos_key_candidates = [
            k for k, v in state.items()
            if isinstance(v, torch.Tensor)
            and k.rsplit(".", 1)[-1] == "instance_positions"
        ]
        morton_order = None
        n_instances = 0
        if pos_key_candidates:
            pos_arr = state[pos_key_candidates[0]].float().cpu().numpy()
            n_instances = pos_arr.shape[0]
            if options.position_format == "tmc13":
                _, perm_t = tmc13.encode_positions_with_perm(
                    torch.from_numpy(pos_arr), quant_step=options.tmc13_quant_step
                )
                morton_order = perm_t.cpu().numpy().astype(np.int64)
            else:
                _, morton_order = position_encode(pos_arr, bits=args.pos_bits)
        rows = []
        for key in sorted(state.keys()):
            v0 = state[key]
            v1 = decoded.get(key)
            if not isinstance(v0, torch.Tensor) or not isinstance(v1, torch.Tensor):
                continue
            # If this is per-instance, sort the original in Morton order
            # to align with what the decoder emitted.
            base = key.rsplit(".", 1)[-1]
            if (
                morton_order is not None
                and base.startswith("instance_")
                and v0.shape[0] == n_instances
            ):
                v0_aligned = v0.detach().cpu()[torch.from_numpy(morton_order)]
            else:
                v0_aligned = v0
            d = _tensor_l2_delta(v0_aligned, v1)
            rows.append((key, d))
        for key, d in rows:
            if d.get("shape_mismatch", False):
                print(f"        {key:65s} SHAPE MISMATCH "
                      f"{d['shape_a']} != {d['shape_b']}")
                continue
            shp = d["shape"]
            print(f"        {key:65s} shape={shp} "
                  f"max_abs={d['max_abs']:.4g} mean_abs={d['mean_abs']:.4g} "
                  f"rel_l2={d['rel_l2']:.4g}")

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        report: Dict[str, Any] = {
            "input": str(args.input),
            "output": str(args.output),
            "step": step,
            "input_size_bytes": in_size,
            "output_size_bytes": out_size,
            "compression_ratio_vs_quantized": in_size / max(out_size, 1),
            "options": asdict(options),
            "have_tmc13": tmc13.is_tmc13_available(),
            "tmc13_version": tmc13.tmc13_version_string(),
            "position_format_used": info["position_format"],
            "tensor_budget": {k: int(v) for k, v in info.get("tensor_budget", {}).items()},
            "tensor_budget_total": int(info.get("tensor_budget_total", 0)),
            "serialized_size_bytes": int(info["serialized_size"]),
            "final_size_bytes": int(info.get("final_size", 0)),
        }
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[omg] wrote byte-budget report to {args.report_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
