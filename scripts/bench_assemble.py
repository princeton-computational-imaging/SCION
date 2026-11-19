#!/usr/bin/env python3
"""Time ``assemble_fused`` (forward and forward+backward) on a trained model, per backend tier.

Uses the real codebook state of a checkpoint instead of synthetic inputs, so the
instance/splat counts and the rows-per-instance distribution are those of an actual scene.

    python scripts/bench_assemble.py --ckpt <run dir or checkpoint> [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from scion.engine.checkpoint import load
from scion.ops import backend
from scion.ops.assemble import assemble_fused


def _inputs(model, requires_grad: bool):
    p = model.get_splat_params_dict()
    model._ensure_layout()
    floats = [p["means"], p["scales"], p["rotations"], p["opacities"], p["features_dc"],
              model.instance_positions, model.instance_rotations,
              model.instance_scales, model.instance_opacities]
    floats = [t.detach().clone().requires_grad_(requires_grad) for t in floats]
    return floats, (model._cand_splat_rows, model._inst_offsets)


def _time_ms(fn, warmup: int = 5, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench(model, tier: str) -> dict:
    os.environ["SCION_BACKEND_assemble_fused"] = tier
    os.environ["SCION_BACKEND_assemble_fused_bwd"] = tier
    backend.reset_backend_cache()
    bounds = (math.log(1e-4), 5.0, 20.0)

    floats, layout = _inputs(model, requires_grad=False)
    with torch.no_grad():
        fwd = _time_ms(lambda: assemble_fused(*floats, *layout, *bounds))

    floats, layout = _inputs(model, requires_grad=True)

    def fwd_bwd():
        out = assemble_fused(*floats, *layout, *bounds)
        sum(o.sum() for o in out).backward()

    return {"forward_ms": fwd, "forward_backward_ms": _time_ms(fwd_bwd)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    model = load(args.ckpt).model.cuda().eval()
    model._ensure_layout()
    res = {
        "checkpoint": args.ckpt,
        "gpu": torch.cuda.get_device_name(),
        "n_instances": int(model.instance_positions.shape[0]),
        "n_gaussians": int(model._layout_total),
        "tiers": {tier: bench(model, tier) for tier in ("cuda", "python")},
    }
    for k in ("forward_ms", "forward_backward_ms"):
        res[f"speedup_{k}"] = res["tiers"]["python"][k] / res["tiers"]["cuda"][k]
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
