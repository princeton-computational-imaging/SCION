"""Render paper tables from run.json records.

    python -m experiments.paper.render <target|all> --runs-root DIR --out DIR [--level L]

Writes ``<out>/tables/*.tex`` (and ``<out>/figures/*`` where a target has a
figure). A target whose records are incomplete or inconsistent is refused:
nothing is written for it, the offending records are listed, and the exit
code is 1. Figure targets only print their generator command.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional

from scion.util.run_index import LEVELS

from experiments.paper.common import BASELINES, Baselines, Ctx, RenderError
from experiments.paper.targets import load_registry


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("target", help="a target from experiments/paper/targets.yaml, or `all`")
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="paper root (gets tables/, figures/)")
    p.add_argument("--level", choices=LEVELS, default="quantized",
                   help="headline compression level (default: quantized)")
    p.add_argument("--baselines", type=Path, default=BASELINES)
    args = p.parse_args(argv)

    reg = load_registry()
    if args.target != "all" and args.target not in reg.targets:
        p.error(f"unknown target {args.target!r}; choose from: all, {', '.join(reg.targets)}")
    names = list(reg.targets) if args.target == "all" else [args.target]
    baselines = Baselines(args.baselines)
    failed = []
    for name in names:
        t = reg.targets[name]
        if not t.renderer:
            print(f"[render] {name}: figure target, generator: {t.command or 'none'} ({t.manual})")
            continue
        ctx = Ctx(reg, t, args.runs_root, args.out, args.level, baselines)
        try:
            outputs = importlib.import_module(f"experiments.paper.{t.renderer}").render(ctx)
        except RenderError as e:
            print(f"[render] {name}: REFUSED\n{e}", file=sys.stderr)
            failed.append(name)
            continue
        for rel, text in outputs.items():
            path = args.out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        print(f"[render] {name}: wrote {', '.join(outputs)}")
    for w in sorted(baselines.warnings):
        print(f"[render] warning: {w}", file=sys.stderr)
    if failed:
        print(f"[render] refused: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
