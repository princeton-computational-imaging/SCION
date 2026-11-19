#!/usr/bin/env bash
# Build tmc3, the MPEG G-PCC reference codec (MPEG-PCC-TMC13) that compression
# level `codec` uses for instance positions. Pinned to release-v23.0-rc2
# (a3d15c5), the version OMG and HAC++ use.
#
#   bash scripts/setup/build_tmc13.sh     # prints the tmc3 path on stdout
#
# Idempotent: uses the same lookup as scion/compression/tmc13.py
# ($TMC13_BINARY, then the build dir below, then tmc3 on PATH) and only
# clones + builds when none of them exists. Any binary it finds or builds
# must report the pinned version, else it exits non-zero. Build dir:
# ${TMC13_SRC:-${XDG_CACHE_HOME:-~/.cache}/scion/mpeg-pcc-tmc13};
# with a custom TMC13_SRC, export TMC13_BINARY=<printed path> afterwards.
# Needs git, cmake and a C++ compiler.
set -euo pipefail

TAG=release-v23.0-rc2
COMMIT=a3d15c5
SRC="${TMC13_SRC:-${XDG_CACHE_HOME:-$HOME/.cache}/scion/mpeg-pcc-tmc13}"
BIN="$SRC/build/tmc3/tmc3"

use() {  # print $1 and exit 0 iff it reports the pinned version
  local v
  v="$("$1" --help 2>&1 | head -n 1 || true)"
  if [[ "$v" != *"$TAG-"*"-g$COMMIT"* ]]; then
    echo "error: $1 reports '$v', expected $TAG ($COMMIT). Point TMC13_BINARY at" \
         "a pinned build or unset it and re-run this script." >&2
    exit 1
  fi
  echo "$1"
  exit 0
}

if [[ -n "${TMC13_BINARY:-}" ]]; then
  [[ -x "$TMC13_BINARY" ]] || { echo "error: TMC13_BINARY=$TMC13_BINARY is not executable" >&2; exit 1; }
  use "$TMC13_BINARY"
fi
if [[ -x "$BIN" ]]; then use "$BIN"; fi
if command -v tmc3 >/dev/null; then use "$(command -v tmc3)"; fi

if [[ ! -d "$SRC/.git" ]]; then
  git clone --quiet --depth 1 --branch "$TAG" https://github.com/MPEGGroup/mpeg-pcc-tmc13 "$SRC" >&2
fi
head="$(git -C "$SRC" rev-parse --short=7 HEAD)"
if [[ "$head" != "$COMMIT" ]]; then
  echo "error: $SRC is at $head, expected $TAG ($COMMIT)" >&2; exit 1
fi
# The policy flag lets CMake >= 4 configure tmc13's old cmake_minimum_required.
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >&2
cmake --build "$SRC/build" --target tmc3 -j "$(nproc 2>/dev/null || echo 4)" >&2
use "$BIN"
