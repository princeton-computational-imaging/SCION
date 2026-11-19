#!/usr/bin/env bash
# Download and extract the paper's datasets into $DATA_ROOT (default data/),
# one directory per scene, in the layout the archives ship:
#
#   mipnerf360  Mip-NeRF 360 (9 scenes)                   <scene>/images*/, <scene>/sparse/0/  (COLMAP)
#   tandt       Tanks & Temples + Deep Blending (4 scenes) <scene>/images/,  <scene>/sparse/0/  (COLMAP)
#   synthetic   Synthetic-NeRF (8 scenes)                  <scene>/transforms_*.json, train/ val/ test/ (Blender)
#   campus      SCION campus ID/OOD (2 scenes)             <scene>/images*/, <scene>/sparse/0/, <scene>/split.json
#
# Usage: bash scripts/download_data.sh <mipnerf360|tandt|synthetic|campus|all>
#
# Archives go to $DATA_ROOT/_archives/ and are SHA256-checked; delete them once
# extracted. Synthetic-NeRF is only on Google Drive, so download nerf_synthetic.zip
# by hand into that folder. Scenes that already exist are skipped.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
ARCHIVES="$DATA_ROOT/_archives"

# fetch <archive> <sha256> <url, or the page for a manual download> [manual]
fetch() {
    local path="$ARCHIVES/$1"
    mkdir -p "$ARCHIVES"
    if [[ ! -f "$path" ]]; then
        if [[ "${4:-}" == manual ]]; then
            echo "error: $path is missing; download $1 from $3 and re-run" >&2
            exit 1
        fi
        echo "downloading $3"
        curl -fL --retry 5 -C - -o "$path.part" "$3"
        mv "$path.part" "$path"
    fi
    echo "checking $path"
    echo "$2  $path" | sha256sum -c --quiet - \
        || { echo "error: $path does not match its SHA256; delete it and re-run" >&2; exit 1; }
}

# extract <archive> <folder in the archive> <marker file> <scene>...
extract() {
    local zip="$ARCHIVES/$1" prefix="$2" marker="$3" scene
    shift 3
    for scene in "$@"; do
        if [[ -e "$DATA_ROOT/$scene" ]]; then
            [[ -f "$DATA_ROOT/$scene/$marker" ]] \
                || { echo "error: $DATA_ROOT/$scene exists but has no $marker; move it away" >&2; exit 1; }
            echo "$DATA_ROOT/$scene exists, skipping"
            continue
        fi
        echo "extracting $scene"
        rm -rf "$DATA_ROOT/_unpack"
        unzip -q "$zip" "$prefix$scene/*" -d "$DATA_ROOT/_unpack"
        mv "$DATA_ROOT/_unpack/$prefix$scene" "$DATA_ROOT/$scene"
        rm -rf "$DATA_ROOT/_unpack"
    done
}

mipnerf360() {
    fetch 360_v2.zip 77332bf4eba3b8ca0c7f70130849b1e394efdd60d8f20efa6f217081d08a8b2a \
        https://storage.googleapis.com/gresearch/refraw360/360_v2.zip
    fetch 360_extra_scenes.zip f8d42b372d7cc589928c3d22849f958c1e2ae948c21e89c6a32c904cefba2fa4 \
        https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
    extract 360_v2.zip "" sparse/0/cameras.bin bicycle bonsai counter garden kitchen room stump
    extract 360_extra_scenes.zip "" sparse/0/cameras.bin flowers treehill
}

tandt() {
    fetch tandt_db.zip 816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1 \
        https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
    extract tandt_db.zip tandt/ sparse/0/cameras.bin train truck
    extract tandt_db.zip db/ sparse/0/cameras.bin drjohnson playroom
}

synthetic() {
    fetch nerf_synthetic.zip 554fed5d4884028f95e0da422f99f5fc071fffefc9f2fc72c5f9361801df2599 \
        https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 manual
    extract nerf_synthetic.zip nerf_synthetic/ transforms_train.json \
        chair drums ficus hotdog lego materials mic ship
}

# The campus scenes are a Hugging Face dataset; `hf` comes with the environment.
campus() {
    local scene
    for scene in mather_sundial mccosh_walkway; do
        if [[ -e "$DATA_ROOT/$scene" ]]; then
            echo "$DATA_ROOT/$scene exists, skipping"
            continue
        fi
        hf download pci-lab/scion-data --repo-type dataset \
            --include "$scene/*" --local-dir "$DATA_ROOT"
    done
}

case "${1:-}" in
    mipnerf360|tandt|synthetic|campus) "$1" ;;
    all) mipnerf360; tandt; synthetic; campus ;;
    *) echo "usage: $0 <mipnerf360|tandt|synthetic|campus|all>" >&2; exit 2 ;;
esac
