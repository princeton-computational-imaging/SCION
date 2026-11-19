#!/usr/bin/env bash
# Turn a folder of photos into a SCION scene with COLMAP, the steps of 3DGS's convert.py:
#
#   bash scripts/colmap_scene.sh data/my_scene              # photos in data/my_scene/input/
#   bash scripts/colmap_scene.sh data/my_scene sequential   # video frames
#
# Writes data/my_scene/images/ (undistorted, one shared pinhole camera), images_{2,4,8}/
# and the COLMAP model in sparse/0/. Needs `colmap` on PATH.
set -euo pipefail

S=${1:?usage: $0 <scene_dir> [exhaustive|sequential]}
MATCHER=${2:-exhaustive}
[[ -d $S/input ]] || { echo "error: put the photos in $S/input/" >&2; exit 1; }
[[ ! -e $S/sparse ]] || { echo "error: $S/sparse exists; remove it to start over" >&2; exit 1; }

colmap feature_extractor --database_path "$S/database.db" --image_path "$S/input" \
    --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV
colmap "${MATCHER}_matcher" --database_path "$S/database.db"
mkdir -p "$S/distorted"
colmap mapper --database_path "$S/database.db" --image_path "$S/input" --output_path "$S/distorted"
colmap image_undistorter --image_path "$S/input" --input_path "$S/distorted/0" --output_path "$S"
mkdir "$S/sparse/0" && mv "$S"/sparse/*.bin "$S/sparse/0/"

python - "$S" <<'EOF'
import sys
from pathlib import Path
from PIL import Image

scene = Path(sys.argv[1])
for f in (2, 4, 8):
    (scene / f"images_{f}").mkdir(exist_ok=True)
    for p in sorted((scene / "images").iterdir()):
        im = Image.open(p)
        im.resize((round(im.width / f), round(im.height / f)), Image.LANCZOS).save(
            scene / f"images_{f}" / p.name, quality=95)
EOF
echo "done: train with scene=$(basename "$S") (data root $(dirname "$S"))"
