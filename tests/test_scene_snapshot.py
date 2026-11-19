"""Scene loading against a frozen snapshot: scion.data builds the recorded
scene for every paper dataset (train/eval frames, downscale, poses,
intrinsics, seed points). Reference: tests/fixtures/snapshot/scenes.json.
Synthetic-NeRF seed points come from COLMAP triangulation at load time
(scion.data.synthetic), so lego's snapshot has no points.

Data root: ``$SCION_SNAPSHOT_DATA/data`` when set, else this
checkout's ``data/``; scenes that are not on disk are skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from scion.data import load_scene
from scion.engine.run import scene_kwargs
from scion.util.provenance import compose

REPO = Path(__file__).resolve().parents[1]
REF = json.loads((REPO / "tests" / "fixtures" / "snapshot" / "scenes.json").read_text())
_env = os.environ.get("SCION_SNAPSHOT_DATA")
DATA = Path(_env) / "data" if _env else REPO / "data"


def _sha(paths):
    return hashlib.sha256("\n".join(Path(p).name for p in paths).encode()).hexdigest()


@pytest.mark.parametrize("key", sorted(REF))
def test_scene_matches_snapshot(key):
    dataset, name = key.split("/")
    if not (DATA / name).is_dir():
        pytest.skip(f"{DATA / name} missing")
    ref = REF[key]
    scene = load_scene(DATA / name, **scene_kwargs(compose(["method=codebook", f"dataset={dataset}",
                                                           f"scene={name}"])))
    train = [scene.image_paths[i] for i in scene.train_indices]
    assert _sha(train) == ref["train_names_sha256"] and len(train) == ref["n_train"]
    assert _sha(scene.image_paths[i] for i in scene.eval_indices) == ref["eval_names_sha256"]
    assert scene.downscale_factor == ref["downscale_factor"]

    cams = scene.train_cameras()[:5]
    c2w = torch.stack([c.cam_to_world[:3, :4] for c in cams]).float()
    torch.testing.assert_close(c2w, torch.tensor(ref["c2w"]), rtol=0, atol=1e-5)
    K = torch.tensor([[c.K[0, 0], c.K[1, 1], c.K[0, 2], c.K[1, 2]] for c in cams])
    torch.testing.assert_close(K, torch.tensor(ref["fxfycxcy"]), rtol=0, atol=1e-3)
    assert [[c.width, c.height] for c in cams] == ref["wh"]

    if ref["points"] is not None:
        pts = scene.sparse_points_3d
        assert pts.shape[0] == ref["points"]["n"]
        assert pts.double().sum(0).tolist() == pytest.approx(ref["points"]["sum"], rel=1e-5)
        assert scene.sparse_points_rgb.long().sum(0).tolist() == ref["points"]["rgb_sum"]
