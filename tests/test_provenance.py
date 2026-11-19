"""scion.util.provenance: only code is part of a job's identity
(git_sha / tree_is_dirty); run records and docs are not, so committing them
neither invalidates finished jobs nor dirties the tree."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from scion.util.provenance import eval_images_sha256, git_sha, tree_is_dirty

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   cwd=repo, check=True, capture_output=True)


def test_records_do_not_change_sha_or_dirtiness(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "scion").mkdir()
    (tmp_path / "scion" / "code.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "code")
    sha = git_sha(tmp_path)
    assert len(sha) == 40 and not tree_is_dirty(tmp_path)

    record = tmp_path / "evals" / "runs" / "ours_small" / "run.json"
    record.parent.mkdir(parents=True)
    record.write_text("{}")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "records")
    record.write_text('{"status": "DONE"}')        # re-run over a tracked record
    assert git_sha(tmp_path) == sha and not tree_is_dirty(tmp_path)

    (tmp_path / "README.md").write_text("docs\n")  # a docs-only commit
    (tmp_path / "scion" / "NOTES.md").write_text("docs\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "docs")
    assert git_sha(tmp_path) == sha and not tree_is_dirty(tmp_path)

    (tmp_path / "scion" / "code.py").write_text("x = 2\n")
    assert tree_is_dirty(tmp_path)
    _git(tmp_path, "commit", "-qam", "change code")
    assert git_sha(tmp_path) != sha and not tree_is_dirty(tmp_path)


def test_untracked_code_is_dirty_untracked_elsewhere_is_not(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / "notes.txt").write_text("scratch")            # outside the code dirs
    (tmp_path / "config" / "__pycache__").mkdir(parents=True)
    (tmp_path / "config" / "__pycache__" / "x.pyc").write_text("")  # ignored
    assert not tree_is_dirty(tmp_path)
    (tmp_path / "config" / "dataset_variant").mkdir()
    (tmp_path / "config" / "dataset_variant" / "new.yaml").write_text("k: 1\n")
    assert tree_is_dirty(tmp_path)  # an untracked config would silently change jobs


def test_eval_images_sha256_is_order_free():
    assert eval_images_sha256(["b.png", "a.png"]) == eval_images_sha256(["a.png", "b.png"])


def test_data_sha256_covers_camera_model_and_seed_cloud_not_images(tmp_path):
    from scion.util.provenance import data_sha256

    with pytest.raises(FileNotFoundError):
        data_sha256(tmp_path)
    (tmp_path / "transforms.json").write_text('{"frames": []}')
    base = data_sha256(tmp_path)
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(b"x")
    assert data_sha256(tmp_path) == base                     # images are not hashed
    (tmp_path / "colmap_points.ply").write_bytes(b"ply")
    with_ply = data_sha256(tmp_path)
    (tmp_path / "colmap_points.ply").write_bytes(b"ply2")
    assert len({base, with_ply, data_sha256(tmp_path)}) == 3
    colmap = tmp_path / "colmap"                             # a COLMAP scene hashes its sparse/0 model
    (colmap / "sparse" / "0").mkdir(parents=True)
    (colmap / "sparse" / "0" / "cameras.bin").write_bytes(b"c")
    first = data_sha256(colmap)
    (colmap / "sparse" / "0" / "images.bin").write_bytes(b"i")
    assert data_sha256(colmap) != first
