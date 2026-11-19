"""The Synthetic-NeRF seed: Blender poses handed to COLMAP (scion.data.synthetic)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from scion.data.colmap import _qvec2rotmat
from scion.data.synthetic import colmap_pose


def test_colmap_pose_sees_a_point_where_the_blender_camera_does():
    c2w = np.eye(4)
    c2w[:3, :3] = Rotation.random(random_state=1).as_matrix()
    c2w[:3, 3] = [0.5, -1.0, 4.0]
    in_gl_camera = np.array([0.3, 0.2, -2.0])        # OpenGL: right, up, 2 in front (-z)
    world = c2w[:3, :3] @ in_gl_camera + c2w[:3, 3]
    q, t = colmap_pose(c2w[:3])
    assert np.allclose(_qvec2rotmat(q) @ world + t, [0.3, -0.2, 2.0])  # OpenCV: right, down, front (+z)
