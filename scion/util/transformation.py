import math
import torch
import torch.nn.functional as F

def random_quat_upper_cone(N, cone_angle_degrees=45.0, device=None):
    """
    Generates N random quaternions such that:
      - The rotated local +Z axis lies within a cone of half-angle `cone_angle_degrees`
        around the world +Z axis.
      - cone_angle_degrees = 180 => uniform upper hemisphere
      - cone_angle_degrees = 0   => no tilt (always vertical)

    Args:
        N (int): number of quaternions
        cone_angle_degrees (float): half-angle of the cone around +Z (0–180)
        device (torch.device or None): optional device
    
    Returns:
        Tensor of shape (N, 4) of unit quaternions (w, x, y, z)
    """
    device = device or "cpu"
    
    # Convert to radians
    cone = math.radians(cone_angle_degrees)
    
    # Uniform direction within a spherical cap (cone)
    # cos(theta) ∈ [cos(cone), 1]
    u = torch.rand(N, device=device)
    cos_theta = 1 - u * (1 - math.cos(cone))
    sin_theta = torch.sqrt(1 - cos_theta * cos_theta)
    
    # random azimuth φ
    phi = 2 * math.pi * torch.rand(N, device=device)

    # Target direction for rotated +Z axis
    dir_x = sin_theta * torch.cos(phi)
    dir_y = sin_theta * torch.sin(phi)
    dir_z = cos_theta
    dirs = torch.stack([dir_x, dir_y, dir_z], dim=-1)   # [N, 3]

    # Create quaternion that rotates (0,0,1) into dirs
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(dirs)
    
    # Rotation axis = cross(e_z, dir)
    axis = torch.cross(z_axis, dirs)
    axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)

    # Handle zero-tilt cases where axis is undefined
    axis_safe = torch.where(axis_norm > 1e-8, axis / axis_norm, torch.tensor([1.0, 0.0, 0.0], device=device))

    # angle between +Z and dir is theta
    theta = torch.acos(torch.clamp(dir_z, -1.0, 1.0))  # rotation angle
    
    half = theta / 2
    sin_half = torch.sin(half)

    # quaternion rotating +Z → dir
    q = torch.stack([
        torch.cos(half),                           # w
        axis_safe[:, 0] * sin_half,               # x
        axis_safe[:, 1] * sin_half,               # y
        axis_safe[:, 2] * sin_half,               # z
    ], dim=-1)

    # Optional: add random twist around target direction
    # (makes grass look more natural)
    twist_angle = 2 * math.pi * torch.rand(N, device=device)
    half_twist = twist_angle / 2
    sin_twist = torch.sin(half_twist)

    q_twist = torch.stack([
        torch.cos(half_twist),
        dirs[:, 0] * sin_twist,
        dirs[:, 1] * sin_twist,
        dirs[:, 2] * sin_twist,
    ], dim=-1)

    # Compose rotations: q_total = q_twist ⊗ q_dir
    # quaternion multiplication
    w1, x1, y1, z1 = q_twist.unbind(-1)                        # q1 = twist
    w2, x2, y2, z2 = q.unbind(-1)                              # q2 = align
    q_out = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

    # Normalize (numerical safety)
    q_out = q_out / torch.linalg.norm(q_out, dim=-1, keepdim=True)

    return q_out

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def quat_mul_normalized(q1, q2):
    """Hamilton product of two quaternion tensors, with result normalized."""
    return F.normalize(quat_mul(q1, q2), dim=-1)

def quat_rotate(q, v):
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)
