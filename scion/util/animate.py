"""Wind animation utilities for codebook Gaussian splatting models.

Wind rotations are applied to whole codebook instances before Gaussian
assembly, producing coherent per-blade sway; a per-splat height-dependent
bend gives the blade curvature.

Key features:
    - Critically damped spring dynamics for temporal lag and rebound
    - Per-splat height-dependent bend with quadratic z-falloff
    - Separate gust (amplitude) and turbulence (phase) fields
    - Per-instance random phase jitter
    - Locally varying bend axis from a noise field
    - Tip flutter for close-range detail
    - Noise-based phase offset
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class AnimationConfig:
    """Configuration for noise-driven wind animation.

    Controls wind direction, strength, turbulence, spring dynamics,
    and per-instance variation for rendering animated sequences.
    """

    # Wind direction and movement
    wind_direction: Tuple[float, float] = (1.0, 0.0)
    """XY wind direction vector (will be normalized)."""

    wind_speed: float = 1.0
    """Speed at which the wind pattern moves through space."""

    wind_strength: float = 15.0
    """Maximum sway angle in degrees."""

    # Oscillation
    sway_frequency: float = 0.8
    """Base sway oscillation frequency in Hz."""

    phase_fbm_scale: float = 0.2
    """XY multiplier before FBM for the base spatial phase field ``phase_offset``."""

    # Turbulence (Perlin noise) - used for phase modulation only
    turbulence_scale: float = 0.5
    """Spatial frequency of noise field. Higher = more spatial variation."""

    turbulence_octaves: int = 3
    """Number of noise octaves for fractal detail."""

    turbulence_persistence: float = 0.5
    """Amplitude decay per octave (0-1). Lower = smoother."""

    turbulence_lacunarity: float = 2.0
    """Frequency multiplier per octave."""

    # Per-instance variation
    height_influence: float = 1.0
    """How much instance height affects sway magnitude (0=none, 1=full)."""

    scale_influence: float = 0.5
    """How much instance scale affects sway magnitude."""

    # Rendering
    fps: float = 30.0
    """Frames per second for output video."""

    duration: float = 5.0
    """Animation duration in seconds."""

    # Spring dynamics
    spring_omega: float = 6.0
    """Natural frequency for critically damped spring. Higher = faster response."""

    # Gust field
    gust_scale: float = 0.15
    """Spatial frequency for slow gust amplitude field."""

    gust_speed_factor: float = 0.3
    """Gust movement speed relative to wind_speed."""

    # Axis perturbation
    axis_perturbation: float = 0.35
    """Per-instance bend axis perturbation weight (0 = global axis)."""

    # Tip flutter
    flutter_amplitude: float = 2.0
    """Tip flutter max angle in degrees."""

    flutter_frequency_factor: float = 3.0
    """Flutter frequency as multiple of base sway_frequency."""

    freq_jitter: float = 0.0
    """Fractional ±jitter on sway_frequency per instance (e.g. 0.15 = ±15%)."""

    @property
    def wind_dir_normalized(self) -> Tuple[float, float]:
        """Return normalized wind direction."""
        dx, dy = self.wind_direction
        mag = math.sqrt(dx * dx + dy * dy) + 1e-8
        return (dx / mag, dy / mag)

    @property
    def num_frames(self) -> int:
        """Total number of frames."""
        return int(self.fps * self.duration)


@dataclass
class WindAnimationState:
    """Persistent per-instance state for spring-based wind animation.

    Stores the spring angle/velocity, random phase jitter and per-instance
    frequency multiplier that persist across frames for temporal coherence.
    """

    angle: torch.Tensor  # [N] primary bend angle (radians)
    ang_vel: torch.Tensor  # [N] primary angular velocity
    phase_jitter: torch.Tensor  # [N] random phase offset per instance
    freq_scale: torch.Tensor  # [N] sway frequency multiplier (1 ± freq_jitter)

    @classmethod
    def create(
        cls,
        N: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        freq_jitter: float = 0.0,
    ) -> "WindAnimationState":
        freq_scale = 1.0 + (torch.rand(N, device=device, dtype=dtype) * 2 - 1) * freq_jitter
        return cls(
            angle=torch.zeros(N, device=device, dtype=dtype),
            ang_vel=torch.zeros(N, device=device, dtype=dtype),
            phase_jitter=torch.rand(N, device=device, dtype=dtype) * 2 * math.pi,
            freq_scale=freq_scale,
        )


@dataclass
class WindDisplacement:
    """Per-instance wind displacement data for per-splat Gaussian assembly.

    Contains angles and axes that are applied per-splat with height-dependent
    scaling in the Gaussian assembly step.
    """

    primary_angle: torch.Tensor  # [N] primary bend angle in radians
    primary_axis: torch.Tensor  # [N, 3] per-instance primary bend axis
    flutter_angle: torch.Tensor  # [N] tip flutter angle in radians


# ---------------------------------------------------------------------------
# Perlin noise
# ---------------------------------------------------------------------------


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    """Quintic smoothstep for Perlin interpolation: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _hash_2d(ix: torch.Tensor, iy: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Hash 2D integer grid coordinates to pseudo-random gradient angles.

    Uses sinusoidal hashing (standard in GPU shaders) for simplicity and
    full GPU compatibility without lookup tables.
    """
    p = (ix.float() + seed * 17.0) * 127.1 + iy.float() * 311.7
    return torch.frac(torch.sin(p) * 43758.5453) * 2.0 * math.pi


def perlin_noise_2d(coords: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Compute 2D Perlin noise.

    Args:
        coords: [..., 2] tensor of (x, y) positions.
        seed: Integer seed for reproducible noise patterns.

    Returns:
        [...] tensor of noise values in approximately [-1, 1].
    """
    x0 = coords[..., 0].floor().long()
    y0 = coords[..., 1].floor().long()
    x1 = x0 + 1
    y1 = y0 + 1

    fx = coords[..., 0] - x0.float()
    fy = coords[..., 1] - y0.float()

    u = _smoothstep(fx)
    v = _smoothstep(fy)

    # Gradient angles at four corners
    a00 = _hash_2d(x0, y0, seed)
    a10 = _hash_2d(x1, y0, seed)
    a01 = _hash_2d(x0, y1, seed)
    a11 = _hash_2d(x1, y1, seed)

    # Gradient dot products
    g00 = torch.cos(a00) * fx + torch.sin(a00) * fy
    g10 = torch.cos(a10) * (fx - 1) + torch.sin(a10) * fy
    g01 = torch.cos(a01) * fx + torch.sin(a01) * (fy - 1)
    g11 = torch.cos(a11) * (fx - 1) + torch.sin(a11) * (fy - 1)

    # Bilinear interpolation
    nx0 = g00 + u * (g10 - g00)
    nx1 = g01 + u * (g11 - g01)
    return nx0 + v * (nx1 - nx0)


def fbm_noise_2d(
    coords: torch.Tensor,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> torch.Tensor:
    """Fractal Brownian Motion using layered Perlin noise.

    Combines multiple octaves of Perlin noise at increasing frequencies
    and decreasing amplitudes for natural-looking turbulence.

    Args:
        coords: [..., 2] tensor of positions.
        octaves: Number of noise layers.
        persistence: Amplitude decay per octave (0-1).
        lacunarity: Frequency multiplier per octave.
        seed: Base seed (offset per octave).

    Returns:
        [...] tensor of noise values, normalized to approximately [-1, 1].
    """
    value = torch.zeros(coords.shape[:-1], device=coords.device, dtype=coords.dtype)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for i in range(octaves):
        value = value + amplitude * perlin_noise_2d(coords * frequency, seed=seed + i * 31)
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return value / max_amplitude


# ---------------------------------------------------------------------------
# Spring dynamics
# ---------------------------------------------------------------------------


def _step_critically_damped(
    angle: torch.Tensor,
    ang_vel: torch.Tensor,
    target: torch.Tensor,
    dt: float,
    omega_n: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Step a critically damped spring (zeta=1) towards target angle.

    Semi-implicit Euler integration. Critically damped means fastest
    convergence without oscillation, producing elastic rebound when the
    target changes.

    Args:
        angle: [N] current angle.
        ang_vel: [N] current angular velocity.
        target: [N] target angle to spring towards.
        dt: Time step in seconds.
        omega_n: Natural frequency (higher = faster response).

    Returns:
        Tuple of (new_angle, new_ang_vel).
    """
    error = angle - target
    ang_vel = ang_vel - (2.0 * omega_n * ang_vel + omega_n * omega_n * error) * dt
    angle = angle + ang_vel * dt
    return angle, ang_vel


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def _axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to unit quaternion.

    Args:
        axis: [..., 3] unit axis vectors.
        angle: [...] angles in radians.

    Returns:
        [..., 4] quaternions (w, x, y, z).
    """
    half = angle * 0.5
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)
    return torch.stack(
        [
            cos_half,
            axis[..., 0] * sin_half,
            axis[..., 1] * sin_half,
            axis[..., 2] * sin_half,
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Instance-level wind animation (CodebookModel)
# ---------------------------------------------------------------------------


def compute_wind_displacement(
    xy: torch.Tensor,
    t: float,
    config: AnimationConfig,
    heights: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    state: Optional[WindAnimationState] = None,
    dt: Optional[float] = None,
) -> Tuple[WindDisplacement, Optional[WindAnimationState]]:
    """Compute per-instance wind displacement for codebook instances.

    Combines noise-based phase, gust amplitude modulation, locally varying
    bend axes, tip flutter, and optionally critically damped spring dynamics
    for temporal coherence.

    Args:
        xy: [N, 2] instance XY positions.
        t: Time in seconds.
        config: Animation parameters.
        heights: [N] or [N, 1] instance heights (optional).
        scales: [N] or [N, 1] instance log-scales (optional).
        state: Persistent spring state from previous frame (optional).
            If None, creates a new state (fresh random phase/frequency
            jitter, used for this frame too) seeded to the current target.
            Pass the returned state back on later frames.
        dt: Time step between frames in seconds (required for spring dynamics).

    Returns:
        Tuple of (WindDisplacement, WindAnimationState).
    """
    N = xy.shape[0]
    device = xy.device
    wind_dx, wind_dy = config.wind_dir_normalized

    # noise-based phase offset
    phase_offset = fbm_noise_2d(xy * config.phase_fbm_scale, octaves=2, seed=300) * 2 * math.pi

    # per-instance random phase jitter. Create the state up front so
    # the jitter (and frequency scale) apply from the first frame on.
    new_state = state
    if new_state is None:
        new_state = WindAnimationState.create(
            N, device, xy.dtype, freq_jitter=config.freq_jitter,
        )
    phase_offset = phase_offset + new_state.phase_jitter

    # slow gust field controls amplitude envelope
    gust_xy = xy * config.gust_scale
    gust_xy = gust_xy.clone()
    gust_xy[..., 0] = gust_xy[..., 0] + wind_dx * t * config.wind_speed * config.gust_speed_factor
    gust_xy[..., 1] = gust_xy[..., 1] + wind_dy * t * config.wind_speed * config.gust_speed_factor
    gust = fbm_noise_2d(gust_xy, octaves=2, seed=100)
    gust_amplitude = 0.3 + 0.7 * (gust * 0.5 + 0.5).clamp(0, 1)  # [0.3, 1.0]

    # fast turbulence field for phase modulation only
    turb_xy = xy.clone() * config.turbulence_scale
    turb_xy[..., 0] = turb_xy[..., 0] + wind_dx * t * config.wind_speed
    turb_xy[..., 1] = turb_xy[..., 1] + wind_dy * t * config.wind_speed
    turbulence = fbm_noise_2d(
        turb_xy,
        octaves=config.turbulence_octaves,
        persistence=config.turbulence_persistence,
        lacunarity=config.turbulence_lacunarity,
    )

    # Height influence: taller instances sway more
    height_factor = torch.ones(N, device=device)
    if heights is not None and config.height_influence > 0:
        h = heights.squeeze(-1) if heights.dim() > 1 else heights
        h_range = h.max() - h.min()
        if h_range > 1e-6:
            h_norm = (h - h.min()) / (h_range + 1e-8)
            height_factor = (1.0 - config.height_influence) + config.height_influence * h_norm
        height_factor = 0.3 + 0.7 * height_factor  # clamp to [0.3, 1.0]

    # Scale influence: larger instances sway more
    scale_factor = torch.ones(N, device=device)
    if scales is not None and config.scale_influence > 0:
        s = scales.squeeze(-1) if scales.dim() > 1 else scales
        scale_factor = 1.0 + config.scale_influence * (torch.exp(s).clamp(max=3.0) - 1.0)

    max_angle_rad = math.radians(config.wind_strength)

    freq = config.sway_frequency * new_state.freq_scale  # [N]; 1.0 without jitter

    # Primary sway: sinusoidal modulated by turbulence phase
    primary_sway = torch.sin(
        2.0 * math.pi * freq * t + phase_offset + turbulence * math.pi
    )
    primary_target = max_angle_rad * primary_sway * gust_amplitude * height_factor * scale_factor

    # critically damped spring dynamics
    if state is None:
        # Seed spring to current target (no initial jerk on first frame)
        new_state.angle = primary_target.clone()
        primary_angle = primary_target
    elif dt is not None and dt > 0:
        new_state.angle, new_state.ang_vel = _step_critically_damped(
            new_state.angle, new_state.ang_vel, primary_target, dt, config.spring_omega
        )
        primary_angle = new_state.angle
    else:
        # No spring (stateless mode)
        primary_angle = primary_target

    # locally varying bend axis from low-freq noise field
    base_primary = torch.tensor([-wind_dy, wind_dx, 0.0], device=device, dtype=xy.dtype)

    if config.axis_perturbation > 0:
        axis_noise_x = fbm_noise_2d(xy * 0.1, octaves=2, seed=200)
        axis_noise_y = fbm_noise_2d(xy * 0.1, octaves=2, seed=201)
        w = config.axis_perturbation
        primary_axis = torch.stack(
            [
                base_primary[0] + w * axis_noise_x,
                base_primary[1] + w * axis_noise_y,
                torch.zeros(N, device=device, dtype=xy.dtype),
            ],
            dim=-1,
        )
        primary_axis = primary_axis / (primary_axis.norm(dim=-1, keepdim=True) + 1e-8)
    else:
        primary_axis = base_primary.unsqueeze(0).expand(N, -1)

    # tip flutter (small, high-freq, applied with z^1.5 in assembly)
    flutter_sway = torch.sin(
        2.0 * math.pi * freq * config.flutter_frequency_factor * t
        + phase_offset * 1.7
        + turbulence * math.pi * 0.5
    )
    flutter_angle = math.radians(config.flutter_amplitude) * flutter_sway

    wind = WindDisplacement(
        primary_angle=primary_angle,
        primary_axis=primary_axis,
        flutter_angle=flutter_angle,
    )

    return wind, new_state
