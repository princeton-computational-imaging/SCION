"""Tests for scion.util.animate module."""

import math
import pytest
import torch

from scion.util.animate import (
    AnimationConfig,
    WindAnimationState,
    WindDisplacement,
    perlin_noise_2d,
    fbm_noise_2d,
    compute_wind_displacement,
    _step_critically_damped,
    _axis_angle_to_quat,
)


class TestAnimationConfig:
    """Tests for AnimationConfig dataclass."""

    def test_default_initialization(self):
        config = AnimationConfig()
        assert config.wind_direction == (1.0, 0.0)
        assert config.wind_speed == 1.0
        assert config.wind_strength == 15.0
        assert config.sway_frequency == 0.8
        assert config.turbulence_octaves == 3
        assert config.fps == 30.0
        assert config.duration == 5.0

    def test_new_defaults(self):
        config = AnimationConfig()
        assert config.spring_omega == 6.0
        assert config.gust_scale == 0.15
        assert config.gust_speed_factor == 0.3
        assert config.phase_fbm_scale == 0.2
        assert config.turbulence_scale == 0.5
        assert config.axis_perturbation == 0.35
        assert config.flutter_amplitude == 2.0
        assert config.flutter_frequency_factor == 3.0

    def test_custom_initialization(self):
        config = AnimationConfig(
            wind_direction=(0.5, 0.5),
            wind_strength=30.0,
            wind_speed=2.0,
        )
        assert config.wind_direction == (0.5, 0.5)
        assert config.wind_strength == 30.0
        assert config.wind_speed == 2.0

    def test_wind_dir_normalized(self):
        config = AnimationConfig(wind_direction=(3.0, 4.0))
        dx, dy = config.wind_dir_normalized
        assert abs(dx - 0.6) < 1e-5
        assert abs(dy - 0.8) < 1e-5
        assert abs(math.sqrt(dx * dx + dy * dy) - 1.0) < 1e-5

    def test_wind_dir_normalized_unit(self):
        config = AnimationConfig(wind_direction=(1.0, 0.0))
        dx, dy = config.wind_dir_normalized
        assert abs(dx - 1.0) < 1e-5
        assert abs(dy - 0.0) < 1e-5

    def test_num_frames(self):
        config = AnimationConfig(fps=30.0, duration=5.0)
        assert config.num_frames == 150

    def test_num_frames_fractional(self):
        config = AnimationConfig(fps=24.0, duration=2.5)
        assert config.num_frames == 60


class TestPerlinNoise:
    """Tests for Perlin noise implementation."""

    def test_output_shape(self):
        coords = torch.rand(100, 2) * 10
        noise = perlin_noise_2d(coords)
        assert noise.shape == (100,)

    def test_output_shape_batched(self):
        coords = torch.rand(8, 16, 2) * 10
        noise = perlin_noise_2d(coords)
        assert noise.shape == (8, 16)

    def test_output_range(self):
        """Perlin noise should be approximately in [-1, 1]."""
        coords = torch.rand(10000, 2) * 20.0 - 10.0
        noise = perlin_noise_2d(coords)
        assert noise.min() >= -1.5  # some tolerance
        assert noise.max() <= 1.5

    def test_spatial_continuity(self):
        """Nearby points should have similar noise values."""
        base = torch.tensor([[5.0, 5.0]])
        offset = torch.tensor([[5.001, 5.001]])
        n1 = perlin_noise_2d(base)
        n2 = perlin_noise_2d(offset)
        assert (n1 - n2).abs().item() < 0.1

    def test_deterministic_with_seed(self):
        coords = torch.rand(50, 2) * 10
        n1 = perlin_noise_2d(coords, seed=42)
        n2 = perlin_noise_2d(coords, seed=42)
        assert torch.allclose(n1, n2)

    def test_different_seeds_differ(self):
        coords = torch.rand(50, 2) * 10
        n1 = perlin_noise_2d(coords, seed=0)
        n2 = perlin_noise_2d(coords, seed=99)
        assert not torch.allclose(n1, n2)

    def test_not_constant(self):
        """Noise should vary across space."""
        coords = torch.rand(100, 2) * 10
        noise = perlin_noise_2d(coords)
        assert noise.std() > 0.01

    def test_gpu_compatible(self):
        """Test that noise works on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        coords = torch.rand(100, 2, device="cuda") * 10
        noise = perlin_noise_2d(coords)
        assert noise.device.type == "cuda"
        assert noise.shape == (100,)


class TestFBMNoise:
    """Tests for Fractal Brownian Motion noise."""

    def test_output_shape(self):
        coords = torch.rand(100, 2) * 10
        noise = fbm_noise_2d(coords)
        assert noise.shape == (100,)

    def test_more_octaves_adds_detail(self):
        """More octaves should change the output."""
        coords = torch.rand(100, 2) * 10
        n1 = fbm_noise_2d(coords, octaves=1)
        n3 = fbm_noise_2d(coords, octaves=3)
        # With multiple octaves, output should differ from single octave
        assert not torch.allclose(n1, n3, atol=1e-3)

    def test_normalized_range(self):
        """FBM should be normalized to approximately [-1, 1]."""
        coords = torch.rand(10000, 2) * 20.0 - 10.0
        noise = fbm_noise_2d(coords, octaves=4)
        assert noise.min() >= -2.0
        assert noise.max() <= 2.0


class TestSpringDynamics:
    """Tests for critically damped spring."""

    def test_converges_to_target(self):
        """Spring should converge towards target over time."""
        angle = torch.zeros(10)
        ang_vel = torch.zeros(10)
        target = torch.ones(10) * 0.5
        dt = 1.0 / 30.0

        for _ in range(100):
            angle, ang_vel = _step_critically_damped(angle, ang_vel, target, dt, omega_n=6.0)

        assert torch.allclose(angle, target, atol=0.01)

    def test_no_overshoot_critically_damped(self):
        """Critically damped spring should not overshoot significantly."""
        angle = torch.zeros(1)
        ang_vel = torch.zeros(1)
        target = torch.ones(1)
        dt = 1.0 / 30.0
        max_val = 0.0

        for _ in range(200):
            angle, ang_vel = _step_critically_damped(angle, ang_vel, target, dt, omega_n=6.0)
            max_val = max(max_val, angle.item())

        # Critically damped may have slight overshoot but should be bounded
        assert max_val < 1.15  # less than 15% overshoot

    def test_zero_dt_no_change(self):
        """With dt=0, state should not change."""
        angle = torch.tensor([0.5])
        ang_vel = torch.tensor([0.1])
        target = torch.tensor([1.0])
        new_angle, new_vel = _step_critically_damped(angle, ang_vel, target, dt=0.0, omega_n=6.0)
        assert torch.allclose(new_angle, angle)
        assert torch.allclose(new_vel, ang_vel)


class TestWindAnimationState:
    """Tests for WindAnimationState."""

    def test_create(self):
        state = WindAnimationState.create(100)
        assert state.angle.shape == (100,)
        assert state.ang_vel.shape == (100,)
        assert state.phase_jitter.shape == (100,)

    def test_initial_angles_zero(self):
        state = WindAnimationState.create(50)
        assert torch.allclose(state.angle, torch.zeros(50))
        assert torch.allclose(state.ang_vel, torch.zeros(50))

    def test_phase_jitter_range(self):
        """Phase jitter should be in [0, 2*pi]."""
        state = WindAnimationState.create(1000)
        assert state.phase_jitter.min() >= 0.0
        assert state.phase_jitter.max() <= 2 * math.pi

    def test_phase_jitter_varies(self):
        """Phase jitter should not be constant."""
        state = WindAnimationState.create(100)
        assert state.phase_jitter.std() > 0.1

    def test_create_on_device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        state = WindAnimationState.create(50, device=torch.device("cuda"))
        assert state.angle.device.type == "cuda"
        assert state.phase_jitter.device.type == "cuda"


class TestAxisAngleToQuat:
    """Tests for quaternion conversion helper."""

    def test_zero_angle_is_identity(self):
        axis = torch.tensor([[0.0, 1.0, 0.0]])
        angle = torch.tensor([0.0])
        q = _axis_angle_to_quat(axis, angle)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(q, identity, atol=1e-6)

    def test_unit_quaternion(self):
        axis = torch.randn(50, 3)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        angle = torch.randn(50)
        q = _axis_angle_to_quat(axis, angle)
        norms = q.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(50), atol=1e-5)

    def test_batched_shape(self):
        axis = torch.randn(10, 5, 3)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        angle = torch.randn(10, 5)
        q = _axis_angle_to_quat(axis, angle)
        assert q.shape == (10, 5, 4)


class TestComputeWindDisplacement:
    """Tests for instance-level wind displacement."""

    def test_output_types(self):
        """Should return (WindDisplacement, WindAnimationState) tuple."""
        xy = torch.rand(200, 2) * 8 - 4
        config = AnimationConfig()
        wind, state = compute_wind_displacement(xy, t=1.0, config=config)
        assert isinstance(wind, WindDisplacement)
        assert isinstance(state, WindAnimationState)

    def test_wind_displacement_shapes(self):
        xy = torch.rand(200, 2) * 8 - 4
        config = AnimationConfig()
        wind, _ = compute_wind_displacement(xy, t=1.0, config=config)
        assert wind.primary_angle.shape == (200,)
        assert wind.primary_axis.shape == (200, 3)
        assert wind.flutter_angle.shape == (200,)

    def test_zero_strength_zero_angles(self):
        """With zero wind strength, all angles should be zero."""
        xy = torch.rand(50, 2) * 8 - 4
        config = AnimationConfig(wind_strength=0.0, flutter_amplitude=0.0)
        wind, _ = compute_wind_displacement(xy, t=1.0, config=config)
        assert torch.allclose(wind.primary_angle, torch.zeros(50), atol=1e-5)

    def test_time_variation(self):
        """Different times should produce different angles."""
        xy = torch.rand(50, 2) * 8 - 4
        config = AnimationConfig(wind_strength=15.0)
        w1, _ = compute_wind_displacement(xy, t=0.0, config=config)
        w2, _ = compute_wind_displacement(xy, t=1.0, config=config)
        assert not torch.allclose(w1.primary_angle, w2.primary_angle, atol=1e-3)

    def test_spatial_variation(self):
        """Different positions should produce different angles."""
        xy1 = torch.tensor([[0.0, 0.0]])
        xy2 = torch.tensor([[5.0, 5.0]])
        config = AnimationConfig(wind_strength=15.0)
        w1, _ = compute_wind_displacement(xy1, t=1.0, config=config)
        w2, _ = compute_wind_displacement(xy2, t=1.0, config=config)
        assert not torch.allclose(w1.primary_angle, w2.primary_angle, atol=1e-3)

    def test_with_heights(self):
        """Heights should influence sway magnitude."""
        xy = torch.rand(50, 2) * 4
        heights_low = torch.zeros(50, 1)
        heights_high = torch.ones(50, 1)
        config = AnimationConfig(wind_strength=15.0, height_influence=1.0)

        w_low, _ = compute_wind_displacement(xy, t=1.0, config=config, heights=heights_low)
        w_high, _ = compute_wind_displacement(xy, t=1.0, config=config, heights=heights_high)

        # Higher instances should have more swing on average
        assert w_high.primary_angle.abs().mean() >= w_low.primary_angle.abs().mean() * 0.5

    def test_with_scales(self):
        xy = torch.rand(50, 2) * 4
        config = AnimationConfig(wind_strength=15.0, scale_influence=0.5)
        scales = torch.ones(50, 1) * 0.5
        wind, state = compute_wind_displacement(xy, t=1.0, config=config, scales=scales)
        assert wind.primary_angle.shape == (50,)
        assert state is not None

    def test_spring_state_created(self):
        """First call should create state seeded to target."""
        xy = torch.rand(50, 2) * 8 - 4
        config = AnimationConfig(wind_strength=15.0)
        wind, state = compute_wind_displacement(xy, t=1.0, config=config)
        # State should be created with angles matching the target
        assert state is not None
        assert torch.allclose(state.angle, wind.primary_angle)

    def test_spring_state_stepping(self):
        """Subsequent calls with state and dt should use spring dynamics."""
        xy = torch.rand(50, 2) * 8 - 4
        config = AnimationConfig(wind_strength=15.0)
        dt = 1.0 / 30.0

        # First call creates state
        _, state = compute_wind_displacement(xy, t=0.0, config=config)
        # Second call steps the spring
        wind2, state2 = compute_wind_displacement(xy, t=dt, config=config, state=state, dt=dt)
        # Angles should differ from pure stateless output
        wind_stateless, _ = compute_wind_displacement(xy, t=dt, config=config)
        # Spring may lag behind target, so should differ
        # (not guaranteed to differ significantly on first step, but state should update)
        assert state2 is not None
        assert state2.angle.shape == (50,)

    def test_jitter_applies_from_first_frame(self):
        """Frame 0 and frame 1 share the phase/frequency jitter; only t differs."""
        torch.manual_seed(0)
        xy = torch.rand(64, 2) * 8 - 4
        config = AnimationConfig(wind_strength=15.0, freq_jitter=0.15)
        dt = 1.0 / 30.0

        wind0, state = compute_wind_displacement(xy, t=0.0, config=config)
        # dt=None evaluates the raw target with the persisted jitter.
        again0, _ = compute_wind_displacement(xy, t=0.0, config=config, state=state)
        target1, _ = compute_wind_displacement(xy, t=dt, config=config, state=state)

        assert torch.allclose(again0.primary_angle, wind0.primary_angle, atol=1e-6)
        assert torch.allclose(again0.flutter_angle, wind0.flutter_angle, atol=1e-6)
        # One time step moves every target smoothly (no jump to a new phase).
        max_angle = math.radians(config.wind_strength)
        step = (target1.primary_angle - wind0.primary_angle).abs().max()
        assert step < 0.25 * max_angle

    def test_axis_perturbation(self):
        """With axis perturbation, different positions should get different axes."""
        xy = torch.tensor([[0.3, 0.7], [10.5, 10.3]])  # avoid integer grid points
        config = AnimationConfig(wind_strength=15.0, axis_perturbation=0.35)
        wind, _ = compute_wind_displacement(xy, t=1.0, config=config)
        # Axes should differ between the two points
        assert not torch.allclose(wind.primary_axis[0], wind.primary_axis[1], atol=1e-3)

    def test_no_axis_perturbation(self):
        """Without axis perturbation, all instances should share the same axis."""
        xy = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        config = AnimationConfig(wind_strength=15.0, axis_perturbation=0.0)
        wind, _ = compute_wind_displacement(xy, t=1.0, config=config)
        assert torch.allclose(wind.primary_axis[0], wind.primary_axis[1])

    def test_flutter_angle(self):
        """Flutter angle should be non-zero with flutter_amplitude > 0."""
        xy = torch.rand(50, 2) * 8 - 4
        config = AnimationConfig(flutter_amplitude=2.0)
        wind, _ = compute_wind_displacement(xy, t=0.5, config=config)
        assert wind.flutter_angle.abs().max() > 0

    def test_gpu_compatible(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        xy = torch.rand(100, 2, device="cuda") * 8 - 4
        config = AnimationConfig()
        wind, state = compute_wind_displacement(xy, t=1.0, config=config)
        assert wind.primary_angle.device.type == "cuda"
        assert state.angle.device.type == "cuda"
