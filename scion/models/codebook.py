"""CodebookModel: Flat codebook with hard integer assignments.

Plain ``nn.Module``. K codebook entries, each with variable S_k Gaussians.
N instances map to codebook entries via hard integer assignments.

Storage:
  - Codebook: Pre-allocated flat [T_max, dims] tensors.
  - splat_candidate_id[T_max]: long buffer mapping each splat row to its
    candidate index (-1 = free/unused).
  - free_list[T_max] + free_count: LIFO stack of available row indices.
  - Instances: [N, dims] tensors with position/rotation/scale/opacity
  - Assignments: [N] long buffer mapping instances to candidates

Pre-allocated buffers avoid autograd shape-mismatch crashes when density
control adds or frees splat rows. Free rows are never read during assembly.

Every parameter is trained by gradient descent; density control (MH splat
births, instance clone / subdivide / cull) runs through the pluggable
strategies in ``scion.strategies`` (docs/codebook_architecture.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from scion.camera import viewmat_from_c2w
from scion.rasterizer import rasterize as _rasterize
from scion.engine.image_loss import (
    SSIMLoss,
    composite_with_background as _composite_with_background,
    gt_image_to_float as _gt_image_to_float,
)
from scion.models.codebook_base import (
    CodebookMixin, GaussianBundle, SPLAT_FREE,
)
from scion.losses.discriminator import DiscriminatorConfig
from scion.util.transformation import random_quat_upper_cone
from scion.util.clustering import kmeans, kmeans_farthest_point
from scion.losses.codebook_losses import compactness_loss
from scion.ops import (
    codebook_compression_ratio,
    codebook_description_length,
)


@dataclass
class CodebookModelConfig:
    """Configuration for CodebookModel."""

    # ------------------------------------------------------------------
    # Rendering, seed init and image-loss settings.
    # ------------------------------------------------------------------

    # Background / SH
    background_color: str = "random"  # "random" | "white" | "black"
    sh_degree: int = 3
    sh_degree_interval: int = 1000

    # Init: seed point cloud, or ``num_random`` uniform points when random_init.
    random_init: bool = False
    num_random: int = 50000
    random_scale: float = 10.0

    # Loss / regularization.
    ssim_lambda: float = 0.2
    use_scale_regularization: bool = True
    max_gauss_ratio: float = 10.0

    # Rasterization / output.
    rasterize_mode: str = "classic"  # "classic" | "antialiased"
    output_depth_during_training: bool = False

    # Auto-downscale schedule used by ``_downscale_if_required``.
    num_downscales: int = 2
    resolution_schedule: int = 3000

    # ------------------------------------------------------------------
    # Codebook-specific fields below.
    # ------------------------------------------------------------------

    # Strategy selection
    splat_strategy: str = "mh"          # "mh" | "none" | "pkg.module:Class"
    instance_strategy: str = "default"  # "default" | "none" | "pkg.module:Class"

    # Codebook structure
    num_candidates: int = 1024  # K: number of codebook entries (initial).
    max_candidates: int = 4096  # Sizes the splat buffer (T_max >= max_candidates * max_splats_per_candidate).
    
    initial_splats_per_candidate: int = 4  # Initial S_k: number of Gaussians per entry at startup (uniform).
    max_splats_per_candidate: int = 256  # Maximum S_k per candidate after growth.
    
    num_instances: float = 100000  # N initial instances. -1 = full sparse PC; (0,1] = fraction of sparse PC; >1 = absolute cap (still bounded by PC size).
    max_num_instances: int = 200000  # Hard cap on instance count during growth.

    # Splat initialization (per-candidate local coordinates)
    splat_init_center: list = field(default_factory=lambda: [0.5, 0.5, 0.0])  # Offset before scaling (0.5 = centered, 0.0 = one-sided).
    splat_init_spread: list = field(default_factory=lambda: [0.1, 0.1, 0.1])  # Spatial extent of initial splat positions around local origin.
    splat_init_scale: list = field(default_factory=lambda: [2.0, 2.0, 2.0])  # Scale multiplier for kNN-based initial splat scales.
    candidate_cone_angle: float = 10.0  # Cone angle (degrees) for random initial splat quaternions.

    # Instance initialization
    cluster_color_weight: float = 2.0  # Color weight for position+color k-means init (higher = more color-dominant).

    # Direct DINO init: the oracle is built by projecting the initial
    # instance positions into the training views and sampling the per-image
    # DINO feature maps bilinearly.
    dino_direct_model: str = "dinov3_vitl16"
    dino_direct_resolution: int = 832
    dino_direct_pca_dim: int = 64
    dino_direct_distance_alpha: float = 0.0  # 0 = uniform per-view; >0 = exp(-α·dist) falloff.
    dino_direct_scene_dir: Optional[str] = None  # Scene dir (must be set by caller; config/method/codebook.yaml: ${data}).

    # Splat strategy timing
    splat_refine_start: int = 1050  # Step to start splat relocate/add.
    splat_refine_every: int = 100   # Steps between splat relocate/add operations.
    splat_refine_stop: int = 25000  # Step to stop splat relocate/add.
    splat_min_opacity: float = 0.05  # Sigmoid-space threshold for dead splat relocation.

    # Instance strategy timing
    instance_refine_start: int = 1050  # Step to start instance-level moves.
    instance_refine_every: int = 100   # Steps between instance-level moves.
    instance_refine_stop: int = 25000  # Step to stop instance-level moves.

    # MHSplatStrategy: relocate + MH birth proposals (per MH-3DGS paper)
    # Timing uses shared splat_refine_start/every/stop (overloaded per strategy YAML).
    splat_birth_accept_threshold: float = 0.5  # Importance centre for sigmoid acceptance.

    # Importance fusion weights (paper: 0.8·z(opa) + 0.5·z(SSIM/grad) + 0.5·z(L1)).
    mh_splat_w_opa: float = 0.8
    mh_splat_w_grad: float = 0.5
    mh_splat_w_l1: float = 0.5
    # Coarse + fine proposal scales (anneal max→min over the densification window).
    mh_splat_coarse_scale_max: float = 10.0
    mh_splat_coarse_scale_min: float = 5.0
    mh_splat_fine_scale_max: float = 2.0
    mh_splat_fine_scale_min: float = 1.0
    # Per-candidate batch sizes (anneal full→0 over the window).
    mh_splat_coarse_batch_per_cand: int = 2
    mh_splat_fine_batch_per_cand: int = 6
    # Voxel-density penalty: voxel_factor = 1 / (1 + voxel_alpha · V).
    mh_splat_voxel_alpha: float = 5.0
    mh_splat_voxel_size_max: float = 0.05
    mh_splat_voxel_size_min: float = 0.005
    mh_splat_use_voxel_penalty: bool = True
    # Per-parent density damping: std *= 1 / (1 + log(1 + denom)).
    mh_splat_use_density_factor: bool = True

    # DefaultInstanceStrategy: gradient-based instance clone/cull
    default_inst_grow_grad: float = 0.001  # Per-instance gradient threshold for cloning.
    default_inst_grow_scale: float = 0.5  # Max scale for clone eligibility.
    default_inst_enable_clone: bool = True  # Gate small-instance cloning.
    default_inst_enable_subdivide: bool = True  # Gate large-instance subdivision.
    default_inst_prune_opa: float = 0.01  # Sigmoid-opacity threshold for culling.
    default_inst_subdivide_M: int = 2  # Children per parent for large-scale subdivide (2..8).
    default_inst_subdivide_max_parents: int = 16  # Per-firing cap on subdivide parents — prevents geometric N growth across consecutive refinements.
    default_inst_subdivide_noise_scale: float = 0.2  # Position jitter σ = noise_scale * exp(scale). 1.0 places children many scene-extents away for large-scale parents; 0.2 keeps them inside the parent ellipsoid.
    default_inst_clone_max_batch: int = 512  # Per-firing cap on clone count — prevents N doubling in one refinement.
    default_inst_clone_noise_scale: float = 0.1  # Multiplier on per-instance scale for clone displacement noise (clone_instances_by_mask).
    default_inst_max_scale_log: float = 4.0  # Instances with log-scale above this are skipped for grow ops (their exp(scale) would push children far outside the scene).
    # Logit-space offset subtracted from each cloned instance's opacity so
    # co-located clones don't α-compose to a higher joint α than the parent
    # alone (the "over-bright round table" failure mode when clones cluster
    # on compact objects).  ``0`` copies the parent opacity verbatim;
    # ``log(2) ≈ 0.693`` halves σ so two overlapping clones approximately
    # recover the parent's α for σ in [0.3, 0.7], matching the splat-split
    # path's 2-child α correction.  Bump toward 1.0+ for sharper damping on
    # scenes with many high-opacity instances.
    default_inst_clone_opa_offset: float = 0.0

    # Loss
    lpips_lambda: float = 0.0  # Weight for LPIPS perceptual loss.
    splat_opacity_reg: float = 0.1  # Weight for splat opacity sparsity regularization.
    splat_scale_reg: float = 0.2  # Weight for splat scale regularization.
    compactness_reg_weight: float = 2e-4  # Weight for within-candidate splat compactness regularization.

    # Instance-level regularization
    instance_opacity_reg: float = 0.02  # Weight for instance opacity sparsity regularization.
    instance_scale_reg: float = 1.0  # Weight for instance scale regularization (pushes toward unit scale).
    instance_min_opacity: float = 0.1  # Sigmoid-space opacity below which an instance is considered dead.

    # Instance scale *tail* penalty.  The mean-based ``instance_scale_reg``
    # above is dominated by the healthy bulk (mean(|log_scale|) ≈ 0.004 in a
    # typical run) and cannot see a handful of blow-up instances whose
    # log_scale grows to +4 (exp ≈ 55× init).  This relu-squared penalty
    # fires only when an instance's log_scale exceeds
    # ``instance_scale_log_cap``.  SUMMED (not averaged) so the per-blob
    # contribution scales with how many instances cross the cap and how
    # far — ``.mean()`` over N=38k would dilute a few blob instances to
    # 1e-4 of total loss and have no bite.
    instance_scale_log_cap: float = 1.0     # exp(1.0) ≈ 2.7× init linear scale
    instance_scale_cap_weight: float = 0.1  # one blob at excess=3 ⇒ ~0.9 loss (≈ main_loss)

    # Splat-opacity *tail* penalty.  ExpG2 showed the blob mechanism just
    # shifted to saturated splats (σ(opa) → 0.988) once instance scale was
    # capped — the mean-based ``splat_opacity_reg`` can't see a handful
    # of over-opaque splats drowning the rest.  relu-squared above
    # ``splat_opa_logit_cap``, SUMMED (same rationale as instance cap).
    #
    # **Tightened 2026-04-19** — drop-correlation analysis identified
    # ``splat_opa_sig_max`` as the #1 leading indicator of PSNR collapse
    # (rises from 0.15 baseline through 0.5 at cliff−200 to 0.95 at
    # cliff).  Previous cap σ(3.0)=0.953 only bit at saturation, too
    # late.  New cap σ(2.0)=0.881 starts pushing back well before blobs
    # form.  Weight 5× higher for real gradient bite.
    splat_opa_logit_cap: float = 2.0        # σ(2) ≈ 0.881
    splat_opa_cap_weight: float = 0.1      # one saturated splat at excess 3 ⇒ 0.9 loss (≈ main_loss)

    # DINO feature oracle
    dino_lookup_backend: str = "voxel_hash"  # Lookup backend: "brute_force" or "voxel_hash".
    dino_knn_k: int = 1  # kNN neighbours for position-to-feature lookup (1=nearest, >1=IDW).
    dino_voxel_size: float = 0.0  # Voxel size for sparse lookup (0 = auto from scene density).
    dino_voxel_neighbor_radius: int = 1  # Search radius in voxel cells before brute-force fallback.

    # DINO-initialized clustering (init-only — DinoFeatureOracle has no
    # training-time methods; see scion/util/dino_oracle.py).
    init_assignment_mode: str = "dino"  # "dino" | "position_color" | "color" | "random_uniform".
    dino_init_position_weight: float = 0.0  # Position weight in DINO clustering (0 = pure semantic).

    # LPIPS loss schedule
    lpips_every: int = 4  # Compute LPIPS every N steps (1 = every step).
    lpips_warmup: int = 0  # Skip LPIPS until self.step >= lpips_warmup.
    lpips_net: str = "vgg"  # LPIPS backbone: "vgg" or "alex".

    # Adversarial training (see scion/losses/discriminator.py:DiscriminatorConfig)
    # Projected discriminator: frozen VGG-16 backbone + small learned heads.
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)


class CodebookModel(CodebookMixin, nn.Module):
    """Flat codebook model with hard assignments and MH codebook transitions.

    The render pipeline (viewmat extraction, background selection, gsplat
    call, output assembly) lives in :meth:`get_outputs`, calling into
    :func:`scion.rasterizer.rasterize`.

    Uses pre-allocated [T_max, dims] buffers for splat parameters so that
    split/merge operations never resize tensors, avoiding autograd shape
    mismatch errors.
    """

    config: CodebookModelConfig

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: CodebookModelConfig,
        num_train_data: int = 0,
        *,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        train_scene=None,
    ) -> None:
        """Construct the codebook model.

        ``seed_points`` is the ``(xyz, rgb uint8)`` seed cloud. ``train_scene`` (a
        :class:`scion.data.Scene`) supplies the train views for DINO
        init; without it they are loaded from ``dino_direct_scene_dir``.
        """
        nn.Module.__init__(self)
        self.config = config
        self.num_train_data = num_train_data
        self.seed_points = seed_points
        self._train_scene = train_scene
        # Empty parameter whose device is the model's device.
        self.device_indicator_param = nn.Parameter(torch.empty(0))

        self.populate_modules()

    @property
    def device(self) -> torch.device:
        return self.device_indicator_param.device

    # ------------------------------------------------------------------
    # Strategies use _-prefixed names, CodebookMixin un-prefixed ones;
    # these properties bridge them.
    # ------------------------------------------------------------------

    @property
    def _num_candidates(self):
        return self.num_candidates

    @_num_candidates.setter
    def _num_candidates(self, v):
        self.num_candidates = v

    @property
    def _num_instances(self):
        return self.num_instances

    @_num_instances.setter
    def _num_instances(self, v):
        self.num_instances = v

    @property
    def _T_max(self):
        return self.T_max

    @_T_max.setter
    def _T_max(self, v):
        self.T_max = v

    @property
    def _binoms(self):
        return self._binoms_buf

    @_binoms.setter
    def _binoms(self, v):
        self._binoms_buf = v

    @property
    def _instance_size_changed(self):
        return self.instance_size_changed

    @_instance_size_changed.setter
    def _instance_size_changed(self, v):
        self.instance_size_changed = v

    @property
    def _cached_grad_ema(self):
        return self.cached_grad_ema

    @_cached_grad_ema.setter
    def _cached_grad_ema(self, v):
        self.cached_grad_ema = v

    @property
    def _gauss_cache(self):
        return self.gauss_cache

    @_gauss_cache.setter
    def _gauss_cache(self, v):
        self.gauss_cache = v

    @property
    def _gauss_cache_step(self):
        return self.gauss_cache_step

    @_gauss_cache_step.setter
    def _gauss_cache_step(self, v):
        self.gauss_cache_step = v

    def populate_modules(self):
        """Initialize codebook + instances from the seed point cloud."""
        K, N, T_max = self._populate_from_pointcloud()
        self._populate_metrics_and_aux()
        self._populate_losses()
        self._populate_strategies(K, N, T_max)

    def _seed_points_and_colors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Seed positions ``[P, 3]`` and RGB in ``[0, 1]`` ``[P, 3]``.

        The seed point cloud, or ``num_random`` uniform points in a cube of
        side ``random_scale`` when ``random_init`` is set or no seed is given.
        Points without seed colours get random ones.
        """
        cfg = self.config
        use_seed = self.seed_points is not None and not cfg.random_init
        if use_seed:
            means = self.seed_points[0].clone()
        else:
            means = (torch.rand((cfg.num_random, 3)) - 0.5) * cfg.random_scale
        n = means.shape[0]
        # Unused draws; they keep the RNG stream, and so seeded runs,
        # identical to the ones behind the paper's numbers.
        for _ in range(3):
            torch.rand(n)
        has_rgb = self.seed_points is not None and self.seed_points[1].shape[0] > 0
        random_rgb = None if (use_seed and has_rgb) else torch.rand(n, 3)
        rgb = self.seed_points[1] / 255.0 if has_rgb else torch.sigmoid(random_rgb)
        return means, rgb.to(means.device)

    def _populate_metrics_and_aux(self):
        """Set up the PSNR / SSIM metrics and the eval background colour."""
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        cfg = self.config
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIMLoss(data_range=1.0, channel=3)
        # Not read by the model; building it advances the RNG, so it stays to
        # keep seeded runs (discriminator init) identical to the paper's.
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        self.last_size: Tuple[int, int] = (0, 0)
        self.info: Dict = {}


        if cfg.background_color == "random":
            # Fixed eval background for random-background training.
            self.background_color = torch.tensor([0.1490, 0.1647, 0.2157])
        elif cfg.background_color == "white":
            self.background_color = torch.ones(3)
        elif cfg.background_color == "black":
            self.background_color = torch.zeros(3)
        else:
            raise ValueError(
                f"Unknown background color {cfg.background_color!r}"
            )

    # ------------------------------------------------------------------
    # Clustering dispatch
    # ------------------------------------------------------------------

    def _cluster_features(self, features: torch.Tensor, K: int, spherical: bool) -> torch.Tensor:
        """FAISS k-means on per-instance features."""
        return kmeans(features, K, spherical=spherical)

    def _cluster_position_color(self, positions: torch.Tensor, colors: torch.Tensor, K: int) -> torch.Tensor:
        """k-means on position + weighted colour."""
        return kmeans_farthest_point(
            positions, colors, K,
            color_weight=self.config.cluster_color_weight,
        )

    def _cluster_color(self, colors: torch.Tensor, K: int) -> torch.Tensor:
        """Cluster raw point-cloud RGB only, using Euclidean FAISS k-means."""
        return self._cluster_features(colors, K, spherical=False)

    # ------------------------------------------------------------------
    # DINO oracle construction (init-time)
    # ------------------------------------------------------------------

    def _build_dino_oracle_direct(
        self, positions: torch.Tensor, dev,
    ) -> "DinoFeatureOracle":
        from scion.util.dino_oracle import DinoFeatureOracle

        scene_dir = self.config.dino_direct_scene_dir
        if scene_dir is None:
            raise ValueError(
                "Direct DINO init requires model.dino_direct_scene_dir "
                "to locate the scene directory."
            )
        oracle_device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else str(dev)
        )
        return DinoFeatureOracle.from_training_views(
            positions.detach(),
            scene_dir=scene_dir,
            dino_model=self.config.dino_direct_model,
            resolution=self.config.dino_direct_resolution,
            pca_dim=self.config.dino_direct_pca_dim,
            distance_alpha=self.config.dino_direct_distance_alpha,
            backend=self.config.dino_lookup_backend,
            k=self.config.dino_knn_k,
            device=oracle_device,
            voxel_size=self.config.dino_voxel_size,
            voxel_neighbor_radius=self.config.dino_voxel_neighbor_radius,
            scene=self._train_scene,
        )

    def _ensure_dino_oracle_for_positions(
        self, positions: torch.Tensor, dev,
    ) -> None:
        """Build the direct oracle now that ``positions`` is available.

        No-op when the oracle was already built.
        """
        if self.dino_oracle is not None:
            return
        self.dino_oracle = self._build_dino_oracle_direct(positions, dev)

    # ------------------------------------------------------------------
    # populate_modules sub-methods
    # ------------------------------------------------------------------

    def _populate_from_pointcloud(self) -> Tuple[int, int, int]:
        """Initialize instances + codebook from the seed point cloud.

        Returns (K, N, T_max).
        """
        K = self.config.num_candidates
        S = self.config.initial_splats_per_candidate
        max_K = self.config.max_candidates
        max_S = self.config.max_splats_per_candidate
        T_active = K * S
        # Buffer must accommodate splat growth (MCMC) + candidate splits.
        # Use max_candidates * max_splats_per_candidate as the ceiling.
        T_max = max(int(T_active * 1.5) + 1024, max_K * max_S)

        init_means, init_colors_rgb = self._seed_points_and_colors()
        dev = init_means.device

        N_pts = init_means.shape[0]
        # num_instances semantics:
        #   < 0     -> use the full sparse point cloud, uncapped
        #   (0, 1]  -> fraction of N_pts
        #   > 1     -> absolute target count; subsampled if below N_pts,
        #              upsampled (with replacement) if above N_pts
        ni = self.config.num_instances
        if ni < 0:
            N = N_pts
        elif 0 < ni <= 1:
            N = max(1, int(round(ni * N_pts)))
        else:
            N = int(ni)

        # Construction is deferred until after instance positions are known
        # so we can project them straight into the training views.
        self.dino_oracle = None

        # Point-cloud subsampling / upsampling initialization
        if N_pts > N:
            idx = torch.randperm(N_pts)[:N]
        elif N > N_pts:
            # num_instances asked for more instances than the sparse point
            # cloud has points. Upsample with replacement (every point used
            # at least once, extras sampled uniformly) rather than silently
            # capping at N_pts.
            extra = torch.randint(0, N_pts, (N - N_pts,))
            idx = torch.cat([torch.arange(N_pts), extra])
        else:
            idx = torch.arange(N_pts)
            N = N_pts

        assignment_mode = self.config.init_assignment_mode
        valid_assignment_modes = {"dino", "position_color", "color", "random_uniform"}
        if assignment_mode not in valid_assignment_modes:
            raise ValueError(
                f"Unknown init_assignment_mode={assignment_mode!r}; expected one of "
                f"{sorted(valid_assignment_modes)}"
            )

        if assignment_mode == "dino":
            self._ensure_dino_oracle_for_positions(init_means[idx], dev)
            if self.dino_oracle is None:
                raise RuntimeError(
                    "init_assignment_mode='dino' requested, but the DINO oracle "
                    "could not be constructed"
                )

        if assignment_mode == "dino":
            dino_feats = self.dino_oracle.lookup(init_means[idx])
            features = F.normalize(dino_feats, dim=-1)
            if self.config.dino_init_position_weight > 0:
                pos_scaled = init_means[idx] * self.config.dino_init_position_weight
                features = F.normalize(torch.cat([features, pos_scaled], dim=-1), dim=-1)
            assignments = self._cluster_features(features, K, spherical=True)
            print(
                f"[CodebookModel] DINO-initialized clustering (kmeans): "
                f"{K} clusters from {dino_feats.shape[1]}D features"
            )
        elif assignment_mode == "position_color":
            assignments = self._cluster_position_color(init_means[idx], init_colors_rgb[idx], K)
        elif assignment_mode == "color":
            assignments = self._cluster_color(init_colors_rgb[idx], K)
            print(
                f"[CodebookModel] Point-cloud RGB clustering "
                f"(kmeans): {K} clusters"
            )
        else:
            # IID uniform assignments are the intended no-structure control:
            # unlike a balanced permutation, this does not inject an implicit
            # equal-cluster-size prior. machine.seed makes it reproducible.
            assignments = torch.randint(0, K, (N,), device=dev)
            print(
                f"[CodebookModel] Uniform-random initial assignments: "
                f"{N} instances across {K} candidates"
            )
        # Clamp before logit: saturated channels (rgb = 0 or 255) would land
        # at ±inf. Averaged into per-cluster means and passed through sigmoid,
        # they would lock candidates at RGB-cube corners for the rest of
        # training (sigmoid gradient ≈ 0 at saturation).
        instance_colors = torch.logit(
            init_colors_rgb[idx].clamp(1e-5, 1 - 1e-5)
        )

        self.register_parameter(
            "instance_positions", Parameter(init_means[idx].clone())
        )
        self.register_parameter(
            "instance_rotations",
            Parameter(
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
                .unsqueeze(0)
                .expand(N, -1)
                .contiguous()
            ),
        )

        # Common instance parameters
        self.register_parameter("instance_scales", Parameter(torch.zeros(N, 1, device=dev)))
        self.register_parameter(
            "instance_opacities",
            Parameter(torch.logit(torch.full((N, 1), 0.5, device=dev))),
        )
        self.register_buffer("instance_assignments", assignments.to(dev))

        # Pre-allocate flat codebook buffers [T_max, dims]
        splat_means = torch.zeros(T_max, 3)
        splat_scales = torch.zeros(T_max, 3)
        splat_quats = torch.zeros(T_max, 4)
        splat_quats[:, 0] = 1.0  # identity quaternion
        splat_opacities = torch.logit(torch.full((T_max, 1), 0.1))
        splat_features_dc = torch.zeros(T_max, 3)

        center = torch.tensor(self.config.splat_init_center, dtype=torch.float32)
        spread = torch.tensor(self.config.splat_init_spread, dtype=torch.float32)
        splat_init_scale = torch.tensor(self.config.splat_init_scale, dtype=torch.float32)

        splat_means[:T_active] = (torch.rand(T_active, 3) - center) * spread
        splat_quats[:T_active] = random_quat_upper_cone(
            T_active, cone_angle_degrees=self.config.candidate_cone_angle
        )
        # kNN-based scale initialization (GPU)
        pts = splat_means[:T_active].detach().reshape(-1, 3).to(dev)
        dists_all = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)
        dists_all.fill_diagonal_(float("inf"))
        knn_dists, _ = dists_all.topk(3, dim=-1, largest=False)
        avg_dist = knn_dists.mean(dim=-1, keepdim=True).cpu()
        splat_scales[:T_active] = torch.log(
            splat_init_scale * avg_dist.repeat(1, 3)
        )

        # Per-cluster colors: fully vectorized
        C = instance_colors.shape[-1]
        asgn_dev = assignments.to(dev)
        colors_dev = instance_colors.to(dev)
        cluster_sums = torch.zeros(K, C, device=dev)
        cluster_counts = torch.zeros(K, device=dev)
        cluster_sums.scatter_add_(0, asgn_dev.unsqueeze(-1).expand(-1, C), colors_dev)
        cluster_counts.scatter_add_(0, asgn_dev, torch.ones(asgn_dev.shape[0], device=dev))
        alive = cluster_counts > 0
        cluster_means = torch.zeros(K, C, device=dev)
        cluster_means[alive] = cluster_sums[alive] / cluster_counts[alive].unsqueeze(-1)
        base_colors = cluster_means.unsqueeze(1).expand(K, S, C).reshape(K * S, C).cpu()
        splat_features_dc[:T_active] = base_colors + 0.1 * torch.randn(T_active, C)

        self.register_parameter("splat_means", Parameter(splat_means))
        self.register_parameter("splat_scales", Parameter(splat_scales))
        self.register_parameter("splat_quats", Parameter(splat_quats))
        self.register_parameter("splat_opacities", Parameter(splat_opacities))
        self.register_parameter("splat_features_dc", Parameter(splat_features_dc))

        # Build splat_candidate_id: first T_active rows mapped to candidates
        splat_candidate_id = torch.full((T_max,), SPLAT_FREE, dtype=torch.long)
        for k in range(K):
            splat_candidate_id[k * S : (k + 1) * S] = k
        self.register_buffer("splat_candidate_id", splat_candidate_id)

        # Free-list: rows from T_active to T_max-1
        free_list = torch.zeros(T_max, dtype=torch.long)
        free_list[:T_max - T_active] = torch.arange(T_active, T_max)
        self.register_buffer("free_list", free_list)
        self.free_count = T_max - T_active

        return K, N, T_max

    def _populate_losses(self):
        """Initialize LPIPS perceptual loss and adversarial discriminator."""
        self.lpips_loss = lpips.LPIPS(net=self.config.lpips_net)

        self.discriminator = None
        self._d_optimizer = None
        self._d_step_count = 0
        self._adv_metrics: Dict[str, float] = {}
        self._d_last_metrics: Dict[str, float] = {}
        if self.config.discriminator.enabled:
            from scion.losses.discriminator import ProjectedDiscriminator, ProjectedDiscriminatorConfig
            d_cfg = self.config.discriminator
            d_config = ProjectedDiscriminatorConfig(
                use_spectral_norm=d_cfg.use_spectral_norm,
                ada_target=d_cfg.ada_target,
                ada_speed=d_cfg.ada_speed,
                ada_interval=d_cfg.ada_interval,
                ada_max_p=d_cfg.ada_max_p,
                vgg_scales=tuple(d_cfg.vgg_scales),
            )
            self.discriminator = ProjectedDiscriminator(d_config)
            self._d_optimizer = torch.optim.Adam(
                self.discriminator.trainable_parameters(),
                lr=self.config.discriminator.lr,
                betas=(self.config.discriminator.beta1, 0.999),
            )
            # torch.compile(fullgraph=True) hits dynamo's recompile limit once
            # the R1-every and ADA branches produce enough graph variants, and
            # can trigger inplace-version autograd errors between D(real) and
            # D(fake). The discriminator is tiny; compile gain is marginal.
            self._d_forward = self.discriminator.forward
        else:
            self._d_forward = None

    def _populate_strategies(self, K: int, N: int, T_max: int):
        """Initialize CodebookMixin, densification strategies, and assembly cache."""
        # Initialize CodebookMixin (register_buffers=False: splat_candidate_id,
        # free_list, and instance_assignments already created by
        # _populate_from_pointcloud).
        self.init_codebook(
            spatial_dim=3,
            rot_dim=4,
            volume_exp=1.0 / 3.0,
            p_splat=14,
            p_inst=9,
            K=K,
            N=N,
            T_max=T_max,
            register_buffers=False,
        )

        # Codebook densification strategy (configurable via strategy fields)
        from scion.strategies import (
            MHSplatStrategy,
            DefaultInstanceStrategy,
            CompositeStrategy,
        )
        from scion.strategies.base import strategy_class
        # "none" is a sentinel producing None — CompositeStrategy skips None
        # slots, giving a true no-op for that level (used as a null baseline).
        _none_sentinel = lambda: None  # noqa: E731
        _splat_strategies = {
            "mh": MHSplatStrategy,
            "none": _none_sentinel,
        }
        _instance_strategies = {
            "default": DefaultInstanceStrategy,
            "none": _none_sentinel,
        }

        def _make(registry, name, key):
            # "pkg.mod:Class" loads a user strategy (docs/extending_strategies.md).
            if name not in registry and ":" in name:
                return strategy_class(name, key)()
            if name not in registry:
                raise ValueError(f"{key}={name!r}: expected one of {sorted(registry)} or 'pkg.mod:Class'")
            return registry[name]()

        self.codebook_strategy = CompositeStrategy(
            splat=_make(_splat_strategies, self.config.splat_strategy, "splat_strategy"),
            instance=_make(_instance_strategies, self.config.instance_strategy, "instance_strategy"),
        )

        # Internal state
        self._assembly_global_idx: Optional[torch.Tensor] = None
        self._assembly_inst_idx: Optional[torch.Tensor] = None
        self._last_assignment_per_gaussian: Optional[torch.Tensor] = None
        # Scratch state populated by ``load_state_dict`` when a compact
        # checkpoint is detected; applied after ``nn.Module.load_state_dict``.
        self._pending_compact_load: Optional[Dict[str, int]] = None

        # Precompute assembly index tensors (must be after init_codebook + offsets)
        self.recompute_assembly_indices()

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    def _apply(self, fn):
        """Recompute cached assembly index tensors after .to(device).

        _cached_inst_idx and _cached_global_idx are plain attributes (not
        registered buffers) so they don't auto-move with the model.  After
        super()._apply moves the registered buffers, recompute so indices
        land on the correct device.  Also moves the DINO oracle if present.
        """
        result = super()._apply(fn)
        if self.instance_assignments.numel() > 0:
            self.recompute_assembly_indices()
        if self.dino_oracle is not None:
            new_device = str(self.instance_assignments.device)
            self.dino_oracle.to(new_device)
        return result

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _get_assembled(self) -> GaussianBundle:
        """Get assembled world-space gaussians, cached per step."""
        if self._gauss_cache is not None and self._gauss_cache_step == self.step:
            return self._gauss_cache
        bundle = self._assemble_gaussians()
        self._gauss_cache = bundle
        self._gauss_cache_step = self.step
        return bundle

    def _assemble_gaussians(self) -> GaussianBundle:
        """Codebook lookup + instance transforms -> world-space GaussianBundle.

        Delegates to CodebookMixin.assemble_gaussians() for the core transform,
        caches index maps for strategy gradient reverse-mapping, and wraps
        the result in a GaussianBundle.
        """
        d = self.assemble_gaussians()

        # Cache index maps for strategy gradient reverse-mapping
        self._assembly_inst_idx = self._cached_inst_idx
        self._assembly_global_idx = self._cached_global_idx
        self._last_assignment_per_gaussian = self.instance_assignments[self._cached_inst_idx]

        return GaussianBundle(
            means=d["means"],
            scales=d["scales"],
            quats=d["rotations"],
            opacities=d["opacities"],
            features_dc=d["features_dc"],
            features_rest=torch.zeros(d["means"].shape[0], 0, 3, device=d["means"].device),
        )

    # ------------------------------------------------------------------
    # World-space Gaussian views
    # ------------------------------------------------------------------

    @property
    def num_points(self):
        """Total world-space gaussians = sum of S_k over all instances."""
        return self._cached_total

    @property
    def means(self):
        return self._get_assembled().means

    @property
    def scales(self):
        return self._get_assembled().scales

    @property
    def quats(self):
        return self._get_assembled().quats

    @property
    def opacities(self):
        return self._get_assembled().opacities

    @property
    def features_dc(self):
        return self._get_assembled().features_dc

    @property
    def features_rest(self):
        return self._get_assembled().features_rest

    # ------------------------------------------------------------------
    # Checkpoint saving / loading  (compact format)
    # ------------------------------------------------------------------
    #
    # CodebookModel pre-allocates flat ``[T_max, dims]`` splat buffers sized
    # to ``max_candidates * max_splats_per_candidate`` (often >100k rows) even
    # when only a few thousand rows are active.  A naive
    # ``torch.save(model.state_dict())`` serialises every unused row,
    # producing ~421 MB checkpoints dominated by zeros / ``SPLAT_FREE``.
    #
    # The compact format strips those unused rows entirely (T_max := T_active)
    # and serialises only the surviving data. Compact state dicts are tagged
    # via the metadata key ``__codebook_meta__.T_max``; they are for
    # inference (no free rows to grow into). Training-state resume saves the
    # uncompacted ``nn.Module.state_dict``, which loads through the same
    # resize-to-match path below.
    #
    # Layout (all keys prefixed by whatever prefix torch's
    # nn.Module.state_dict passes in):
    #   splat_means, splat_scales, splat_quats, splat_opacities,
    #   splat_features_dc            -> shape [T_active, dims]
    #   splat_candidate_id           -> shape [T_active], all entries
    #                                   != SPLAT_FREE (free rows dropped)
    #   free_list                    -> OMITTED (no free rows to track)
    #   instance_*                   -> shape [num_instances, dims] (no change
    #                                   at the data level — these buffers are
    #                                   already sized exactly to N)
    #   __codebook_meta__.T_max / .N_max / .num_candidates /
    #   .num_instances              -> scalars; T_max == T_active.
    # ------------------------------------------------------------------

    _COMPACT_META_PREFIX = "__codebook_meta__."

    # Splat-level Parameters + the single splat-level buffer that share the
    # [T_max, ...] row layout.  Compact save strips down to active rows.
    _SPLAT_ROW_KEYS = (
        "splat_means", "splat_scales", "splat_quats",
        "splat_opacities", "splat_features_dc",
        "splat_candidate_id",
    )

    def state_dict(self, *args, **kwargs):
        """Compact state_dict: strip inactive splat rows + metadata header.

        Matches ``nn.Module.state_dict`` signature so callers pass through
        unchanged, and respects its ``prefix`` argument.
        """
        sd = super().state_dict(*args, **kwargs)

        # Recover the prefix the caller used so we can rewrite keys in place.
        # torch may pass ``prefix`` positionally (arg index 1) or as kw.
        prefix: str = kwargs.get("prefix", "")
        if not prefix and len(args) >= 2 and isinstance(args[1], str):
            prefix = args[1]
        # Second heuristic: every key shares a common prefix in nested
        # contexts — sniff if it matches one of ours.
        if not prefix and sd:
            sample = next(iter(sd))
            for name in self._SPLAT_ROW_KEYS:
                if sample.endswith(name):
                    prefix = sample[: -len(name)]
                    break

        p = prefix  # alias
        cid_key = f"{p}splat_candidate_id"
        if cid_key not in sd:
            # Unexpected (model not fully populated) — return the raw state dict.
            return sd

        cid = sd[cid_key]
        active_mask = cid != SPLAT_FREE
        active_idx = active_mask.nonzero(as_tuple=True)[0].to(torch.long).contiguous()

        # Compact every splat-row tensor to active rows only.
        for name in self._SPLAT_ROW_KEYS:
            key = f"{p}{name}"
            if key in sd:
                sd[key] = sd[key][active_idx].contiguous()

        # Drop free_list entirely — a compact state dict has no free rows.
        fl_key = f"{p}free_list"
        if fl_key in sd:
            del sd[fl_key]

        # Trim instance-level tensors to num_instances (usually already exact).
        N = int(self._num_instances)
        inst_trim_names = [
            "instance_positions", "instance_rotations",
            "instance_scales", "instance_opacities",
            "instance_assignments",
        ]
        for name in inst_trim_names:
            key = f"{p}{name}"
            if key in sd and sd[key].shape[0] > N:
                sd[key] = sd[key][:N].contiguous()

        # N_max: track the nominal cap so future expansion code can
        # pre-allocate to it.  Fall back to current N if config missing.
        N_max = int(getattr(self.config, "max_num_instances", N) or N)

        # Metadata header — single ordered dict entry per field, so
        # ``load_state_dict`` can detect the compact format without scanning.
        # ``T_max`` collapses to T_active: the saved rows ARE the active set.
        T_active = int(active_idx.numel())
        meta = {
            "T_max": torch.tensor(T_active, dtype=torch.long),
            "N_max": torch.tensor(N_max, dtype=torch.long),
            "num_candidates": torch.tensor(int(self._num_candidates), dtype=torch.long),
            "num_instances": torch.tensor(N, dtype=torch.long),
        }
        for name, val in meta.items():
            sd[f"{p}{self._COMPACT_META_PREFIX}{name}"] = val

        return sd

    def _detect_compact_format(self, state_dict) -> bool:
        """Return True iff ``state_dict`` carries compact metadata header."""
        suffix = f"{self._COMPACT_META_PREFIX}T_max"
        return any(k.endswith(suffix) for k in state_dict.keys())

    def _extract_compact_meta(self, state_dict) -> Dict[str, torch.Tensor]:
        """Pop compact metadata keys from *state_dict* and return them.

        Keys are returned unprefixed (``T_max``, ``N_max``, ...).  The
        input dict is mutated in place.
        """
        meta: Dict[str, torch.Tensor] = {}
        prefix_len_suffix = self._COMPACT_META_PREFIX
        matched_prefix = ""
        to_del = []
        for key in list(state_dict.keys()):
            idx = key.find(prefix_len_suffix)
            if idx == -1:
                continue
            matched_prefix = key[:idx]
            field = key[idx + len(prefix_len_suffix):]
            meta[field] = state_dict[key]
            to_del.append(key)
        for key in to_del:
            del state_dict[key]
        meta["__prefix__"] = matched_prefix  # reuse key for prefix stash
        return meta

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True, **kwargs):
        """Load a compact (``state_dict``) or uncompacted state dict.

        Parameters and buffers are resized to the saved shapes before
        ``nn.Module.load_state_dict``. Compact state dicts (detected via the
        ``__codebook_meta__.T_max`` key) also set T_max / N / K from their
        metadata and get an empty free list.
        """
        self.step = self.config.max_num_iterations if hasattr(self.config, "max_num_iterations") else 30000

        # Work on a mutable copy so we can rewrite / strip keys without
        # disturbing the caller's dict (which may be any Mapping).
        state_dict = dict(state_dict)

        is_compact = self._detect_compact_format(state_dict)

        if is_compact:
            meta = self._extract_compact_meta(state_dict)
            prefix = meta.pop("__prefix__", "")
            T_max = int(meta["T_max"].item())
            N = int(meta["num_instances"].item())
            K = int(meta["num_candidates"].item())
            # T_max == T_active: every saved row is active, the free list is empty.
            free_count = 0
            state_dict[f"{prefix}free_list"] = torch.zeros(0, dtype=torch.long)

            # Cache scalar state to apply post-load.
            self._pending_compact_load = {
                "T_max": T_max,
                "N": N,
                "K": K,
                "free_count": free_count,
            }
        else:
            self._pending_compact_load = None

        # --- Resize parameters / buffers to the saved shapes ---
        def _resize_param(key: str) -> None:
            if key not in state_dict:
                return
            new = state_dict[key]
            cur = getattr(self, key)
            if new.shape != cur.shape:
                setattr(self, key, torch.nn.Parameter(
                    torch.zeros_like(new, device=cur.device)
                ))

        for key in ("splat_means", "splat_scales", "splat_quats",
                    "splat_opacities", "splat_features_dc",
                    "instance_positions", "instance_rotations",
                    "instance_scales", "instance_opacities"):
            _resize_param(key)

        for key in ("splat_candidate_id", "free_list", "instance_assignments"):
            if key in state_dict:
                cur = getattr(self, key, None)
                new = state_dict[key]
                if cur is None or cur.shape != new.shape:
                    dev = cur.device if cur is not None else new.device
                    self.register_buffer(
                        key, torch.zeros_like(new, device=dev),
                    )

        # If the state_dict has no frozen-aux keys (DINO oracle / LPIPS /
        # Discriminator / VGG) we're loading a "compact-only" checkpoint —
        # those modules will keep their init state, which is correct because
        # they're deterministic & scene-only (reloaded from disk/torchvision
        # in setup()). Force strict=False so torch doesn't raise on the
        # missing keys.
        aux_markers = ("dino_oracle", "lpips", "loss_fn_vgg",
                       "discriminator", "perceptual", "vgg")
        has_aux = any(any(m in k for m in aux_markers) for k in state_dict)
        if not has_aux:
            strict = False
        torch.nn.Module.load_state_dict(self, state_dict, strict=strict, **kwargs)

        # Scalar / tracked state not stored explicitly in the checkpoint:
        # derive from the loaded buffers (compact path overrides below).
        if "instance_assignments" in state_dict:
            self.num_candidates = int(self.instance_assignments.max().item()) + 1
        if "splat_candidate_id" in state_dict:
            self.free_count = int((self.splat_candidate_id == SPLAT_FREE).sum().item())

        if self._pending_compact_load is not None:
            pend = self._pending_compact_load
            self.T_max = pend["T_max"]
            self.num_instances = pend["N"]
            self.num_candidates = pend["K"]
            self.free_count = pend["free_count"]
            self._last_replaced_T_max = pend["T_max"]
            self._pending_compact_load = None

        # Rebuild forward map + per-splat indices so _get_assembled works.
        self.recompute_assembly_indices()
        # Any previously-cached bundle points to stale pre-load tensors.
        self._gauss_cache = None
        self._gauss_cache_step = -1

        # Assembly-index recompute happens automatically in _apply() after
        # the caller moves the model to its target device.

    # ------------------------------------------------------------------
    # Checkpoint size (W&B telemetry; run.json sizes come from
    # scion.compression.levels.evaluate_levels)
    # ------------------------------------------------------------------

    def size_bytes(self, level: str = "raw") -> int:
        """Byte length of this model's artifact at compression ``level``.

        ``raw`` holds only the learned splat / instance tensors (no frozen
        DINO / LPIPS / discriminator weights); see
        :mod:`scion.compression.levels`.
        """
        from scion.compression.levels import encode

        return len(encode(self.state_dict(), level)[0])

    # ------------------------------------------------------------------
    # Forward / outputs
    # ------------------------------------------------------------------

    def get_outputs(self, camera):
        """Forward pass of a :class:`scion.Camera`.

        Pipeline:
          1. Extract viewmat / K / W / H from the camera (downscaled by the
             resolution schedule while training).
          2. SH or RGB colour path.
          3. Per-frame background colour + ``gsplat`` rasterization.
          4. Output dict (``rgb``, ``depth``, ``accumulation``, ``background``).
        """
        c2w = camera.cam_to_world
        opacities_crop = self.opacities
        means_crop = self.means
        features_dc_crop = self.features_dc
        features_rest_crop = self.features_rest
        scales_crop = self.scales
        quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        # (1) Extract viewmat + K + size
        camera_scale_fac = self._get_downscale_factor()
        device = means_crop.device
        c2w_b = c2w.unsqueeze(0) if c2w.dim() == 2 else c2w
        viewmat = viewmat_from_c2w(c2w_b).to(device)
        scale = 1.0 / float(camera_scale_fac)
        K_mat = camera.K.to(device).clone()
        K_mat[0, 0] *= scale
        K_mat[1, 1] *= scale
        K_mat[0, 2] *= scale
        K_mat[1, 2] *= scale
        K_mat = K_mat.unsqueeze(0)
        # floor, like the GT box filter
        W = int(camera.width * scale)
        H = int(camera.height * scale)
        self.last_size = (H, W)

        if self.config.rasterize_mode not in ("antialiased", "classic"):
            raise ValueError(f"Unknown rasterize_mode: {self.config.rasterize_mode!r}")

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        # (2) SH-aware color path
        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree,
            )
            colors_for_render = colors_crop
        else:
            colors_for_render = torch.sigmoid(colors_crop).squeeze(1)  # [N, 3]
            sh_degree_to_use = None

        # (3) Background + rasterize
        background = self._get_background_color()

        rgb, alpha, depth_im, background, info = _rasterize(
            means=means_crop,
            quats=quats_crop,
            scales_log=scales_crop,
            opacities_logit=opacities_crop,
            colors=colors_for_render,
            viewmat=viewmat,
            K=K_mat,
            width=W,
            height=H,
            background=background,
            sh_degree=sh_degree_to_use,
            rasterize_mode=self.config.rasterize_mode,
            render_mode=render_mode,
            near_plane=0.01,
            far_plane=1e10,
            absgrad=True,
            packed=False,
        )
        self.info = info

        # (4) Output dict — broadcast background to image shape on eval
        # so callers can use it as a per-pixel reference.
        if background.shape[0] == 3 and not self.training:
            background_out = background.expand(H, W, 3)
        else:
            background_out = background

        return {
            "rgb": rgb,
            "depth": depth_im,
            "accumulation": alpha,
            "background": background_out,
        }

    # ------------------------------------------------------------------
    # Resolution schedule / background
    # ------------------------------------------------------------------

    def _get_downscale_factor(self) -> int:
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        return 1

    def _downscale_if_required(self, image: torch.Tensor) -> torch.Tensor:
        d = self._get_downscale_factor()
        if d > 1:
            # Box-filter resize (cv2 "area" equivalent via a uniform d x d
            # convolution with stride d).
            image = image.to(torch.float32)
            weight = (1.0 / (d * d)) * torch.ones(
                (1, 1, d, d), dtype=torch.float32, device=image.device,
            )
            return (
                F.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d)
                .squeeze(1)
                .permute(1, 2, 0)
            )
        return image

    def _get_background_color(self) -> torch.Tensor:
        cfg = self.config
        if cfg.background_color == "random":
            if self.training:
                return torch.rand(3, device=self.device)
            return self.background_color.to(self.device)
        if cfg.background_color == "white":
            return torch.ones(3, device=self.device)
        if cfg.background_color == "black":
            return torch.zeros(3, device=self.device)
        raise ValueError(f"Unknown background color {cfg.background_color!r}")

    def get_gt_img(self, image: torch.Tensor) -> torch.Tensor:
        """uint8/float ground-truth, downscaled to match the render output."""
        image = _gt_image_to_float(image, self.device)
        return self._downscale_if_required(image)

    def composite_with_background(
        self, image: torch.Tensor, background: torch.Tensor,
    ) -> torch.Tensor:
        return _composite_with_background(image, background)

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()

    def forward(self, camera):
        return self.get_outputs(camera)

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Codebook splat + instance parameters, one group each."""
        groups: Dict[str, List[Parameter]] = {
            "splat_means": [self.splat_means],
            "splat_features_dc": [self.splat_features_dc],
            "splat_opacities": [self.splat_opacities],
            "splat_scales": [self.splat_scales],
            "splat_quats": [self.splat_quats],
            "instance_positions": [self.instance_positions],
            "instance_rotations": [self.instance_rotations],
            "instance_scales": [self.instance_scales],
            "instance_opacities": [self.instance_opacities],
        }
        return groups

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_splat_params_dict(self) -> Dict[str, nn.Parameter]:
        """Get flat parameter dict with canonical keys for CodebookMixin."""
        return {
            "means": self.splat_means,
            "scales": self.splat_scales,
            "rotations": self.splat_quats,
            "opacities": self.splat_opacities,
            "features_dc": self.splat_features_dc,
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Training losses: L1 + SSIM + LPIPS + regularizers."""
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        # Optional masking
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(gt_img.device)
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        # L1 + SSIM
        l1_pix = torch.abs(gt_img - pred_img)  # [H, W, 3]
        Ll1 = l1_pix.mean()
        # Per-pixel L1 map for the MH birth importance term (strategies/splat.py).
        if self.training:
            self._last_l1_map = l1_pix.mean(dim=-1).detach()  # [H, W]
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1).unsqueeze(0),
            pred_img.permute(2, 0, 1).unsqueeze(0),
        )
        main_loss = (
            (1 - self.config.ssim_lambda) * Ll1
            + self.config.ssim_lambda * simloss
        )
        loss_dict = {"main_loss": main_loss}

        # LPIPS (lpips package expects [-1,1]; clamp to [0,1] then rescale)
        if (
            self.config.lpips_lambda > 0
            and self.step >= self.config.lpips_warmup
            and self.step % self.config.lpips_every == 0
        ):
            gt_clamped = torch.clamp(gt_img, 0.0, 1.0) * 2 - 1
            pred_clamped = torch.clamp(pred_img, 0.0, 1.0) * 2 - 1
            lpips_val = self.lpips_loss(
                gt_clamped.permute(2, 0, 1)[None, ...],
                pred_clamped.permute(2, 0, 1)[None, ...],
            )
            loss_dict["lpips_loss"] = self.config.lpips_lambda * lpips_val

        # Opacity regularization (active region only)
        active_rows = self.get_active_rows()
        if self.config.splat_opacity_reg > 0:
            loss_dict["opacity_reg"] = (
                self.config.splat_opacity_reg
                * torch.sigmoid(self.splat_opacities[active_rows]).abs().mean()
            )

        # Splat-opacity TAIL penalty — catches blob-driving saturated splats
        # that mean-based ``opacity_reg`` misses.  relu-squared on logit
        # above ``splat_opa_logit_cap``, summed over active splats.
        if self.config.splat_opa_cap_weight > 0 and active_rows.numel() > 0:
            cap = float(self.config.splat_opa_logit_cap)
            excess = (self.splat_opacities[active_rows, 0] - cap).clamp(min=0.0)
            loss_dict["splat_opa_cap_reg"] = (
                self.config.splat_opa_cap_weight * excess.pow(2).sum()
            )

        # Scale regularization: penalize elongated splats (aspect ratio)
        if self.config.use_scale_regularization and self.step % 5 == 0:
            scale_exp = torch.exp(self.splat_scales[active_rows])
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = self.config.splat_scale_reg * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
        loss_dict["scale_reg"] = scale_reg

        # Instance opacity regularization
        if self.config.instance_opacity_reg > 0:
            loss_dict["instance_opacity_reg"] = (
                self.config.instance_opacity_reg
                * torch.sigmoid(self.instance_opacities).abs().mean()
            )

        # Instance scale regularization
        if self.config.instance_scale_reg > 0:
            loss_dict["instance_scale_reg"] = (
                self.config.instance_scale_reg
                * self.instance_scales.abs().mean()
            )

        # Instance scale TAIL penalty — catches the blob-driving outlier
        # instances whose log_scale grows to +4 while the mean-based
        # instance_scale_reg above stays ≈ 0.004 (blind to the tail).
        # relu-squared above ``instance_scale_log_cap``, SUMMED so the
        # per-blob contribution scales with count × severity.
        if self.config.instance_scale_cap_weight > 0:
            cap = float(self.config.instance_scale_log_cap)
            excess = (self.instance_scales - cap).clamp(min=0.0)
            loss_dict["instance_scale_cap_reg"] = (
                self.config.instance_scale_cap_weight * excess.pow(2).sum()
            )

        # Compactness regularization: penalize variance of splat means within each candidate.
        # Use forward-map sorted rows so compactness_loss sees contiguous candidate segments.
        if self.config.compactness_reg_weight > 0:
            T_active = self._cached_T_active
            if T_active > 0:
                sorted_means = self.splat_means[self._fw_sorted_rows]
                closs = compactness_loss(
                    sorted_means,
                    self._fw_offsets,
                    self._num_candidates,
                    T_active=T_active,
                )
            else:
                closs = torch.zeros(1, device=self.device)
            loss_dict["compactness_reg"] = self.config.compactness_reg_weight * closs

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Metrics: PSNR + assignment entropy + codebook stats."""
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        metrics: Dict[str, torch.Tensor] = {}
        metrics["psnr"] = self.psnr(outputs["rgb"], gt_rgb)
        metrics["gaussian_count"] = torch.tensor(float(self.num_points))

        # Assignment entropy: -sum p(k) log p(k)
        K = self._num_candidates
        counts = self.candidate_usage_counts()
        N_total = counts.sum().clamp(min=1).float()
        p = counts.float() / N_total
        p = p.clamp(min=1e-10)
        entropy = -(p * p.log()).sum()
        metrics["assignment_entropy"] = entropy

        metrics["num_candidates"] = torch.tensor(float(K))
        metrics["num_instances"] = torch.tensor(float(self._num_instances))

        sk = self._candidate_sizes.float()
        N = int(counts.sum().item())
        metrics["mean_splats_per_candidate"] = sk.mean() if sk.numel() > 0 else torch.tensor(0.0)

        # --- Scale / opacity percentile diagnostics (blob-collapse analysis) ---
        # Per-splat / per-instance / composite world-scale statistics help
        # attribute scene bloat to the right parameter family:
        #   splat_scale_*   — static codebook template scale (log-space)
        #   inst_scale_*    — per-instance multiplicative scale (log-space)
        #   world_scale_*   — sum of the two, exponentiated; what gsplat sees
        # The p99 / max are the tail behaviour that dominates blob render.
        active_rows = self.get_active_rows() if hasattr(self, "get_active_rows") else None
        if active_rows is not None and active_rows.numel() > 0:
            splat_scl = self.splat_scales[active_rows]
            # Per-axis max log-scale (geometric extent along longest axis).
            splat_scl_max_axis = splat_scl.max(dim=-1).values
            metrics["splat_scale_log_p99"] = torch.quantile(
                splat_scl_max_axis.float(), 0.99,
            )
            metrics["splat_scale_log_max"] = splat_scl_max_axis.max().float()
        if self._num_instances > 0:
            inst_scl = self.instance_scales.data
            inst_scl_max_axis = inst_scl.max(dim=-1).values if inst_scl.dim() > 1 else inst_scl
            metrics["inst_scale_log_p99"] = torch.quantile(
                inst_scl_max_axis.float(), 0.99,
            )
            metrics["inst_scale_log_max"] = inst_scl_max_axis.max().float()
            inst_opa_sig = torch.sigmoid(self.instance_opacities.data[:, 0])
            metrics["inst_opa_sig_p99"] = torch.quantile(inst_opa_sig.float(), 0.99)
            metrics["inst_opa_sig_max"] = inst_opa_sig.max().float()
        # World-scale (what the rasteriser sees) — only safe when the
        # assembly cache is populated.
        gauss = getattr(self, "gauss_cache", None)
        if gauss is not None and getattr(gauss, "scales", None) is not None:
            ws = gauss.scales
            if ws.numel() > 0:
                ws_max_axis = ws.max(dim=-1).values
                metrics["world_scale_log_p99"] = torch.quantile(ws_max_axis.float(), 0.99)
                metrics["world_scale_log_max"] = ws_max_axis.max().float()
        # Active splat opacity percentile (raw splat bank, before instance
        # combination).  Shows blob-forming splats directly.
        if active_rows is not None and active_rows.numel() > 0:
            splat_opa_sig = torch.sigmoid(self.splat_opacities[active_rows, 0])
            metrics["splat_opa_sig_p99"] = torch.quantile(
                splat_opa_sig.float(), 0.99,
            )
            metrics["splat_opa_sig_max"] = splat_opa_sig.max().float()

        # Description length: storage cost of codebook representation
        metrics["description_length"] = codebook_description_length(sk, N)
        # Compression ratio: codebook storage / naive per-instance storage
        metrics["compression_ratio"] = codebook_compression_ratio(
            sk, self.instance_assignments, N,
        )

        strategy_metrics = getattr(self, "_last_strategy_metrics", {})
        for key, value in strategy_metrics.items():
            try:
                metrics[f"strategy/{key}"] = torch.tensor(
                    float(value), device=self.device
                )
            except (TypeError, ValueError):
                continue

        # Gradient-informed metrics (from cached Adam exp_avg)
        grad_ema = getattr(self, "_cached_grad_ema", {})
        if "instance_positions" in grad_ema and grad_ema["instance_positions"].shape[0] == self.instance_assignments.shape[0]:
            from scion.ops import (
                compute_gradient_disagreement,
                compute_instance_urgency,
            )
            pos_ema = grad_ema["instance_positions"]
            scale_ema = grad_ema.get("instance_scales")
            urgency = compute_instance_urgency(pos_ema, scale_ema)
            disagree = compute_gradient_disagreement(
                pos_ema, self.instance_assignments, K,
            )
            metrics["grad_urgency_mean"] = urgency.mean()
            metrics["grad_urgency_p90"] = torch.quantile(urgency, 0.9)
            metrics["grad_disagree_mean"] = disagree.mean()
            metrics["grad_disagree_max"] = disagree.max()

        # Adversarial training metrics
        for k, v in self._adv_metrics.items():
            metrics[k] = torch.tensor(v)
        self._adv_metrics = {}

        return metrics

    # ------------------------------------------------------------------
    # Adversarial training
    # ------------------------------------------------------------------

    def _extract_patches(
        self,
        pred_img: torch.Tensor,
        gt_img: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample random native-resolution patches for the discriminator.

        If mask is provided, patch centers are sampled from masked pixels.
        Otherwise, patches are sampled uniformly from the full image.

        Args:
            pred_img: Predicted image [H, W, 3].
            gt_img:   Ground-truth image [H, W, 3].
            mask:     Optional binary mask [H, W, 1].

        Returns:
            (fake_patches, real_patches) each [N, 3, patch_size, patch_size],
            or (None, None) if the image/region is too small.
        """
        patch_size = self.config.discriminator.patch_size
        num_patches = self.config.discriminator.num_patches
        H, W = pred_img.shape[:2]
        half = patch_size // 2

        if H < patch_size or W < patch_size:
            return None, None

        if mask is not None:
            mask_binary = (mask[..., 0] > 0.5)
            mask_coords = mask_binary.nonzero(as_tuple=False)
            if mask_coords.shape[0] == 0:
                return None, None
            mask_coords[:, 0].clamp_(half, H - patch_size + half)
            mask_coords[:, 1].clamp_(half, W - patch_size + half)
            mask_coords = mask_coords.unique(dim=0)
            if mask_coords.shape[0] == 0:
                return None, None
            indices = torch.randint(0, mask_coords.shape[0], (num_patches,))
            centers = mask_coords[indices]
        else:
            rows = torch.randint(half, H - patch_size + half + 1, (num_patches,))
            cols = torch.randint(half, W - patch_size + half + 1, (num_patches,))
            centers = torch.stack([rows, cols], dim=1)

        # Vectorized patch extraction via advanced indexing (0 GPU syncs).
        r0 = (centers[:, 0] - half).long()  # [N] top-left rows
        c0 = (centers[:, 1] - half).long()  # [N] top-left cols
        dr = torch.arange(patch_size, device=centers.device)  # [P]
        dc = torch.arange(patch_size, device=centers.device)  # [P]
        row_idx = r0[:, None, None] + dr[None, :, None]      # [N, P, P]
        col_idx = c0[:, None, None] + dc[None, None, :]      # [N, P, P]

        fake_patches = pred_img[row_idx, col_idx]  # [N, P, P, C]
        real_patches = gt_img[row_idx, col_idx]     # [N, P, P, C]

        if mask is not None:
            mask_patches = mask[row_idx, col_idx]   # [N, P, P, C]
            fake_patches = fake_patches * mask_patches
            real_patches = real_patches * mask_patches

        return fake_patches.permute(0, 3, 1, 2), real_patches.permute(0, 3, 1, 2)

    def _discriminator_step(
        self,
        real_patches: torch.Tensor,
        fake_patches: torch.Tensor,
        step: int,
    ) -> float:
        """Run one discriminator update: instance noise, ADA, the configured GAN loss
        (``gan_loss_type``), and lazy R1 every ``r1_every`` steps (``r1_gamma`` 0 disables it).

        Both inputs must be detached from the generator graph.

        Args:
            real_patches: [N, 3, H, W], detached.
            fake_patches: [N, 3, H, W], detached.
            step: Current training step.

        Returns:
            D accuracy on real patches.
        """
        from scion.losses.discriminator import (
            discriminator_hinge_loss,
            discriminator_bce_loss,
        )

        self._d_step_count += 1

        # Instance noise (decays linearly to 0 by step 10000)
        noise_std = self.config.discriminator.instance_noise
        if noise_std > 0.0:
            decay = max(0.0, 1.0 - step / 10000.0)
            effective_noise = noise_std * decay
            if effective_noise > 0.0:
                real_patches = real_patches + torch.randn_like(real_patches) * effective_noise
                fake_patches = fake_patches + torch.randn_like(fake_patches) * effective_noise

        # ADA: augment real+fake jointly
        n = real_patches.shape[0]
        both = torch.cat([real_patches, fake_patches], dim=0)
        both = self.discriminator.augment(both)
        real_patches, fake_patches = both[:n], both[n:]

        # R1 lazy penalty check
        apply_r1 = (
            self.config.discriminator.r1_gamma > 0.0
            and (self._d_step_count % self.config.discriminator.r1_every == 0)
        )

        if apply_r1:
            real_patches = real_patches.requires_grad_(True)
            # R1 needs create_graph=True which is incompatible with compiled
            # forward's donated buffers — use eager forward for real patches.
            d_real = self.discriminator(real_patches)
        else:
            d_real = self._d_forward(real_patches)
        d_fake = self._d_forward(fake_patches)

        # ADA: update augmentation probability
        real_acc = (d_real > 0).float().mean().item()
        fake_acc = (d_fake < 0).float().mean().item()
        self.discriminator.ada.update(real_acc)

        # Record D's own view of real/fake for telemetry (mean logits + accs)
        self._d_last_metrics = {
            "d_real_acc": real_acc,
            "d_fake_acc": fake_acc,
            "d_real_mean": d_real.detach().mean().item(),
            "d_fake_mean": d_fake.detach().mean().item(),
            "d_real_minus_fake": (d_real.detach().mean() - d_fake.detach().mean()).item(),
        }

        # GAN loss dispatch (hinge | bce | bce_smooth)
        gan_type = self.config.discriminator.gan_loss_type
        if gan_type == "hinge":
            d_loss = discriminator_hinge_loss(d_real, d_fake)
        else:
            eps = (
                self.config.discriminator.label_smoothing
                if gan_type == "bce_smooth" else 0.0
            )
            d_loss = discriminator_bce_loss(d_real, d_fake, eps)

        # R1 gradient penalty (lazy, with compensation)
        if apply_r1:
            grad_real = torch.autograd.grad(
                outputs=d_real.sum(),
                inputs=real_patches,
                create_graph=True,
            )[0]
            r1_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
            compensation = self.config.discriminator.r1_every / 2.0
            d_loss = d_loss + (self.config.discriminator.r1_gamma / 2.0) * r1_penalty * compensation

        # D optimizer step (isolated from G graph)
        self._d_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        self._d_optimizer.step()

        return real_acc

    def adversarial_step(
        self,
        pred_img: torch.Tensor,
        gt_img: torch.Tensor,
        mask: Optional[torch.Tensor],
        step: int,
    ) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
        """Run one D update and compute G loss (returned for trainer backward).

        The D step uses detached patches with its own optimizer. The G loss is
        computed with frozen D params so that the trainer's backward only
        accumulates gradients for the generator (codebook model).

        Called from the trainer's loss step, outside the compiled model
        forward+loss path.

        Args:
            pred_img: Predicted image [H, W, 3] (with gradients to generator).
            gt_img:   Ground-truth image [H, W, 3].
            mask:     Optional mask [H, W, 1] for patch center selection.
            step:     Current training step.

        Returns:
            Tuple of (metrics_dict, g_loss_tensor). g_loss is None if patches
            could not be extracted.
        """
        from scion.losses.discriminator import (
            generator_hinge_loss,
            generator_bce_loss,
        )

        fake_patches, real_patches = self._extract_patches(pred_img, gt_img, mask)
        if fake_patches is None:
            return {}, None

        # D step(s) — detached inputs isolate from G graph; own optimizer.
        # d_steps_per_update > 1 resamples fresh patches per inner step so
        # D sees more diverse examples per G step (cheap: no extra G forward).
        if step % self.config.discriminator.update_freq == 0:
            n_inner = max(1, self.config.discriminator.d_steps_per_update)
            self._discriminator_step(
                real_patches.detach(), fake_patches.detach(), step,
            )
            for _ in range(n_inner - 1):
                fp_i, rp_i = self._extract_patches(pred_img, gt_img, mask)
                if fp_i is None:
                    break
                self._discriminator_step(
                    rp_i.detach(), fp_i.detach(), step,
                )

        # G step — freeze D so gradients only flow to generator
        for p in self.discriminator.parameters():
            p.requires_grad_(False)

        fake_aug = self.discriminator.augment(fake_patches)
        d_fake_g = self._d_forward(fake_aug)
        gan_type = self.config.discriminator.gan_loss_type
        if gan_type == "hinge":
            adv = generator_hinge_loss(d_fake_g)
        else:
            eps = (
                self.config.discriminator.label_smoothing
                if gan_type == "bce_smooth" else 0.0
            )
            adv = generator_bce_loss(d_fake_g, eps)
        g_loss = self.config.discriminator.loss_weight * adv

        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        metrics = {
            "d_fake": d_fake_g.mean().item(),
            "ada_p": self.discriminator.ada_p,
            "adv_loss": g_loss.item(),
        }
        # Surface the D step's own view of real vs. fake (populated when
        # _discriminator_step fires this step; otherwise reuse the last one).
        metrics.update(getattr(self, "_d_last_metrics", {}))
        return metrics, g_loss

    # ------------------------------------------------------------------
    # Helpers for strategies
    # ------------------------------------------------------------------

    def candidate_usage_counts(self) -> torch.Tensor:
        """Count instances per candidate."""
        return torch.bincount(
            self.instance_assignments, minlength=self._num_candidates
        )
