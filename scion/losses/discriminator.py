"""Projected (VGG-16) discriminator for adversarial training.

ProjectedDiscriminator: frozen pre-trained VGG-16 backbone with lightweight
learnable projection heads at the feature scales in ``vgg_scales`` (Sauer et al. 2021),
with Adaptive Discriminator Augmentation (Karras et al. 2020). ~86K learnable
parameters with the shipped two scales, while leveraging rich ImageNet features.

References:
    - Karras et al., "Training Generative Adversarial Networks with Limited Data" (StyleGAN2-ADA)
    - Sauer et al., "Projected GANs Converge Faster" (NeurIPS 2021)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleADA(nn.Module):
    """Pure-PyTorch Adaptive Discriminator Augmentation (Karras et al. 2020).

    Replaces kornia's ``AdaptiveDiscriminatorAugmentation`` with a minimal,
    torch.compile-friendly implementation.  Augmentation probability ``p``
    adapts based on discriminator accuracy on real samples: *p* increases when
    D overfits (real_acc > target) and decreases otherwise.

    Augmentations (all differentiable):
        1. Horizontal flip  (p=0.5 within selected samples)
        2. Random 90-degree rotation  (uniform {0, 1, 2, 3})
        3. Brightness jitter  (+/- 0.2)
        4. Contrast jitter  (0.8 - 1.2)
        5. Gaussian noise  (std=0.1)
    """

    def __init__(
        self,
        initial_p: float = 0.0,
        adjustment_speed: float = 0.02,
        max_p: float = 0.6,
        target_real_acc: float = 0.6,
        update_every: int = 4,
        ema_lambda: float = 0.99,
    ) -> None:
        super().__init__()
        self.p: float = initial_p
        self.adjustment_speed: float = adjustment_speed
        self.max_p: float = max_p
        self.target_real_acc: float = target_real_acc
        self.update_every: int = update_every
        self.ema_lambda: float = ema_lambda
        self.real_acc_ema: float = 0.0
        self._update_counter: int = 0

    def update(self, real_acc: float) -> None:
        """Update adaptive probability based on D accuracy on real samples."""
        self._update_counter += 1
        self.real_acc_ema = self.ema_lambda * self.real_acc_ema + (1.0 - self.ema_lambda) * real_acc
        if self._update_counter % self.update_every == 0:
            if self.real_acc_ema > self.target_real_acc:
                self.p = min(self.p + self.adjustment_speed, self.max_p)
            else:
                self.p = max(self.p - self.adjustment_speed, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply differentiable augmentations with probability ``self.p``.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Augmented tensor, same shape as x.
        """
        if self.p == 0.0:
            return x

        B = x.shape[0]
        device = x.device

        # Per-sample selection mask
        mask = torch.rand(B, device=device) < self.p  # [B]

        if not mask.any():
            return x

        selected = x[mask]  # [S, C, H, W]

        # 1. Horizontal flip (p=0.5)
        flip_mask = torch.rand(selected.shape[0], device=device) < 0.5
        if flip_mask.any():
            selected[flip_mask] = selected[flip_mask].flip(-1)

        # 2. Random 90-degree rotation (uniform {0,1,2,3})
        k = torch.randint(0, 4, (selected.shape[0],), device=device)
        for rot in range(1, 4):
            rot_mask = k == rot
            if rot_mask.any():
                selected[rot_mask] = torch.rot90(selected[rot_mask], rot, [-2, -1])

        # 3. Brightness jitter (+/- 0.2)
        brightness = (torch.rand(selected.shape[0], 1, 1, 1, device=device) - 0.5) * 0.4
        selected = selected + brightness

        # 4. Contrast jitter (0.8 - 1.2)
        contrast = 0.8 + torch.rand(selected.shape[0], 1, 1, 1, device=device) * 0.4
        mean = selected.mean(dim=[-2, -1], keepdim=True)
        selected = mean + (selected - mean) * contrast

        # 5. Gaussian noise (std=0.1)
        selected = selected + torch.randn_like(selected) * 0.1

        out = x.clone()
        out[mask] = selected
        return out


@dataclass
class DiscriminatorConfig:
    """Full discriminator config: architecture + training + ADA.

    Loaded from config/discriminator/*.yaml via Hydra and referenced by
    CodebookModelConfig. The defaults below are fallbacks for programmatic use;
    the shipped values (e.g. ``gan_loss_type: bce_smooth``, ``r1_gamma: 0``,
    ``patch_size: 64``) are in ``config/discriminator/default.yaml``.
    """

    # Master toggle
    enabled: bool = False
    """Enable adversarial training."""

    # Generator-side loss
    loss_weight: float = 0.1
    """Weight applied to the generator adversarial loss."""

    warmup_step: int = 2000
    """Number of training steps before adversarial loss activates."""

    # Discriminator optimizer
    lr: float = 2e-4
    """Discriminator Adam learning rate."""

    beta1: float = 0.0
    """Adam beta1 for discriminator (0.0 recommended for GAN stability)."""

    # Architecture
    use_spectral_norm: bool = False
    """Apply spectral normalization to the learnable projection / head layers."""

    # Training loop
    update_freq: int = 1
    """Run discriminator once every N generator steps."""

    d_steps_per_update: int = 1
    """Inner-loop count: how many D steps to take each time the outer
    update_freq gate fires. Each inner step resamples fresh patches from
    the same (pred, gt) pair so D sees more diverse examples per G step
    without needing a fresh G forward pass."""

    r1_gamma: float = 1.0
    """R1 gradient penalty weight."""

    r1_every: int = 4
    """Apply R1 gradient penalty every N discriminator steps."""

    instance_noise: float = 0.1
    """Std of Gaussian noise added to D inputs (decays linearly to 0)."""

    # Patch sampling
    patch_size: int = 128
    """Size of random crops at native resolution from the masked region."""

    num_patches: int = 64
    """Number of random patches per discriminator step."""

    # Adaptive Discriminator Augmentation (Karras et al. 2020)
    ada_target: float = 0.6
    """Target D(real) sign accuracy. Higher = more augmentation."""

    ada_speed: float = 0.02
    """Step size for adjusting augmentation probability p."""

    ada_interval: int = 4
    """Update p every N discriminator steps."""

    ada_max_p: float = 0.6
    """Maximum augmentation probability."""

    # GAN loss type — hinge collapses the discriminator to a near-constant
    # function under R1 regularization (see commit log). BCE with label
    # smoothing is non-saturating and keeps a meaningful margin.
    gan_loss_type: str = "hinge"
    """One of "hinge" | "bce" | "bce_smooth"."""

    label_smoothing: float = 0.1
    """Smoothing applied to BCE targets when gan_loss_type == "bce_smooth"
    (real -> 1-eps, fake -> eps). Ignored for other loss types."""

    # Which VGG feature scales
    # (relu1_2, relu2_2, relu3_3, relu4_3 -> indices 0..3) the projected
    # discriminator aggregates. Restricting to (0,) keeps D focused on
    # the finest/highest-zoom scale so its task doesn't drift from
    # "coarse-blur detection" (early training) to "high-frequency detail
    # detection" (late training) as the generator sharpens.
    vgg_scales: tuple = (0,)
    """VGG feature scales aggregated by ProjectedDiscriminator."""


# ---------------------------------------------------------------------------
# Projected GAN discriminator (frozen VGG-16 backbone)
# ---------------------------------------------------------------------------


@dataclass
class ProjectedDiscriminatorConfig:
    """Configuration for the Projected GAN discriminator with frozen VGG-16."""

    proj_channels: int = 64
    """Number of channels in each 1x1 projection layer."""

    use_spectral_norm: bool = True
    """Apply spectral normalization to learnable layers."""

    vgg_scales: tuple = (0,)
    """Which VGG scales (indices into [relu1_2, relu2_2, relu3_3, relu4_3])
    are aggregated. Default (0,) = finest/highest-zoom only, so D's task
    doesn't drift between coarse (deep) and fine (shallow) features as the
    generator converges. Pass (0, 1, 2, 3) to restore all four."""

    # Adaptive Discriminator Augmentation
    ada_target: float = 0.6
    """Target D(real) sign accuracy."""

    ada_speed: float = 0.01
    """Step size for adjusting augmentation probability p."""

    ada_interval: int = 4
    """Update p every N discriminator steps."""

    ada_max_p: float = 0.8
    """Maximum augmentation probability."""


class ProjectedDiscriminator(nn.Module):
    """Projected GAN discriminator with frozen pre-trained VGG-16 backbone.

    Uses a frozen VGG-16 to extract multi-scale features (relu1_2 through
    relu4_3), then applies lightweight learnable 1x1 projections and small
    2-layer conv heads at each scale. Final output is per-scale logits resized
    to the coarsest spatial resolution and summed.

    Architecture per scale:
        VGG feature [B, C_i, H_i, W_i]  (frozen)
        -> Conv2d(C_i, proj_ch, 1) + LeakyReLU  (projection)
        -> Conv2d(proj_ch, proj_ch, 3, pad=1) + LeakyReLU  (head layer 1)
        -> Conv2d(proj_ch, 1, 1)  (head layer 2 -> logits)

    VGG-16 feature channels: [64, 128, 256, 512] at relu1_2/2_2/3_3/4_3.
    Learnable parameters: ~86K with proj_channels=64 and vgg_scales=[0, 1] (backbone is frozen).

    References:
        Sauer et al., "Projected GANs Converge Faster" (NeurIPS 2021)
    """

    _VGG_CHANNELS = [64, 128, 256, 512]

    def __init__(self, config: ProjectedDiscriminatorConfig):
        super().__init__()
        self.config = config

        from scion.losses.vgg import VGGBackboneExtractor

        self.backbone = VGGBackboneExtractor()

        num_scales = len(self._VGG_CHANNELS)
        # Which scales (indices into [relu1_2, relu2_2, relu3_3, relu4_3])
        # the discriminator actually aggregates. Default (0,) gives D the
        # shallowest/finest-detail features only — freezes its "zoom level"
        # so G can't escape by shifting differences to a scale D ignores.
        self.active_scales = tuple(config.vgg_scales)
        for s in self.active_scales:
            if not (0 <= s < num_scales):
                raise ValueError(
                    f"vgg_scales index {s} out of range [0, {num_scales})"
                )
        use_sn = config.use_spectral_norm
        proj_ch = config.proj_channels

        def _sn(module: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(module) if use_sn else module

        # Per-scale 1x1 projectors: C_i -> proj_channels
        self.projectors = nn.ModuleList()
        for c_in in self._VGG_CHANNELS:
            self.projectors.append(nn.Sequential(
                _sn(nn.Conv2d(c_in, proj_ch, kernel_size=1)),
                nn.LeakyReLU(0.2),
            ))

        # Per-scale discriminator heads: proj_channels -> 1
        self.heads = nn.ModuleList()
        for _ in range(num_scales):
            self.heads.append(nn.Sequential(
                _sn(nn.Conv2d(proj_ch, proj_ch, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2),
                _sn(nn.Conv2d(proj_ch, 1, kernel_size=1)),
            ))

        # Adaptive Discriminator Augmentation
        self.ada = SimpleADA(
            initial_p=0.0,
            adjustment_speed=config.ada_speed,
            max_p=config.ada_max_p,
            target_real_acc=config.ada_target,
            update_every=config.ada_interval,
        )

    def augment(self, x: torch.Tensor, real_acc: float | None = None) -> torch.Tensor:
        """Apply ADA augmentation on raw RGB patches (before backbone).

        Args:
            x: Input tensor [B, 3, H, W].
            real_acc: D's accuracy on real samples for adaptive p update.

        Returns:
            Augmented tensor, same shape as x.
        """
        if real_acc is not None:
            self.ada.update(real_acc)
        return self.ada(x)

    @property
    def ada_p(self) -> float:
        """Current ADA augmentation probability."""
        return self.ada.p

    def trainable_parameters(self):
        """Return only learnable parameters (projectors + heads, not backbone)."""
        return itertools.chain(self.projectors.parameters(), self.heads.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: backbone → project → head per scale → sum logits.

        Args:
            x: Input image tensor [B, 3, H, W], values in [0, 1].

        Returns:
            Summed logits [B, 1, H_min, W_min] where H_min, W_min are the
            spatial dimensions of the coarsest VGG feature map.
        """
        feats = self.backbone(x)  # list of [B, C_i, H_i, W_i]

        logits_list = []
        for i, (feat, proj, head) in enumerate(
            zip(feats, self.projectors, self.heads)
        ):
            if i not in self.active_scales:
                continue
            projected = proj(feat)    # [B, proj_ch, H_i, W_i]
            logit = head(projected)   # [B, 1, H_i, W_i]
            logits_list.append(logit)

        # Resize all to coarsest spatial resolution and sum
        min_h = min(l.shape[2] for l in logits_list)
        min_w = min(l.shape[3] for l in logits_list)

        summed = torch.zeros(
            x.shape[0], 1, min_h, min_w, device=x.device, dtype=x.dtype,
        )
        for logit in logits_list:
            if logit.shape[2] != min_h or logit.shape[3] != min_w:
                logit = F.interpolate(
                    logit, size=(min_h, min_w), mode="bilinear", align_corners=False,
                )
            summed = summed + logit

        return summed


# ---------------------------------------------------------------------------
# Hinge loss functions
# ---------------------------------------------------------------------------

def generator_hinge_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss: G wants D to classify fake as real (positive).

    Args:
        d_fake: Discriminator output on generated images [B, 1, H', W'].

    Returns:
        Scalar loss. Minimizing this pushes d_fake toward positive values.
    """
    return -d_fake.mean()


def discriminator_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss: D wants positive for real, negative for fake.

    Args:
        d_real: Discriminator output on real images [B, 1, H', W'].
        d_fake: Discriminator output on generated images [B, 1, H', W'].

    Returns:
        Scalar loss. D is penalized if d_real < 1 or d_fake > -1.
    """
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def generator_bce_loss(d_fake: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Non-saturating BCE generator loss: G pushes sigmoid(d_fake) toward 1-eps.

    Unlike hinge, BCE has non-vanishing gradient even when D confidently
    classifies fakes — prevents the saturation modes that pin d_fake_mean.
    """
    target = torch.full_like(d_fake, 1.0 - eps)
    return F.binary_cross_entropy_with_logits(d_fake, target)


def discriminator_bce_loss(
    d_real: torch.Tensor, d_fake: torch.Tensor, eps: float = 0.0
) -> torch.Tensor:
    """BCE discriminator loss with optional label smoothing.

    Targets: real -> 1-eps, fake -> eps. eps=0.1 is the standard smoothing
    strength; eps=0 recovers plain BCE (brittle under overconfidence).
    """
    t_real = torch.full_like(d_real, 1.0 - eps)
    t_fake = torch.full_like(d_fake, eps)
    return (
        F.binary_cross_entropy_with_logits(d_real, t_real)
        + F.binary_cross_entropy_with_logits(d_fake, t_fake)
    )
