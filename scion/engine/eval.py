"""Eval-split scoring of a trained CodebookModel: the numbers the paper reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

_LPIPS: dict = {}


def ssim(pred: torch.Tensor, gt: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM as 3DGS's ``utils/loss_utils.ssim`` computes it: Gaussian window,
    zero padding, mean over pixels and channels. ``(B, C, H, W)`` in ``[0, 1]``."""
    g = torch.exp(-((torch.arange(window, device=pred.device) - window // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    c = pred.shape[1]
    w = (g[:, None] @ g[None, :]).expand(c, 1, window, window).contiguous()
    f = lambda x: F.conv2d(x, w, padding=window // 2, groups=c)
    mu_p, mu_g = f(pred), f(gt)
    var_p, var_g, cov = f(pred * pred) - mu_p ** 2, f(gt * gt) - mu_g ** 2, f(pred * gt) - mu_p * mu_g
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_p * mu_g + c1) * (2 * cov + c2))
            / ((mu_p ** 2 + mu_g ** 2 + c1) * (var_p + var_g + c2))).mean()


def image_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """PSNR / SSIM / LPIPS the way 3DGS's ``metrics.py`` scores them:
    :func:`ssim`, and LPIPS-VGG on ``[0, 1]`` inputs (not rescaled to
    ``[-1, 1]``, like 3DGS's lpipsPyTorch call). ``(1, 3, H, W)`` in ``[0, 1]``."""
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    key = str(pred.device)
    if key not in _LPIPS:
        _LPIPS[key] = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(pred.device).eval()
    return {"psnr": float(-10 * torch.log10(F.mse_loss(pred, gt))), "ssim": float(ssim(pred, gt)),
            "lpips": float(_LPIPS[key](pred, gt))}


def eval_views(model, scene, out_dir: Optional[Path], loader=None):
    """Render the eval split: ``(rows, fps)``, one ``{psnr, ssim, lpips}`` row
    per eval view in eval order, and the render-only frames per second.

    The ``eval_fn`` contract of
    :func:`scion.compression.levels.evaluate_levels`, and the numbers
    the paper reports: :func:`image_metrics` (3DGS's convention) on the GT
    composited onto the eval background. With ``out_dir``, writes
    ``eval_img_<NNNN>.png`` (GT | render) per view.
    """
    import time as _time

    import torchvision.utils as vutils

    from scion.data.loader import FullImageLoader

    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    loader = loader or FullImageLoader(scene, split="eval", shuffle=False, device=str(model.device))
    model.eval()
    rows: list[dict] = []
    render_s = 0.0
    with torch.no_grad():
        for i in range(len(loader)):
            batch = loader.next_eval(i)
            _sync()
            t0 = _time.perf_counter()
            outputs = model.get_outputs(batch["camera"])
            _sync()
            render_s += _time.perf_counter() - t0
            gt = model.composite_with_background(
                model.get_gt_img(batch["image"]), outputs["background"]
            )
            pred = outputs["rgb"]
            gt_b = torch.moveaxis(gt, -1, 0)[None]
            pred_b = torch.moveaxis(pred, -1, 0)[None]
            rows.append(image_metrics(pred_b, gt_b))
            if out_dir is not None:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                vutils.save_image(
                    torch.cat([gt, pred], dim=1).permute(2, 0, 1).cpu(),
                    Path(out_dir) / f"eval_img_{i:04d}.png",
                )
    return rows, len(rows) / render_s
