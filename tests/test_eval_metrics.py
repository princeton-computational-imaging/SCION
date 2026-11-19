"""The paper metrics follow 3DGS's metrics.py (checked against HAC++'s own
SSIM/LPIPS on its renders when this was written; see scion.engine.eval)."""
import torch

from scion.engine.eval import ssim


def test_ssim_basics():
    x = torch.rand(1, 3, 32, 40, generator=torch.Generator().manual_seed(0))
    noisy = (x + 0.1 * torch.randn(x.shape, generator=torch.Generator().manual_seed(1))).clamp(0, 1)
    assert torch.isclose(ssim(x, x), torch.tensor(1.0))
    assert 0.5 < float(ssim(noisy, x)) < 0.99
    assert torch.isclose(ssim(noisy, x), ssim(x, noisy))
