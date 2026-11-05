"""Image-quality metrics with the conventions of the 3DGS evaluation protocol.

PSNR on [0, 1] images, SSIM with an 11x11 Gaussian window (sigma 1.5), and
LPIPS with the VGG backbone, as reported by 3DGS, Mip-NeRF 360 and 3DGS-MCMC.
"""

import torch
import torch.nn.functional as F


def psnr(pred, gt):
    """PSNR of [H, W, 3] images in [0, 1]."""
    return (-10 * torch.log10(F.mse_loss(pred, gt))).item()


def _gaussian_1d(size=11, sigma=1.5, device='cpu'):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def ssim_map(pred, gt):
    """Per-pixel SSIM of [B, 3, H, W] images in [0, 1]; returns [B, 3, H, W].

    The 11x11 Gaussian window is separable; all five local moments are filtered in
    one grouped convolution pair with zero padding, identical to the 3DGS 2D window.
    """
    g = _gaussian_1d(device=pred.device)
    # permuted [H, W, 3] inputs would make cat() return channels-last memory, which slows depthwise conv
    x = torch.cat([pred, gt, pred * pred, gt * gt, pred * gt], 1).contiguous(memory_format=torch.contiguous_format)  # [B, 15, H, W]
    x = F.conv2d(x, g.view(1, 1, 1, 11).expand(15, 1, 1, 11), padding=(0, 5), groups=15)
    x = F.conv2d(x, g.view(1, 1, 11, 1).expand(15, 1, 11, 1), padding=(5, 0), groups=15)
    mu_x, mu_y, exx, eyy, exy = x.chunk(5, 1)
    sxx, syy, sxy = exx - mu_x ** 2, eyy - mu_y ** 2, exy - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


def ssim(pred, gt):
    """SSIM of [H, W, 3] images in [0, 1]."""
    return ssim_map(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).mean().item()


class LPIPS:
    """LPIPS-VGG on [H, W, 3] images in [0, 1]."""

    def __init__(self, device):
        import lpips
        self.model = lpips.LPIPS(net='vgg', verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, pred, gt):
        to_nchw = lambda x: x.permute(2, 0, 1)[None] * 2 - 1
        return self.model(to_nchw(pred), to_nchw(gt)).item()
