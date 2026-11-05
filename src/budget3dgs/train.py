"""Budgeted 3D Gaussian splatting training and evaluation on gsplat.

All methods share initialization, optimizer, schedules, losses and evaluation;
they differ only in the density-control strategy:

* ``mcmc``         gsplat's native 3DGS-MCMC strategy with ``cap_max = budget``.
* ``rvr``          residual-variance relocation, the Budget-3DGS strategy.
* ``abl-opacity``  our strategy with MCMC's opacity targets (implementation control).
* ``abl-error``    attributed L1 residual targets instead of residual variance.
* ``abl-recycle``  RVR plus reclaiming the lowest-visible-mass Gaussians.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import MCMCStrategy, rasterization, spherical_harmonics
from PIL import Image
from scipy.spatial import cKDTree

from .data import load_colmap_scene
from .export import save_ply
from .metrics import LPIPS, psnr, ssim, ssim_map
from .strategy import BudgetStrategy

METHODS = ('mcmc', 'rvr', 'abl-opacity', 'abl-error', 'abl-recycle')
SH_C0 = 0.28209479177387814


# ── Config ───────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────
    data: str = 'data/tandt/truck'
    images_dir: str = 'images'
    downsample: int = 1
    test_every: int = 8
    output: str = 'runs/truck/rvr-b100000-s0'

    # ── Method ────────────────────────────────────────
    method: str = 'rvr'
    budget: int = 100_000
    recycle: float = 0.01            # used by abl-recycle only

    # ── Optimization (3DGS-MCMC defaults of gsplat) ───
    steps: int = 30_000
    sh_degree: int = 3
    init_opacity: float = 0.5
    init_scale: float = 0.1
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    ssim_lambda: float = 0.2
    refine_stop: int = 25_000
    seed: int = 0

    # ── Outputs ───────────────────────────────────────
    save_renders: int = 4
    save_ply: bool = True


# ── Model ────────────────────────────────────────────

def init_gaussians(scene, cfg, device):
    """Initialize from SfM points; uniformly subsample when they exceed the budget."""
    points, colors = scene.points, scene.colors
    if len(points) > cfg.budget:
        keep = torch.randperm(len(points), generator=torch.Generator().manual_seed(cfg.seed))[:cfg.budget]
        points, colors = points[keep.to(points.device)], colors[keep.to(points.device)]
    n = len(points)

    # scale from the mean distance to the 3 nearest neighbours, as in 3DGS
    dist, _ = cKDTree(points.cpu().numpy()).query(points.cpu().numpy(), k=4)
    dist = torch.from_numpy(dist[:, 1:].mean(1)).float().to(device).clamp_min(1e-7)
    params = {
        'means': points.clone(),
        'scales': torch.log(dist * cfg.init_scale)[:, None].repeat(1, 3),
        'quats': F.normalize(torch.rand(n, 4, device=device), dim=-1),
        'opacities': torch.logit(torch.full((n,), cfg.init_opacity, device=device)),
        'sh0': ((colors - 0.5) / SH_C0)[:, None, :],
        'shN': torch.zeros(n, (cfg.sh_degree + 1) ** 2 - 1, 3, device=device),
    }
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.contiguous()) for k, v in params.items()})

    # 3DGS learning rates; positions scale with the scene extent
    lrs = {'means': 1.6e-4 * scene.scene_scale, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2,
           'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{'params': params[k], 'lr': lr, 'name': k}], eps=1e-15, fused=True) for k, lr in lrs.items()}
    return params, optimizers


def render(params, viewmat, K, width, height, sh_degree, probe=None):
    """Rasterize one view; returns [H, W, 3] color, [H, W, K] probe image (or None), info.

    With ``probe`` ([N, K] zeros, requires_grad) the colors are evaluated here and K
    extra channels are rasterized through the same compositing weights. Because the
    probe values are zero they add nothing to the image and no gradient to geometry.
    """
    coeffs = torch.cat([params['sh0'], params['shN']], 1)  # [N, K, 3]
    common = dict(means=params['means'], quats=params['quats'], scales=torch.exp(params['scales']),
                  opacities=torch.sigmoid(params['opacities']), viewmats=viewmat[None], Ks=K[None],
                  width=width, height=height, packed=False)
    if probe is None:
        out, _, info = rasterization(colors=coeffs, sh_degree=sh_degree, **common)
        return out[0], None, info
    campos = torch.linalg.inv(viewmat)[:3, 3]
    rgb = spherical_harmonics(sh_degree, params['means'] - campos, coeffs)
    rgb = torch.clamp_min(rgb + 0.5, 0.0)  # same activation as gsplat's SH path
    out, _, info = rasterization(colors=torch.cat([rgb, probe], -1), **common)  # [1, H, W, 3 + K]
    return out[0, ..., :3], out[0, ..., 3:], info


# ── Loss / Objective ─────────────────────────────────

@torch.compile
def photometric_loss(image, gt, ssim_lambda: float):
    """3DGS objective 0.8 L1 + 0.2 D-SSIM on [H, W, 3] images."""
    l1 = (image - gt).abs().mean()
    d_ssim = 1 - ssim_map(image.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).mean()
    return (1 - ssim_lambda) * l1 + ssim_lambda * d_ssim


# ── Execution / Training ─────────────────────────────

def make_strategy(cfg):
    if cfg.method == 'mcmc':
        return MCMCStrategy(cap_max=cfg.budget, refine_stop_iter=cfg.refine_stop)
    target, recycle = {'rvr': ('variance', 0.0), 'abl-opacity': ('opacity', 0.0), 'abl-error': ('error', 0.0),
                       'abl-recycle': ('variance', cfg.recycle)}[cfg.method]
    return BudgetStrategy(cfg.budget, target=target, recycle=recycle, refine_stop=cfg.refine_stop)


def train(cfg: TrainConfig):
    """Train one budgeted model, evaluate on held-out views and write all outputs to ``cfg.output``."""
    device = 'cuda'
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(cfg.output)
    out.mkdir(parents=True)  # refuses to overwrite an existing run
    (out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))

    # ── Scene, parameters and strategy ────────────────
    scene = load_colmap_scene(cfg.data, cfg.images_dir, cfg.downsample, cfg.test_every, device)
    params, optimizers = init_gaussians(scene, cfg, device)
    strategy = make_strategy(cfg)
    probes = isinstance(strategy, BudgetStrategy) and strategy.uses_probes
    state = (strategy.initialize_state(len(params['means']), device) if isinstance(strategy, BudgetStrategy)
             else strategy.initialize_state())
    means_lr = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'], gamma=0.01 ** (1 / cfg.steps))
    order = torch.randperm(cfg.steps, generator=torch.Generator().manual_seed(cfg.seed)) % len(scene.train_ids)

    # ── Optimization loop ─────────────────────────────
    log = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        v = scene.train_ids[order[step]]
        gt = scene.images[v].to(device, non_blocking=True).float() / 255  # [H, W, 3]
        sh_degree = min(step // 1000, cfg.sh_degree)
        use_probe = probes and step < cfg.refine_stop  # attribution is only consumed by refinement
        probe = torch.zeros(len(params['means']), strategy.num_probes, device=device, requires_grad=True) if use_probe else None
        image, probe_image, info = render(params, scene.viewmats[v], scene.Ks[v], scene.width, scene.height, sh_degree, probe)

        loss = photometric_loss(image, gt, cfg.ssim_lambda)
        loss = loss + cfg.opacity_reg * torch.sigmoid(params['opacities']).mean() + cfg.scale_reg * torch.exp(params['scales']).mean()
        if use_probe:
            # zero-valued probes; d/d(probe_ik) = sum_p w_ip f_k(p)
            fields = strategy.probe_fields((image - gt).detach())  # [H, W, K]
            loss = loss + (probe_image * fields).sum()
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_lr.step()
        lr = means_lr.get_last_lr()[0]
        if isinstance(strategy, BudgetStrategy):
            if use_probe:
                strategy.accumulate(state, probe.grad)
            strategy.step_post_backward(params, optimizers, state, step, lr)
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, lr=lr)

        if step % 500 == 0 or step == cfg.steps - 1:
            relocation = strategy.last if isinstance(strategy, BudgetStrategy) else {}
            log.append({'step': step, 'loss': loss.item(), 'n': len(params['means']), 'time': time.perf_counter() - t0} | relocation)
            print(f"[INFO] {scene.name} {cfg.method} B={cfg.budget} step {step:5d} loss {log[-1]['loss']:.4f} "
                  f"N={log[-1]['n']} {log[-1]['time']:.0f}s", flush=True)
    torch.cuda.synchronize()
    train_time = time.perf_counter() - t0

    # ── Evaluation and outputs ────────────────────────
    metrics = evaluate(params, scene, cfg, out)
    metrics |= {'scene': scene.name, 'method': cfg.method, 'budget': cfg.budget, 'seed': cfg.seed,
                'num_gaussians': len(params['means']), 'train_time_s': train_time,
                'peak_memory_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'width': scene.width, 'height': scene.height}
    (out / 'log.json').write_text(json.dumps(log))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    torch.save({k: v.detach().cpu() for k, v in params.items()}, out / 'gaussians.pt')
    if cfg.save_ply:
        save_ply(params, out / 'point_cloud.ply')
    print(f"[INFO] {scene.name} {cfg.method} B={cfg.budget}: PSNR {metrics['psnr']:.2f} SSIM {metrics['ssim']:.3f} "
          f"LPIPS {metrics['lpips']:.3f} ({train_time / 60:.1f} min)", flush=True)
    return metrics


# ── Evaluation ───────────────────────────────────────

@torch.no_grad()
def evaluate(params, scene, cfg, out):
    """Held-out PSNR / SSIM / LPIPS-VGG and render FPS; saves the first renders as PNG."""
    lpips_fn = LPIPS('cuda')
    (out / 'renders').mkdir()
    scores, times = [], []
    for j, v in enumerate(scene.test_ids):
        gt = scene.images[v].cuda().float() / 255
        torch.cuda.synchronize()
        t = time.perf_counter()
        image, _, _ = render(params, scene.viewmats[v], scene.Ks[v], scene.width, scene.height, cfg.sh_degree)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        image = image.clamp(0, 1)
        scores.append((psnr(image, gt), ssim(image, gt), lpips_fn(image, gt)))
        if j < cfg.save_renders:
            Image.fromarray((image.cpu().numpy() * 255).round().astype(np.uint8)).save(out / 'renders' / f'{j:03d}.png')
            Image.fromarray(scene.images[v].numpy()).save(out / 'renders' / f'gt_{j:03d}.png')
    psnr_, ssim_, lpips_ = np.mean(scores, 0)
    return {'psnr': float(psnr_), 'ssim': float(ssim_), 'lpips': float(lpips_), 'fps': float(1 / np.median(times)),
            'per_view': [list(s) for s in scores]}
