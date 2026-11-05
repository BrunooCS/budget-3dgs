"""Probe-channel attribution equals brute-force compositing-weight sums and is gradient-neutral."""

import torch
from gsplat import rasterization

from budget3dgs.strategy import BudgetStrategy
from budget3dgs.train import render


def _scene(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    params = {'means': torch.randn(n, 3, generator=g) * 0.5 + torch.tensor([0, 0, 4.0]),
              'scales': torch.full((n, 3), -1.5) + 0.3 * torch.randn(n, 3, generator=g),
              'quats': torch.randn(n, 4, generator=g), 'opacities': torch.randn(n, generator=g),
              'sh0': torch.randn(n, 1, 3, generator=g) * 0.5, 'shN': torch.zeros(n, 15, 3)}
    params = {k: v.cuda().requires_grad_() for k, v in params.items()}
    K = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device='cuda')
    return params, torch.eye(4, device='cuda'), K


def test_probe_matches_brute_force_weights():
    params, viewmat, K = _scene()
    n = len(params['means'])
    error = torch.rand(64, 64, device='cuda')
    probe = torch.zeros(n, 2, device='cuda', requires_grad=True)
    _, probe_image, _ = render(params, viewmat, K, 64, 64, 0, probe)
    ((probe_image[..., 0] * error).sum() + probe_image[..., 1].sum()).backward()

    # one-hot colors render each Gaussian's compositing weight w_ip as its own channel: [H, W, N]
    weights, _, _ = rasterization(params['means'], params['quats'], torch.exp(params['scales']),
                                  torch.sigmoid(params['opacities']), torch.eye(n, device='cuda'),
                                  viewmat[None], K[None], 64, 64, packed=False)
    weights = weights[0].detach()
    torch.testing.assert_close(probe.grad[:, 0], (weights * error[..., None]).sum((0, 1)), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(probe.grad[:, 1], weights.sum((0, 1)), rtol=1e-4, atol=1e-4)


def test_probe_leaves_scene_gradients_unchanged():
    grads = []
    for use_probe in (False, True):
        params, viewmat, K = _scene()
        probe = torch.zeros(len(params['means']), 2, device='cuda', requires_grad=True)
        image, probe_image, _ = render(params, viewmat, K, 64, 64, 0, probe)
        loss = image.square().mean()
        if use_probe:
            loss = loss + (probe_image[..., 0] * torch.rand(64, 64, device='cuda')).sum() + probe_image[..., 1].sum()
        loss.backward()
        grads.append(torch.cat([params[k].grad.flatten() for k in ('means', 'scales', 'quats', 'opacities', 'sh0')]))
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-6)


def test_footprint_variance_matches_brute_force():
    params, viewmat, K = _scene(seed=1)
    n = len(params['means'])
    strategy = BudgetStrategy(budget=n, target='variance')
    state = strategy.initialize_state(n, 'cuda')
    residual = torch.randn(64, 64, 3, device='cuda')
    probe = torch.zeros(n, strategy.num_probes, device='cuda', requires_grad=True)
    _, probe_image, _ = render(params, viewmat, K, 64, 64, 0, probe)
    (probe_image * strategy.probe_fields(residual)).sum().backward()
    strategy.accumulate(state, probe.grad)

    weights, _, _ = rasterization(params['means'], params['quats'], torch.exp(params['scales']),
                                  torch.sigmoid(params['opacities']), torch.eye(n, device='cuda'),
                                  viewmat[None], K[None], 64, 64, packed=False)
    w = weights[0].detach().reshape(-1, n)                    # [HW, N]
    r = residual.reshape(-1, 3)                               # [HW, 3]
    mass = w.sum(0)
    mean = (w.T @ r) / mass[:, None].clamp_min(1e-6)          # [N, 3] weighted mean residual
    variance = (w.T @ r.square().sum(-1)) - mass * mean.square().sum(-1)
    torch.testing.assert_close(state['error'], variance.clamp_min(0), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(state['mass'], mass, rtol=1e-4, atol=1e-4)
