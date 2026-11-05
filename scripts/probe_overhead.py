"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cost of the Budget-3DGS probe attribution per training iteration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Times render + loss + backward (+ Adam step) for one training view with and
without the five zero-valued probe channels and their accumulation.

1) Load a scene and initialize ``budget`` Gaussians from its SfM points.
2) Time ``iters`` iterations of each variant after warm-up, interleaved twice.
3) Print milliseconds per iteration and the relative overhead.

Run on an otherwise idle GPU; concurrent jobs distort the timings.

Example:
    python scripts/probe_overhead.py --data data/tandt/truck --budget 100000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import time

import torch

from budget3dgs.data import load_colmap_scene
from budget3dgs.strategy import BudgetStrategy
from budget3dgs.train import TrainConfig, init_gaussians, photometric_loss, render


def main():
    """Print per-iteration time with and without the probe channels."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--images_dir', default='images')
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--budget', type=int, default=100_000)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    scene = load_colmap_scene(args.data, args.images_dir, args.downsample)
    params, optimizers = init_gaussians(scene, TrainConfig(budget=args.budget), 'cuda')
    strategy = BudgetStrategy(args.budget)
    state = strategy.initialize_state(len(params['means']), 'cuda')

    def iteration(i, probes):
        v = scene.train_ids[i % len(scene.train_ids)]
        gt = scene.images[v].cuda().float() / 255
        probe = torch.zeros(len(params['means']), strategy.num_probes, device='cuda', requires_grad=True) if probes else None
        image, probe_image, _ = render(params, scene.viewmats[v], scene.Ks[v], scene.width, scene.height, 3, probe)
        loss = photometric_loss(image, gt, 0.2)
        if probes:
            loss = loss + (probe_image * strategy.probe_fields((image - gt).detach())).sum()
        loss.backward()
        if probes:
            strategy.accumulate(state, probe.grad)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

    times = {False: [], True: []}
    for probes in (False, True, False, True):
        for i in range(10):
            iteration(i, probes)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for i in range(args.iters):
            iteration(i, probes)
        torch.cuda.synchronize()
        times[probes].append((time.perf_counter() - t) / args.iters * 1000)
    base, rvr = min(times[False]), min(times[True])
    print(f'[INFO] {scene.name} {scene.width}x{scene.height}, {len(params["means"])} Gaussians: '
          f'{base:.1f} ms/iter without probes, {rvr:.1f} ms/iter with probes (+{100 * (rvr / base - 1):.0f}%)')


if __name__ == '__main__':
    main()
