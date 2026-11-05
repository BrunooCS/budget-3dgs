"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Where does each strategy spend its Gaussians?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For one held-out view and runs of the same scene and budget, plots per method:
the render, its absolute error, and the density of *visible* Gaussian centers.

1) Render each run with the zero-valued probe channels; their gradients give each
   Gaussian's visible mass M_i in this view (sum of its compositing weights).
2) Keep Gaussians with M_i > 0.5 pixel and histogram their projected centers on
   a coarse grid; occluded or out-of-frame primitives are not counted.
3) Write a PNG grid (rows: methods; columns: render | error | density) and a
   final row comparing the last run with the first: ground truth, error reduction
   and log2 ratio of Gaussian-smoothed densities, both on the same grid.

Example:
    python scripts/allocation_figure.py runs/gpu_6gb/truck/mcmc-b100000-s0 runs/gpu_6gb/truck/rvr-b100000-s0 \
        --labels "3DGS-MCMC" "Budget-3DGS" --output docs/assets/allocation_truck.png
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from budget3dgs.data import load_colmap_scene
from budget3dgs.metrics import psnr
from budget3dgs.train import render


def main():
    """Render, error and visible-center density for each run on held-out view ``--view``."""
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--view', type=int, default=0)
    parser.add_argument('--cell', type=int, default=8)
    args = parser.parse_args()

    cfg = json.loads((Path(args.runs[0]) / 'config.json').read_text())
    scene = load_colmap_scene(cfg['data'], cfg['images_dir'], cfg['downsample'], cfg['test_every'])
    v = scene.test_ids[args.view]
    gt = scene.images[v].cuda().float() / 255
    h, w = scene.height, scene.width

    # ── Render and attribute visible mass per run ─────
    rows = []
    for run in args.runs:
        params = {k: x.cuda() for k, x in torch.load(Path(run) / 'gaussians.pt', weights_only=True).items()}
        probe = torch.zeros(len(params['means']), 2, device='cuda', requires_grad=True)
        image, probe_image, info = render(params, scene.viewmats[v], scene.Ks[v], w, h, cfg['sh_degree'], probe)
        probe_image[..., 1].sum().backward()
        visible = probe.grad[:, 1] > 0.5                       # [N] visible mass above half a pixel
        xy = info['means2d'][0][visible].detach().cpu().numpy()  # [N_vis, 2] pixel coordinates
        density, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=(h // args.cell, w // args.cell), range=((0, h), (0, w)))
        image = image.detach().clamp(0, 1)
        rows.append((image.cpu().numpy(), (image - gt).abs().mean(-1).cpu().numpy(), density, psnr(image, gt), int(visible.sum())))

    # ── Plot ──────────────────────────────────────────
    vmax_err, vmax_den = max(r[1].max() for r in rows) * 0.5, max(r[2].max() for r in rows)
    fig, axes = plt.subplots(len(rows) + 1, 3, figsize=(12, (len(rows) + 1) * (4 * h / w + 0.35)), squeeze=False)
    for ax_row, (image, err, density, score, n_vis), label in zip(axes, rows, args.labels):
        ax_row[0].imshow(image)
        ax_row[0].set_title(f'{label}: render ({score:.2f} dB)', fontsize=9)
        ax_row[1].imshow(err, cmap='inferno', vmin=0, vmax=vmax_err)
        ax_row[1].set_title('|error|', fontsize=9)
        ax_row[2].imshow(np.log1p(density), cmap='viridis', vmin=0, vmax=np.log1p(vmax_den))
        ax_row[2].set_title(f'visible Gaussian centers (log, {n_vis / 1000:.0f}K visible)', fontsize=9)

    # Last run vs first run on the density grid: [h, w] -> [h // cell, w // cell]
    hc, wc = h // args.cell, w // args.cell
    pool = lambda e: e[:hc * args.cell, :wc * args.cell].reshape(hc, args.cell, wc, args.cell).mean((1, 3))
    gain = pool(rows[0][1]) - pool(rows[-1][1])
    smooth = lambda d: gaussian_filter(d, 2.0) + 0.5  # counts per cell are sparse; compare local densities
    ratio = np.log2(smooth(rows[-1][2]) / smooth(rows[0][2]))
    ax_row = axes[-1]
    ax_row[0].imshow(gt.cpu().numpy())
    ax_row[0].set_title('ground truth (held-out view)', fontsize=9)
    lim = np.abs(gain).max() * 0.6
    ax_row[1].imshow(gain, cmap='RdBu', vmin=-lim, vmax=lim)
    ax_row[1].set_title(f'error reduction, {args.labels[0]} - {args.labels[-1]} (blue: lower)', fontsize=9)
    ax_row[2].imshow(ratio, cmap='RdBu', vmin=-1, vmax=1)
    ax_row[2].set_title(f'log2 visible-center ratio, {args.labels[-1]} / {args.labels[0]} (blue: more)', fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f'[INFO] wrote {args.output}')


if __name__ == '__main__':
    main()
