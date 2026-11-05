"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Export trained models for the browser viewer (docs/viewer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Copies the 3DGS-MCMC and Budget-3DGS PLYs of each scene at one budget into
``<output>/scenes`` and writes ``<output>/scenes.json`` with a start camera and
the held-out PSNR of every model.

1) Read the COLMAP cameras of the scene (no images are loaded).
2) Start camera: the first held-out view. World up is the mean camera up axis;
   the orbit target lies on the view axis at the median depth of the Gaussians.
3) Copy ``point_cloud.ply`` of each method and record PSNR, size and count.

Example:
    python scripts/export_viewer.py runs/gpu_6gb --scenes garden kitchen truck --budget 50000 --output docs/viewer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from budget3dgs.data import _qvec_to_rotmat, read_colmap

METHODS = {'mcmc': '3DGS-MCMC', 'rvr': 'Budget-3DGS'}


def start_camera(data, means):
    """Position, orbit target and up vector (world coordinates) from the first held-out COLMAP view."""
    _, records, _, _ = read_colmap(Path(data) / 'sparse' / '0')
    records.sort(key=lambda r: r[3])
    Rs = np.stack([_qvec_to_rotmat(q) for q, _, _, _ in records])  # [V, 3, 3] world -> camera
    up = -Rs[:, 1, :].mean(0)                                        # COLMAP camera y points down
    R, t = Rs[0], records[0][1]
    depth = (means @ R.T + t)[:, 2]                                  # [N] camera-space z
    center = -R.T @ t
    target = center + R[2] * np.median(depth[depth > 0])
    return center.tolist(), target.tolist(), (up / np.linalg.norm(up)).tolist()


def main():
    """Copy PLYs and write the viewer manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument('runs')
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--budget', type=int, default=50_000)
    parser.add_argument('--output', default='docs/viewer')
    args = parser.parse_args()

    out = Path(args.output)
    (out / 'scenes').mkdir(parents=True, exist_ok=True)
    manifest = []
    for scene in args.scenes:
        runs = {m: Path(args.runs) / scene / f'{m}-b{args.budget}-s0' for m in METHODS}
        cfg = json.loads((runs['rvr'] / 'config.json').read_text())
        means = torch.load(runs['rvr'] / 'gaussians.pt', weights_only=True)['means'].numpy()  # [N, 3]
        position, target, up = start_camera(cfg['data'], means)
        models = []
        for m, label in METHODS.items():
            name = f'{scene}-{m}.ply'
            shutil.copyfile(runs[m] / 'point_cloud.ply', out / 'scenes' / name)
            metrics = json.loads((runs[m] / 'metrics.json').read_text())
            models.append({'label': label, 'file': f'scenes/{name}', 'psnr': round(metrics['psnr'], 2),
                           'gaussians': metrics['num_gaussians'], 'mb': round((out / 'scenes' / name).stat().st_size / 1e6, 1)})
        manifest.append({'name': scene, 'position': position, 'target': target, 'up': up, 'models': models})
        print(f'[INFO] {scene}: ' + ', '.join(f"{m['label']} {m['psnr']} dB" for m in models))
    (out / 'scenes.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
