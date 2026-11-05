"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Novel-view video of one or more trained runs, side by side
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Renders trained models of the same scene along one smooth camera
path and writes an MP4 and a GIF with the panels side by side.

1) Load each run's config, scene cameras and final Gaussians.
2) Build the path: slerp rotations / interpolate centers through every k-th
   training camera in capture order, so the path stays inside the captured volume.
3) Render all runs at each pose with full SH and write the videos.

Example:
    python scripts/render_video.py runs/gpu_6gb/truck/mcmc-b100000-s0 runs/gpu_6gb/truck/rvr-b100000-s0 \
        --labels "3DGS-MCMC" "Budget-3DGS" --output docs/assets/truck_100k
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from budget3dgs.data import load_colmap_scene
from budget3dgs.train import render


def load_run(run):
    cfg = json.loads((Path(run) / 'config.json').read_text())
    params = {k: v.cuda() for k, v in torch.load(Path(run) / 'gaussians.pt', weights_only=True).items()}
    return cfg, params


def camera_path(viewmats, keyframe_every, frames):
    """Interpolated world->camera matrices [F, 4, 4] through every k-th camera."""
    c2w = torch.linalg.inv(viewmats).cpu().numpy()[::keyframe_every]
    t = np.linspace(0, 1, len(c2w))
    s = np.linspace(0, 1, frames)
    rotations = Slerp(t, Rotation.from_matrix(c2w[:, :3, :3]))(s).as_matrix()
    centers = np.stack([np.interp(s, t, c2w[:, :3, 3][:, i]) for i in range(3)], -1)
    poses = np.tile(np.eye(4), (frames, 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = rotations, centers
    return torch.linalg.inv(torch.tensor(poses, dtype=torch.float32, device='cuda'))


@torch.no_grad()
def main():
    """Render the runs along a shared path and write ``<output>.mp4`` and ``<output>.gif``."""
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=240)
    parser.add_argument('--keyframe_every', type=int, default=10)
    parser.add_argument('--gif_width', type=int, default=960)
    args = parser.parse_args()

    # 1) Runs and the shared scene
    runs = [load_run(r) for r in args.runs]
    cfg = runs[0][0]
    scene = load_colmap_scene(cfg['data'], cfg['images_dir'], cfg['downsample'], cfg['test_every'])
    K, w, h = scene.Ks[0], scene.width, scene.height

    # 2) Path through the training cameras
    poses = camera_path(scene.viewmats[scene.train_ids], args.keyframe_every, args.frames)

    # 3) Render panels and label them
    frames = []
    for viewmat in poses:
        panels = []
        for (run_cfg, params), label in zip(runs, args.labels):
            image, _, _ = render(params, viewmat, K, w, h, run_cfg['sh_degree'])
            panel = Image.fromarray((image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            draw = ImageDraw.Draw(panel)
            draw.rectangle([0, 0, w, 22], fill=(0, 0, 0))
            draw.text((8, 5), f"{label}  |  {len(params['means']) / 1000:.0f}K Gaussians", fill=(255, 255, 255))
            panels.append(np.asarray(panel))
        frame = np.concatenate(panels, 1)  # [h, n_runs * w, 3]
        frames.append(frame[:h // 2 * 2, :frame.shape[1] // 2 * 2])  # libx264 needs even dimensions

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out.with_suffix('.mp4'), frames, fps=30, quality=8, macro_block_size=1)
    scale = args.gif_width / frames[0].shape[1]
    small = [np.asarray(Image.fromarray(f).resize((args.gif_width, round(f.shape[0] * scale)), Image.LANCZOS)) for f in frames[::2]]
    imageio.mimwrite(out.with_suffix('.gif'), small, duration=1000 / 15, loop=0)
    print(f'[INFO] wrote {out}.mp4 and {out}.gif ({len(frames)} frames)')


if __name__ == '__main__':
    main()
