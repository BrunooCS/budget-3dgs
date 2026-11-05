"""COLMAP scene loading with the standard 3DGS evaluation protocol.

Cameras use OpenCV axes (+x right, +y down, +z forward). Images are sorted by
name; every ``test_every``-th image (index % test_every == 0) is held out, as in
3DGS, Mip-NeRF 360 and gsplat. Initialization uses the COLMAP sparse points and
colors, again matching the published protocol. Images stay in pinned host memory
as uint8 and only the current view is copied to the GPU, which keeps several
hundred views of a scene off a 6 GB device.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ── COLMAP Parsing ───────────────────────────────────

_MODELS = {0: ('SIMPLE_PINHOLE', 3), 1: ('PINHOLE', 4), 2: ('SIMPLE_RADIAL', 4), 3: ('RADIAL', 5), 4: ('OPENCV', 8)}


def _unpack(f, code):
    return struct.unpack('<' + code, f.read(struct.calcsize('<' + code)))


def read_colmap(sparse: Path):
    """Read cameras.bin, images.bin and points3D.bin of a COLMAP model."""
    cameras, images = {}, []
    with (sparse / 'cameras.bin').open('rb') as f:
        for _ in range(_unpack(f, 'Q')[0]):
            cam_id, model_id, w, h = _unpack(f, 'iiQQ')
            model, count = _MODELS[model_id]
            cameras[cam_id] = (model, w, h, _unpack(f, 'd' * count))
    with (sparse / 'images.bin').open('rb') as f:
        for _ in range(_unpack(f, 'Q')[0]):
            values = _unpack(f, 'idddddddi')
            name = b''
            while (c := f.read(1)) != b'\x00':
                name += c
            f.seek(_unpack(f, 'Q')[0] * 24, 1)  # skip 2D observations
            images.append((np.array(values[1:5]), np.array(values[5:8]), values[8], name.decode()))
    with (sparse / 'points3D.bin').open('rb') as f:
        n = _unpack(f, 'Q')[0]
        xyz, rgb = np.empty((n, 3), np.float32), np.empty((n, 3), np.float32)
        for i in range(n):
            values = _unpack(f, 'QdddBBBd')
            xyz[i], rgb[i] = values[1:4], np.array(values[4:7]) / 255
            f.seek(_unpack(f, 'Q')[0] * 8, 1)  # skip track
    return cameras, images, xyz, rgb


def _intrinsics(model, params):
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
        f, cx, cy = params[:3]
        return f, f, cx, cy
    return params[:4]  # PINHOLE / OPENCV: fx, fy, cx, cy


def _qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                     [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                     [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])


# ── Scene ────────────────────────────────────────────

@dataclass
class Scene:
    name: str
    images: torch.Tensor      # [V, H, W, 3] uint8, pinned host memory
    viewmats: torch.Tensor    # [V, 4, 4] world -> camera
    Ks: torch.Tensor          # [V, 3, 3]
    train_ids: list
    test_ids: list
    points: torch.Tensor      # [P, 3] SfM points
    colors: torch.Tensor      # [P, 3] in [0, 1]
    scene_scale: float

    @property
    def width(self):
        return self.images.shape[2]

    @property
    def height(self):
        return self.images.shape[1]


def load_colmap_scene(root, images_dir='images', downsample=1, test_every=8, device='cuda'):
    """Load an undistorted COLMAP capture: ``root/{images_dir}/*`` and ``root/sparse/0``.

    ``images_dir`` selects pre-downsampled folders (e.g. ``images_4`` for Mip-NeRF 360
    outdoor scenes) and ``downsample`` further reduces them with Lanczos filtering;
    intrinsics are rescaled to the loaded resolution.
    """
    root = Path(root)
    cameras, records, xyz, rgb = read_colmap(root / 'sparse' / '0')
    records.sort(key=lambda r: r[3])

    # 1) Images and intrinsics at the loaded resolution
    files = sorted((root / images_dir).iterdir())
    by_stem = {p.stem: p for p in files}
    images, viewmats, Ks = [], [], []
    for qvec, tvec, cam_id, name in records:
        model, w0, h0, params = cameras[cam_id]
        image = Image.open(by_stem[Path(name).stem]).convert('RGB')
        image = np.asarray(image.resize((round(image.width / downsample), round(image.height / downsample)), Image.LANCZOS))
        h, w = image.shape[:2]
        fx, fy, cx, cy = _intrinsics(model, params)
        sx, sy = w / w0, h / h0
        Ks.append([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
        viewmat = np.eye(4)
        viewmat[:3, :3], viewmat[:3, 3] = _qvec_to_rotmat(qvec), tvec
        viewmats.append(viewmat)
        images.append(image)

    # 2) Scene scale from camera centers (3DGS "nerf normalization" radius)
    viewmats = torch.tensor(np.stack(viewmats), dtype=torch.float32)
    centers = torch.linalg.inv(viewmats)[:, :3, 3]
    scene_scale = 1.1 * (centers - centers.mean(0)).norm(dim=-1).max().item()

    ids = list(range(len(records)))
    return Scene(root.name, torch.from_numpy(np.stack(images)).pin_memory(), viewmats.to(device),
                 torch.tensor(Ks, dtype=torch.float32, device=device),
                 [i for i in ids if i % test_every], [i for i in ids if i % test_every == 0],
                 torch.from_numpy(xyz).to(device), torch.from_numpy(rgb).to(device), scene_scale)
