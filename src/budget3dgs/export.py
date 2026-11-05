"""Export trained Gaussians in the standard 3DGS PLY layout.

The file opens in common splat viewers (e.g. SuperSplat, the INRIA SIBR viewer).
"""

import numpy as np
from plyfile import PlyData, PlyElement


def save_ply(params, path):
    """Write means, normals (zero), SH DC/rest, logit opacity, log scales and quaternions."""
    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    n = len(p['means'])
    sh0 = p['sh0'].transpose(0, 2, 1).reshape(n, -1)    # [N, 1, 3] -> [N, 3]
    shN = p['shN'].transpose(0, 2, 1).reshape(n, -1)    # [N, K-1, 3] -> [N, 3*(K-1)], channel-major as in 3DGS
    columns = [p['means'], np.zeros((n, 3)), sh0, shN, p['opacities'][:, None], p['scales'], p['quats']]
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(shN.shape[1])]
             + ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])
    data = np.ascontiguousarray(np.concatenate(columns, 1), dtype=np.float32)
    vertices = data.view([(name, 'f4') for name in names])[:, 0]  # [N, C] float32 -> [N] structured
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(path))
