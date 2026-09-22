"""Shared geometry utilities for the keypoint head.

Intentionally free-functions so neither head's class hierarchy is touched.
"""
import torch


def gt_transform_batch(batch_position_current, batch_position_other):
    """Compute per-sample relative 2-D rigid transform from ego-pose lists.

    Verbatim port of KeypointHead.getGTTransformBatch — do NOT diverge.
    """
    R_list, t_list, scale_list = [], [], []
    for b in range(len(batch_position_current)):
        relative_transformation = torch.linalg.inv(
            torch.tensor(batch_position_current[b])
        ).matmul(torch.tensor(batch_position_other[b]))
        orthogonal_projection = torch.eye(4)
        orthogonal_projection[2, 2] = 0
        relative_transformation = orthogonal_projection.matmul(relative_transformation)

        R = relative_transformation[:3, :3]
        t = relative_transformation[:3, 3]
        scale = torch.norm(t)
        t = t[:2]
        R = R[:2, :2]

        R_list.append(R)
        t_list.append(t)
        scale_list.append(scale)
    return R_list, t_list, scale_list


def estimate_rigid_2d(pts0, pts1):
    """Least-squares 2-D rigid transform (R, t) mapping pts0 -> pts1 (Kabsch/Procrustes).

    Args:
        pts0: [N, 2] source points
        pts1: [N, 2] target points

    Returns:
        R: [2, 2] rotation matrix
        t: [2]   translation vector
    """
    if pts0.shape[0] < 2:
        return torch.eye(2, device=pts0.device, dtype=pts0.dtype), \
               torch.zeros(2, device=pts0.device, dtype=pts0.dtype)
    c0 = pts0.mean(0)
    c1 = pts1.mean(0)
    H = (pts0 - c0).t() @ (pts1 - c1)
    U, _, V = torch.linalg.svd(H)
    R = V.t() @ U.t()
    if torch.det(R) < 0:
        V2 = V.clone()
        V2[-1] = -V2[-1]
        R = V2.t() @ U.t()
    t = c1 - R @ c0
    return R, t


def grid_warp_to_homography(R, t, dim):
    """Convert the grid-unit (row,col) warp (R,t) into a 3x3 homography acting on
    NORMALIZED (x,y) in [0,1].

    The rest of the codebase warps in grid/cell (row,col) units centered at
    (dim-1)/2 with the invSE2-forward convention (see repeatability_eval._warp_coords
    (x,y)=(col_n,row_n) in [0,1]. This builds T such that
    `bev_warp_ops._warp_xy(xy_norm, T)` reproduces `_warp_coords` exactly.

    Normalized<->cell mapping: cell = norm * (dim-1), i.e. row=(dim-1)*y, col=(dim-1)*x.
    Composition: T = inv(M) @ (Tc @ invSE2 @ Tc^-1) @ M, where M maps
    [x,y,1]->[row,col,1] and the middle factor is the centered grid warp.
    """
    R = torch.as_tensor(R, dtype=torch.float64)
    t = torch.as_tensor(t, dtype=torch.float64)
    s = float(dim - 1)
    c = s / 2.0
    SE2 = torch.eye(3, dtype=torch.float64)
    SE2[:2, :2] = R[:2, :2]
    SE2[:2, 2] = t[:2]
    invSE2 = torch.inverse(SE2)
    Tc = torch.eye(3, dtype=torch.float64); Tc[0, 2] = c; Tc[1, 2] = c
    Tcn = torch.eye(3, dtype=torch.float64); Tcn[0, 2] = -c; Tcn[1, 2] = -c
    G = Tc @ invSE2 @ Tcn                                  # centered grid warp (row,col)
    M = torch.tensor([[0.0, s, 0.0], [s, 0.0, 0.0], [0.0, 0.0, 1.0]],
                     dtype=torch.float64)                  # [x,y,1] -> [row,col,1]
    T_AB = torch.inverse(M) @ G @ M
    return T_AB.to(torch.float32)
