"""Keypoint extraction and matching helpers shared by the KITTI evaluation scripts.

Detector and matcher are kept identical to what the model is trained with:
non-maximum suppression on the score map, the same validity mask, and mutual
nearest-neighbour matching on the descriptors.
"""
import numpy as np
import torch

from bevmatch.keypoint_ops import keypoint_valid_mask, plateau_safe_nms


def peak_cells(head, heatmap_logits, topk):
    """Score logits [1,1,H,W] (or [H,W]) -> top-K interior local maxima as (row, col).

    Uses the same NMS and validity mask as the training-time detector, so the
    evaluation sees exactly the keypoints the model was optimized to produce.
    """
    h = heatmap_logits
    if h.dim() == 4:
        h = h[0, 0]
    elif h.dim() == 3:
        h = h[0]
    dim = h.shape[-1]
    valid = keypoint_valid_mask(
        dim,
        border=getattr(head, 'kpt_mask_border', 8),
        center_halfwidth=getattr(head, 'kpt_mask_center_halfwidth', 4),
    ).to(h.device)
    prob = torch.sigmoid(h.float())
    nms = plateau_safe_nms(prob, k=5) & valid
    ys, xs = torch.nonzero(nms, as_tuple=True)
    if ys.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=h.device)
    top = torch.topk(prob[ys, xs], k=min(topk, ys.numel())).indices
    return torch.stack([ys[top], xs[top]], 1).long()


def mutual_nn_matches(desc0, c0, desc1, c1, binary=False):
    """Mutual nearest neighbours between the cells of two frames.

    Float descriptors are compared by cosine distance after L2 normalization;
    binary descriptors by Hamming distance. desc: [1, D, H, W]. Returns [(i, j), ...].
    """
    if c0.shape[0] == 0 or c1.shape[0] == 0:
        return []
    d0 = desc0[0, :, c0[:, 0], c0[:, 1]].T
    d1 = desc1[0, :, c1[:, 0], c1[:, 1]].T
    if binary:
        dist = (d0.unsqueeze(1) != d1.unsqueeze(0)).sum(-1)
    else:
        d0 = torch.nn.functional.normalize(d0.float(), dim=1)
        d1 = torch.nn.functional.normalize(d1.float(), dim=1)
        dist = 1.0 - (d0 @ d1.T)
    nn12 = dist.argmin(1)
    nn21 = dist.argmin(0)
    return [(i, int(nn12[i])) for i in range(nn12.shape[0])
            if int(nn21[int(nn12[i])]) == i]


def cells_to_xy(cells, dim, res):
    """(row, col) cells -> metric BEV coordinates (x, y), with x from row and y from col."""
    cn = np.asarray(cells, dtype=np.float32)
    half = dim / 2.0
    return np.stack([(cn[:, 0] - half) * res, (cn[:, 1] - half) * res], 1).astype(np.float32)
