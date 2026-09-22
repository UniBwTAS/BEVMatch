"""BEV warping helpers used by the keypoint head. Pure functions, no modules."""
import atexit
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import linear_sum_assignment


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def refine_reference(ref, delta):
    """Iterative refinement in the logit domain."""
    return torch.sigmoid(inverse_sigmoid(ref) + delta)


def decode_location(ref, offset, r_max):
    """Bounded local offset around the reference, clamped to normalized [0,1]."""
    return (ref + (2.0 * torch.sigmoid(offset) - 1.0) * r_max).clamp(0.0, 1.0)


def normalized_to_metres(xy_norm, extent_m=108.0):
    """Map normalized [0,1] BEV coords to centered metres (loss/host side only)."""
    return (xy_norm - 0.5) * extent_m


def reference_grid(num_queries=256, device=None):
    """A fixed near-square grid of normalized reference points. Deterministic."""
    side = int(round(num_queries ** 0.5))
    assert side * side == num_queries, "num_queries must be a perfect square (e.g. 256)"
    lin = (torch.arange(side, device=device).float() + 0.5) / side  # cell centers
    ys, xs = torch.meshgrid(lin, lin, indexing='ij')
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # [num_queries, 2]


def sample_descriptor_field(field, xy_norm):
    """Bilinear-sample a dense descriptor field at sparse predicted locations.

    field   [B,C,H,W]  dense descriptor map (e.g. the frozen 128-D field).
    xy_norm [B,Q,2]    predicted locations, (x=col, y=row) in [0,1].
    returns [B,Q,C]    fp32, L2-normalized descriptors.

    Convention is locked to the rest of the head (and the ONNX export):
    grid = xy*2-1, (x,y) ordering, bilinear, align_corners=True, zeros pad.
    """
    grid = (xy_norm.float() * 2 - 1).unsqueeze(1)            # [B,1,Q,2], (x,y)
    d = F.grid_sample(field.float(), grid, mode='bilinear',
                      align_corners=True, padding_mode='zeros')  # [B,C,1,Q]
    d = d.squeeze(2).transpose(1, 2)                         # [B,Q,C]
    return F.normalize(d, dim=-1)


def stratified_topk_cells(sal, row, col, dim, t_max, n_regions=9):
    """Spatially-stratified top-K cell selection.

    Global top-K by saliency clusters all targets into the densest region (a
    building wall, a dense return), teaching the head to pile its landmarks
    there (observed: 9x9 coverage 0.12, ~10/81 regions). Instead, partition the
    BEV into an n_regions x n_regions grid and select round-robin by within-
    region saliency rank: every non-empty region contributes its strongest cell
    first, then its second-strongest, etc., until t_max is reached. This spreads
    targets across all *structured* regions (empty regions contribute nothing —
    distribution follows real saliency, it never fills void) yet still fills to
    t_max where structure exists.

    Args:
        sal:  [M] saliency per candidate cell (already artifact-masked).
        row:  [M] long, BEV row of each candidate (0..dim-1).
        col:  [M] long, BEV col of each candidate.
        dim:  BEV side length in cells.
        t_max: number of cells to return.
        n_regions: regions per axis (default 9 -> 81 regions, matches the
                   spatial_coverage diagnostic).
    Returns:
        [min(t_max, M)] long indices into the input arrays, ordered by
        (within-region rank asc, saliency desc).
    """
    M = sal.shape[0]
    if M == 0:
        return torch.empty(0, dtype=torch.long, device=sal.device)
    rs = (dim + n_regions - 1) // n_regions                      # region size in cells
    reg = (row.long() // rs) * n_regions + (col.long() // rs)    # [M] region id
    order = torch.argsort(sal, descending=True)                  # global saliency-desc order
    reg_ord = reg[order]
    # within-region rank along the saliency-desc order (0 = region's strongest)
    within = torch.zeros(M, dtype=torch.long, device=sal.device)
    for r in torch.unique(reg_ord):
        m = reg_ord == r
        within[m] = torch.arange(int(m.sum()), device=sal.device)
    # stable sort by within-rank keeps the saliency-desc order inside each rank,
    # so rank-0 cells come first (strongest region first), then all rank-1, ...
    final = torch.argsort(within, stable=True)
    return order[final][:t_max]


_LSA_POOL = ThreadPoolExecutor(max_workers=8)
atexit.register(_LSA_POOL.shutdown, wait=False)
_BIG = 1e5


@torch.no_grad()
def build_match_cost(pred_xy, pred_score, pred_desc, tgt_xy, tgt_desc, tgt_valid,
                     lam_loc, lam_cls, lam_desc):
    """Build batched Hungarian cost matrix.

    Descriptors are L2-normalized internally so callers need not pre-normalize.
    """
    # all in fp32; coords normalized [0,1]
    pred_xy = pred_xy.float(); tgt_xy = tgt_xy.float()
    loc = torch.cdist(pred_xy, tgt_xy, p=1)                      # [B,Q,T]
    # single-class (landmark) cost: a per-query confidence penalty, identical across
    # target columns by design — loc+desc decide WHICH target.
    cls = (-torch.log(torch.sigmoid(pred_score).float() + 1e-6))[..., None]  # [B,Q,1]
    pred_desc = F.normalize(pred_desc.float(), dim=-1)
    tgt_desc = F.normalize(tgt_desc.float(), dim=-1)
    desc = 1.0 - torch.bmm(pred_desc, tgt_desc.transpose(1, 2))  # [B,Q,T]
    C = lam_loc * loc + lam_cls * cls + lam_desc * desc
    invalid = ~tgt_valid[:, None, :]                            # [B,1,T]
    return C.masked_fill(invalid, _BIG)


@torch.no_grad()
def hungarian_assign(cost, tgt_valid, use_threads=True):
    B, Q, T = cost.shape
    cnp = cost.detach().to('cpu', dtype=torch.float64).numpy()
    valid_np = tgt_valid.detach().cpu().numpy()

    def solve(b):
        r, c = linear_sum_assignment(cnp[b])
        keep = valid_np[b][c]                                   # drop padded targets
        return r[keep], c[keep]

    results = list(_LSA_POOL.map(solve, range(B))) if use_threads else [solve(b) for b in range(B)]
    col_idx = torch.full((B, Q), -1, dtype=torch.long)
    match_mask = torch.zeros((B, Q), dtype=torch.bool)
    for b, (r, c) in enumerate(results):
        col_idx[b, r] = torch.from_numpy(c)
        match_mask[b, r] = True
    return None, col_idx.to(cost.device), match_mask.to(cost.device)


def _warp_xy(xy, T):
    """Apply a 3x3 homogeneous warp to normalized xy [N,2] -> [N,2]."""
    ones = torch.ones(xy.shape[0], 1, dtype=xy.dtype, device=xy.device)
    h = torch.cat([xy, ones], dim=1) @ T.t()
    return h[:, :2] / h[:, 2:3].clamp(min=1e-6)


def random_bev_warp(max_rot_deg=30.0, max_trans_cells=30.0, scale_range=(0.9, 1.1),
                    rng=None, device=None):
    """Random in-plane BEV affine (R[2x2], t[2] in grid cells) for synthetic-homography
    covariance training. R = scale * rotation (optionally reflected). Deterministic if rng
    (a python Random) is given; else uses torch.rand (NOT available in workflow scripts but
    fine in training)."""
    import math
    if rng is not None:
        th = (rng.random() * 2 - 1) * max_rot_deg * math.pi / 180.0
        s = scale_range[0] + rng.random() * (scale_range[1] - scale_range[0])
        tx = (rng.random() * 2 - 1) * max_trans_cells
        ty = (rng.random() * 2 - 1) * max_trans_cells
    else:
        th = (torch.rand(()).item() * 2 - 1) * max_rot_deg * math.pi / 180.0
        s = scale_range[0] + torch.rand(()).item() * (scale_range[1] - scale_range[0])
        tx = (torch.rand(()).item() * 2 - 1) * max_trans_cells
        ty = (torch.rand(()).item() * 2 - 1) * max_trans_cells
    c, sn = math.cos(th), math.sin(th)
    R = torch.tensor([[s * c, -s * sn], [s * sn, s * c]], dtype=torch.float32, device=device)
    t = torch.tensor([tx, ty], dtype=torch.float32, device=device)
    return R, t


def warp_feature_map(feat, R, t, dim):
    """Spatially warp a feature map [C,H,W] so CONTENT moves frame0->frame1 by the affine
    (R,t) — i.e. content at cell c0 lands at cell c1 where (c0->c1) is exactly the mapping
    `generate_correspondence(dim,R,t)` produces. Used to build the synthetic second view for
    R2D2-style covariance training (the score head must warp the same way -> rewards
    content-based detection, penalizes fixed-frame / conv-boundary responses)."""
    from .keypoint_geometry import grid_warp_to_homography
    C, H, W = feat.shape
    Hmat = grid_warp_to_homography(R, t, dim).to(feat.device).float()    # forward, normalized xy
    Hinv = torch.linalg.inv(Hmat)
    ys, xs = torch.meshgrid(torch.arange(dim, device=feat.device),
                            torch.arange(dim, device=feat.device), indexing='ij')
    xy = torch.stack([xs.reshape(-1).float() / (dim - 1),
                      ys.reshape(-1).float() / (dim - 1)], dim=1)         # [HW,2] output (x,y)
    src = _warp_xy(xy, Hinv)                                             # sample-from input coords
    grid = (src * 2 - 1).reshape(1, dim, dim, 2)
    return F.grid_sample(feat[None].float(), grid, mode='bilinear',
                         align_corners=True, padding_mode='zeros')[0]


def persistence_gate(target_xy, neighbour_xy_list, T_list, tol_norm, k):
    """Keep targets that re-appear (within tol) in >= k spread-out neighbour frames."""
    counts = torch.zeros(target_xy.shape[0])
    for neigh_xy, T in zip(neighbour_xy_list, T_list):
        if neigh_xy.numel() == 0:
            continue
        warped = _warp_xy(target_xy, T)                      # current -> neighbour frame
        d = torch.cdist(warped, neigh_xy.float(), p=2)        # [Ntgt, Nneigh]
        counts += (d.min(dim=1).values <= tol_norm).float()
    return counts >= k
