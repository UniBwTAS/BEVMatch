"""Pure, dependency-light tensor ops for BEV keypoints.

No torch.nn, no torch.distributed, no SummaryWriter — every function takes
plain tensors plus explicit dim/bev_resolution so it is unit-testable without
building KeypointHead. Convention: (row, col), flat idx = row * dim + col.
"""
import torch
import torch.nn.functional as F


def _as_t(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.float64)
    return torch.as_tensor(x, dtype=torch.float64)


def compose_aug_warp(ego2global_A, ego2global_B, aug_current, aug_other,
                     bev_resolution):
    """Augmentation-aware BEV warp current->other, in grid units.

    W = aug_other @ inv(ego2global_A) @ ego2global_B @ inv(aug_current),
    z-killed, then R_grid = W[:2,:2], t_grid = W[:2,3] / bev_resolution.
    All matrices are metric 4x4 (lidar_aug_matrix = raw->aug). Returns
    (R_grid [2,2], t_grid [2]) as float32.
    """
    A = _as_t(ego2global_A)
    B = _as_t(ego2global_B)
    augc = _as_t(aug_current)
    augo = _as_t(aug_other)

    W = augo @ torch.inverse(A) @ B @ torch.inverse(augc)

    # Orthographic z-kill (drop the z row/col contribution), matching legacy.
    ortho = torch.eye(4, dtype=torch.float64)
    ortho[2, 2] = 0.0
    W = ortho @ W

    R_grid = W[:2, :2].to(torch.float32)
    t_grid = (W[:2, 3] / float(bev_resolution)).to(torch.float32)
    return R_grid, t_grid


def warp_points(points, H):
    """points [N,2] (row,col) homogeneous-warped by H [3,3]."""
    N = points.shape[0]
    ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)
    pts = torch.cat([points, ones], dim=1).permute(1, 0)
    pts = H.to(points.dtype) @ pts
    pts = pts[0:2, :] / pts[2:, :]
    return pts.permute(1, 0)


def generate_correspondence(dim, R, t):
    """Affine correspondence on a dim x dim grid. R is 2x2 (may include
    scale/reflection), t is length-2 (grid units). Centered at (dim-1)/2.
    Returns (corr0, corr1) flat idx = row*dim + col over bijective,
    in-bounds cells.
    """
    dim = int(dim)
    device = R.device if isinstance(R, torch.Tensor) else torch.device("cpu")
    grid_row, grid_col = torch.meshgrid(
        torch.arange(dim, device=device),
        torch.arange(dim, device=device),
        indexing="ij",
    )
    points0 = torch.stack([grid_row.flatten(), grid_col.flatten()], dim=1).float()

    SE2 = torch.eye(3, device=device, dtype=torch.float32)
    SE2[:2, :2] = R[:2, :2].to(device).float()
    SE2[:2, 2] = t[:2].to(device).float()
    invSE2 = torch.inverse(SE2)

    center = (dim - 1) / 2.0  # exact reflection center (cell-center convention)
    pts_centered = points0 - center

    # Round in the INTEGER grid frame, not the half-integer centered frame
    # (torch.round on x.5 is banker's rounding -> breaks even-dim identity).
    warped = torch.round(warp_points(pts_centered, invSE2) + center)
    pts_rec = torch.round(warp_points(warped - center, SE2) + center)

    warped = warped.long()
    pts_rec = pts_rec.long()
    points0 = points0.long()

    bij = (points0[:, 0] == pts_rec[:, 0]) & (points0[:, 1] == pts_rec[:, 1])
    src = points0[bij]
    tgt = warped[bij]
    ib = (tgt[:, 0] >= 0) & (tgt[:, 1] >= 0) & (tgt[:, 0] < dim) & (tgt[:, 1] < dim)
    src, tgt = src[ib], tgt[ib]

    corr0 = (src[:, 0] * dim + src[:, 1]).long()
    corr1 = (tgt[:, 0] * dim + tgt[:, 1]).long()
    return corr0, corr1


def windowed_correlation_unfold(desc0_flat, desc1, c0, c1, dim, k):
    """Legacy reference: uses F.unfold over ALL HW cells then indexes c1.

    desc0_flat: [HW, D] (L2-normalized). desc1: [D, H, W] (L2-normalized).
    c0, c1: [M] flat indices. Returns [M, k*k].
    Kept for equivalence testing; superseded by windowed_correlation (gather).
    """
    D = desc1.shape[0]
    pad = k // 2
    # Unfold frame1 into kxk windows per cell: [D*k*k, HW]
    unf = F.unfold(desc1.unsqueeze(0), kernel_size=k, padding=pad)  # [1, D*k*k, HW]
    unf = unf.squeeze(0).reshape(D, k * k, dim * dim)  # [D, k*k, HW]
    win = unf[:, :, c1]                                # [D, k*k, M]
    q = desc0_flat[c0].T                               # [D, M]
    # cosine; eps guards zero-padded border windows (zero-norm -> NaN otherwise)
    win = win / (win.norm(dim=0, keepdim=True) + 1e-8)
    q = q / (q.norm(dim=0, keepdim=True) + 1e-8)
    sims = torch.einsum("dm,dkm->mk", q, win)          # [M, k*k]
    return sims


def windowed_correlation(desc0_flat, desc1, c0, c1, dim, k):
    """Cosine sim of desc0_flat[c0] vs the kxk desc1 window around c1.

    Gather-based implementation: only extracts the M×k×k neighborhood cells
    instead of unfolding the full H×W grid (~6x less memory for dim=180,D=128,k=5).
    Numerically identical to windowed_correlation_unfold (same zero-padding semantics).

    desc0_flat: [HW, D] (L2-normalized). desc1: [D, H, W] (L2-normalized).
    c0, c1: [M] flat indices. Returns [M, k*k].
    """
    D = desc1.shape[0]
    pad = k // 2
    device = desc1.device

    # Zero-pad desc1 to match F.unfold's padding semantics
    desc1_p = F.pad(desc1, (pad, pad, pad, pad))  # [D, H+2pad, W+2pad]

    # Convert flat c1 indices to (row, col) in the UNPADDED grid
    r = c1 // dim  # [M]
    c = c1 % dim   # [M]

    # In the padded grid, the k×k window for cell (r,c) starts at (r, c)
    # (because padding shifts every cell by pad). Kernel offsets: 0..k-1 for both axes.
    # This exactly replicates F.unfold's row-major kernel order: ki = kh*k + kw.
    krange = torch.arange(k, device=device)
    rr = r[:, None, None] + krange[None, :, None]  # [M, k, 1]
    cc = c[:, None, None] + krange[None, None, :]  # [M, 1, k]

    # Gather: desc1_p[:, rr, cc] broadcasts to [D, M, k, k]
    win = desc1_p[:, rr, cc]             # [D, M, k, k]
    win = win.permute(0, 2, 3, 1)        # [D, k, k, M]  (row-major kernel order)
    win = win.reshape(D, k * k, -1)      # [D, k*k, M]   matching unfold's layout

    q = desc0_flat[c0].T                 # [D, M]
    # cosine; eps guards zero-padded border windows (zero-norm -> NaN otherwise)
    win = win / (win.norm(dim=0, keepdim=True) + 1e-8)
    q = q / (q.norm(dim=0, keepdim=True) + 1e-8)
    sims = torch.einsum("dm,dkm->mk", q, win)  # [M, k*k]
    return sims


def _window_argmax_and_contrast(sims, k):
    """sims [M, k*k] -> (is_center_argmax [M] bool, contrast [M])."""
    center = (k * k) // 2
    top2 = torch.topk(sims, k=min(2, sims.shape[1]), dim=1).values  # [M,2]
    top1 = top2[:, 0]
    second = top2[:, 1] if top2.shape[1] > 1 else top1
    win_min = sims.min(dim=1).values
    is_center = sims.argmax(dim=1) == center
    contrast = (top1 - second) / (top1 - win_min + 1e-6)
    return is_center, contrast


def build_geometric_targets(score_a, score_b, R, t, dim, k=5, valid_mask=None):
    """GEOMETRIC-ONLY keypoint targets (the SuperPoint-head fix).

    A cell is a target iff it is a local-max of score_a AND its warp-image cell
    (via the known affine (R,t)) is a local-max of score_b — i.e. the same point
    survives the real viewpoint change. The descriptor and feature-norm are NOT
    consulted (that is the saliency leak the prior head had via build_keypoint_targets).

    score_a, score_b: [H,W] in [0,1] (sigmoid). Returns y_flat [H*W] {0,1}.
    """
    import torch.nn.functional as F
    H, W = score_a.shape
    pad = k // 2

    def _local_max(s):
        # local-max AND strictly above the window mean -> excludes flat plateaus
        # (a flat region equals its own maxpool everywhere; the >mean test kills it).
        mx = F.max_pool2d(s[None, None], k, 1, pad)[0, 0]
        mean = F.avg_pool2d(s[None, None], k, 1, pad)[0, 0]
        return ((s == mx) & (s > mean)).reshape(-1)

    lm_a = _local_max(score_a)
    lm_b = _local_max(score_b)
    corr0, corr1 = generate_correspondence(dim, R, t)            # bijective flat idx
    corr0 = corr0.to(score_a.device); corr1 = corr1.to(score_a.device)  # R,t may be CPU
    keep = lm_a[corr0] & lm_b[corr1]                             # mutual local-max under warp
    if valid_mask is not None:
        keep = keep & valid_mask.reshape(-1)[corr0]
    y = torch.zeros(H * W, device=score_a.device)
    y[corr0[keep]] = 1.0
    return y


def build_geometric_targets_pair(score_a, score_b, R, t, dim, k=5, valid_mask=None,
                                 min_score=0.0, max_targets=None,
                                 desc_a=None, desc_b=None, reliability_weight=0.0):
    """Warp-consistent geometric targets for BOTH frames.

    Same mutual-local-max-under-warp criterion as build_geometric_targets, but returns
    (y0, y1) where y0 marks frame-A target cells and y1 marks their warp-image cells in
    frame B — guaranteed bijective (a target in A has its partner lit in B). This is the
    REPEATABILITY-driven replacement for build_keypoint_targets (which used descriptor
    correlation = saliency leak). score_a, score_b: [H,W] in [0,1]. Returns (y0,y1) flat.

    Over-fire guard (config D lesson: a noisy early score map has hundreds of local
    maxima -> dense scatter -> repeatability collapse):
      min_score:   a mutual-max must also score >= min_score in BOTH frames (kills weak
                   noise maxima; on an untrained map nearly all scores are low so few pass).
      max_targets: keep only the top-N mutual-maxima by min(score_a,score_b) (hard density
                   cap). None = unlimited (original behavior).
    """
    import torch.nn.functional as F
    H, W = score_a.shape
    pad = k // 2

    def _local_max(s):
        mx = F.max_pool2d(s[None, None], k, 1, pad)[0, 0]
        mean = F.avg_pool2d(s[None, None], k, 1, pad)[0, 0]
        return ((s == mx) & (s > mean)).reshape(-1)

    sa = score_a.reshape(-1); sb = score_b.reshape(-1)
    lm_a = _local_max(score_a)
    lm_b = _local_max(score_b)
    corr0, corr1 = generate_correspondence(dim, R, t)
    corr0 = corr0.to(score_a.device); corr1 = corr1.to(score_a.device)
    keep = lm_a[corr0] & lm_b[corr1]
    if valid_mask is not None:
        vm = valid_mask.reshape(-1)
        keep = keep & vm[corr0] & vm[corr1]
    # per-correspondence strength = min of the two endpoint scores
    strength = torch.minimum(sa[corr0], sb[corr1])
    if reliability_weight > 0.0 and desc_a is not None:
        desc_a = desc_a.detach().float()
        desc_b = desc_b.detach().float()
        D = desc_a.shape[0]
        da = torch.nn.functional.normalize(desc_a.reshape(D, -1).T, dim=1)  # [HW, D]
        db = torch.nn.functional.normalize(desc_b.reshape(D, -1).T, dim=1)
        # cosine between each A-cell's descriptor and its GEOMETRIC partner's descriptor in B
        rel = (da[corr0] * db[corr1]).sum(dim=1).clamp(min=0.0)   # [Ncorr] in [0,1]
        # blend: reliability_weight=0 -> pure strength; =1 -> strength*rel; interpolate
        strength = strength * ((1.0 - reliability_weight) + reliability_weight * rel)
    if min_score > 0.0:
        keep = keep & (strength >= min_score)
    if max_targets is not None and int(keep.sum()) > max_targets:
        kept_idx = keep.nonzero(as_tuple=True)[0]
        topv, topk = torch.topk(strength[kept_idx], max_targets)
        new_keep = torch.zeros_like(keep)
        new_keep[kept_idx[topk]] = True
        keep = new_keep
    y0 = torch.zeros(H * W, device=score_a.device)
    y1 = torch.zeros(H * W, device=score_a.device)
    y0[corr0[keep]] = 1.0
    y1[corr1[keep]] = 1.0
    return y0, y1


def plateau_safe_nms(prob, k=5):
    """Keep strict local maxima; deterministic tie-break via position epsilon.

    prob: [H,W] in [0,1]. Returns a bool keep-mask [H,W].
    """
    H, W = prob.shape
    device = prob.device
    # position epsilon so no two cells tie exactly (lowest linear index wins)
    lin = torch.arange(H * W, device=device, dtype=prob.dtype).reshape(H, W)
    eps = (1.0 / (H * W + 1.0))
    p = prob + eps * (1.0 - lin / (H * W))  # earlier index = larger bump
    pad = k // 2
    pooled = F.max_pool2d(p.unsqueeze(0).unsqueeze(0), k, stride=1, padding=pad)
    pooled = pooled.squeeze(0).squeeze(0)
    return p >= pooled  # strict max wins; ties broken by eps


def extract_keypoints(heatmap_logits, tau, cap=80, k=5):
    """sigmoid -> threshold tau -> plateau-safe NMS -> top-cap.

    Returns (coords [N,2] long (row,col), scores [N]). May be empty.
    """
    prob = torch.sigmoid(heatmap_logits.float())
    H, W = prob.shape
    keep = plateau_safe_nms(prob, k=k) & (prob > tau)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    if ys.numel() == 0:
        return (torch.zeros((0, 2), dtype=torch.long, device=prob.device),
                torch.zeros((0,), device=prob.device))
    sc = prob[ys, xs]
    if sc.numel() > cap:
        top = torch.topk(sc, k=cap).indices
        ys, xs, sc = ys[top], xs[top], sc[top]
    coords = torch.stack([ys, xs], dim=1).long()
    return coords, sc


def keypoint_valid_mask(dim, border=8, center_halfwidth=4, device=None):
    """Boolean [dim,dim] mask of cells where keypoints may fire / be supervised.

    False on (a) a `border`-cell frame around the BEV grid and (b) a full-height
    vertical band of half-width `center_halfwidth` over the center column (x=0).
    Both are FIXED sensor/projection artifacts — the conv zero-pad edges and the
    ego/LiDAR-origin axis — that sit at the same cells in every frame, so the
    repeatability/recall objective rewards detecting them. Masking them removes
    that degenerate shortcut and forces the head onto the real interior
    structure (proven distributed: ~67% of z-scored feature variance is interior
    while the head was firing 0% of keypoints there). True == allowed/interior.
    """
    m = torch.ones((dim, dim), dtype=torch.bool, device=device)
    b = int(border)
    if b > 0:
        m[:b, :] = False
        m[-b:, :] = False
        m[:, :b] = False
        m[:, -b:] = False
    c = int(center_halfwidth)
    if c > 0:
        cx = dim // 2
        m[:, max(0, cx - c):cx + c + 1] = False
    return m


def build_keypoint_targets(desc0, desc1, R, t, dim, k, rho, k_target, valid_flat=None):
    """Geometry-derived R2D2 targets via windowed correlation.

    desc0/desc1: [D,H,W] L2-normalized. Returns (y0, y1) flat [HW] in {0,1}.
    valid_flat: optional bool [HW] mask; correspondences whose endpoint falls on
    a masked (False) cell are never selected as targets — so the artifact cells
    cannot be supervised as keypoints.
    """
    device = desc0.device
    HW = dim * dim
    d0f = desc0.reshape(desc0.shape[0], -1).T  # [HW, D]
    d1f = desc1.reshape(desc1.shape[0], -1).T

    c0, c1 = generate_correspondence(dim, R.to(device), t.to(device))
    y0 = torch.zeros(HW, device=device)
    y1 = torch.zeros(HW, device=device)
    if c0.numel() == 0:
        return y0, y1

    # forward: desc0[c0] vs window around c1 in desc1
    sims_f = windowed_correlation(d0f, desc1, c0, c1, dim, k)
    ctr_f, con_f = _window_argmax_and_contrast(sims_f, k)
    # backward: desc1[c1] vs window around c0 in desc0
    sims_b = windowed_correlation(d1f, desc0, c1, c0, dim, k)
    ctr_b, con_b = _window_argmax_and_contrast(sims_b, k)

    keep = ctr_f & ctr_b & (con_f >= rho) & (con_b >= rho)
    if valid_flat is not None:
        vf = valid_flat.to(device)
        keep = keep & vf[c0] & vf[c1]   # drop correspondences on artifact cells
    if keep.sum() == 0:
        return y0, y1

    score = sims_f[:, (k * k) // 2]  # center self-similarity as confidence
    kept_idx = torch.nonzero(keep).squeeze(1)
    kept_score = score[kept_idx]
    if kept_idx.numel() > k_target:
        top = torch.topk(kept_score, k=k_target).indices
        kept_idx = kept_idx[top]

    y0[c0[kept_idx]] = 1.0
    y1[c1[kept_idx]] = 1.0
    return y0, y1


def build_superpoint_targets(cells, dim, sp_cell):
    """Teacher landmark (row,col) integer cells -> per-region class label map for the
    SuperPoint detector. Returns LongTensor [dim//sp_cell, dim//sp_cell] with values in
    [0, sp_cell**2]: a region's label is the SUB-INDEX (row%sp_cell)*sp_cell + (col%sp_cell)
    of a teacher landmark in it, or the DUSTBIN index sp_cell**2 if the region has none.
    On collision (>1 landmark in a region) the FIRST wins (cells order). cells: LongTensor
    [N,2] (row,col), may be empty. Out-of-range cells are ignored (teacher cells are interior)."""
    g = dim // sp_cell
    dustbin = sp_cell ** 2
    labels = torch.full((g, g), dustbin, dtype=torch.long)
    if cells.numel() == 0:
        return labels
    for i in range(cells.shape[0]):
        r = int(cells[i, 0].item())
        c = int(cells[i, 1].item())
        gr, gc = r // sp_cell, c // sp_cell
        if gr < 0 or gr >= g or gc < 0 or gc >= g:
            continue
        if labels[gr, gc].item() == dustbin:  # first-wins: only assign if still dustbin
            labels[gr, gc] = (r % sp_cell) * sp_cell + (c % sp_cell)
    return labels


def superpoint_logits_to_heatmap(sp_logits, sp_cell):
    """[B, sp_cell**2+1, g, g] raw logits -> dense [B,1,dim,dim] LOGIT heatmap, drop-in for
    the existing sigmoid-based extract_keypoints/eval. Softmax over channel dim, DROP the
    dustbin channel (last), take the sp_cell**2 keypoint-prob channels, PixelShuffle(sp_cell)
    to [B,1,g*sp_cell, g*sp_cell], then convert prob p -> logit log(p+eps)-log(1-p+eps) so
    downstream sigmoid() recovers p. eps=1e-6. Use torch.nn.functional.pixel_shuffle.
    NB PixelShuffle channel order: out[..,h*r+i, w*r+j] = in[.., i*r+j, h,w], which MATCHES
    the sub-index (row_off=i, col_off=j) used in build_superpoint_targets."""
    eps = 1e-6
    p = torch.softmax(sp_logits, dim=1)[:, :sp_cell ** 2]  # [B, sp_cell**2, g, g]
    dense_p = F.pixel_shuffle(p, sp_cell)  # [B, 1, dim, dim]
    return torch.log(dense_p + eps) - torch.log(1 - dense_p + eps)


def xy_to_cells(xy, dim, bev_res):
    """BEV metres (x,y, lidar frame) -> integer (row,col) cells. row=x, col=y, origin at center.

    Matches the INTRINSIC lidar-BEV voxelizer convention (verified by the synthetic point
    probe tools/check_voxel_axis.py: a point at lidar x=+30 m lights up BEV ROW=140, y=+30 m
    lights up BEV COL=141). The previous mapping (row=y, col=x) was TRANSPOSED relative to the
    feature field: it placed teacher targets on the diagonal-flipped cell, which a conv head
    cannot follow (a transpose is not a translation) -> it fell back to position priors
    (stripe artifacts, position-locked descriptor, stationary-only detection). gen_corr /
    compose_aug_warp already use this row=x convention (that is why the March self-supervised
    long-distance matching was stable)."""
    import torch
    half = dim / 2.0
    row = torch.round(xy[:, 0] / bev_res + half).long()  # row <- x
    col = torch.round(xy[:, 1] / bev_res + half).long()  # col <- y
    cells = torch.stack([row, col], 1).clamp(0, dim - 1)
    return cells


def render_gaussian_heatmap(cells, dim, sigma=2.0):
    """Render unit-peak Gaussians at integer (row,col) cells -> [dim,dim] in [0,1] (max-pooled overlap)."""
    import torch
    hm = torch.zeros(dim, dim, device=cells.device)
    if cells.numel() == 0:
        return hm
    ys = torch.arange(dim, device=cells.device)
    yy, xx = torch.meshgrid(ys, ys, indexing='ij')
    for r, c in cells.tolist():
        g = torch.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma ** 2))
        hm = torch.maximum(hm, g)
    return hm
