"""Pure keypoint loss terms (all on logits). No nn.Module, no dist."""
import torch
import torch.nn.functional as F


def recall_hinge_loss(h_flat, y_flat, n):
    """Encourage >= n_eff targets to fire. Zero if no targets.

    Uses a DIFFERENTIABLE soft count (sum of sigmoid), NOT a hard
    (sigmoid>0.5) threshold — the hard version has zero gradient and would
    train nothing (the spec's primary anti-collapse term must be live).
    h_flat, y_flat: [HW] logits / {0,1} targets.

    IMPORTANT: call with n >= k_target (the target builder's cap). If n <
    num_targets, n_eff caps below num_targets and a UNIFORM sigmoid=0.5 map
    satisfies the hinge (soft_fired = 0.5*num_t >= n_eff), giving zero gradient
    — the head then sits on the flat 0.5 plateau. With n = k_target,
    n_eff = num_t and the plateau yields recall = 0.5*num_t > 0, a real
    gradient pushing every geometry target toward sigmoid->1.
    """
    num_t = int(y_flat.sum().item())
    if num_t == 0:
        return h_flat.new_zeros(())
    n_eff = min(n, num_t)
    tgt_logits = h_flat[y_flat > 0.5]
    soft_fired = torch.sigmoid(tgt_logits).sum()  # differentiable count
    return F.relu(h_flat.new_tensor(float(n_eff)) - soft_fired)


def _gather_windows(x, k):
    """x [H,W] -> [HW, k*k] logit windows (zero-pad)."""
    H, W = x.shape
    pad = k // 2
    unf = F.unfold(x.unsqueeze(0).unsqueeze(0), kernel_size=k, padding=pad)
    return unf.squeeze(0).T  # [HW, k*k]


def repeatability_cosine_loss(h0, h1, c0, c1, dim, k, eps=1e-6):
    """1 - mean per-window cosine between h0 windows at c0 and h1 windows at c1.

    Constant windows -> centered vector ~0 -> cosine ~0 -> loss ~1 (worst).
    """
    if c0.numel() == 0:
        return h0.new_zeros(())
    w0 = _gather_windows(h0, k)[c0]  # [M, k*k]
    w1 = _gather_windows(h1, k)[c1]
    w0 = w0 - w0.mean(dim=1, keepdim=True)
    w1 = w1 - w1.mean(dim=1, keepdim=True)
    num = (w0 * w1).sum(dim=1)
    den = w0.norm(dim=1) * w1.norm(dim=1) + eps
    cos = num / den
    return (1.0 - cos).mean()


def peakiness_ce_loss(h, targets_flat, dim, k):
    """Cross-entropy that the center is the argmax of its k*k logit window,
    evaluated only at target cells.
    """
    idx = torch.nonzero(targets_flat > 0.5).squeeze(1)
    if idx.numel() == 0:
        return h.new_zeros(())
    win = _gather_windows(h, k)[idx]   # [T, k*k]
    center = (k * k) // 2
    logp = F.log_softmax(win, dim=1)
    return -logp[:, center].mean()


def peakiness_maxmean_loss(score, valid_mask=None, k=5):
    """R2D2 peakiness anti-collapse: reward each k*k window's (max - mean).

    score: [H,W] in [0,1] (already sigmoid'd). Loss = -mean_windows(max - mean),
    so a flat/smooth map (max~=mean) is the WORST and a locally-peaked map is best.
    This is the primary anti-collapse term; without it a pure repeatability loss
    collapses to a constant/blob map (perfectly "repeatable", useless). Optional
    valid_mask [H,W] bool restricts the window-center set to interior cells.
    """
    win = _gather_windows(score, k)               # [HW, k*k]
    peaky = win.max(dim=1).values - win.mean(dim=1)   # [HW]
    if valid_mask is not None:
        m = valid_mask.reshape(-1).float()
        return -(peaky * m).sum() / m.sum().clamp(min=1.0)
    return -peaky.mean()


def repulsion_loss_logits(h, coords, min_distance=4.0):
    """Penalize co-activation of nearby strong cells (logit space).

    h: [H,W] logits. coords: [N,2] candidate peak locations (long).
    """
    if coords.shape[0] < 2:
        return h.new_zeros(())
    xy = coords.float()
    d = torch.cdist(xy, xy)  # [N,N]
    near = (d < min_distance).float() - torch.eye(coords.shape[0], device=h.device)
    near = near.clamp(min=0.0)
    s = torch.sigmoid(h[coords[:, 0], coords[:, 1]])
    pair = s.unsqueeze(0) * s.unsqueeze(1)
    denom = near.sum().clamp(min=1.0)
    return (near * pair).sum() / denom


def supervised_focal_loss(score_logits, target_hm, alpha=2.0, beta=4.0, eps=1e-6):
    """CenterNet/CornerNet penalty-reduced focal loss for supervised landmark training.

    Trains student's detection head against teacher's landmark heatmap.
    score_logits, target_hm: [H,W] logits and {0,1} target heatmap.
    alpha, beta: focal exponents (positive/negative hard-negative penalty).
    """
    # Cast to float32 to prevent bf16 AMP overflow in log-based loss computation
    score_logits = score_logits.float()
    target_hm = target_hm.float()
    p = torch.sigmoid(score_logits).clamp(eps, 1 - eps)
    pos = (target_hm == 1.0).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target_hm) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def superpoint_detection_loss(sp_logits, target_labels, dustbin_weight=1.0):
    """Cross-entropy detector loss. sp_logits [C, g, g] (C = sp_cell**2+1, raw logits for ONE
    sample), target_labels [g, g] long in [0,C-1]. Returns scalar = F.cross_entropy over the
    g*g regions (mean). Cast logits to .float() first (bf16-safety, same as supervised_focal_loss).

    dustbin_weight (<1.0) down-weights the DUSTBIN class (last index C-1 = the "no-keypoint"
    region) in the CE. ~96% of regions are dustbin, so the unweighted CE is dustbin-dominated and
    the head under-predicts keypoints (tiny heatmap scores, low recall). 1.0 = original behaviour."""
    C = sp_logits.shape[0]
    weight = None
    if dustbin_weight != 1.0:
        weight = torch.ones(C, device=sp_logits.device, dtype=torch.float32)
        weight[C - 1] = dustbin_weight
    return F.cross_entropy(sp_logits.float().reshape(C, -1).T, target_labels.reshape(-1),
                           weight=weight)


def binary_selfsup_descriptor_loss(z_a, z_b_pos, z_b_neg, margin=0.4, conf_weight=0.05,
                                    local_neg_cells=None):
    """Self-sup binary descriptor loss with REAL far-cell negatives + confidence.

    z_a:     [K, D] frame-A raw logits at corr cells.
    z_b_pos: [K, D] frame-B raw logits at the MATCHED corr cells.
    z_b_neg: [M, D] frame-B raw logits at random FAR (non-corr) cells (negative pool).
    local_neg_cells: optional [L, D] frame-B raw logits at cells SPATIALLY-NEAR the
        correspondences (within a small radius) but NOT themselves correspondences.
        Added to the negative pool so the hardest-negative mining must push each
        anchor away from its own near-identical neighbourhood — the flat/featureless
        discrimination signal. None (default) -> byte-identical to prior behaviour.

    soft-Hamming hd(x,y) = 0.5*(1 - cos(tanh x, tanh y)).
    Positives pulled to 0; each anchor pushed >margin from its HARDEST (nearest)
    negative in the pool; confidence pushes |tanh(z)|->1. Returns scalar.

    Cast to float32 for bf16-safety.
    """
    K = z_a.shape[0]
    if K < 1:
        _extra = local_neg_cells.sum() if local_neg_cells is not None else 0.0
        return (z_a.sum() + z_b_pos.sum() + z_b_neg.sum() + _extra) * 0.0

    # bf16-safe: work in float32
    z_a = z_a.float()
    z_b_pos = z_b_pos.float()
    z_b_neg = z_b_neg.float()

    # Local/flat hard-negative mining: augment the far-cell negative pool with the
    # spatially-near non-correspondence cells. In flat areas these are geometrically
    # near-identical to the anchor's true match, so making them negatives forces the
    # descriptor to find whatever subtle distinguishing signal exists.
    if local_neg_cells is not None and local_neg_cells.shape[0] >= 1:
        z_b_neg = torch.cat([z_b_neg, local_neg_cells.float()], dim=0)

    a = torch.tanh(z_a)           # [K, D]
    bp = torch.tanh(z_b_pos)      # [K, D]
    bn = torch.tanh(z_b_neg)      # [M, D]

    an = F.normalize(a, dim=1)
    bpn = F.normalize(bp, dim=1)
    bnn = F.normalize(bn, dim=1)

    pos = (0.5 * (1.0 - (an * bpn).sum(1))).mean()   # [K] soft-Hamming to matched

    # Hardest negative per anchor: min soft-Hamming over the negative pool.
    # If the pool is empty (e.g. identity pose on tiny BEV grid where all cells
    # are correspondences), skip the neg term — pos + conf still train the head.
    M = bn.shape[0]
    if M >= 1:
        sim = an @ bnn.T                               # [K, M] cosine
        hd_neg = 0.5 * (1.0 - sim)                    # [K, M] soft-Hamming
        hardest = hd_neg.min(dim=1).values             # [K] nearest (hardest) negative
        neg = F.relu(margin - hardest).mean()
    else:
        neg = z_a.new_zeros(())

    conf = (1.0 - a.abs()).mean()                      # push |tanh(z)| -> 1

    return pos + neg + conf_weight * conf


def binary_descriptor_metric_loss(z_a, z_b, match, margin=0.4):
    """Metric loss for binary descriptors via soft-Hamming on track correspondences.

    Pulls matched keypoints (same track_id) together, pushes mismatches apart via margin.
    z_a, z_b: [N,D] and [M,D] logits (pre-binarization, tanh domain).
    match: [K,2] LongTensor of (i in a, j in b) pairs with the same track_id.
    margin: margin for negative hard samples (default 0.4).

    Returns: scalar loss. Tanh-soft-codes descriptors; uses normalized soft-Hamming
    = 0.5*(1 - cosine_similarity). Positives -> 0, negatives -> >margin via in-batch permutation.
    """
    if match.numel() == 0:
        return (z_a.sum() + z_b.sum()) * 0.0

    a = torch.tanh(z_a)
    b = torch.tanh(z_b)
    ai, bj = match[:, 0], match[:, 1]

    def hd(x, y):
        return 0.5 * (1 - F.cosine_similarity(x, y, dim=-1))

    pos = hd(a[ai], b[bj]).mean()  # same track -> 0
    perm = torch.randperm(len(bj), device=z_a.device)
    neg = F.relu(margin - hd(a[ai], b[bj][perm])).mean()  # different -> > margin
    return pos + neg
