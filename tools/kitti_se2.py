"""Pure-geometry SE(2) helpers for KITTI keypoint odometry (no GPU / no dataset)."""
import numpy as np
import cv2


def se2(x, y, theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]], dtype=np.float64)


def se2_inv(T):
    R = T[:2, :2]; t = T[:2, 2]
    Ti = np.eye(3, dtype=np.float64)
    Ti[:2, :2] = R.T
    Ti[:2, 2] = -R.T @ t
    return Ti


def pose_to_xytheta(T):
    """SE(2) matrix -> (x, y, theta) edge/measurement vector."""
    return np.array([T[0, 2], T[1, 2], float(np.arctan2(T[1, 0], T[0, 0]))])


def ransac_se2(xy0, xy1, thr_m):
    """Full rigid SE(2) mapping xy0 -> xy1 (scale from the partial-affine similarity is
    dropped). Returns (T[3,3], inlier_mask) or (None, None)."""
    xy0 = np.asarray(xy0, np.float32); xy1 = np.asarray(xy1, np.float32)
    if xy0.shape[0] < 3:
        return None, None
    M, inl = cv2.estimateAffinePartial2D(
        xy0, xy1, method=cv2.RANSAC, ransacReprojThreshold=thr_m,
        maxIters=5000, confidence=0.999)
    if M is None:
        return None, None
    a, b = M[0, 0], M[0, 1]                     # [s*cosθ, -s*sinθ]
    theta = np.arctan2(-b, a)                    # scale-independent angle
    T = se2(float(M[0, 2]), float(M[1, 2]), float(theta))
    mask = inl.ravel().astype(bool) if inl is not None else np.zeros(xy0.shape[0], bool)
    return T, mask


def accumulate(rel_T_list):
    W = [np.eye(3, dtype=np.float64)]
    for T in rel_T_list:
        W.append(W[-1] @ se2_inv(np.asarray(T, np.float64)))
    return np.stack(W, 0)


def positions(world_poses):
    return np.asarray(world_poses, np.float64)[:, :2, 2].copy()


def umeyama_align_2d(est_xy, gt_xy):
    """Rigid 2D (rotation+translation, no scale) alignment of est onto gt. Returns
    (aligned_est, ate_rmse)."""
    est = np.asarray(est_xy, np.float64); gt = np.asarray(gt_xy, np.float64)
    mu_e = est.mean(0); mu_g = gt.mean(0)
    E = est - mu_e; G = gt - mu_g
    H = E.T @ G
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = mu_g - R @ mu_e
    aligned = (R @ est.T).T + t
    ate = float(np.sqrt(((aligned - gt) ** 2).sum(1).mean()))
    return aligned, ate
