"""KITTI-Odometrie-Metriken (RTE %, RRE deg/100m, KITTI-Segmentprotokoll 100..800m) fuer die
BEV-SE(2)-VO-Caches — gegen die OFFIZIELLE KITTI-GT (sequences/poses/<seq>.txt, Kamera-Frame,
via Tr in den LiDAR-Frame transformiert) und optional gegen die SemanticKITTI/SuMa-Posen.

2D-Deklaration: unsere VO ist SE(2) im BEV; Fehler werden in der xy-Ebene berechnet (Standard in
der BEV-VO-Literatur). GT wird auf xy+Yaw projiziert.

Aufruf: python tools/kitti_rte_rre.py <cache_dir> <seq> [stride]
  cache_dir: enthaelt rel_poses.npy (VO, [x,y,theta] je strided Schritt) [+ traj_est.npy]
  seq: z.B. 09; stride: Frame-Stride des Laufs (default 1)
"""
import sys, os
import os
import numpy as np

ROOT = os.environ.get("KITTI_SEQ_ROOT", "data/kitti_odometry/dataset/sequences")
SEG_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]


def load_calib_Tr(seq):
    for line in open(f"{ROOT}/{seq}/calib.txt"):
        if line.startswith('Tr'):
            v = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            Tr = np.eye(4); Tr[:3, :] = v
            return Tr
    raise RuntimeError('Tr nicht gefunden')


def load_poses_cam(path, Tr):
    """KITTI-Posen (Kamera-Frame) -> LiDAR-Frame (wie create_kitti_odometry_infos)."""
    Tr_inv = np.linalg.inv(Tr)
    P = []
    for line in open(path):
        vals = np.array([float(x) for x in line.split()])
        if vals.size != 12:
            continue
        M = np.eye(4); M[:3, :] = vals.reshape(3, 4)
        P.append(Tr_inv @ M @ Tr)
    return np.stack(P, 0)


def se2_from_rel(rel):
    """rel_poses.npy [N,3] (x,y,theta) -> absolute SE(2)-Posen [N+1,3,3] (wie accumulate im VO)."""
    W = [np.eye(3)]
    for (x, y, th) in rel:
        T = np.array([[np.cos(th), -np.sin(th), x], [np.sin(th), np.cos(th), y], [0, 0, 1]])
        W.append(W[-1] @ np.linalg.inv(T))
    return np.stack(W, 0)


def kitti_errors(est_poses, gt_poses):
    """KITTI-Protokoll: fuer jedes Startframe und jede Segmentlaenge L in SEG_LENGTHS den
    Endframe per GT-Bogenlaenge finden; Fehler der Relativpose (est vs gt), normiert auf L."""
    gx = gt_poses[:, :2]
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gx, axis=0), axis=1))])
    t_errs, r_errs = [], []
    for i in range(len(gt_poses)):
        for L in SEG_LENGTHS:
            j = np.searchsorted(d, d[i] + L)
            if j >= len(gt_poses):
                break
            def rel(P, a, b):
                Ta = np.eye(3); Ta[:2, :2] = rot2(P[a, 2]); Ta[:2, 2] = P[a, :2]
                Tb = np.eye(3); Tb[:2, :2] = rot2(P[b, 2]); Tb[:2, 2] = P[b, :2]
                return np.linalg.inv(Ta) @ Tb
            Rg = rel(gt_poses, i, j); Re = rel(est_poses, i, j)
            E = np.linalg.inv(Rg) @ Re
            t_err = np.linalg.norm(E[:2, 2])
            r_err = abs(np.arctan2(E[1, 0], E[0, 0]))
            t_errs.append(t_err / L * 100.0)          # Prozent
            r_errs.append(np.degrees(r_err) / L * 100.0)  # deg pro 100m
    return (float(np.mean(t_errs)) if t_errs else float('nan'),
            float(np.mean(r_errs)) if r_errs else float('nan'), len(t_errs))


def rot2(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def kitti_errors_se3(est3, gt4):
    """SE(3)-embedded-Protokoll (DWPVO-konform): est-SE(2) wird mit z=0, Roll=Pitch=0 in SE(3)
    eingebettet und gegen die VOLLE 3D-GT (inkl. Hoehe) mit dem KITTI-Segmentprotokoll bewertet."""
    est3 = np.asarray(est3)
    if est3.ndim == 3:            # echte SE(3)-Posen (6-DoF-Lift)
        E4 = est3
    else:                          # SE(2)-Einbettung (z=0, Roll=Pitch=0)
        E4 = []
        for (x, y, th) in est3:
            T = np.eye(4); T[:2, :2] = rot2(th); T[0, 3] = x; T[1, 3] = y
            E4.append(T)
        E4 = np.stack(E4, 0)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt4[:, :3, 3], axis=0), axis=1))])
    t_errs, r_errs = [], []
    for i in range(len(gt4)):
        for L in SEG_LENGTHS:
            j = np.searchsorted(d, d[i] + L)
            if j >= len(gt4):
                break
            Rg = np.linalg.inv(gt4[i]) @ gt4[j]
            Re = np.linalg.inv(E4[i]) @ E4[j]
            E = np.linalg.inv(Rg) @ Re
            t_err = np.linalg.norm(E[:3, 3])
            c = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            r_err = np.arccos(c)
            t_errs.append(t_err / L * 100.0)
            r_errs.append(np.degrees(r_err) / L * 100.0)
    return (float(np.mean(t_errs)) if t_errs else float('nan'),
            float(np.mean(r_errs)) if r_errs else float('nan'), len(t_errs))


def to_xyyaw(P4):
    """[N,4,4] LiDAR-Frame -> [N,3] (x,y,yaw) in der BEV-Ebene."""
    xy = P4[:, :2, 3]
    yaw = np.arctan2(P4[:, 1, 0], P4[:, 0, 0])
    return np.column_stack([xy, yaw])


def umeyama_ate(est_xy, gt_xy, with_scale=False, return_scale=False):
    """ATE nach Umeyama-Ausrichtung.

    with_scale=False (Default) -> starre SE(2)-Ausrichtung, die geschaetzte Skala bleibt
    unangetastet. Das ist die richtige Variante fuer ein metrisches BEV-Gitter: die Skala
    kommt aus der Zellgroesse und dem LiDAR und darf nicht nachtraeglich gefittet werden.
    with_scale=True -> Sim(2), also mit geschaetztem Skalenfaktor; nur zum Vergleich, denn
    Baselines ohne metrische Referenz erhalten diese Freiheit ebenfalls nicht.
    """
    mu_e, mu_g = est_xy.mean(0), gt_xy.mean(0)
    E, G = est_xy - mu_e, gt_xy - mu_g
    U, S, Vt = np.linalg.svd(G.T @ E / len(E))
    D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = 1.0
    if with_scale and (E ** 2).sum():
        s = np.trace(np.diag(S) @ D) / (E ** 2).mean(0).sum()
    aligned = (s * (R @ E.T)).T + mu_g
    ate = float(np.sqrt(((aligned - gt_xy) ** 2).sum(1).mean()))
    return (ate, float(s)) if return_scale else ate


def main():
    cache, seq = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    est4 = None
    se3_path = os.path.join(cache, 'rel_se3.npy')
    if os.path.exists(se3_path):                              # 6-DoF-Lift-Cache
        rel4 = np.load(se3_path)
        W = [np.eye(4)]
        for T in rel4:
            W.append(W[-1] @ np.linalg.inv(T))
        est4 = np.stack(W, 0)
        est3 = np.column_stack([est4[:, 0, 3], est4[:, 1, 3],
                                np.arctan2(est4[:, 1, 0], est4[:, 0, 0])])
        print(f"[rte] 6-DoF-Cache: {len(est4)} SE(3)-Posen")
    else:
        rel = np.load(os.path.join(cache, 'rel_poses.npy'))
        est = se2_from_rel(rel)                               # [F,3,3]
        est3 = np.column_stack([est[:, 0, 2], est[:, 1, 2],
                                np.arctan2(est[:, 1, 0], est[:, 0, 0])])
    Tr = load_calib_Tr(seq)
    refs = {'official-GT': f"{ROOT}/poses/{seq}.txt", 'SuMa': f"{ROOT}/{seq}/poses.txt"}
    print(f"[rte] cache={cache} seq={seq} stride={stride} est-Frames={len(est3)}")
    kf = None
    kf_path = os.path.join(cache, 'kf_indices.npy')
    if os.path.exists(kf_path):
        kf = np.load(kf_path)
        print(f"[rte] adaptives Keyframing: {len(kf)} Keyframes")
    for name, path in refs.items():
        if not os.path.exists(path):
            print(f"[rte] {name}: fehlt"); continue
        gt_all = load_poses_cam(path, Tr)
        gt4 = gt_all[kf] if kf is not None else gt_all[::stride]
        n = min(len(gt4), len(est3))
        gt3 = to_xyyaw(gt4[:n]); e3 = est3[:n]
        # est ist relativ zum Startframe; GT ebenfalls auf Startframe normieren
        T0 = np.eye(3); T0[:2, :2] = rot2(gt3[0, 2]); T0[:2, 2] = gt3[0, :2]
        gt_n = []
        for p in gt3:
            T = np.eye(3); T[:2, :2] = rot2(p[2]); T[:2, 2] = p[:2]
            Tn = np.linalg.inv(T0) @ T
            gt_n.append([Tn[0, 2], Tn[1, 2], np.arctan2(Tn[1, 0], Tn[0, 0])])
        gt_n = np.array(gt_n)
        rte, rre, nseg = kitti_errors(e3, gt_n)
        ate = umeyama_ate(e3[:, :2], gt_n[:, :2])                      # SE(2), ohne Skalenfit
        ate_sim, sc = umeyama_ate(e3[:, :2], gt_n[:, :2], with_scale=True, return_scale=True)
        # SE(3)-Protokoll: echte 6-DoF-Posen falls vorhanden, sonst SE(2)-Einbettung
        gt4n = np.linalg.inv(gt4[0])[None] @ gt4[:n]
        rte3, rre3, _ = kitti_errors_se3(est4[:n] if est4 is not None else e3, gt4n)
        # DWPVO-Stil: SE(3)-Umeyama-Alignment (ohne Skale) der GANZEN Trajektorie VOR der
        # Segmentmetrik — kippt die planare Trajektorie in die mittlere Steigungsebene.
        Epos = (est4[:n, :3, 3] if est4 is not None
                else np.column_stack([e3[:, 0], e3[:, 1], np.zeros(n)]))
        Gpos = gt4n[:, :3, 3]
        muE, muG = Epos.mean(0), Gpos.mean(0)
        U, S, Vt = np.linalg.svd((Gpos - muG).T @ (Epos - muE))
        D3 = np.eye(3); D3[2, 2] = np.sign(np.linalg.det(U @ Vt))
        Ral = U @ D3 @ Vt
        tal = muG - Ral @ muE
        Tal = np.eye(4); Tal[:3, :3] = Ral; Tal[:3, 3] = tal
        if est4 is not None:
            Eal = Tal[None] @ est4[:n]
        else:
            Eal = []
            for (x, y, th) in e3:
                T = np.eye(4); T[:2, :2] = rot2(th); T[0, 3] = x; T[1, 3] = y
                Eal.append(Tal @ T)
            Eal = np.stack(Eal, 0)
        rteA, rreA, _ = kitti_errors_se3(Eal, gt4n)
        print(f"[rte] {name:12s} n={n:5d}  2D: RTE={rte:6.2f}% RRE={rre:6.3f}  "
              f"SE3-emb: RTE={rte3:6.2f}% RRE={rre3:6.3f}  "
              f"SE3-aligned: RTE={rteA:6.2f}% RRE={rreA:6.3f}  "
              f"ATE2D(SE2)={ate:6.2f} m  [Sim2={ate_sim:6.2f} m, Skala={sc:.4f}]  (Seg={nseg})")


if __name__ == '__main__':
    main()
