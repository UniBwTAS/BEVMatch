"""LiDAR-Referenzbaselines (L2L) fuer beliebige mmdet3d-Infos -- ohne mmdet3d, reine CPU.

Zweck: Tab. II (in-house / Waymo) enthaelt bisher nur unsere eigenen Zahlen; Reviewer 8 P1
bemaengelt die fehlenden Baselines. Dieses Skript erzeugt sie mit exakt den Einstellungen, mit
denen die nuScenes-Zahlen der Tab. I entstanden sind (`bevfusion/evaluator.py`):

    voxel_size                     0.1 m
    max_correspondence_distance    0.5 m
    GICP                           registration_generalized_icp, Start = Identitaet (kein
                                   GT-Vorwissen), max_iteration 200, rel_fitness/rmse 1e-6
    FPFH                           Normalen r=2*voxel, FPFH r=5*voxel/max_nn=100, danach
                                   Fast Global Registration, dist_thresh = 1.5*voxel,
                                   iteration_number 64

Die Paarbildung entspricht `_find_random_sample_in_range` des Dataloaders: Vorwaertssuche
ueber hoechstens 150 Frames, |dt| <= 20 s, Distanz-/Winkelfenster je Bin, zufaellige Wahl unter
den Kandidaten mit festem Seed. Timestamps in Mikrosekunden (Waymo) werden erkannt und
umgerechnet.

Gemessen wird im Paper-Protokoll: RTE/RRE nur ueber ERFOLGREICHE Registrierungen (Erfolg =
Translationsfehler < |GT-Translation| + 0.1 m), Recall ueber alle Paare bei 0.5m/5deg und
2m/10deg.

Aufruf:
  python tools/baseline_lidar_cpu.py --infos <val.pkl> --data-root <dir> \
      --methods gicp,fpfh --bins close,far --workers 32 --out lidar_inhouse.json
"""
import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import open3d as o3d

BINS = {
    'close': (0.0, 5.0, 10.0),
    'mid': (5.0, 10.0, 30.0),
    'far': (10.0, 20.0, 50.0),
}
MAX_OFFSET, MAX_DT_S = 150, 20.0
VOXEL = 0.1
ICP_MAX_CORR = 0.5
PC_RANGE = 54.0          # wie das Modell: +-54 m, damit die Baseline denselben Ausschnitt sieht

_G = {}


def load_frames(infos_path):
    with open(infos_path, 'rb') as fh:
        info = pickle.load(fh)
    dl = info.get('data_list', info) if isinstance(info, dict) else info
    out = []
    for it in dl:
        lp = it.get('lidar_points') or {}
        if not lp.get('lidar_path'):
            continue
        # Waymo-Infos fuehren kein lidar2ego. Da fuer die RELATIVE Pose beide Frames identisch
        # behandelt werden, ist die Identitaet hier zulaessig: eine feste Zusatztransformation
        # kuerzt sich in inv(L_a) @ L_b nicht heraus, veraendert aber beide Seiten gleich und
        # entspricht damit der Konvention, die auch das Modell auf diesem Datensatz benutzt.
        l2e = lp.get('lidar2ego')
        out.append(dict(
            ego2global=np.asarray(it['ego2global'], np.float64),
            timestamp=float(it.get('timestamp', len(out))),
            lidar_path=lp['lidar_path'],
            num_feats=int(lp.get('num_pts_feats', 5)),
            lidar2ego=np.asarray(l2e, np.float64) if l2e is not None else np.eye(4),
        ))
    return out


def rel_pose_lidar(a, b):
    la = a['ego2global'] @ a['lidar2ego']
    lb = b['ego2global'] @ b['lidar2ego']
    return np.linalg.inv(la) @ lb


def build_pairs(frames, bin_name, seed=0):
    lo, hi, amax = BINS[bin_name]
    rng = np.random.RandomState(seed)
    n = len(frames)
    sec = np.array([f['timestamp'] for f in frames])
    sec = np.where(sec > 1e10, sec / 1e6, sec)          # Waymo liefert Mikrosekunden
    # Vektorisiert wie in tools/count_eval_pairs.py: die Schleifenvariante braucht bei 16k+
    # Frames mit 150 Offsets Minuten, weil sie pro Kandidat eine 4x4-Inversion rechnet.
    # Der Abstand zweier Posen ist rotationsinvariant, also genuegen die globalen xy-Positionen;
    # der Relativwinkel ist die (umlaufkorrigierte) Yaw-Differenz.
    poses = np.stack([f['ego2global'] for f in frames])
    xy = poses[:, :2, 3]
    yaw = np.degrees(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    pairs = []
    for i in range(n):
        top = min(i + MAX_OFFSET, n - 1)
        if top <= i:
            continue
        j = np.arange(i + 1, top + 1)
        ok = np.abs(sec[j] - sec[i]) <= MAX_DT_S
        if not ok.any():
            continue
        j = j[ok]
        d = np.linalg.norm(xy[j] - xy[i], axis=1)
        a = np.abs((yaw[j] - yaw[i] + 180.0) % 360.0 - 180.0)
        sel = j[(d >= lo) & (d <= hi) & (a <= amax)]
        if sel.size:
            pairs.append((i, int(rng.choice(sel))))
    return pairs


def load_cloud(frame, data_root):
    p = frame['lidar_path']
    base = os.path.basename(p)
    for cand in (os.path.join(data_root, p),
                 os.path.join(data_root, 'samples', 'LIDAR_TOP', base),
                 os.path.join(data_root, 'training', 'velodyne', base),
                 os.path.join(data_root, 'velodyne', base), p):
        if os.path.exists(cand):
            p = cand
            break
    else:
        raise FileNotFoundError(frame['lidar_path'])
    pts = np.fromfile(p, dtype=np.float32)
    f = frame['num_feats']
    pts = pts.reshape(-1, f)[:, :3].astype(np.float64)
    m = (np.abs(pts[:, 0]) <= PC_RANGE) & (np.abs(pts[:, 1]) <= PC_RANGE)
    return pts[m]


def to_pcd(pts, down=True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd.voxel_down_sample(VOXEL) if down else pcd


def reg_gicp(src, dst):
    """Generalized ICP ab Identitaet -- kein Vorwissen ueber die Pose."""
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 2, max_nn=30))
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 2, max_nn=30))
    res = o3d.pipelines.registration.registration_generalized_icp(
        src, dst, ICP_MAX_CORR, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200, relative_fitness=1e-6, relative_rmse=1e-6))
    return np.asarray(res.transformation)


def reg_fpfh(src, dst):
    """FPFH-Merkmale + Fast Global Registration, wie im Evaluator."""
    for p in (src, dst):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 2, max_nn=30))
    fs = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 5, max_nn=100))
    fd = o3d.pipelines.registration.compute_fpfh_feature(
        dst, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 5, max_nn=100))
    res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src, dst, fs, fd,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=VOXEL * 1.5,
            iteration_number=64, maximum_tuple_count=1000))
    return np.asarray(res.transformation)


def reg_ransac(src, dst):
    """FPFH-Merkmale + RANSAC -- die Variante, die der Tabellenname 'R.-FPFH' behauptet.

    Unterschied zu FGR: RANSAC zieht wiederholt zufaellige Korrespondenz-Tripel, bewertet jede
    Hypothese ueber ihre Inlier-Zahl und behaelt die beste -- Ausreisser werden also verworfen.
    FGR zieht keine Hypothesen, sondern minimiert eine robuste Zielfunktion (Geman-McClure) mit
    schrittweise verschaerfter Toleranz, gewichtet Ausreisser also herunter. RANSAC ist
    stochastisch und langsamer, liefert dafuer eine explizite Inlier-Menge.
    Parameter wie im Open3D-Referenzablauf: dist = 1.5*voxel, ransac_n = 3, Kanten- und
    Distanzpruefung, Abbruch bei 100k Iterationen / Konfidenz 0.999.
    """
    for p_ in (src, dst):
        p_.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 2, max_nn=30))
    fs = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 5, max_nn=100))
    fd = o3d.pipelines.registration.compute_fpfh_feature(
        dst, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 5, max_nn=100))
    thr = VOXEL * 1.5
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, dst, fs, fd, True, thr,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thr)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return np.asarray(res.transformation)


def _init(infos_path, data_root, methods, threads, voxel):
    global VOXEL
    VOXEL = voxel
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    _G['frames'] = load_frames(infos_path)
    _G['root'] = data_root
    _G['methods'] = methods


def _run_pair(task):
    bin_name, i, j = task
    frames, root = _G['frames'], _G['root']
    try:
        a = to_pcd(load_cloud(frames[i], root))
        b = to_pcd(load_cloud(frames[j], root))
    except Exception:
        return None
    T_gt = rel_pose_lidar(frames[i], frames[j])
    gt_t = T_gt[:2, 3]
    gt_a = np.degrees(np.arctan2(T_gt[1, 0], T_gt[0, 0]))
    gt_d = float(np.linalg.norm(gt_t))
    out = {}
    for m in _G['methods']:
        try:
            T = {'gicp': reg_gicp, 'fpfh': reg_fpfh, 'ransac': reg_ransac}[m](b, a)
        except Exception:
            continue
        te = float(np.linalg.norm(T[:2, 3] - gt_t))
        re = float(abs((np.degrees(np.arctan2(T[1, 0], T[0, 0])) - gt_a + 180) % 360 - 180))
        out[m] = (te, re, gt_d)
    return (bin_name, out) if out else None


def summarize(per_bin):
    summary = {}
    for b, per in per_bin.items():
        summary[b] = {}
        for meth, vals in per.items():
            e = np.array(vals)
            succ = e[:, 0] < e[:, 2] + 0.1          # Erfolgskriterium des Papers
            es = e[succ] if succ.any() else e
            summary[b][meth] = dict(
                n=len(e), n_success=int(succ.sum()),
                RTE=float(es[:, 0].mean()), RRE=float(es[:, 1].mean()),
                recall_05_5=float(100 * np.mean((e[:, 0] < 0.5) & (e[:, 1] < 5))),
                recall_2_10=float(100 * np.mean((e[:, 0] < 2.0) & (e[:, 1] < 10))))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infos', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--methods', default='gicp,fpfh', help='gicp | fpfh (FGR) | ransac (RANSAC+FPFH)')
    ap.add_argument('--bins', default='close,far')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0, help='Paare je Bin (Stichprobe)')
    ap.add_argument('--voxel', type=float, default=0.1,
                    help='Voxelgroesse. 0.1 wie bei nuScenes; dichtere Sensoren (VLS-128, Waymo) '
                         'brauchen groebere Werte, damit FPFH auf eine vergleichbare Punktzahl kommt')
    ap.add_argument('--shard', default='')
    ap.add_argument('--out', default='baseline_lidar_results.json')
    a = ap.parse_args()

    global VOXEL
    VOXEL = a.voxel
    methods = [m.strip() for m in a.methods.split(',') if m.strip()]
    frames = load_frames(a.infos)
    print(f'{len(frames)} Frames mit LiDAR', flush=True)

    tasks = []
    for b in [x.strip() for x in a.bins.split(',') if x.strip()]:
        pr = build_pairs(frames, b)
        if a.limit and len(pr) > a.limit:
            # gleichmaessig ueber die Sequenz ausduennen statt die ersten N zu nehmen,
            # sonst deckt die Stichprobe nur den Anfang der Aufnahme ab
            idx = np.linspace(0, len(pr) - 1, a.limit).astype(int)
            pr = [pr[k] for k in idx]
        print(f'  {b}: {len(pr)} Paare', flush=True)
        tasks += [(b, i, j) for i, j in pr]
    if a.shard:
        i, n = (int(x) for x in a.shard.split('/'))
        tasks = tasks[i::n]
        print(f'  Shard {i}/{n}: {len(tasks)} Paare', flush=True)

    per_bin = {b: {m: [] for m in methods} for b in BINS}
    if a.workers > 1:
        with Pool(a.workers, initializer=_init,
                  initargs=(a.infos, a.data_root, methods, a.threads, a.voxel)) as pool:
            for k, res in enumerate(pool.imap_unordered(_run_pair, tasks, chunksize=4)):
                if res:
                    b, out = res
                    for m, v in out.items():
                        per_bin[b][m].append(v)
                if (k + 1) % 200 == 0:
                    print(f'  {k+1}/{len(tasks)}', flush=True)
    else:
        # Sequenziell. Open3D parallelisiert GICP/FPFH intern ueber OpenMP, und ein
        # fork-basierter Prozess-Pool blockierte hier reproduzierbar (Jobs liefen Stunden
        # ohne eine einzige Iteration). Die Parallelitaet kommt stattdessen ueber --shard
        # aus mehreren Slurm-Jobs, was zusaetzlich gegen Einzeljob-Ausfaelle robust ist.
        _init(a.infos, a.data_root, methods, a.threads, a.voxel)
        t_start = time.time()
        for k, task in enumerate(tasks):
            res = _run_pair(task)
            if res:
                b, out = res
                for m, v in out.items():
                    per_bin[b][m].append(v)
            if (k + 1) % 50 == 0:
                el = time.time() - t_start
                print(f'  {k+1}/{len(tasks)}  {el/(k+1):.2f}s/Paar  '
                      f'ETA {(len(tasks)-k-1)*el/(k+1)/60:.0f} min', flush=True)

    per_bin = {b: {m: v for m, v in per.items() if v} for b, per in per_bin.items()}
    per_bin = {b: per for b, per in per_bin.items() if per}
    summary = summarize(per_bin)
    with open(a.out, 'w') as fh:
        json.dump(dict(summary=summary, raw={b: {m: v for m, v in p.items()}
                                             for b, p in per_bin.items()}), fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'-> {a.out}', flush=True)


if __name__ == '__main__':
    main()
