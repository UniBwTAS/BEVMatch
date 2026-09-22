"""Eigenstaendige Auswertung der Bild-Baselines (SC2SC) -- ohne mmdet3d, CPU-tauglich.

Warum eigenstaendig: die Bild-Baselines brauchen weder das BEVFusion-Modell noch die
Punktwolken. Der regulaere Eval-Pfad laedt beides und verlangt CUDA, obwohl fuer
SIFT/SuperPoint/LoFTR/DeDoDe nur ein Kamerabildpaar, die Intrinsik und die GT-Pose noetig
sind. Ohne diesen Ballast laeuft die Auswertung auf freien CPU-Kernen und laesst sich ueber
Prozesse parallelisieren (DeDoDe skaliert NICHT ueber Threads -- 32 Threads sind langsamer
als 8, deshalb Prozess- statt Thread-Parallelitaet).

Nachgebildet werden exakt:
  * Paarbildung wie *_dataset_keypoints.py::_find_random_sample_in_range
    (Vorwaerts-Suche <=150 Frames, |dt| <= 20 s, Distanz-/Winkelfenster je Bin,
     zufaellige Wahl unter den Kandidaten -- hier mit festem Seed)
  * Bildvorverarbeitung wie ImageAug3D im Testmodus
    (resize 0.48 -> crop; die Intrinsik wird identisch mittransformiert)
  * Posenschaetzung wie im PoseEvaluator: gegenseitiges NN-Matching, Essentialmatrix,
    recoverPose, Skala aus der GT-Translation, Transformation ins LiDAR-Frame

Aufruf:
  python tools/baseline_matchers_cpu.py --infos <val.pkl> --data-root <dir> \
      --methods dedode,superpoint,loftr --bins close,far --workers 16 --out results.json
"""
import argparse
import json
import os
import pickle
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from PIL import Image

BINS = {                       # Name -> (min_dist, max_dist, max_angle_deg)
    'close': (0.0, 5.0, 10.0),
    'mid': (5.0, 10.0, 30.0),
    'far': (10.0, 20.0, 50.0),
}
FINAL_DIM = (256, 704)         # (H, W) wie in der Config; per --final-dim ueberschreibbar
CAM_KEY = 'CAM_FRONT'
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], np.float32)
RESIZE = 0.48                  # nuScenes-Eval-Config; andere Datensaetze per --resize
                               #   in-house  final_dim 256x704, resize 0.314
                               #   Waymo     final_dim 384x640, resize 0.30
MAX_OFFSET, MAX_DT_S = 150, 20.0
RANSAC_THR = 1.0

_MODELS = {}                   # je Prozess einmal geladen
LEGACY_PATH = [False]          # True = Bildweg des urspruenglichen Evaluators


# --------------------------------------------------------------------------- Daten

def load_frames(infos_path):
    with open(infos_path, 'rb') as fh:
        info = pickle.load(fh)
    dl = info.get('data_list', info) if isinstance(info, dict) else info
    out = []
    for it in dl:
        cams = it.get('images') or {}
        cam = cams.get(CAM_KEY)
        if cam is None:
            continue
        out.append(dict(
            ego2global=np.asarray(it['ego2global'], np.float64),
            timestamp=float(it.get('timestamp', len(out))),
            img_path=cam['img_path'],
            # Waymo liefert cam2img als 3x4; fuer die Essentialmatrix wird die 3x3-Intrinsik
            # gebraucht, die dort im linken Block steht.
            cam2img=np.asarray(cam['cam2img'], np.float64)[:3, :3],
            lidar2cam=np.asarray(cam['lidar2cam'], np.float64),
            lidar2ego=np.asarray(it['lidar_points'].get('lidar2ego', np.eye(4)), np.float64),
        ))
    return out


def rel_pose_lidar(a, b):
    """Relative Pose von Frame a nach b im LiDAR-Frame (wie lidar2global im Modell)."""
    la = a['ego2global'] @ a['lidar2ego']
    lb = b['ego2global'] @ b['lidar2ego']
    return np.linalg.inv(la) @ lb


def build_pairs(frames, bin_name, seed=0):
    """Paare wie der Dataloader: Vorwaertssuche, Zeitfenster, Distanz-/Winkelfenster."""
    lo, hi, amax = BINS[bin_name]
    rng = np.random.RandomState(seed)
    n = len(frames)
    sec = np.array([f['timestamp'] for f in frames])
    sec = np.where(sec > 1e10, sec / 1e6, sec)
    pairs = []
    for i in range(n):
        cand = []
        for j in range(i + 1, min(i + MAX_OFFSET + 1, n)):
            if abs(sec[j] - sec[i]) > MAX_DT_S:
                continue
            T = np.linalg.inv(frames[i]['ego2global']) @ frames[j]['ego2global']
            P = np.eye(4); P[2, 2] = 0                       # 2D-Projektion wie im Dataset
            T = P @ T
            dist = float(np.linalg.norm(T[:3, 3]))
            ang = abs(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
            if lo <= dist <= hi and ang <= amax:
                cand.append(j)
        if cand:
            pairs.append((i, int(rng.choice(cand))))
    return pairs


def load_image_and_K(frame, data_root):
    """Bild laden und wie ImageAug3D (Testmodus) resizen/croppen; K mittransformieren."""
    p = frame['img_path']
    for cand in (os.path.join(data_root, p),
                 os.path.join(data_root, 'samples', CAM_KEY, os.path.basename(p)),
                 os.path.join(data_root, 'image_0', os.path.basename(p)),
                 os.path.join(data_root, 'training', 'image_0', os.path.basename(p)),
                 p):
        if os.path.exists(cand):
            p = cand
            break
    else:
        raise FileNotFoundError(frame['img_path'])
    img = Image.open(p).convert('RGB')
    W, H = img.size
    fH, fW = FINAL_DIM
    newW, newH = int(W * RESIZE), int(H * RESIZE)
    crop_h = int(newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    img = img.resize((newW, newH)).crop((crop_w, crop_h, crop_w + fW, crop_h + fH))
    K = frame['cam2img'].copy()
    K[:2] *= RESIZE
    K[0, 2] -= crop_w
    K[1, 2] -= crop_h
    arr = np.asarray(img, np.uint8)
    if LEGACY_PATH[0]:
        # Bildweg des urspruenglichen Evaluators nachbilden: die Pipeline normalisiert mit
        # ImageNet-Statistiken, _extract_images castet den float-Tensor danach direkt nach
        # uint8 (der max()<=1.0-Zweig greift nicht) -- negative Werte laufen dabei ueber.
        f = (arr.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
        arr = f.astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        arr = np.stack([arr] * 3, -1)
    return arr, K


# --------------------------------------------------------------------------- Matcher

def get_model(method):
    if method in _MODELS:
        return _MODELS[method]
    import kornia.feature as KF
    if method == 'dedode':
        m = KF.DeDoDe.from_pretrained(detector_weights='L-upright',
                                      descriptor_weights='B-upright').eval()
    elif method == 'loftr':
        m = KF.LoFTR(pretrained='outdoor').eval()
    else:
        m = None       # SIFT laeuft ueber OpenCV, kein Torch-Modell noetig
    _MODELS[method] = m
    return m


def to_tensor(img):
    t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0
    return t[None]


def mutual_nn_hamming(d0, d1):
    """Gegenseitiges Matching binaerer Deskriptoren (ORB) ueber die Hamming-Distanz."""
    a = np.unpackbits(d0.astype(np.uint8), axis=1).astype(np.float32)
    b = np.unpackbits(d1.astype(np.uint8), axis=1).astype(np.float32)
    D = a @ (1 - b).T + (1 - a) @ b.T                     # Hamming-Distanz ueber Bit-Matrizen
    i0 = D.argmin(1); i1 = D.argmin(0)
    keep = i1[i0] == np.arange(len(a))
    return np.stack([np.arange(len(a))[keep], i0[keep]], 1)


def mutual_nn(d0, d1, ratio=0.7):
    """Gegenseitiges NN-Matching mit Ratio-Test (wie nn_match_two_way)."""
    d0 = d0 / (np.linalg.norm(d0, axis=1, keepdims=True) + 1e-12)
    d1 = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-12)
    S = d0 @ d1.T
    dmat = np.sqrt(np.clip(2 - 2 * S, 0, None))
    i0 = dmat.argmin(1)
    v0 = dmat[np.arange(len(d0)), i0]
    i1 = dmat.argmin(0)
    keep = (i1[i0] == np.arange(len(d0))) & (v0 < ratio)
    return np.stack([np.arange(len(d0))[keep], i0[keep]], 1)


def match_pair(img_a, img_b, method):
    """Liefert (pts_a, pts_b) korrespondierende Pixelkoordinaten."""
    ta, tb = to_tensor(img_a), to_tensor(img_b)
    m = get_model(method)
    if method == 'dedode':
        with torch.no_grad():
            ka, _, da = m(ta, n=2048)
            kb, _, db = m(tb, n=2048)
        idx = mutual_nn(da[0].numpy(), db[0].numpy())
        if len(idx) < 8:
            return None, None
        return ka[0].numpy()[idx[:, 0]], kb[0].numpy()[idx[:, 1]]
    if method == 'loftr':
        import kornia.color as KC
        with torch.no_grad():
            out = m({'image0': KC.rgb_to_grayscale(ta), 'image1': KC.rgb_to_grayscale(tb)})
        pa, pb, conf = out['keypoints0'].numpy(), out['keypoints1'].numpy(), out['confidence'].numpy()
        sel = conf > 0.5
        if sel.sum() < 8:
            return None, None
        return pa[sel], pb[sel]
    if method == 'sift':
        g_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        g_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        s = cv2.SIFT_create(2048)
        ka, da = s.detectAndCompute(g_a, None)
        kb, db = s.detectAndCompute(g_b, None)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return None, None
        idx = mutual_nn(da.astype(np.float32), db.astype(np.float32))
        if len(idx) < 8:
            return None, None
        return (np.array([ka[i].pt for i in idx[:, 0]], np.float32),
                np.array([kb[i].pt for i in idx[:, 1]], np.float32))
    raise ValueError(method)


def pose_from_matches(pts_a, pts_b, K, gt_scale, lidar2cam):
    """Essentialmatrix -> recoverPose -> Skala aus GT -> SE(2) im LiDAR-Frame."""
    E, mask = cv2.findEssentialMat(pts_b, pts_a, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=RANSAC_THR)
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, _ = cv2.recoverPose(E, pts_b, pts_a, K, mask=mask)
    t = t.flatten() * float(gt_scale)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    cam2lidar = np.linalg.inv(lidar2cam)
    T_l = cam2lidar @ T @ lidar2cam
    return float(np.arctan2(T_l[1, 0], T_l[0, 0])), T_l[:2, 3]


# --------------------------------------------------------------------------- Worker

_CTX = {}


def _init(infos_path, data_root, methods, threads, legacy=False):
    torch.set_num_threads(threads)
    LEGACY_PATH[0] = legacy
    _CTX['frames'] = load_frames(infos_path)
    _CTX['data_root'] = data_root
    _CTX['methods'] = methods


def _run_pair(args):
    bin_name, i, j = args
    fr = _CTX['frames']
    out = {}
    try:
        img_a, K = load_image_and_K(fr[i], _CTX['data_root'])
        img_b, _ = load_image_and_K(fr[j], _CTX['data_root'])
    except Exception:
        return bin_name, out
    T_gt = rel_pose_lidar(fr[i], fr[j])
    gt_ang = float(np.arctan2(T_gt[1, 0], T_gt[0, 0]))
    gt_t = T_gt[:2, 3]
    gt_scale = float(np.linalg.norm(T_gt[:3, 3]))
    for meth in _CTX['methods']:
        try:
            pa, pb = match_pair(img_a, img_b, meth)
            if pa is None:
                continue
            res = pose_from_matches(pa, pb, K, gt_scale, fr[i]['lidar2cam'])
            if res is None:
                continue
            ang, t = res
            da = np.degrees(np.arctan2(np.sin(ang - gt_ang), np.cos(ang - gt_ang)))
            out[meth] = (float(np.linalg.norm(t - gt_t)), float(abs(da)), float(np.linalg.norm(gt_t)))
        except Exception:
            continue
    return bin_name, out




def _sp_model(device):
    from transformers import AutoImageProcessor, SuperPointForKeypointDetection
    rid = "magic-leap-community/superpoint"
    if 'sp' not in _MODELS:
        _MODELS['sp'] = (AutoImageProcessor.from_pretrained(rid),
                         SuperPointForKeypointDetection.from_pretrained(rid).to(device).eval())
    return _MODELS['sp']


def extract_features(method, frames, need, data_root, device, batch, io_workers):
    """dict[frame_idx] = (keypoints[N,2], descriptors[N,D]) -- jeder Frame genau einmal.

    dedode/superpoint laufen batchweise auf der GPU, sift/orb ueber OpenCV in CPU-Threads.
    """
    from concurrent.futures import ThreadPoolExecutor
    feats, Ks = {}, {}
    dev = torch.device(device)

    if method in ('sift', 'orb'):
        det = cv2.SIFT_create(2048) if method == 'sift' else cv2.ORB_create(nfeatures=2048)
        def one(i):
            im, K = load_image_and_K(frames[i], data_root)
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            kp, de = det.detectAndCompute(g, None)
            if de is None or len(kp) < 8:
                return i, None, K
            return i, (np.array([k.pt for k in kp], np.float32), de), K
        with ThreadPoolExecutor(io_workers) as ex:
            for n, (i, f, K) in enumerate(ex.map(one, need), 1):
                if f is not None:
                    feats[i] = f
                Ks[i] = K
                if n % 2000 == 0:
                    print(f'[gpu] {method}: {n}/{len(need)}', flush=True)
        return feats, Ks

    import kornia.feature as KF
    if method == 'dedode':
        model = KF.DeDoDe.from_pretrained(detector_weights='L-upright',
                                          descriptor_weights='B-upright').to(dev).eval()
    elif method == 'superpoint':
        proc, model = _sp_model(dev)
    else:
        raise ValueError(method)

    pool = ThreadPoolExecutor(io_workers)
    for b0 in range(0, len(need), batch):
        chunk = need[b0:b0 + batch]
        loaded = list(pool.map(lambda i: load_image_and_K(frames[i], data_root), chunk))
        for k, i in enumerate(chunk):
            Ks[i] = loaded[k][1]
        if method == 'dedode':
            x = torch.stack([to_tensor(im)[0] for im, _ in loaded]).to(dev)
            with torch.no_grad():
                kp, _, de = model(x, n=2048)
            for k, i in enumerate(chunk):
                feats[i] = (kp[k].cpu().numpy(), de[k].half().cpu().numpy().astype(np.float32))
        else:
            from PIL import Image as PImage
            inp = proc([PImage.fromarray(im) for im, _ in loaded], return_tensors='pt').to(dev)
            with torch.no_grad():
                out = model(**inp)
            # Bildgroessen muessen auf demselben Device liegen wie die Modellausgabe
            sizes = torch.tensor([[im.shape[0], im.shape[1]] for im, _ in loaded], device=dev)
            res = proc.post_process_keypoint_detection(out, sizes)
            for k, i in enumerate(chunk):
                feats[i] = (res[k]['keypoints'].cpu().numpy().astype(np.float32),
                            res[k]['descriptors'].cpu().numpy().astype(np.float32))
        if (b0 // batch) % 20 == 0:
            print(f'[gpu] {method}: {b0 + len(chunk)}/{len(need)}', flush=True)
    pool.shutdown()
    return feats, Ks


# --------------------------------------------------------------------------- GPU-Pfad

def run_gpu_batched(frames, tasks, methods, data_root, device='cuda', batch=32, io_workers=16):
    """Batchweise Auswertung aller Kamera-Baselines.

    Zwei Hebel gegenueber dem Prozess-Pool: (1) jeder Frame wird pro Methode genau EINMAL
    durch den Detektor geschickt, obwohl er in mehreren Paaren vorkommt; (2) die Inferenz
    laeuft batchweise. Bildladen und RANSAC bleiben auf der CPU, in Threads parallel zur GPU.
    LoFTR ist detector-free und braucht beide Bilder gemeinsam -- daher ein eigener Pfad
    ohne Feature-Wiederverwendung.
    """
    from concurrent.futures import ThreadPoolExecutor
    need = sorted({i for _, i, _ in tasks} | {j for _, _, j in tasks})
    print(f'[gpu] {len(tasks)} Paare, {len(need)} eindeutige Frames '
          f'(spart {100*(1-len(need)/max(1,2*len(tasks))):.0f}% der Bildinferenzen)', flush=True)
    acc = {}

    for method in methods:
        if method == 'romav2':
            acc = _run_romav2(frames, tasks, data_root, io_workers, acc)
            continue
        if method == 'loftr':
            acc = _run_loftr(frames, tasks, data_root, device, batch, io_workers, acc)
            continue
        feats, Ks = extract_features(method, frames, need, data_root, device, batch, io_workers)

        def solve(t, _m=method, _f=feats, _K=Ks):
            bin_name, i, j = t
            if i not in _f or j not in _f:
                return bin_name, None
            ka, da = _f[i]; kb, db = _f[j]
            idx = (mutual_nn_hamming(da, db) if _m == 'orb'
                   else mutual_nn(da.astype(np.float32), db.astype(np.float32)))
            if len(idx) < 8:
                return bin_name, None
            T_gt = rel_pose_lidar(frames[i], frames[j])
            res = pose_from_matches(ka[idx[:, 0]], kb[idx[:, 1]], _K[i],
                                    float(np.linalg.norm(T_gt[:3, 3])), frames[i]['lidar2cam'])
            if res is None:
                return bin_name, None
            ang, tt = res
            gt_ang = float(np.arctan2(T_gt[1, 0], T_gt[0, 0]))
            d = np.degrees(np.arctan2(np.sin(ang - gt_ang), np.cos(ang - gt_ang)))
            return bin_name, (float(np.linalg.norm(tt - T_gt[:2, 3])), float(abs(d)),
                              float(np.linalg.norm(T_gt[:2, 3])))

        with ThreadPoolExecutor(io_workers) as ex:
            for n, (b, r) in enumerate(ex.map(solve, tasks), 1):
                if r is not None:
                    acc.setdefault(b, {}).setdefault(method, []).append(r)
                if n % 5000 == 0:
                    print(f'[gpu] {method}: Posen {n}/{len(tasks)}', flush=True)
        print(f'[gpu] {method} fertig', flush=True)
    return acc



def _run_romav2(frames, tasks, data_root, io_workers, acc):
    """RoMa v2 (2025): dense Matcher, verarbeitet Bildpaare gemeinsam.

    Laeuft nur in der separaten Umgebung `romav2` (Python >= 3.10, torch >= 2.5) und
    zwingend auf der GPU -- auf CPU ist der DINOv3-Backbone unbrauchbar langsam.
    Gemessen: ~0.55 s/Paar auf einer A100.
    """
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image as PImage
    from romav2 import RoMaV2
    model = RoMaV2()
    pool = ThreadPoolExecutor(io_workers)
    for n, (bin_name, i, j) in enumerate(tasks, 1):
        try:
            (ia, Ka), (ib, _) = list(pool.map(
                lambda t: load_image_and_K(frames[t], data_root), (i, j)))
            preds = model.match(PImage.fromarray(ia), PImage.fromarray(ib))
            sampled = model.sample(preds, 2000)[0]
            kA, kB = model.to_pixel_coordinates(sampled, ia.shape[0], ia.shape[1],
                                                ib.shape[0], ib.shape[1])
            pa = kA.detach().cpu().numpy().astype(np.float32)
            pb = kB.detach().cpu().numpy().astype(np.float32)
            if len(pa) < 8:
                continue
            T_gt = rel_pose_lidar(frames[i], frames[j])
            res = pose_from_matches(pa, pb, Ka, float(np.linalg.norm(T_gt[:3, 3])),
                                    frames[i]['lidar2cam'])
            if res is None:
                continue
            ang, tt = res
            gt_ang = float(np.arctan2(T_gt[1, 0], T_gt[0, 0]))
            d = np.degrees(np.arctan2(np.sin(ang - gt_ang), np.cos(ang - gt_ang)))
            acc.setdefault(bin_name, {}).setdefault('romav2', []).append(
                (float(np.linalg.norm(tt - T_gt[:2, 3])), float(abs(d)),
                 float(np.linalg.norm(T_gt[:2, 3]))))
        except Exception as e:
            if n <= 3:
                print(f'[gpu] romav2 Paar {n}: {type(e).__name__}: {str(e)[:120]}', flush=True)
            continue
        if n % 500 == 0:
            print(f'[gpu] romav2: {n}/{len(tasks)}', flush=True)
    pool.shutdown()
    print('[gpu] romav2 fertig', flush=True)
    return acc


def _run_loftr(frames, tasks, data_root, device, batch, io_workers, acc):
    """LoFTR: detector-free, verarbeitet Bildpaare gemeinsam (kein Feature-Cache moeglich)."""
    from concurrent.futures import ThreadPoolExecutor
    import kornia.feature as KF
    import kornia.color as KC
    dev = torch.device(device)
    model = KF.LoFTR(pretrained='outdoor').to(dev).eval()
    pool = ThreadPoolExecutor(io_workers)
    for b0 in range(0, len(tasks), batch):
        chunk = tasks[b0:b0 + batch]
        loaded = list(pool.map(lambda t: (load_image_and_K(frames[t[1]], data_root),
                                          load_image_and_K(frames[t[2]], data_root)), chunk))
        for (bin_name, i, j), ((ia, Ka), (ib, _)) in zip(chunk, loaded):
            ta = KC.rgb_to_grayscale(to_tensor(ia)).to(dev)
            tb = KC.rgb_to_grayscale(to_tensor(ib)).to(dev)
            with torch.no_grad():
                out = model({'image0': ta, 'image1': tb})
            pa = out['keypoints0'].cpu().numpy(); pb = out['keypoints1'].cpu().numpy()
            conf = out['confidence'].cpu().numpy()
            sel = conf > 0.5
            if sel.sum() < 8:
                continue
            T_gt = rel_pose_lidar(frames[i], frames[j])
            res = pose_from_matches(pa[sel], pb[sel], Ka, float(np.linalg.norm(T_gt[:3, 3])),
                                    frames[i]['lidar2cam'])
            if res is None:
                continue
            ang, tt = res
            gt_ang = float(np.arctan2(T_gt[1, 0], T_gt[0, 0]))
            d = np.degrees(np.arctan2(np.sin(ang - gt_ang), np.cos(ang - gt_ang)))
            acc.setdefault(bin_name, {}).setdefault('loftr', []).append(
                (float(np.linalg.norm(tt - T_gt[:2, 3])), float(abs(d)),
                 float(np.linalg.norm(T_gt[:2, 3]))))
        if (b0 // batch) % 20 == 0:
            print(f'[gpu] loftr: {b0 + len(chunk)}/{len(tasks)}', flush=True)
    pool.shutdown()
    print('[gpu] loftr fertig', flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infos', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--methods', default='dedode', help='dedode,loftr,sift')
    ap.add_argument('--bins', default='close,far')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='0 = alle Paare je Bin')
    ap.add_argument('--legacy-images', action='store_true',
                    help='Bildweg des urspruenglichen Evaluators nachbilden (ImageNet-Norm -> uint8-Cast)')
    ap.add_argument('--device', default='cpu', help='cpu | cuda (cuda nutzt den Batch-Pfad)')
    ap.add_argument('--shard', default='', help='i/n -- nur Teil i von n der Paare bearbeiten')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--final-dim', default='', help='H,W -- Zielgroesse nach dem Crop (Default nuScenes 256,704)')
    ap.add_argument('--resize', type=float, default=0.0, help='Resize-Faktor der ImageAug3D-Testphase')
    ap.add_argument('--cam', default='CAM_FRONT', help='Kamera-Key in den Infos')
    ap.add_argument('--out', default='baseline_cpu_results.json')
    a = ap.parse_args()

    global FINAL_DIM, RESIZE, CAM_KEY
    if a.final_dim:
        # 'x' als Trenner zulassen: ein Komma in --export=...,FINAL_DIM=256,704 wuerde von
        # Slurm als Variablentrenner gelesen und die Groesse zerreissen.
        FINAL_DIM = tuple(int(v) for v in a.final_dim.replace('x', ',').split(','))
    if a.resize:
        RESIZE = a.resize
    CAM_KEY = a.cam
    print(f'Bildvorverarbeitung: final_dim={FINAL_DIM} resize={RESIZE} cam={CAM_KEY}', flush=True)
    methods = [m.strip() for m in a.methods.split(',') if m.strip()]
    frames = load_frames(a.infos)
    print(f'[base] {len(frames)} Frames, Methoden={methods}', flush=True)

    tasks = []
    for b in [x.strip() for x in a.bins.split(',') if x.strip()]:
        pr = build_pairs(frames, b)
        if a.limit:
            pr = pr[::max(1, len(pr) // a.limit)][:a.limit]
        print(f'[base] Bin {b}: {len(pr)} Paare', flush=True)
        tasks += [(b, i, j) for i, j in pr]

    if a.shard:
        si, sn = (int(x) for x in a.shard.split('/'))
        tasks = tasks[si::sn]                       # deterministische Aufteilung ueber alle Bins
        print(f'[base] Shard {si}/{sn}: {len(tasks)} Paare', flush=True)

    acc = {}
    if a.device.startswith('cuda'):
        acc = run_gpu_batched(frames, tasks, methods, a.data_root,
                              device=a.device, batch=a.batch_size, io_workers=a.workers)
    else:
      with Pool(a.workers, initializer=_init,
                initargs=(a.infos, a.data_root, methods, a.threads, a.legacy_images)) as pool:
        for k, (b, res) in enumerate(pool.imap_unordered(_run_pair, tasks, chunksize=4), 1):
            for meth, v in res.items():
                acc.setdefault(b, {}).setdefault(meth, []).append(v)
            if k % 200 == 0:
                print(f'[base] {k}/{len(tasks)}', flush=True)

    summary = {}
    for b, per in acc.items():
        summary[b] = {}
        for meth, vals in per.items():
            e = np.array(vals)
            # Das Paper mittelt RTE/RRE nur ueber ERFOLGREICHE Registrierungen (Erfolg =
            # Translationsfehler unter GT-Translation + 0.1 m); Recall zaehlt dagegen ueber
            # alle Paare und erfasst so die Fehlschlaege. Beide Varianten werden berichtet.
            succ = (e[:, 0] < e[:, 2] + 0.1) if e.shape[1] > 2 else np.ones(len(e), bool)
            es = e[succ] if succ.any() else e
            summary[b][meth] = dict(
                n=len(e), n_success=int(succ.sum()),
                RTE=float(es[:, 0].mean()), RRE=float(es[:, 1].mean()),
                RTE_all=float(e[:, 0].mean()), RRE_all=float(e[:, 1].mean()),
                recall_05_5=float(100 * np.mean((e[:, 0] < 0.5) & (e[:, 1] < 5))),
                recall_2_10=float(100 * np.mean((e[:, 0] < 2.0) & (e[:, 1] < 10))))
            s = summary[b][meth]
            print(f"[base] {b:5s} {meth:11s} n={s['n']:5d} RTE={s['RTE']:.2f} m "
                  f"RRE={s['RRE']:.2f} deg R@0.5/5={s['recall_05_5']:.1f}% "
                  f"R@2/10={s['recall_2_10']:.1f}%", flush=True)
    json.dump(summary, open(a.out, 'w'), indent=1)
    print(f'[base] -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
