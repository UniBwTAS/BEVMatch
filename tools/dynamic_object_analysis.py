"""Dynamik-Analyse (R8 T4) auf nuScenes: Wie stark beeinflussen bewegte Objekte BEVMatch?

Fuer jedes Frame-Paar:
  1. Keypoints + Descriptors beider Frames, MNN-Matches, RANSAC-SE(2)  (= Standard-Pipeline)
  2. Keypoints werden als "dynamisch" markiert, wenn sie im BEV-Footprint einer GT-Box mit
     |velocity| > V_DYN liegen (rotierte Box, Punkt-in-Box-Test im Box-Koordinatensystem).
  3. Statistiken: Anteil dynamischer Keypoints / tentativer Matches / RANSAC-Inlier.
  4. Zweiter Durchlauf mit VOR RANSAC entfernten dynamischen Matches -> RTE/RRE-Vergleich.

Ergebnis beantwortet: filtert RANSAC dynamische Korrespondenzen bereits heraus?

Env: CFG, CKPT, N_FRAMES (300), TOPK (200), V_DYN (0.5 m/s), OUT (dynamic_analysis.txt)
"""
import os
import numpy as np
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29887')
os.environ.setdefault('RANK', '0'); os.environ.setdefault('WORLD_SIZE', '1'); os.environ.setdefault('LOCAL_RANK', '0')
import torch, torch.distributed as dist
torch.cuda.set_device(0)
if not dist.is_initialized():
    dist.init_process_group('nccl', rank=0, world_size=1)
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
import bevmatch  # noqa: F401  (registriert Modell und Heads)
from mmdet3d.registry import MODELS
from keypoint_extraction import (peak_cells as _peak_cells,
                                 mutual_nn_matches as _mutual_nn_matches,
                                 cells_to_xy as _cells_to_xy)
from kitti_se2 import ransac_se2

CFG = os.environ['CFG']; CKPT = os.environ['CKPT']
N = int(os.environ.get('N_FRAMES', '300')); TOPK = int(os.environ.get('TOPK', '200'))
V_DYN = float(os.environ.get('V_DYN', '0.5')); OUT = os.environ.get('OUT', 'dynamic_analysis.txt')

cfg = Config.fromfile(CFG); init_default_scope('mmdet3d')
model = MODELS.build(cfg.model).cuda().eval()
load_checkpoint(model, CKPT, map_location='cuda', strict=False)
head = model.keypoint_head
cfg.val_dataloader.batch_size = 1; cfg.val_dataloader.num_workers = 4
cfg.val_dataloader.persistent_workers = False
dl = Runner.build_dataloader(cfg.val_dataloader)

# GT-Boxen kommen NICHT ueber meta_keys -> direkt aus dem Infos-PKL, Schluessel = sample_idx
import pickle
_ds = cfg.val_dataloader.dataset
_pkl = os.path.join(_ds['data_root'], _ds['ann_file'])
_info = pickle.load(open(_pkl, 'rb'))
_dl_list = _info.get('data_list', _info)
INST = {}
for _it in _dl_list:
    for _k in ('sample_idx', 'token'):
        if _k in _it:
            INST[_it[_k]] = _it.get('instances', [])
print(f'[dyn] GT-Instanzen fuer {len(INST)} Frames geladen', flush=True)


def dynamic_mask(xy, meta):
    """True fuer Keypoints im BEV-Footprint einer bewegten GT-Box."""
    ins = INST.get(meta.get('sample_idx'), INST.get(meta.get('token'), []))
    m = np.zeros(len(xy), bool)
    for o in ins:
        v = o.get('velocity')
        if v is None or not np.isfinite(np.asarray(v, float)).all():
            continue
        if float(np.linalg.norm(np.asarray(v, float)[:2])) < V_DYN:
            continue
        b = o.get('bbox_3d')
        if b is None or len(b) < 7:
            continue
        cx, cy, _, l, w, _, yaw = b[:7]
        c, s = np.cos(-yaw), np.sin(-yaw)
        d = xy - np.array([cx, cy])
        xl = d[:, 0] * c - d[:, 1] * s
        yl = d[:, 0] * s + d[:, 1] * c
        m |= (np.abs(xl) <= l / 2 + 0.6) & (np.abs(yl) <= w / 2 + 0.6)   # +1 Zelle Toleranz
    return m


def rel_gt(meta_c, meta_o):
    a = np.asarray(meta_c['ego2global'], np.float64); b = np.asarray(meta_o['ego2global'], np.float64)
    return np.linalg.inv(b) @ a


st = dict(n=0, kp=0, kp_dyn=0, mt=0, mt_dyn=0, inl=0, inl_dyn=0,
          err=[], err_clean=[], aerr=[], aerr_clean=[], n_dynframe=0)
for i, data in enumerate(dl):
    if st['n'] >= N:
        break
    with torch.no_grad():
        pd = model.data_preprocessor(data, False); inp = pd['inputs']; sm = pd['data_samples']
        mc = [s.metainfo for s in sm['current']]; mo = [s.metainfo for s in sm['other']]
        T = rel_gt(mc[0], mo[0])
        if np.linalg.norm(T[:2, 3]) < 1.0:      # degenerierte Paare (other==current) ueberspringen
            continue
        f0 = model.extract_feat(inp['current'], mc, drop_cam=False, drop_lidar=False)
        f1 = model.extract_feat(inp['other'], mo, drop_cam=False, drop_lidar=False)
        f0 = f0[0] if isinstance(f0, (list, tuple)) else f0
        f1 = f1[0] if isinstance(f1, (list, tuple)) else f1
        k0, d0, _ = head.forward(f0[0:1], None); k1, d1, _ = head.forward(f1[0:1], None)
    c0 = _peak_cells(head, k0, TOPK); c1 = _peak_cells(head, k1, TOPK)
    if len(c0) < 4 or len(c1) < 4:
        continue
    dim, res = head.dim_grid, head.bev_resolution
    xy0 = _cells_to_xy(c0.cpu().numpy(), dim, res); xy1 = _cells_to_xy(c1.cpu().numpy(), dim, res)
    dyn0 = dynamic_mask(xy0, mc[0]); dyn1 = dynamic_mask(xy1, mo[0])
    m = _mutual_nn_matches(d0, c0, d1, c1, binary=False)
    if len(m) < 4:
        continue
    A = np.array([xy0[a] for a, _ in m]); B = np.array([xy1[b] for _, b in m])
    mdyn = np.array([dyn0[a] or dyn1[b] for a, b in m], bool)
    Te, inl = ransac_se2(A, B, 1.0)
    if Te is None:
        continue

    def err(Tp):
        dt = float(np.linalg.norm(np.array([Tp[0, 2], Tp[1, 2]]) - T[:2, 3]))
        da = abs(np.degrees(np.arctan2(Tp[1, 0], Tp[0, 0]) - np.arctan2(T[1, 0], T[0, 0])))
        return dt, (da if da < 180 else 360 - da)
    e, ae = err(Te)
    st['err'].append(e); st['aerr'].append(ae)
    # zweiter Durchlauf: dynamische Matches VOR RANSAC entfernen
    keep = ~mdyn
    if keep.sum() >= 4:
        Tc, _ = ransac_se2(A[keep], B[keep], 1.0)
        if Tc is not None:
            ec, aec = err(Tc)
            st['err_clean'].append(ec); st['aerr_clean'].append(aec)
        else:
            st['err_clean'].append(e); st['aerr_clean'].append(ae)
    else:
        st['err_clean'].append(e); st['aerr_clean'].append(ae)
    st['kp'] += len(xy0); st['kp_dyn'] += int(dyn0.sum())
    st['mt'] += len(m); st['mt_dyn'] += int(mdyn.sum())
    st['inl'] += int(inl.sum()); st['inl_dyn'] += int((inl & mdyn).sum())
    st['n_dynframe'] += int(mdyn.any()); st['n'] += 1
    if st['n'] % 50 == 0:
        print(f"[dyn] {st['n']}/{N}", flush=True)

e = np.array(st['err']); ec = np.array(st['err_clean'])
ae = np.array(st['aerr']); aec = np.array(st['aerr_clean'])
lines = [
    f"frames={st['n']}  (davon mit dynamischen Matches: {st['n_dynframe']})",
    f"dynamische Keypoints:        {100*st['kp_dyn']/max(st['kp'],1):5.2f}%  ({st['kp_dyn']}/{st['kp']})",
    f"dynamische tent. Matches:    {100*st['mt_dyn']/max(st['mt'],1):5.2f}%  ({st['mt_dyn']}/{st['mt']})",
    f"dynamische RANSAC-Inlier:    {100*st['inl_dyn']/max(st['inl'],1):5.2f}%  ({st['inl_dyn']}/{st['inl']})",
    f"median RTE  normal={np.median(e):.3f} m   ohne dyn. Matches={np.median(ec):.3f} m",
    f"median RRE  normal={np.median(ae):.3f}°   ohne dyn. Matches={np.median(aec):.3f}°",
    f"Recall@0.5m/5°  normal={100*np.mean((e<0.5)&(ae<5)):.1f}%   clean={100*np.mean((ec<0.5)&(aec<5)):.1f}%",
    f"Recall@2m/10°   normal={100*np.mean((e<2)&(ae<10)):.1f}%   clean={100*np.mean((ec<2)&(aec<10)):.1f}%",
]
print('\n'.join('[dyn] ' + l for l in lines), flush=True)
open(OUT, 'w').write('\n'.join(lines) + '\n')
print('[dyn] DONE', flush=True)
