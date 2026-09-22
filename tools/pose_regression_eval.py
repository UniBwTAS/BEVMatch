"""Evaluiert die Regressionskoepfe im EXAKT gleichen Protokoll wie BEVMatch in Tabelle I.

Damit der Vergleich Matching gegen Regression misst und nicht zwei verschiedene Messvorschriften:

  * Bins        close = 0-5 m und <=10 deg, far = 10-20 m und <=50 deg relativer Bewegung.
  * RTE/RRE     nur ueber ERFOLGREICHE Registrierungen gemittelt; erfolgreich heisst
                Translationsfehler < |GT-Translation| + 0.1 m (Paper, Sec. IV-A).
  * Recall      ueber alle Paare des Bins, bei 0.5m/5deg und 2m/10deg.

Zusaetzlich werden zwei triviale Referenzen mitgefuehrt, weil ein Regressor sie schlagen MUSS,
um ueberhaupt etwas gelernt zu haben:
  * Null-Praediktor       -- sagt immer (0,0) voraus
  * bester konstanter     -- sagt immer den Mittelwert des Bins voraus (kennt also die Statistik)

Env: CFG, CKPT, HEAD_CKPT (oder HEAD_DIR fuer alle head_*.pth), OUT
"""
import glob
import os

import numpy as np

os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', os.environ.get('MPORT', '29895'))
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('LOCAL_RANK', '0')
import torch
import torch.distributed as dist

torch.cuda.set_device(0)
if not dist.is_initialized():
    dist.init_process_group('nccl', rank=0, world_size=1)
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

import bevmatch  # noqa: F401  (registriert Modell und Heads)
from mmdet3d.registry import MODELS

import sys
from bevmatch.pose_regression import build_head  # noqa: E402

CFG = os.environ['CFG']
CKPT = os.environ['CKPT']
HEAD_DIR = os.environ.get('HEAD_DIR', '')
HEAD_CKPT = os.environ.get('HEAD_CKPT', '')
OUT = os.environ.get('OUT', 'pose_reg_eval.txt')
MAX_PAIRS = int(os.environ.get('MAX_PAIRS', '0'))
# Modalitaetskombination wie im Paper: C2C, C2L, L2L, M2M. Der Kopf wurde auf M2M trainiert;
# die uebrigen Modi zeigen, ob er ueber die Modalitaet hinweg generalisiert -- genau die
# Faehigkeit, auf der die Abgrenzung gegen direkte Regression beruht.
MODE = os.environ.get('MODE', 'M2M').upper()
DROP = {'M2M': ((False, False), (False, False)),
        'C2C': ((False, True), (False, True)),      # beide Frames nur Kamera
        'L2L': ((True, False), (True, False)),      # beide Frames nur LiDAR
        'C2L': ((False, True), (True, False))}[MODE]  # current Kamera, other LiDAR

BINS = [('close', 0.0, 5.0, 10.0), ('far', 10.0, 20.0, 50.0)]

cfg = Config.fromfile(CFG)
init_default_scope('mmdet3d')
model = MODELS.build(cfg.model).cuda().eval()
load_checkpoint(model, CKPT, map_location='cuda', strict=False)
for p in model.parameters():
    p.requires_grad = False


from bevmatch.keypoint_ops import compose_aug_warp  # noqa: E402


def rel_pose_meters(ma, mb):
    """Wie BEVMatch: W = aug_other @ inv(e2g_current) @ e2g_other @ inv(aug_current), z-kill."""
    eye = np.eye(4)
    R, t = compose_aug_warp(ma['ego2global'], mb['ego2global'],
                            ma.get('lidar_aug_matrix', eye),
                            mb.get('lidar_aug_matrix', eye), 1.0)
    return np.asarray(t, dtype=np.float64), float(np.degrees(np.arctan2(float(R[1, 0]), float(R[0, 0]))))


paths = sorted(glob.glob(os.path.join(HEAD_DIR, 'head_*.pth'))) if HEAD_DIR else [HEAD_CKPT]
heads = {}
for p in paths:
    if not p or not os.path.exists(p):
        continue
    sd = torch.load(p, map_location='cpu')
    kw = {}
    if sd.get('pool'):
        kw['pool'] = sd['pool']
    if sd.get('n_rot'):
        kw['n_rot'] = sd['n_rot']
    if sd.get('max_deg'):
        kw['max_deg'] = sd['max_deg']
    h = build_head(sd['arch'], sd['c_in'], sd['res_m'], sd['grid'],
                   sd.get('max_shift', 20), **kw).cuda().eval()
    h.load_state_dict(sd['state_dict'])
    heads[sd['arch']] = h
print(f'geladen: {list(heads)}', flush=True)

cfg.val_dataloader.batch_size = 1
cfg.val_dataloader.num_workers = 4
cfg.val_dataloader.persistent_workers = False
dl = Runner.build_dataloader(cfg.val_dataloader)

rows = {a: [] for a in heads}
gts = []
n = 0
for data in dl:
    if MAX_PAIRS and n >= MAX_PAIRS:
        break
    with torch.no_grad():
        pd = model.data_preprocessor(data, False)
        inp, sm = pd['inputs'], pd['data_samples']
        mc = [s.metainfo for s in sm['current']]
        mo = [s.metainfo for s in sm['other']]
        (c0, l0), (c1, l1) = DROP
        f0 = model.extract_feat(inp['current'], mc, drop_cam=c0, drop_lidar=l0)
        f1 = model.extract_feat(inp['other'], mo, drop_cam=c1, drop_lidar=l1)
        f0 = (f0[0] if isinstance(f0, (list, tuple)) else f0).float()
        f1 = (f1[0] if isinstance(f1, (list, tuple)) else f1).float()
        gt_t, gt_a = rel_pose_meters(mc[0], mo[0])
        gts.append([gt_t[0], gt_t[1], gt_a])
        for a, h in heads.items():
            pt, pa = h(f0, f1)
            pt = pt[0].cpu().numpy()
            ang = np.degrees(np.arctan2(pa[0, 1].item(), pa[0, 0].item()))
            rows[a].append([pt[0], pt[1], ang])
    n += 1

gts = np.array(gts)
mag = np.linalg.norm(gts[:, :2], axis=1)
rot = np.abs(gts[:, 2])

lines = [f'Modus: {MODE}   Paare: {n}', '']
for name, lo, hi, amax in BINS:
    sel = (mag >= lo) & (mag < hi) & (rot <= amax)
    if sel.sum() == 0:
        lines.append(f'{name}: keine Paare')
        continue
    g = gts[sel]
    gm = np.linalg.norm(g[:, :2], axis=1)
    lines.append(f'--- {name} (n={sel.sum()}) ---')
    lines.append(f'  Null-Praediktor     RTE={gm.mean():.3f} m')
    const = g[:, :2].mean(0)
    lines.append(f'  bester konstanter   RTE={np.linalg.norm(g[:, :2]-const, axis=1).mean():.3f} m')
    for a in heads:
        p = np.array(rows[a])[sel]
        te = np.linalg.norm(p[:, :2] - g[:, :2], axis=1)
        re = np.abs((p[:, 2] - g[:, 2] + 180) % 360 - 180)
        ok = te < (gm + 0.1)                      # Erfolgskriterium wie im Paper
        rte = te[ok].mean() if ok.any() else float('nan')
        rre = re[ok].mean() if ok.any() else float('nan')
        r05 = 100.0 * np.mean((te < 0.5) & (re < 5))
        r2 = 100.0 * np.mean((te < 2.0) & (re < 10))
        lines.append(f'  {a:8s} RTE={rte:6.3f} m  RRE={rre:6.3f} deg  '
                     f'R@0.5m/5deg={r05:5.1f}%  R@2m/10deg={r2:5.1f}%  '
                     f'(erfolgreich {100.0*ok.mean():.0f}%, pred-std={p[:, :2].std(0).mean():.3f} vs gt-std={g[:, :2].std(0).mean():.3f})')
    lines.append('')

txt = '\n'.join(lines)
print(txt, flush=True)
with open(OUT, 'w') as fh:
    fh.write(txt + '\n')
