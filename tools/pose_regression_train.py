"""Trainiert mehrere Pose-Regressions-Koepfe GLEICHZEITIG auf denselben eingefrorenen Features.

Ein Encoder-Forward ist teuer, ein Kopf-Update ist billig. Alle Varianten sehen deshalb im selben
Job exakt dieselben Batches in derselben Reihenfolge -- der Vergleich misst die Architektur und
nicht die Datenreihenfolge, und die GPU-Zeit wird einmal statt viermal bezahlt.

Gegen den frueheren Kollaps (Vorhersage-Streuung 0.005 m bei GT-Streuung 3.07 m) sind drei Dinge
eingebaut:

  * Pose wie BEVMatch    -- dieselbe Funktion compose_aug_warp, die der Keypoint-Head fuer seine
    Korrespondenzen nutzt: Richtung current->other UND die Augmentierungsmatrizen. Der alte Kopf
    rechnete inv(other) @ current ohne Augmentierung, also spiegelverkehrt und aug-blind.
  * Kollaps-Diagnostik   -- jede Logzeile fuehrt die Streuung der Vorhersagen gegen die des GT.
    Faellt std(pred)/std(gt) unter ~0.1, sagt der Kopf praktisch eine Konstante voraus. Das ist
    der Zustand, in dem v1/v2 endeten, und er ist am Loss allein nicht zu erkennen.
  * MODE=overfit         -- Sanity-Check auf wenigen Paaren. Ein Kopf, der 32 Paare nicht
    auswendig lernen kann, ist strukturell kaputt; erst wenn das klappt, lohnt volles Training.

Env: CFG, CKPT, MODE(train|overfit), ARCHS (komma-sep.), EPOCHS, LR, MAX_ITERS, OVERFIT_N,
     MAX_SHIFT, OUT_DIR
"""
import os
import time

import numpy as np

os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', os.environ.get('MPORT', '29893'))
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
MODE = os.environ.get('MODE', 'train')
ARCHS = os.environ.get('ARCHS', 'corr,corrcnn,hires,deep').split(',')
EPOCHS = int(os.environ.get('EPOCHS', '3'))
LR = float(os.environ.get('LR', '3e-4'))
MAX_ITERS = int(os.environ.get('MAX_ITERS', '0'))
OVERFIT_N = int(os.environ.get('OVERFIT_N', '32'))
MAX_SHIFT = int(os.environ.get('MAX_SHIFT', '20'))
# Modalitaets-Dropout beim Kopf-Training: ohne ihn sieht der Kopf ausschliesslich M2M-Paare und
# hat nie Gelegenheit, ueber die Modalitaet hinweg zu lernen. Das Matching-Modell wird mit
# progressivem Dropout trainiert; erst mit derselben Behandlung ist der Vergleich fair.
MODALITY_DROPOUT = os.environ.get('MODALITY_DROPOUT', '0') == '1'
# (drop_cam, drop_lidar) je Frame -- C2L bewusst in beiden Richtungen, damit der Kopf nicht
# lernt, die Modalitaet an der Reihenfolge festzumachen.
MODE_TABLE = [((False, False), (False, False)),   # M2M
              ((False, True), (False, True)),     # C2C
              ((True, False), (True, False)),     # L2L
              ((False, True), (True, False)),     # C2L
              ((True, False), (False, True))]     # L2C
OUT_DIR = os.environ.get('OUT_DIR', 'work_dirs_posereg')
os.makedirs(OUT_DIR, exist_ok=True)

cfg = Config.fromfile(CFG)
init_default_scope('mmdet3d')
model = MODELS.build(cfg.model).cuda().eval()
load_checkpoint(model, CKPT, map_location='cuda', strict=False)
for p in model.parameters():
    p.requires_grad = False


from bevmatch.keypoint_ops import compose_aug_warp  # noqa: E402


def batch_targets(sm):
    """Ziel-Pose EXAKT so wie BEVMatch sie als Supervision benutzt.

    compose_aug_warp ist dieselbe Funktion, die der Keypoint-Head fuer seine Korrespondenzen
    aufruft: W = aug_other @ inv(ego2global_current) @ ego2global_other @ inv(aug_current),
    mit orthographischem z-kill. Damit misst der Vergleich Matching gegen Regression und nicht
    zwei verschiedene Pose-Konventionen. Der alte Kopf rechnete inv(other) @ current OHNE die
    Augmentierungsmatrizen -- also in umgekehrter Richtung und ohne die im Batch tatsaechlich
    angewandte Augmentierung. bev_resolution=1.0 liefert die Translation direkt in Metern.
    """
    ts, angs = [], []
    eye = np.eye(4)
    for a, b in zip(sm['current'], sm['other']):
        ma, mb = a.metainfo, b.metainfo
        R, t = compose_aug_warp(ma['ego2global'], mb['ego2global'],
                                ma.get('lidar_aug_matrix', eye),
                                mb.get('lidar_aug_matrix', eye), 1.0)
        ts.append(np.asarray(t, dtype=np.float64))
        angs.append(float(np.arctan2(float(R[1, 0]), float(R[0, 0]))))
    t = torch.tensor(np.array(ts), dtype=torch.float32).cuda()
    a = torch.tensor(np.array(angs), dtype=torch.float32).cuda()
    return t, a


def feats(inp, metas, drop=(False, False)):
    f = model.extract_feat(inp, metas, drop_cam=drop[0], drop_lidar=drop[1])
    return (f[0] if isinstance(f, (list, tuple)) else f).float()


_rng = np.random.RandomState(0)


def pick_modes():
    """Zieht die Modalitaetskombination fuer die naechste Iteration."""
    if not MODALITY_DROPOUT:
        return (False, False), (False, False)
    return MODE_TABLE[_rng.randint(len(MODE_TABLE))]


cfg.train_dataloader.num_workers = 8
dl = Runner.build_dataloader(cfg.train_dataloader)

heads, opts, scheds = {}, {}, {}
res_m = None
log_path = os.path.join(OUT_DIR, f'train_{MODE}.log')
logf = open(log_path, 'a')


def log(msg):
    print(msg, flush=True)
    logf.write(msg + '\n')
    logf.flush()


log(f'=== MODE={MODE} ARCHS={ARCHS} EPOCHS={EPOCHS} LR={LR} MAX_SHIFT={MAX_SHIFT} '
    f'MODALITY_DROPOUT={MODALITY_DROPOUT} ===')

# Im Overfit-Modus einen kleinen, festen Satz Batches im Speicher halten.
cached = []
if MODE == 'overfit':
    for data in dl:
        with torch.no_grad():
            pd = model.data_preprocessor(data, True)
            inp, sm = pd['inputs'], pd['data_samples']
            mc = [s.metainfo for s in sm['current']]
            mo = [s.metainfo for s in sm['other']]
            d0, d1 = pick_modes()
            f0, f1 = feats(inp['current'], mc, d0), feats(inp['other'], mo, d1)
            gt_t, gt_a = batch_targets(sm)
        cached.append((f0.cpu(), f1.cpu(), gt_t.cpu(), gt_a.cpu()))
        if sum(c[0].shape[0] for c in cached) >= OVERFIT_N:
            break
    n_s = sum(c[0].shape[0] for c in cached)
    log(f'[overfit] {len(cached)} Batches / {n_s} Paare zwischengespeichert')

pred_hist = {a: [] for a in ARCHS}
gt_hist = []
t_start = time.time()
total_iters = EPOCHS * (len(cached) if MODE == 'overfit' else (MAX_ITERS or len(dl)))

it_global = 0
for ep in range(EPOCHS):
    source = cached if MODE == 'overfit' else dl
    for it, item in enumerate(source):
        if MODE != 'overfit' and MAX_ITERS and it >= MAX_ITERS:
            break
        if MODE == 'overfit':
            f0, f1, gt_t, gt_a = (x.cuda() for x in item)
        else:
            with torch.no_grad():
                pd = model.data_preprocessor(item, True)
                inp, sm = pd['inputs'], pd['data_samples']
                mc = [s.metainfo for s in sm['current']]
                mo = [s.metainfo for s in sm['other']]
                d0, d1 = pick_modes()
                f0, f1 = feats(inp['current'], mc, d0), feats(inp['other'], mo, d1)
                gt_t, gt_a = batch_targets(sm)

        if res_m is None:
            pc = cfg.point_cloud_range
            res_m = float(pc[3] - pc[0]) / f0.shape[-1]
            log(f'Features: C={f0.shape[1]} Gitter={f0.shape[-1]}  ->  {res_m:.3f} m/Zelle')
            for a in ARCHS:
                h = build_head(a, f0.shape[1], res_m, f0.shape[-1], MAX_SHIFT).cuda()
                heads[a] = h
                opts[a] = torch.optim.AdamW(h.parameters(), lr=LR, weight_decay=1e-4)
                scheds[a] = torch.optim.lr_scheduler.CosineAnnealingLR(opts[a], max(total_iters, 1))
                log(f'  {a:8s} {sum(p.numel() for p in h.parameters())/1e6:.2f}M Parameter')

        gt_vec = torch.stack([torch.cos(gt_a), torch.sin(gt_a)], 1)
        line = []
        for a in ARCHS:
            pt, pa = heads[a](f0, f1)
            # Translation in Metern; Skalierung ueber die typische Bewegungsgroesse normiert,
            # damit Translations- und Winkelterm vergleichbar gewichtet sind.
            loss_t = torch.nn.functional.smooth_l1_loss(pt / 5.0, gt_t / 5.0)
            # Winkelfehler direkt statt L1 auf [cos,sin]: bei den hier auftretenden kleinen
            # Drehungen variiert cos nur um 0.022, sin um 0.416 -- der cos-Term traegt praktisch
            # keinen Gradienten bei und verduennt das Signal. atan2 der Differenz behandelt
            # zugleich den Umlauf korrekt.
            pred_ang = torch.atan2(pa[:, 1], pa[:, 0])
            d_ang = torch.atan2(torch.sin(pred_ang - gt_a), torch.cos(pred_ang - gt_a))
            loss_a = torch.nn.functional.smooth_l1_loss(d_ang, torch.zeros_like(d_ang))
            loss = loss_t + 2.0 * loss_a
            opts[a].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads[a].parameters(), 5.0)
            opts[a].step()
            scheds[a].step()
            line.append(f'{a}={loss.item():.4f}')
            if it_global % 50 == 0:
                pred_hist[a].append(pt.detach().cpu().numpy())
        if it_global % 50 == 0:
            gt_hist.append(gt_t.detach().cpu().numpy())
        if it_global % 50 == 0:
            msg = f'[ep{ep} it{it}] ' + ' '.join(line)
            if len(gt_hist) >= 3:
                g = np.concatenate(gt_hist)
                gsd = g.std(0)
                diag = []
                for a in ARCHS:
                    p = np.concatenate(pred_hist[a])
                    psd = p.std(0)
                    ratio = float(np.mean(psd / np.maximum(gsd, 1e-6)))
                    diag.append(f'{a}:std{ratio:.2f}')
                msg += '  | pred/gt-Streuung ' + ' '.join(diag)
            log(msg)
        it_global += 1

    for a in ARCHS:
        # pool/n_rot mitschreiben: der Kopf rechnet res_m*pool in Meter um, und angles haengt an
        # n_rot. Ein abweichender Default beim Laden wuerde die Evaluation still verfaelschen.
        h = heads[a]
        torch.save({'state_dict': h.state_dict(), 'arch': a, 'c_in': f0.shape[1],
                    'res_m': res_m, 'grid': f0.shape[-1], 'max_shift': MAX_SHIFT,
                    'pool': int(getattr(h, 'pool', 2)),
                    'n_rot': int(h.angles.numel()) if hasattr(h, 'angles') else 0,
                    'max_deg': float(getattr(h, 'max_deg', 0.0))},
                   os.path.join(OUT_DIR, f'head_{a}.pth'))
    log(f'[ep{ep}] gespeichert nach {OUT_DIR}  ({(time.time()-t_start)/60:.1f} min)')

# Abschlussdiagnose: wer sagt eine Konstante voraus, wer nicht?
g = np.concatenate(gt_hist)
log('')
log('--- Abschluss: Streuung der Vorhersagen gegen GT (Kollaps-Test) ---')
log(f'GT-Streuung: x={g.std(0)[0]:.3f} y={g.std(0)[1]:.3f}')
for a in ARCHS:
    p = np.concatenate(pred_hist[a])
    r = p.std(0) / np.maximum(g.std(0), 1e-6)
    verdict = 'KOLLABIERT' if float(np.mean(r)) < 0.1 else ('schwach' if float(np.mean(r)) < 0.4 else 'lernt')
    log(f'  {a:8s} pred-std x={p.std(0)[0]:.3f} y={p.std(0)[1]:.3f}  Verhaeltnis={float(np.mean(r)):.2f}  {verdict}')
logf.close()
