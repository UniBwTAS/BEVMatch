"""Schreibt einen KONSISTENTEN Cache: Zellen, Deskriptoren und SUB-ZELL-verfeinerte
metrische Keypoints stammen aus DEMSELBEN Forward-Pass.

Wichtig: die Inferenz ist auf der GPU nicht bit-deterministisch -- minimal abweichende Scores
aendern die Top-K-Auswahl. Ein zweiter Pass, der nur xy_sub nachtraegt, ordnet die Koordinaten
daher einer anderen Keypoint-Auswahl zu als die bereits gecachten Deskriptoren. Deshalb wird
hier alles gemeinsam neu geschrieben statt ergaenzt.

Fuer die Verfeinerung wird exakt die Implementierung aus kitti_kpt_odometry verwendet
(_cells_to_xy_sc, per Modul-Flag aktiviert), damit die Formel identisch zu den publizierten
Sub-Zell-Zahlen ist -- kein Nachbau. Die Deskriptor-Extraktion entspricht 1:1
kitti_cache_descriptors.py. Berichtet wird, wie stark die Zellauswahl vom alten Cache abweicht.

Env: KPT_ODOM_CFG, CKPT, CACHE (Verzeichnis mit desc_cache.npz)
"""
import os
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29996')
os.environ.setdefault('RANK', '0'); os.environ.setdefault('WORLD_SIZE', '1'); os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np
import kitti_kpt_odometry as ko
from kitti_kpt_odometry import build_model_and_ds, infer_frame, TOPK


def main():
    CFG = os.environ['KPT_ODOM_CFG']; CKPT = os.environ['CKPT']
    CACHE = os.environ.get('CACHE', 'work_dirs_kitti_kpt_desc/odom_seq09_e10')
    npz = os.path.join(CACHE, 'desc_cache.npz')
    z = np.load(npz, allow_pickle=True)
    cells_old = z['cells']; F = len(cells_old)
    ko.SUBCELL = True                       # aktiviert die Verfeinerung in _cells_to_xy_sc
    model, ds = build_model_and_ds(CFG, CKPT)
    head = model.keypoint_head
    import torch
    xy_sub = []; cells_new = []; desc_new = []; mism = 0; kp_diff = 0; kp_tot = 0
    for idx in range(F):
        fr = infer_frame(model, head, ds, idx, TOPK)
        c = fr['c'].cpu().numpy()
        old = np.asarray(cells_old[idx])
        if c.shape == old.shape:
            kp_diff += int((c != old).any(1).sum()); kp_tot += len(c)
            if not np.array_equal(c, old):
                mism += 1
        else:
            mism += 1; kp_tot += len(c); kp_diff += len(c)
        with torch.no_grad():                       # Deskriptoren AUS DIESEM Pass
            d = fr['d'][0, :, c[:, 0], c[:, 1]].T
            d = torch.nn.functional.normalize(d.float(), dim=1)
        desc_new.append(d.cpu().numpy().astype(np.float16))
        cells_new.append(c)
        xy_sub.append(ko._cells_to_xy_sc(fr['h'], c, fr['dim'], fr['res']).astype(np.float32))
        if idx % 200 == 0:
            print(f'[sc] {idx}/{F}', flush=True)
    print(f'[sc] Frames mit abweichender Zellauswahl: {mism}/{F}; '
          f'abweichende Keypoints: {kp_diff}/{kp_tot} ({100*kp_diff/max(kp_tot,1):.2f}%)', flush=True)
    out = {k: z[k] for k in z.keys()}
    out['xy_sub'] = np.array(xy_sub, dtype=object)
    out['cells'] = np.array(cells_new, dtype=object)      # konsistent zu xy_sub und desc
    out['desc'] = np.array(desc_new, dtype=object)
    np.savez_compressed(npz, **out)
    print(f'[sc] xy_sub in {npz} geschrieben', flush=True)
    print('[sc] DONE', flush=True)


if __name__ == '__main__':
    main()
