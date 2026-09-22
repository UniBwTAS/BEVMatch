"""GPU pass: cache per-keypoint DESCRIPTORS (+ cells + GT pose) for KITTI seq frames, so the
descriptor-association SLAM can run offline on CPU. Samples the 128-D descriptor field at the
detected peak cells (desc_field[0,:,row,col].T), L2-normalizes, stores float16.

Env: KPT_ODOM_CFG, CKPT, SEQ_LEN cap via MAX_FRAMES (0=all), FRAME_STRIDE (1), TOPK (256),
OUTDIR (where desc_cache.npz is written)."""
import os
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29994')
os.environ.setdefault('RANK', '0'); os.environ.setdefault('WORLD_SIZE', '1'); os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np
import torch
from kitti_kpt_odometry import build_model_and_ds, infer_frame, TOPK


def main():
    CFG = os.environ['KPT_ODOM_CFG']; CKPT = os.environ['CKPT']
    OUT = os.environ.get('OUTDIR', 'work_dirs_kitti_kpt_desc/odom_seq09_e10')
    MAXF = int(os.environ.get('MAX_FRAMES', '0'))
    STRIDE = max(1, int(os.environ.get('FRAME_STRIDE', '1')))
    os.makedirs(OUT, exist_ok=True)
    model, ds = build_model_and_ds(CFG, CKPT)
    head = model.keypoint_head
    F = len(ds) if MAXF in (0, None) else min(MAXF, len(ds))
    cells = []; descs = []; gt_xy = []
    for idx in range(0, F, STRIDE):
        fr = infer_frame(model, head, ds, idx, TOPK)
        c = fr['c']                                  # LongTensor [N,2] (row,col)
        d = fr['d']                                  # descriptor field [1,D,H,W]
        with torch.no_grad():
            dk = d[0, :, c[:, 0], c[:, 1]].T          # [N,D]
            dk = torch.nn.functional.normalize(dk.float(), dim=1)
        cells.append(c.cpu().numpy().astype(np.int16))
        descs.append(dk.cpu().numpy().astype(np.float16))
        gt_xy.append(fr['l2g'][:2, 3])
        if idx % 100 == 0:
            print(f'[desc] {idx}/{F}', flush=True)
    dim = int(fr['dim'])
    np.savez_compressed(os.path.join(OUT, 'desc_cache.npz'),
                        cells=np.array(cells, dtype=object),
                        desc=np.array(descs, dtype=object),
                        gt_xy=np.asarray(gt_xy, np.float64),
                        dim=dim)
    print(f'[desc] DONE frames={len(cells)} dim={dim} -> {OUT}/desc_cache.npz', flush=True)


if __name__ == '__main__':
    main()
