"""KITTI keypoint VO: per-frame model inference -> consecutive descriptor-MNN + RANSAC SE(2)
-> accumulated 2D trajectory, plus trajectory plotting.
GT poses are used for evaluation/titles ONLY, never in estimation."""
import os
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29993')
os.environ.setdefault('RANK', '0'); os.environ.setdefault('WORLD_SIZE', '1'); os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, torch, torch.distributed as dist
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.dataset import pseudo_collate as collate
import bevmatch  # noqa: F401  (registriert die Komponenten): F401  registers model + heads
from keypoint_extraction import (peak_cells as _peak_cells,
                                 mutual_nn_matches as _mutual_nn_matches,
                                 cells_to_xy as _cells_to_xy)
from kitti_se2 import ransac_se2, accumulate, positions, se2, se2_inv, umeyama_align_2d, pose_to_xytheta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# KeypointHead.__init__ calls dist.get_rank() (rank-0-only logging), so a process group must
# be live even for single-GPU inference -- matches the established pattern in
# tools/kpt_ransac_pairs.py / tools/kitti_smoke.py.
torch.cuda.set_device(0)
if not dist.is_initialized():
    dist.init_process_group('nccl', rank=0, world_size=1)

TOPK = int(os.environ.get('TOPK', '256'))
RANSAC_THR_M = float(os.environ.get('RANSAC_THR_M', '1.0'))


def rel_pose_from_poses(l2g_c, l2g_o):
    """GT relative BEV pose: T = inv(l2g_o) @ l2g_c maps current-lidar xy -> other-lidar xy.
    Returns (rot_deg, trans_m). Local copy of tools/kpt_ransac_pairs.rel_pose_from_poses:
    that module is a top-level script (builds its own model/dataloader at import time from
    MH_CFG/CKPT env vars with no `if __name__` guard), so importing the function from it would
    also execute all of that unrelated setup. This is a pure 5-line function with the same
    signature/semantics; duplicating it here avoids the unwanted import-time side effects."""
    T = np.linalg.inv(np.asarray(l2g_o, np.float64)) @ np.asarray(l2g_c, np.float64)
    rot = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
    trans = float(np.linalg.norm(T[:2, 3]))
    return rot, trans


def build_model_and_ds(cfg_path, ckpt):
    init_default_scope('mmdet3d')
    cfg = Config.fromfile(cfg_path)
    from mmengine.registry import MODELS, DATASETS
    model = MODELS.build(cfg.model).cuda().eval()
    load_checkpoint(model, ckpt, map_location='cuda', strict=False)
    ds_cfg = cfg.val_dataloader.dataset
    if os.environ.get('ANN_FILE'):                 # point the loader at a specific sequence's pkl
        ds_cfg = dict(ds_cfg); ds_cfg['ann_file'] = os.environ['ANN_FILE']
    ds = DATASETS.build(ds_cfg)
    return model, ds


@torch.no_grad()
def infer_frame(model, head, ds, idx, topk):
    batch = collate([ds[idx]])
    data = model.data_preprocessor(batch, training=False)
    inp, sm = data['inputs'], data['data_samples']
    cur = sm['current']; mc = [d.metainfo for d in cur]
    # Modality-Flags des Modells respektieren (camera-only-/lidar-only-Modelle, A7)
    f0 = model.extract_feat(inp['current'], mc,
                            drop_cam=not getattr(model, 'use_camera', True),
                            drop_lidar=not getattr(model, 'use_lidar', True))
    k0, d0, _ = head.forward(f0[0], None)
    h0 = k0[0, 0]; dim = h0.shape[-1]; res = getattr(head, 'bev_resolution', 108.0 / dim)
    c0 = _peak_cells(head, k0, topk)
    xy0 = _cells_to_xy(c0.cpu().numpy(), dim, res)
    l2g = model.lidar2global(mc[0]); l2g = np.asarray(l2g.cpu() if hasattr(l2g, 'cpu') else l2g)
    return dict(h=h0, c=c0, d=d0, xy=xy0, l2g=l2g, dim=dim, res=res)


SUBCELL = os.environ.get('SUBCELL', '0') == '1'


def _cells_to_xy_sc(hlogits, cells_np, dim, res):
    """Zell->Meter, optional mit SUB-ZELL-Verfeinerung (SUBCELL=1): sigmoid-gewichteter
    3x3-Schwerpunkt um jede Peak-Zelle -> kontinuierlicher Offset in [-1,1] Zellen.
    Antwort auf R8 'no sub-cell keypoint localization' — reine Inferenz, kein Retraining."""
    xy = _cells_to_xy(cells_np, dim, res)
    if not SUBCELL or len(cells_np) == 0:
        return xy
    p = torch.sigmoid(hlogits.float()).cpu().numpy()
    off = np.zeros_like(xy)
    for k, (r, c) in enumerate(cells_np):
        r0, r1 = max(r - 1, 0), min(r + 2, dim)
        c0, c1 = max(c - 1, 0), min(c + 2, dim)
        w = p[r0:r1, c0:c1]
        s = w.sum()
        if s > 1e-6:
            rr, cc = np.mgrid[r0:r1, c0:c1]
            off[k] = [(rr * w).sum() / s - r, (cc * w).sum() / s - c]
    return xy + off * res


def run_vo(model, ds, out_dir, max_frames=0, frame_stride=1):
    os.makedirs(out_dir, exist_ok=True)
    head = model.keypoint_head
    F = len(ds) if max_frames in (0, None) else min(max_frames, len(ds))
    frame_stride = max(1, int(frame_stride))
    prev = None; prev_idx = None; rel_list = []; gt_xy = []; failed = []
    hm_u8 = []; cells = []; pair_matches = []; pair_inl = []; pair_txt = []
    last_rel_T = np.eye(3)
    for idx in range(0, F, frame_stride):
        fr = infer_frame(model, head, ds, idx, TOPK)
        gt_xy.append(fr['l2g'][:2, 3])
        hm_u8.append((torch.sigmoid(fr['h'].float()).cpu().numpy() * 255).astype(np.uint8))
        cells.append(fr['c'].cpu().numpy().astype(np.int16))
        if prev is not None:
            matches = _mutual_nn_matches(prev['d'], prev['c'], fr['d'], fr['c'], binary=False)
            if matches:
                xy0 = _cells_to_xy_sc(prev['h'], prev['c'][[m[0] for m in matches]].cpu().numpy(), prev['dim'], prev['res'])
                xy1 = _cells_to_xy_sc(fr['h'], fr['c'][[m[1] for m in matches]].cpu().numpy(), fr['dim'], fr['res'])
                T, mask = ransac_se2(xy0, xy1, RANSAC_THR_M)
            else:
                T, mask = None, None
            if T is None:
                T = last_rel_T.copy(); failed.append(idx)          # constant-velocity fallback
                mask = np.zeros(len(matches), bool)
            last_rel_T = T
            rel_list.append([T[0, 2], T[1, 2], float(np.arctan2(T[1, 0], T[0, 0]))])
            pair_matches.append(np.asarray(matches, np.int32).reshape(-1, 2))
            pair_inl.append(np.asarray(mask, bool))
            rg, tg = rel_pose_from_poses(_h4(prev['l2g']), _h4(fr['l2g']))
            n_in = int(mask.sum())
            pair_txt.append(f'frame {prev_idx}->{idx} (stride {frame_stride})  '
                            f'matches={len(matches)} inliers={n_in}  '
                            f'GT rot={rg:+.1f}deg trans={tg:.2f}m')
        prev = fr; prev_idx = idx
        if idx % 50 == 0:
            print(f'[vo] {idx}/{F} (stride {frame_stride})', flush=True)
    world = accumulate([np.array([[np.cos(th), -np.sin(th), x], [np.sin(th), np.cos(th), y], [0, 0, 1]])
                        for (x, y, th) in rel_list])
    est_xy = positions(world)
    np.save(os.path.join(out_dir, 'traj_est.npy'), est_xy)
    np.save(os.path.join(out_dir, 'traj_gt.npy'), np.asarray(gt_xy, np.float64))
    np.save(os.path.join(out_dir, 'rel_poses.npy'), np.asarray(rel_list, np.float64))
    np.save(os.path.join(out_dir, 'vo_failed.npy'), np.asarray(failed, np.int64))
    np.savez_compressed(os.path.join(out_dir, 'matches_cache.npz'),
                        hm_u8=np.stack(hm_u8, 0),
                        cells=np.array(cells, dtype=object),
                        pair_matches=np.array(pair_matches, dtype=object),
                        pair_inl=np.array(pair_inl, dtype=object),
                        pair_titletxt=np.array(pair_txt, dtype=object))
    print(f'[vo] DONE F={F} failed={len(failed)} est/gt/cache written to {out_dir}', flush=True)
    return est_xy, np.asarray(gt_xy, np.float64)


def _h4(m):
    m = np.asarray(m, np.float64)
    return m if m.shape == (4, 4) else np.eye(4)


def _rel_xytheta(Ti, Tj):
    return pose_to_xytheta(se2_inv(Ti) @ Tj)


