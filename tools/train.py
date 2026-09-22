# Copyright (c) OpenMMLab. All rights reserved.
# Training entry point for BEVMatch, derived from the MMDetection3D tools/train.py
# (Apache 2.0, see NOTICE). Reduced to what the released configs need.
"""Train BEVMatch.

    python tools/train.py configs/bevmatch_nuscenes.py
    python tools/train.py configs/bevmatch_kitti_odometry.py

Distributed training uses the standard MMEngine launcher, e.g.

    torchrun --nproc_per_node=8 tools/train.py <config> --launcher pytorch

The config refers to every component by name; importing ``bevmatch`` below registers
them with the MMDetection3D registry, which is why no further wiring is needed.
"""
import argparse
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet3d.utils import register_all_modules

# Point clouds keep many file descriptors open in the workers.
torch.multiprocessing.set_sharing_strategy('file_system')

register_all_modules(init_default_scope=True)
import mmdet3d.models  # noqa: F401,E402  (populates the model registry)

import bevmatch  # noqa: F401,E402  (registers BEVMatch's own components)


def parse_args():
    p = argparse.ArgumentParser(description='Train BEVMatch')
    p.add_argument('config', help='path to a config file')
    p.add_argument('--work-dir', help='directory for checkpoints and logs')
    p.add_argument('--resume', nargs='?', type=str, const='auto',
                   help='resume from the latest checkpoint, or from the given one')
    p.add_argument('--amp', action='store_true', help='enable mixed-precision training')
    p.add_argument('--auto-scale-lr', action='store_true',
                   help='scale the learning rate to the actual batch size')
    p.add_argument('--cfg-options', nargs='+', action=DictAction,
                   help='override config entries, e.g. --cfg-options model.use_lidar=False')
    p.add_argument('--launcher', default='none',
                   choices=['none', 'pytorch', 'slurm', 'mpi'])
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = p.parse_args()
    # torchrun communicates the rank through the environment.
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = (args.work_dir or cfg.get('work_dir')
                    or osp.join('work_dirs', osp.splitext(osp.basename(args.config))[0]))

    if args.amp:
        # Keep an already configured AmpOptimWrapper untouched.
        if cfg.optim_wrapper.type != 'AmpOptimWrapper':
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    if args.resume is not None:
        cfg.resume = True
        cfg.load_from = None if args.resume == 'auto' else args.resume

    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
