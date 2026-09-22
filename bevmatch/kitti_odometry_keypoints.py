# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.dataset import BaseDataset

from mmdet3d.registry import DATASETS, TRANSFORMS


@TRANSFORMS.register_module()
class WrapSingleImageAsMultiView(BaseTransform):
    """Wrap a single-camera ``results['img']`` array into a length-1 list.

    KITTI odometry has ONE camera, but the BEVFusionKeypoints model's
    camera path (``extract_img_feat``: ``B, N, C, H, W = x.size()``) and
    ``view_transform`` (``DepthLSSTransform.forward`` indexes
    ``lidar2image[b][:, :3, :3]`` etc., i.e. expects a leading per-camera
    axis) are hard-wired for MULTI-VIEW (N-camera) inputs. Running this
    transform right after the single-image loader (e.g. mmcv's
    ``LoadImageFromFile``, which sets ``results['img']`` to a bare
    ``(H, W, 3)`` array) turns it into a 1-element list so
    ``Pack3DDetInputs.pack_single_results`` takes its "multiple imgs"
    branch (``np.stack(results['img'], axis=0)`` -> permute -> a rank-4
    ``(1, C, H, W)`` tensor) instead of the flat rank-3 single-image
    branch. The data preprocessor's ``collate_data`` then stacks the
    batch into ``(B, 1, C, H, W)`` -- exactly the ``N=1`` multi-view shape
    the model expects. Downstream transforms that iterate
    ``for img in results['img']`` (e.g. ``ImageAug3D``) work unchanged on
    a 1-element list.
    """

    def transform(self, results: dict) -> dict:
        if not isinstance(results['img'], list):
            results['img'] = [results['img']]
        return results


@DATASETS.register_module()
class KittiOdometryKeypoints(BaseDataset):
    """KITTI-odometry pair loader for self-supervised keypoint training.

    Consumes the Task-1 info pkl (``data/kitti_odom/kitti_odom_infos_*.pkl``,
    an mmengine ``{'data_list': [...], 'metainfo': {...}}`` file where each
    info dict has ``seq, frame, lidar_path, image_path, ego2global,
    lidar2img, cam2img, cam2lidar``) and emits a ``{current, other}`` frame
    pair: two frames from the SAME sequence whose ego-motion baseline lies
    in ``[min_baseline, max_baseline]`` metres.

    The pair-nesting mirrors ``NuScenesDatasetKeypoints.__getitem__``
    (mmdet3d/datasets/nuscenes_dataset_keypoints.py, ~L346-406) exactly:
    ``__getitem__`` (NOT ``prepare_data``) is overridden, and
    ``data['current']`` / ``data['other']`` are each obtained via
    ``super().__getitem__(idx)`` -- i.e. each is a FULL
    ``{'inputs': ..., 'data_samples': Det3DDataSample}`` sample as produced
    by the normal ``BaseDataset`` pipeline (``Pack3DDetInputs``), not a
    nested ``{'inputs': {'current': ..., 'other': ...}}`` shape. This is
    what the BEVFusionKeypoints model's ``loss()`` and the dataloader
    collate expect: ``batch_inputs_dict["current"]['points']/['imgs']`` and
    ``batch_data_samples["current"]``.

    Args:
        ann_file (str): Path to the Task-1 info pkl.
        pipeline (list[dict]): Data pipeline, typically
            ``LoadPointsFromFile`` + an image loader + ``Pack3DDetInputs``.
        min_baseline (float): Minimum ego-motion translation (metres,
            BEV/xy only) between ``current`` and ``other``. Defaults to 1.0.
        max_baseline (float): Maximum ego-motion translation (metres).
            Defaults to 15.0.
        max_gap (int): Maximum index distance (within the same sequence)
            searched for an ``other`` candidate. Defaults to 20.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[dict],
                 min_baseline: float = 1.0,
                 max_baseline: float = 15.0,
                 max_gap: int = 20,
                 test_mode: bool = False,
                 **kwargs) -> None:
        self.min_baseline = min_baseline
        self.max_baseline = max_baseline
        self.max_gap = max_gap
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Convert a raw Task-1 info dict into the keys the pipeline /
        model metainfo interface expects.

        Sets ``lidar_points={'lidar_path':..., 'lidar2ego': eye(4)}`` (KITTI
        odometry has no separate lidar-to-ego calibration, so ``lidar2ego``
        is identity) and ``img_path`` for the single-camera image loader.
        ``ego2global`` is kept as a top-level 4x4 unchanged.

        ``lidar2img``/``cam2img``/``cam2lidar`` are each wrapped as a
        length-1 LIST of their 4x4 matrix (i.e. ``[matrix]`` instead of
        ``matrix``) -- KITTI odometry has ONE camera, but
        ``BEVFusionKeypoints.setupTransformsCamera``/``extract_img_feat``/
        ``view_transform`` are hard-wired for MULTI-VIEW (N-camera) inputs:
        each per-sample meta value is appended into a batch list and then
        ``np.asarray``'d, so a bare 4x4 collapses to a ``(B, 4, 4)`` tensor
        (missing the camera axis N) while `DepthLSSTransform`'s per-camera
        indexing (``lidar2image[b][:, :3, :3]``) needs ``(B, N, 4, 4)``.
        Wrapping as a 1-element list here (paired with
        ``WrapSingleImageAsMultiView`` on the image side, see this module)
        makes KITTI's single camera present as an ``N=1`` multi-view frame,
        exactly like the multi-camera nuScenes/pad_scenes path but with
        one camera instead of six.
        """
        info = dict(raw_data_info)
        info['lidar_points'] = {
            'lidar_path': info['lidar_path'],
            'lidar2ego': np.eye(4).tolist(),
        }
        info['img_path'] = info['image_path']
        info['lidar2img'] = [np.array(info['lidar2img'], dtype=np.float32)]
        info['cam2img'] = [np.array(info['cam2img'], dtype=np.float32)]
        info['cam2lidar'] = [np.array(info['cam2lidar'], dtype=np.float32)]
        return info

    def _same_seq_range(self, idx: int) -> Tuple[int, int]:
        """Return the inclusive [lo, hi] index range around ``idx`` that
        stays within the same sequence and within ``max_gap`` steps."""
        seq = self.get_data_info(idx)['seq']
        lo = idx
        while (lo - 1 >= 0
               and self.get_data_info(lo - 1)['seq'] == seq
               and idx - (lo - 1) <= self.max_gap):
            lo -= 1
        hi = idx
        n = len(self)
        while (hi + 1 < n
               and self.get_data_info(hi + 1)['seq'] == seq
               and (hi + 1) - idx <= self.max_gap):
            hi += 1
        return lo, hi

    def pick_other(self, idx: int) -> int:
        """Pick a same-sequence frame index whose ego-motion baseline from
        ``idx`` (2D translation norm, BEV) lies in
        ``[min_baseline, max_baseline]`` metres.

        Falls back to ``idx`` itself if no candidate within ``max_gap``
        steps satisfies the baseline range (e.g. a short / stationary
        sequence tail).
        """
        info_i = self.get_data_info(idx)
        Ei = np.array(info_i['ego2global'])
        lo, hi = self._same_seq_range(idx)
        candidates = []
        for j in range(lo, hi + 1):
            if j == idx:
                continue
            Ej = np.array(self.get_data_info(j)['ego2global'])
            baseline = float(np.linalg.norm((np.linalg.inv(Ej) @ Ei)[:2, 3]))
            if self.min_baseline <= baseline <= self.max_baseline:
                candidates.append(j)
        if not candidates:
            return idx
        return int(np.random.choice(candidates))

    def __getitem__(self, idx: int) -> dict:
        """Emit the ``{current, other}`` pair.

        Mirrors ``NuScenesDatasetKeypoints.__getitem__``: builds each half
        of the pair by calling the base ``__getitem__`` (pipeline +
        ``Pack3DDetInputs``) independently for ``idx`` and ``pick_other(idx)``,
        then marks both ``inputs['valid'] = True``.
        """
        other_idx = self.pick_other(idx)

        data = {}
        data['current'] = super().__getitem__(idx)
        data['other'] = super().__getitem__(other_idx)
        data['current']['inputs']['valid'] = True
        data['other']['inputs']['valid'] = True
        return data
