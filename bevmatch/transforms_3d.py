# modify from https://github.com/mit-han-lab/bevfusion
import json
import os
from typing import Any, Dict

import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.datasets import GlobalRotScaleTrans
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # update the calibration matrices
        data['img_aug_matrix'] = transforms
        return data


@TRANSFORMS.register_module()
class BEVFusionRandomFlip3D:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()

        if 'lidar_aug_matrix' not in data:
            data['lidar_aug_matrix'] = np.eye(4)
        data['lidar_aug_matrix'][:3, :] = rotation @ data[
            'lidar_aug_matrix'][:3, :]
        return data


@TRANSFORMS.register_module()
class BEVFusionGlobalRotScaleTrans(GlobalRotScaleTrans):
    """Compared with `GlobalRotScaleTrans`, the augmentation order in this
    class is rotation, translation and scaling (RTS)."""

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._trans_bbox_points(input_dict)
        self._scale_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'T', 'S'])

        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = input_dict['pcd_rotation'].T * input_dict[
            'pcd_scale_factor']
        lidar_augs[:3, 3] = input_dict['pcd_trans'] * \
            input_dict['pcd_scale_factor']

        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = np.eye(4)
        input_dict[
            'lidar_aug_matrix'] = lidar_augs @ input_dict['lidar_aug_matrix']

        return input_dict


@TRANSFORMS.register_module()
class GridMask(BaseTransform):

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def transform(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.length = np.random.randint(1, d)
        else:
            self.length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.length, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.length, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results


# ---------------------------------------------------------------------------
# LiDAR beam-thinning + random box dropout  (ported from BEVFusion project)
# ---------------------------------------------------------------------------

# Hardcoded VLS-128 (Velodyne Alpha Prime) vert_correction in radians, sorted
# ascending by elevation. Source:
#   (ROS velodyne_pointcloud driver)
#       params/VLS128.yaml
# 128 beams spanning -25° to +15°. Dense between -7° and +5° (~90 beams at
# 0.11° spacing); sparse outside (~38 beams at ~1° spacing). The dense band
# is the "horizon eye"; the sparse below-horizon band is what scans
# close-range ground.
VLS128_ELEVATIONS_RAD = np.array([
    -0.4363323129985824, -0.34177037412552963, -0.2799857186049304, -0.23675391303303078,
    -0.2049365607691742, -0.18057176441133332, -0.16133823605435582, -0.1457698991265664,
    -0.13351768777756623, -0.12479104151759457, -0.11955505376161157, -0.11606439525762292,
    -0.11344640137963143, -0.11152653920243766, -0.1096066770252439, -0.10768681484805014,
    -0.10576695267085637, -0.10384709049366261, -0.10192722831646885, -0.10000736613927509,
    -0.09808750396208132, -0.09616764178488756, -0.0942477796076938, -0.09232791743050003,
    -0.09040805525330627, -0.08848819307611251, -0.08656833089891874, -0.08464846872172498,
    -0.08272860654453122, -0.08080874436733745, -0.07888888219014369, -0.07696902001294993,
    -0.07504915783575616, -0.07312929565856241, -0.07120943348136864, -0.06928957130417489,
    -0.06736970912698112, -0.06544984694978735, -0.0635299847725936, -0.06161012259539983,
    -0.05969026041820607, -0.05777039824101231, -0.05585053606381855, -0.05393067388662478,
    -0.05201081170943102, -0.05009094953223726, -0.04817108735504349, -0.046251225177849735,
    -0.044331363000655974, -0.04241150082346221, -0.040491638646268445, -0.038571776469074684,
    -0.03665191429188092, -0.034732052114687155, -0.032812189937493394, -0.030892327760299633,
    -0.02897246558310587, -0.027052603405912107, -0.025132741228718343, -0.023212879051524585,
    -0.02129301687433082, -0.01937315469713706, -0.017453292519943295, -0.015533430342749533,
    -0.013613568165555772, -0.011693705988362009, -0.009773843811168246, -0.007853981633974483,
    -0.005934119456780721, -0.004014257279586958, -0.0020943951023931952, -0.00017453292519943296,
    0.0017453292519943296, 0.003665191429188092, 0.005585053606381855, 0.007504915783575617,
    0.00942477796076938, 0.011344640137963142, 0.013264502315156905, 0.015184364492350668,
    0.01710422666954443, 0.019024088846738195, 0.020943951023931952, 0.022863813201125717,
    0.024783675378319478, 0.026703537555513242, 0.028623399732707003, 0.030543261909900768,
    0.03246312408709453, 0.03438298626428829, 0.03630284844148206, 0.03822271061867582,
    0.04014257279586958, 0.04206243497306335, 0.0439822971502571, 0.04590215932745086,
    0.04782202150464463, 0.04974188368183839, 0.05166174585903215, 0.053581608036225914,
    0.05550147021341968, 0.05742133239061344, 0.059341194567807204, 0.061261056745000965,
    0.06318091892219473, 0.0651007810993885, 0.06702064327658225, 0.06894050545377602,
    0.07086036763096977, 0.07278022980816354, 0.0747000919853573, 0.07661995416255106,
    0.07853981633974483, 0.0804596785169386, 0.08237954069413235, 0.08429940287132612,
    0.08691739674931762, 0.09040805525330627, 0.0947713783832921, 0.10000736613927509,
    0.10611601852125524, 0.11309733552923257, 0.12182398178920421, 0.1322959573011702,
    0.14713125594312199, 0.16929693744344995, 0.20507618710933373, 0.2617993877991494,
], dtype=np.float32)


@TRANSFORMS.register_module()
class KeepBeamsByElevation(BaseTransform):
    """Subsample LiDAR points by beam, inferring beam ID from elevation.

    Specialised for the VLS-128 case: the sensor's beam distribution is
    massively non-uniform — ~90 beams packed at 0.11° spacing between
    -7° and +5° (the "horizon eye"), only ~38 beams across the rest of
    the FOV (sparse, ~1° spacing). This transform **keeps every sparse
    beam as-is and decimates only the dense band**, leaving a roughly
    uniform ~1° angular resolution across the full FOV. With default
    parameters (dense band -7°..+5°, stride 9) the 128-beam point cloud
    is reduced to ~48 beams without losing close-range coverage that
    relies on the sparse below-horizon beams.

    .bin files from rosbag2nuscenes don't carry a per-point ring index
    (they're written as (x, y, z, intensity) only). So we recover the
    beam ID from each point's elevation angle by binary-searching into
    the VLS-128 vert_correction table.

    Apply this transform in BOTH train and test pipelines if the goal
    is to permanently train with a "cheaper" sensor profile. Apply it
    in train only if you want it as augmentation.

    Args:
        dense_band_deg: (low, high) elevation band (degrees) in which to
            decimate beams. Default (-7, 5) targets the VLS-128 dense
            horizon zone.
        dense_stride: keep every Nth beam in the dense band (by sorted
            elevation order). Default 9 → ~1.0° spacing inside the band.
        sensor_height_m: unused for filtering, kept for documentation /
            future use.
    """

    def __init__(self,
                 dense_band_deg=(-7.0, 5.0),
                 dense_stride: int = 9,
                 sensor_height_m: float = 1.7,
                 min_source_beams: int = 64):
        if dense_stride < 1:
            raise ValueError(
                f'dense_stride must be >= 1, got {dense_stride}')
        self.min_source_beams = int(min_source_beams)
        low_deg, high_deg = dense_band_deg
        if low_deg >= high_deg:
            raise ValueError(
                f'dense_band_deg must be (low, high) with low < high, '
                f'got {dense_band_deg}')

        self.dense_band_deg = (float(low_deg), float(high_deg))
        self.dense_stride = int(dense_stride)
        self.sensor_height_m = float(sensor_height_m)

        elevs = VLS128_ELEVATIONS_RAD                       # (128,) ascending
        low_rad = float(np.deg2rad(low_deg))
        high_rad = float(np.deg2rad(high_deg))
        in_dense = (elevs >= low_rad) & (elevs < high_rad)
        keep_mask = np.zeros_like(elevs, dtype=bool)
        keep_mask[~in_dense] = True                         # all sparse beams
        dense_indices = np.where(in_dense)[0]
        keep_mask[dense_indices[::dense_stride]] = True     # decimated dense

        midpoints = (elevs[:-1] + elevs[1:]) / 2.0          # (127,)

        self._midpoints = torch.from_numpy(midpoints).float()
        self._keep_mask = torch.from_numpy(keep_mask)
        self._n_beams_kept = int(keep_mask.sum())
        self._n_beams_total = int(keep_mask.size)

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'dense_band_deg={self.dense_band_deg}, '
                f'dense_stride={self.dense_stride}, '
                f'kept={self._n_beams_kept}/{self._n_beams_total} beams)')

    def transform(self, data):
        points = data.get('points')
        if points is None or len(points) == 0:
            return data
        xy = points.tensor[:, :2]
        z = points.tensor[:, 2]
        r_xy = (xy * xy).sum(dim=1).sqrt().clamp(min=1e-6)
        elev = torch.atan2(z, r_xy)                         # (N,)
        distinct_beams = int(torch.unique(torch.round(elev * 1000)).numel())
        assert distinct_beams >= self.min_source_beams, (
            f"KeepBeamsByElevation expects VLS-128-class input (>= "
            f"{self.min_source_beams} beams); got ~{distinct_beams}. Do NOT "
            f"apply beam thinning to 32-beam nuScenes.")
        midpoints = self._midpoints.to(elev.device)
        keep_mask = self._keep_mask.to(elev.device)
        beam_idx = torch.searchsorted(midpoints, elev)      # 0..127
        beam_idx = beam_idx.clamp(0, self._n_beams_total - 1)
        keep = keep_mask[beam_idx]
        data['points'] = points[keep]
        return data


@TRANSFORMS.register_module()
class RandomLiDARBoxDropout(BaseTransform):
    """3D Cutout / Random Erasing for LiDAR.

    With probability `prob` per iter, picks a random axis-aligned box (in
    the xy plane, all heights) placed randomly anywhere within
    `[-extent_m, +extent_m]^2`, and removes every LiDAR point inside that
    box. GT labels (boxes, seg masks, centers) are unchanged — the model
    is asked to detect whatever was there using cameras alone.

    Unlike `SpatialLiDARDropout` (which always punches the same hole at
    the ego origin, simulating close-range blindness), this picks a
    random hole location each iter, so the model is forced to handle
    "LiDAR is missing somewhere" as a general failure mode rather than
    specifically at close range. Cameras and LiDAR voxelisation are
    otherwise untouched.

    Args:
        prob: probability of applying the cutout each iter.
        box_size_m: either a scalar (fixed side length, in metres) or a
            ``(low, high)`` tuple. If a tuple, the side length is sampled
            uniformly in `[low, high]` on each call.
        extent_m: half-extent of the region the box centre is sampled
            from (centre coords sampled uniformly in
            [-extent + box/2, +extent - box/2] so the full box stays
            inside the ROI).
    """

    def __init__(self,
                 prob: float = 0.5,
                 box_size_m=(3.0, 12.0),
                 extent_m: float = 30.0):
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f'prob must be in [0, 1], got {prob}')
        if isinstance(box_size_m, (tuple, list)):
            if len(box_size_m) != 2:
                raise ValueError(
                    f'box_size_m tuple must have length 2, got '
                    f'{box_size_m}')
            low, high = float(box_size_m[0]), float(box_size_m[1])
            if low <= 0 or high <= 0 or low > high:
                raise ValueError(
                    f'box_size_m range must satisfy 0 < low <= high, '
                    f'got {box_size_m}')
            self._size_low = low
            self._size_high = high
        else:
            size = float(box_size_m)
            if size <= 0:
                raise ValueError(
                    f'box_size_m must be > 0, got {box_size_m}')
            self._size_low = size
            self._size_high = size
        if extent_m <= self._size_high / 2:
            raise ValueError(
                f'extent_m must be > max(box_size_m)/2; got '
                f'extent_m={extent_m}, max box={self._size_high}')
        self.prob = float(prob)
        self.box_size_m = box_size_m
        self.extent_m = float(extent_m)

    def transform(self, data):
        if np.random.random() >= self.prob:
            return data
        points = data.get('points')
        if points is None or len(points) == 0:
            return data
        if self._size_low == self._size_high:
            box_size = self._size_low
        else:
            box_size = float(np.random.uniform(
                self._size_low, self._size_high))
        half = box_size / 2.0
        cx = float(np.random.uniform(-self.extent_m + half,
                                     self.extent_m - half))
        cy = float(np.random.uniform(-self.extent_m + half,
                                     self.extent_m - half))
        xy = points.tensor[:, :2]
        in_box = (
            (xy[:, 0] >= cx - half) & (xy[:, 0] < cx + half) &
            (xy[:, 1] >= cy - half) & (xy[:, 1] < cy + half)
        )
        data['points'] = points[~in_box]
        return data



@TRANSFORMS.register_module()
class LoadBEVMapsFromV3Image(BaseTransform):
    """V3-layout BEV map loader (PNG+JSON for obstacle/semantic, NPZ for terrain).

    Replaces the old single-NPZ ``LoadBEVSegmentationFromImage`` for the
    ``pad_scenesV3`` dataset layout. Produces ``gt_masks_bev`` (existing
    semantics) and, when ``load_terrain=True``, ``gt_terrain_bev`` of shape
    ``(3, H, W)`` = ``[slope_x_ego, slope_y_ego, validity_mask]`` in m/m.
    """

    _CACHE: dict = {}

    def __init__(
            self,
            dataset_root,
            xbound,
            ybound,
            classes,
            label_mapping=None,
            load_terrain=False,
            load_obstacle=False,
            terrain_info_threshold=5.0,
            small_obstacle_max_area=20,
    ):
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(round(patch_h / ybound[2]))
        canvas_w = int(round(patch_w / xbound[2]))
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = list(classes)
        self.label_mapping = dict(label_mapping or [])
        self.dataset_root = dataset_root
        self.load_terrain = bool(load_terrain)
        # When True, additionally emit ``gt_obstacle_bev`` [1,H,W] float from the same warped
        # obstacle ROI used for the semantic ``non_drivable`` mask: 1.0 occupied (obstacle_u8==255),
        # 0.0 free (==0), 255 IGNORE (==127 'unknown'). Grid-aligned to gt_masks_bev (same
        # canvas + INTER_AREA downsample). Back-compat: default False -> key never added.
        self.load_obstacle = bool(load_obstacle)
        self.terrain_info_threshold = float(terrain_info_threshold)
        self.small_obstacle_max_area = int(small_obstacle_max_area)

        self._load_all(dataset_root)

    def _cache_for_root(self, dataset_root):
        return LoadBEVMapsFromV3Image._CACHE[dataset_root]

    def _load_all(self, dataset_root):
        if dataset_root in LoadBEVMapsFromV3Image._CACHE:
            cached = LoadBEVMapsFromV3Image._CACHE[dataset_root]
            if cached['load_terrain'] or not self.load_terrain:
                return
        obs_dir = os.path.join(dataset_root, 'maps', 'obstacle')
        sem_dir = os.path.join(dataset_root, 'maps', 'semantic')
        ter_dir = os.path.join(dataset_root, 'maps', 'terrain')
        if not os.path.isdir(obs_dir):
            raise FileNotFoundError(f'Missing V3 obstacle dir: {obs_dir}')
        if not os.path.isdir(sem_dir):
            raise FileNotFoundError(f'Missing V3 semantic dir: {sem_dir}')
        if self.load_terrain and not os.path.isdir(ter_dir):
            raise FileNotFoundError(f'Missing V3 terrain dir: {ter_dir}')

        locations = {}
        for fn in sorted(os.listdir(obs_dir)):
            if not fn.endswith('.png'):
                continue
            loc = fn[:-4]
            locations[loc] = self._load_one(dataset_root, loc)
        LoadBEVMapsFromV3Image._CACHE[dataset_root] = {
            'locations': locations,
            'load_terrain': self.load_terrain,
        }

    def _load_one(self, dataset_root, loc):
        obs_png = os.path.join(dataset_root, 'maps', 'obstacle', f'{loc}.png')
        obs_json = os.path.join(dataset_root, 'maps', 'obstacle', f'{loc}.json')
        sem_png = os.path.join(
            dataset_root, 'maps', 'semantic', f'{loc}_semantics_label.png')
        sem_json = os.path.join(
            dataset_root, 'maps', 'semantic', f'{loc}_semantics.json')
        # Pre-decoded mirrors (see tools/decode_v3_maps.py). When present
        # they are mmap'd, so all dataloader workers share the same
        # physical pages via the kernel page cache.
        obs_npy = os.path.join(
            dataset_root, 'obstacle_decode', f'{loc}.npy')
        sem_npy = os.path.join(
            dataset_root, 'semantic_decode', f'{loc}_semantics_label.npy')

        # JSON metadata is always canonical.
        for p in (obs_json, sem_json):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f'V3 metadata missing for location {loc!r}: {p}')

        with open(obs_json) as f:
            obs_meta = json.load(f)
        with open(sem_json) as f:
            sem_meta = json.load(f)

        # Per-location bounds/resolution must agree across all sources.
        self._check_meta_match(loc, 'obstacle', obs_meta, 'semantic', sem_meta)

        if os.path.isfile(obs_npy):
            obstacle_u8 = np.load(obs_npy, mmap_mode='r')
        elif os.path.isfile(obs_png):
            obstacle_u8 = np.asarray(Image.open(obs_png), dtype=np.uint8)
        else:
            raise FileNotFoundError(
                f'V3 obstacle pixel data missing for location {loc!r}: '
                f'expected {obs_npy} or {obs_png}')
        if os.path.isfile(sem_npy):
            semantic_u8 = np.load(sem_npy, mmap_mode='r')
        elif os.path.isfile(sem_png):
            semantic_u8 = np.asarray(Image.open(sem_png), dtype=np.uint8)
        else:
            raise FileNotFoundError(
                f'V3 semantic pixel data missing for location {loc!r}: '
                f'expected {sem_npy} or {sem_png}')

        id2label = {int(k): v for k, v in sem_meta['id2label'].items()}
        bounds = np.array(
            [obs_meta['xmin'], obs_meta['xmax'],
             obs_meta['ymin'], obs_meta['ymax']],
            dtype=np.float64,
        )
        resolution = float(obs_meta['resolution'])

        entry = {
            'obstacle_u8': obstacle_u8,
            'semantic_u8': semantic_u8,
            'id2label': id2label,
            'bounds': bounds,
            'resolution': resolution,
        }

        if self.load_terrain:
            ter_decode_dir = os.path.join(dataset_root, 'terrain_decode')
            slope_x_npy = os.path.join(ter_decode_dir, f'{loc}_slope_x.npy')
            slope_y_npy = os.path.join(ter_decode_dir, f'{loc}_slope_y.npy')
            info_npy = os.path.join(ter_decode_dir, f'{loc}_information.npy')
            terr_meta_json = os.path.join(ter_decode_dir, f'{loc}_meta.json')

            if all(os.path.isfile(p) for p in (slope_x_npy, slope_y_npy,
                                                info_npy, terr_meta_json)):
                with open(terr_meta_json) as f:
                    ter_meta = json.load(f)
                ter_bounds = np.asarray(ter_meta['bounds'], dtype=np.float64)
                ter_res = float(ter_meta['resolution'])
                if not np.allclose(ter_bounds, bounds, atol=1e-2):
                    raise ValueError(
                        f'Bounds mismatch for {loc!r}: obstacle={bounds} '
                        f'vs terrain={ter_bounds}')
                if not np.isclose(ter_res, resolution):
                    raise ValueError(
                        f'Resolution mismatch for {loc!r}: obstacle='
                        f'{resolution} vs terrain={ter_res}')
                entry['terrain_slope_x'] = np.load(
                    slope_x_npy, mmap_mode='r')
                entry['terrain_slope_y'] = np.load(
                    slope_y_npy, mmap_mode='r')
                entry['terrain_information'] = np.load(
                    info_npy, mmap_mode='r')
            else:
                ter_npz = os.path.join(
                    dataset_root, 'maps', 'terrain', f'{loc}_terrain.npz')
                if not os.path.isfile(ter_npz):
                    raise FileNotFoundError(
                        f'V3 terrain missing for location {loc!r}: '
                        f'expected {slope_x_npy} (decoded) or '
                        f'{ter_npz} (NPZ)')
                with np.load(ter_npz) as d:
                    ter_bounds = np.asarray(d['bounds'], dtype=np.float64)
                    ter_res = float(d['resolution'])
                    if not np.allclose(ter_bounds, bounds, atol=1e-2):
                        raise ValueError(
                            f'Bounds mismatch for {loc!r}: obstacle={bounds} '
                            f'vs terrain={ter_bounds}')
                    if not np.isclose(ter_res, resolution):
                        raise ValueError(
                            f'Resolution mismatch for {loc!r}: obstacle='
                            f'{resolution} vs terrain={ter_res}')
                    entry['terrain_slope_x'] = d['slope_x'].copy()
                    entry['terrain_slope_y'] = d['slope_y'].copy()
                    entry['terrain_information'] = d['information'].copy()

        return entry

    def _check_meta_match(self, loc, name_a, meta_a, name_b, meta_b):
        for field in ('xmin', 'xmax', 'ymin', 'ymax'):
            if not np.isclose(meta_a[field], meta_b[field], atol=1e-2):
                raise ValueError(
                    f'{field} mismatch for {loc!r}: {name_a}='
                    f'{meta_a[field]} vs {name_b}={meta_b[field]}')
        if not np.isclose(meta_a['resolution'], meta_b['resolution']):
            raise ValueError(
                f'resolution mismatch for {loc!r}: {name_a}='
                f'{meta_a["resolution"]} vs {name_b}={meta_b["resolution"]}')

    def _get_pose(self, results):
        lidar_points = results.get('lidar_points', {})
        if 'lidar2ego' in lidar_points:
            lidar2ego = np.array(lidar_points['lidar2ego'])
        else:
            lidar2ego = np.eye(4)
        if 'ego2global' not in results:
            raise ValueError("'ego2global' not found in results")
        ego2global = np.array(results['ego2global'])
        if lidar2ego.shape == (3, 4):
            t = np.eye(4); t[:3, :] = lidar2ego; lidar2ego = t
        if ego2global.shape == (3, 4):
            t = np.eye(4); t[:3, :] = ego2global; ego2global = t
        lidar2global = ego2global @ lidar2ego
        map_pose = lidar2global[:2, 3]
        rotation = lidar2global[:3, :3]
        v = rotation @ np.array([1, 0, 0])
        yaw = float(np.arctan2(v[1], v[0]))
        return map_pose, yaw

    def transform(self, results):
        if 'location' not in results:
            raise ValueError("'location' not found in results")
        location = results['location']
        cache = self._cache_for_root(self.dataset_root)
        if location not in cache['locations']:
            raise ValueError(f'Map not found for location: {location}')
        entry = cache['locations'][location]

        map_pose, yaw = self._get_pose(results)

        H_global, W_global = entry['obstacle_u8'].shape
        xmin, xmax, ymin, ymax = entry['bounds']
        res = entry['resolution']

        hf = self.patch_size[0] / 2  # x extent (forward)
        hs = self.patch_size[1] / 2  # y extent (left)
        W_roi = max(1, int(round((2.0 * hf) / res)))
        H_roi = max(1, int(round((2.0 * hs) / res)))

        dst = np.array(
            [[0, H_roi - 1],
             [W_roi - 1, H_roi - 1],
             [W_roi - 1, 0],
             [0, 0]],
            dtype=np.float32,
        )
        corners_local = np.array(
            [[hf, -hs], [hf, hs], [-hf, hs], [-hf, -hs]], dtype=np.float32)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        corners_global = (corners_local @ R.T
                          + np.array([map_pose[0], map_pose[1]], dtype=np.float32))
        cols = (corners_global[:, 0] - xmin) / res
        rows = (corners_global[:, 1] - ymin) / res
        src = np.stack([cols, rows], axis=1).astype(np.float32)

        oob = (
            np.any(src[:, 0] < 0) or np.any(src[:, 0] >= W_global) or
            np.any(src[:, 1] < 0) or np.any(src[:, 1] >= H_global)
        )
        if oob:
            results['gt_masks_bev'] = np.zeros(
                (len(self.classes), *self.canvas_size), dtype=bool)
            if self.load_terrain:
                results['gt_terrain_bev'] = np.zeros(
                    (3, *self.canvas_size), dtype=np.float32)
            if self.load_obstacle:
                # No map coverage -> treat as all-free (zeros), same convention as gt_masks_bev.
                results['gt_obstacle_bev'] = np.zeros(
                    (1, *self.canvas_size), dtype=np.float32)
            return results

        M = cv2.getPerspectiveTransform(src, dst)
        roi_obs = self._warp(entry['obstacle_u8'], M, (W_roi, H_roi),
                             interp=cv2.INTER_NEAREST)
        roi_sem = self._warp(entry['semantic_u8'], M, (W_roi, H_roi),
                             interp=cv2.INTER_NEAREST)
        results = self._post_warp_to_labels(
            results, entry['id2label'], roi_obs, roi_sem)

        if self.load_terrain:
            results = self._post_warp_to_terrain(results, entry, M,
                                                 (W_roi, H_roi), yaw)
        return results

    # OpenCV's warpPerspective / remap uses int16 internally for source
    # pixel coordinates. Anything ≥ SHRT_MAX (32767) on either source dim
    # asserts and dies. We crop the source down to the bbox of the actual
    # ROI we need before the warp; the crop is always much smaller than
    # the ROI's pixel extent because we typically warp a 60-70m patch out
    # of a multi-km map.
    _SHRT_MAX = 32767

    @classmethod
    def _warp(cls, src, M, dsize, interp,
              border_value=0, src_size_limit=None):
        limit = src_size_limit if src_size_limit is not None else cls._SHRT_MAX
        H_src, W_src = src.shape[:2]
        if max(H_src, W_src) < limit:
            return cv2.warpPerspective(
                src, M, dsize, flags=interp,
                borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

        # Source exceeds the OpenCV limit. Compute the bbox of the inverse
        # warp on the destination corners, crop the source to that bbox
        # plus a small margin, and shift M to operate on the cropped
        # source.
        W_dst, H_dst = dsize
        dst_corners = np.array(
            [[0.0, 0.0],
             [W_dst - 1.0, 0.0],
             [W_dst - 1.0, H_dst - 1.0],
             [0.0, H_dst - 1.0]],
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        M_inv = np.linalg.inv(M.astype(np.float64))
        src_corners = cv2.perspectiveTransform(dst_corners, M_inv)
        src_corners = src_corners.reshape(-1, 2)

        margin = 8
        x_min = max(0, int(np.floor(src_corners[:, 0].min())) - margin)
        y_min = max(0, int(np.floor(src_corners[:, 1].min())) - margin)
        x_max = min(W_src, int(np.ceil(src_corners[:, 0].max())) + margin + 1)
        y_max = min(H_src, int(np.ceil(src_corners[:, 1].max())) + margin + 1)

        if x_max <= x_min or y_max <= y_min:
            # ROI entirely outside the source — equivalent to OOB at the
            # transform() level. Return all-borderValue output.
            out = np.full((H_dst, W_dst), border_value, dtype=src.dtype)
            return out

        crop_w = x_max - x_min
        crop_h = y_max - y_min
        if max(crop_w, crop_h) >= limit:
            # Even the crop is too large — pathological case. The source
            # map would have to span >32767 px just for the ROI footprint,
            # which is way beyond our 70 m / 0.1 m = 700 px design point.
            raise RuntimeError(
                f'_warp crop bbox {crop_w}x{crop_h} still exceeds '
                f'SHRT_MAX limit {limit}; ROI math may be wrong')

        # Slice; force contiguous if memmap so cv2 doesn't reject it.
        src_crop = src[y_min:y_max, x_min:x_max]
        if not src_crop.flags.c_contiguous:
            src_crop = np.ascontiguousarray(src_crop)

        # Shift M for the new source origin:
        # M @ [u, v, 1]   = dst        in original src coords
        # M' @ [u', v', 1] = dst       in cropped src coords (u' = u-x_min)
        # → M' = M @ T(x_min, y_min)  where T is a translation.
        T_shift = np.array(
            [[1.0, 0.0, float(x_min)],
             [0.0, 1.0, float(y_min)],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        M_crop = M.astype(np.float64) @ T_shift

        return cv2.warpPerspective(
            src_crop, M_crop, dsize, flags=interp,
            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

    def _post_warp_to_labels(self, results, id2label, roi_obs, roi_sem):
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=bool)
        occ = roi_obs == 255
        mapping = self.label_mapping
        canvas_wh = (self.canvas_size[1], self.canvas_size[0])

        def _downsample(mask_high):
            return cv2.resize(mask_high.astype(np.float32), canvas_wh,
                              interpolation=cv2.INTER_AREA) > 0

        for sem_idx, class_name in id2label.items():
            mapped = mapping.get(class_name, class_name)
            if mapped in self.classes:
                labels[self.classes.index(mapped)] |= _downsample(
                    roi_sem == sem_idx)
        if 'non_drivable' in self.classes:
            labels[self.classes.index('non_drivable')] |= _downsample(occ)

        results['gt_masks_bev'] = labels

        if self.load_obstacle:
            # Binary obstacle-occupancy target [1,H,W], grid-aligned to gt_masks_bev.
            #   occupied (roi_obs==255) -> 1.0 ; unknown (==127) -> 255 IGNORE ; free (==0) -> 0.0
            # Downsample each class via the same INTER_AREA>0 rule the semantics use so any
            # high-res occupied/unknown pixel within a 0.6 m cell claims that cell. Occupied wins
            # over unknown on overlap (a real obstacle is never masked out by adjacent unknown).
            occ_lo = _downsample(roi_obs == 255)
            unk_lo = _downsample(roi_obs == 127)
            obstacle = np.zeros(self.canvas_size, dtype=np.float32)
            obstacle[occ_lo] = 1.0
            obstacle[unk_lo & ~occ_lo] = 255.0
            results['gt_obstacle_bev'] = obstacle[None]  # [1,H,W]

        return results

    def _post_warp_to_terrain(self, results, entry, M, roi_size, yaw):
        W_roi, H_roi = roi_size
        sx_g = np.nan_to_num(entry['terrain_slope_x'], nan=0.0)
        sy_g = np.nan_to_num(entry['terrain_slope_y'], nan=0.0)
        info = entry['terrain_information']

        roi_sx_g = self._warp(sx_g, M, (W_roi, H_roi),
                              interp=cv2.INTER_LINEAR, border_value=0.0)
        roi_sy_g = self._warp(sy_g, M, (W_roi, H_roi),
                              interp=cv2.INTER_LINEAR, border_value=0.0)
        # information is an integer observation count; INTER_NEAREST preserves
        # integer semantics so `>= threshold` behaves exactly as "this cell's
        # source was observed enough".
        roi_info = self._warp(info.astype(np.float32), M, (W_roi, H_roi),
                              interp=cv2.INTER_NEAREST, border_value=0.0)

        # Rotate (slope_x, slope_y) from global to ego frame by -yaw.
        c, s = np.cos(-yaw), np.sin(-yaw)
        sx_e = c * roi_sx_g - s * roi_sy_g
        sy_e = s * roi_sx_g + c * roi_sy_g

        valid_hi = (roi_info >= self.terrain_info_threshold).astype(np.float32)

        canvas_wh = (self.canvas_size[1], self.canvas_size[0])
        sx_lo = cv2.resize(sx_e, canvas_wh, interpolation=cv2.INTER_AREA)
        sy_lo = cv2.resize(sy_e, canvas_wh, interpolation=cv2.INTER_AREA)
        valid_lo = (cv2.resize(valid_hi, canvas_wh,
                               interpolation=cv2.INTER_AREA) > 0.5).astype(np.float32)

        # Zero out slope where invalid so downstream code never reads
        # spatially-blended noise as a "real" slope value.
        sx_lo = sx_lo * valid_lo
        sy_lo = sy_lo * valid_lo

        stacked = np.stack([sx_lo, sy_lo, valid_lo], axis=0).astype(np.float32)
        if not np.isfinite(stacked).all():
            raise RuntimeError('gt_terrain_bev produced non-finite values')
        results['gt_terrain_bev'] = stacked
        return results
