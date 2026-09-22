from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.dist import get_rank  # 0 without a process group, e.g. single-GPU training
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from .base_keypoints import Base3DDetectorKeypoints
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmengine.structures import InstanceData
from .ops import Voxelization

try:                                   # nur fuer die Bild-Baselines der nuScenes-Tabellen
    from .evaluator import PoseEvaluator   # noqa: F401
except ImportError:                        # im KITTI-Setup nicht vorhanden und nicht noetig
    PoseEvaluator = None

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import os


@MODELS.register_module()
class BEVFusionKeypoints(Base3DDetectorKeypoints):

    def __init__(
            self,
            data_preprocessor: OptConfigType = None,
            pts_voxel_encoder: Optional[dict] = None,
            pts_middle_encoder: Optional[dict] = None,
            fusion_layer: Optional[dict] = None,
            img_backbone: Optional[dict] = None,
            pts_backbone: Optional[dict] = None,
            view_transform: Optional[dict] = None,
            img_neck: Optional[dict] = None,
            pts_neck: Optional[dict] = None,
            bbox_head: Optional[dict] = None,
            init_cfg: OptMultiConfig = None,
            seg_head: Optional[dict] = None,
            keypoint_head: Optional[dict] = None,
            obstacle_head: Optional[dict] = None,
            use_lidar: Optional[bool] = True,
            use_camera: Optional[bool] = True,
            use_dropout: Optional[bool] = True,
            descriptor_lidar_drop_prob: float = 0.0,
            use_modality_dropout: bool = False,
            modality_dropout_max_prob: float = 0.7,
            only_reference_implementation: Optional[bool] = False,
            **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        self.iteration = 0
        self.val_iteration = 0
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_dropout = use_dropout
        # LEVER 1 (camera-enforcement): fraction of training iters on which the LiDAR
        # BEV branch is dropped (camera kept, incl. its LiDAR-conditioned depth lift) so
        # the descriptor self-sup loss is forced to make CAMERA-only cells matchable.
        # 0.0 = off (existing configs byte-identical). The drop is derived deterministically
        # from self.iteration so every DDP rank makes the same decision without a collective.
        self.descriptor_lidar_drop_prob = float(descriptor_lidar_drop_prob)
        # MARCH-ERA MODALITY DROPOUT (re-enabled, gated): drop EITHER camera or lidar
        # (mutually exclusive) on a fraction of iters that ramps 0 -> modality_dropout_max_prob,
        # so the shared encoder+descriptor must work from each modality alone -> forces camera
        # use. DDP-safe: the 4 drop flags are a deterministic hash of self.iteration (identical
        # on every rank, no broadcast). Default False = existing configs byte-identical.
        self.use_modality_dropout = bool(use_modality_dropout)
        self.modality_dropout_max_prob = float(modality_dropout_max_prob)
        self.only_reference_implementation = only_reference_implementation
        self.evaluator = PoseEvaluator() if PoseEvaluator is not None else None

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None

        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None

        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head) if bbox_head is not None else None
        self.seg_head = MODELS.build(seg_head) if seg_head is not None else None
        self.keypoint_head = MODELS.build(keypoint_head) if keypoint_head is not None else None
        self.obstacle_head = MODELS.build(obstacle_head) if obstacle_head is not None else None

        np.random.seed(42)

        self.init_weights()
        self.set_of_dropouts = [
            (False, False, False, False, "m2m"),  # no dropout (Full)
            (False, True, False, True, "c2c"),  # drop current lidar and other lidar (Camera only)
            (True, False, True, False, "l2l"),  # drop current cam and other cam (Lidar only)
            (False, True, True, False, "c2l")  # drop current lidar and other cam (Current Camera + Other Lidar)
        ]

        # Initialize epoch counter for DA finetuning
        self.epoch_counter = 0
        self.pts_feature_shape = None
        self.img_feature_shape = None
        self.files_removed = False

    def setupTransformsCamera(self, batch_input_metas, imgs):
        # Setup transforms
        lidar2image, camera_intrinsics, camera2lidar = [], [], []
        img_aug_matrix, lidar_aug_matrix = [], []
        for i, meta in enumerate(batch_input_metas):
            lidar2image.append(meta['lidar2img'])
            camera_intrinsics.append(meta['cam2img'])
            camera2lidar.append(meta['cam2lidar'])
            img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
            lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

        lidar2image = imgs.new_tensor(np.asarray(lidar2image))
        camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
        camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
        img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
        lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

        return lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> Dict:

        # Otherwise, proceed with normal training
        batch_input_metas_current = [item.metainfo for item in batch_data_samples["current"]]
        batch_input_metas_other = [item.metainfo for item in batch_data_samples["other"]]

        # Modality dropout disabled ("without dropouts" per design v5). The old
        # per-iter dist.broadcast(flags) + 4 .item() GPU->CPU syncs that kept the
        # drop decision consistent across ranks are removed: with dropout off the
        # flags are always False on every rank, so the collective + syncs were dead
        # work that serialized the CUDA stream each iteration.
        current_cam_drop = current_lidar_drop = False
        other_cam_drop = other_lidar_drop = False

        # CAMERA-ONLY-Training (A7, DWPVO-Vergleichszeile): use_lidar=False im Modell-Config
        # droppt den LiDAR-Zweig KONSTANT fuer beide Frames — der Encoder sieht nie LiDAR.
        # (Analog use_camera=False fuer ein LiDAR-only-Modell.)
        if not getattr(self, 'use_lidar', True):
            current_lidar_drop = other_lidar_drop = True
        if not getattr(self, 'use_camera', True):
            current_cam_drop = other_cam_drop = True

        self.iteration += 1

        # LEVER 1: on a fraction of iters drop the LiDAR BEV (keep camera + its depth lift)
        # so the descriptor must encode camera appearance. Deterministic in self.iteration
        # -> identical across DDP ranks with no broadcast/sync. Both frames dropped together
        # so the (current, other) descriptor pair is camera-only and stays matchable.
        if self.training and self.descriptor_lidar_drop_prob > 0.0:
            _h = (self.iteration * 2654435761) % 1000  # cheap deterministic hash -> [0,1000)
            if _h < int(self.descriptor_lidar_drop_prob * 1000):
                current_lidar_drop = other_lidar_drop = True

        # MARCH-ERA symmetric modality dropout (gated, DDP-safe deterministic-in-iteration).
        # Schedule mirrors the original getDrop: 0 for the first 500 iters, then ramps to
        # modality_dropout_max_prob by iter 2000. Each modality dropped independently; if both
        # would drop, keep one (deterministic tiebreak) so a frame never loses all sensors.
        if self.training and self.use_modality_dropout:
            warmup_iters, ramp_iters = 500, 2000
            if self.iteration <= warmup_iters:
                prob = 0.0
            else:
                t = min(1.0, (self.iteration - warmup_iters) / (ramp_iters - warmup_iters))
                prob = self.modality_dropout_max_prob * t
            thr = int(prob * 1000)

            def _hb(salt):  # deterministic per-(iteration,salt) Bernoulli(prob), same on all ranks
                return ((self.iteration * 2654435761) ^ (salt * 40503)) % 1000 < thr

            cc, cl = _hb(1), _hb(2)
            if cc and cl:  # keep one so the current frame retains a modality
                cc, cl = (False, True) if (self.iteration % 2 == 0) else (True, False)
            oc, ol = _hb(3), _hb(4)
            if oc and ol:
                oc, ol = (False, True) if (self.iteration % 2 == 0) else (True, False)
            current_cam_drop, current_lidar_drop = cc, cl
            other_cam_drop, other_lidar_drop = oc, ol

        batch_input_poses_current = [self.lidar2global(item) for item in batch_input_metas_current]
        batch_input_poses_other = [self.lidar2global(item) for item in batch_input_metas_other]

        aug_current = [m.get('lidar_aug_matrix', None) for m in batch_input_metas_current]
        aug_other = [m.get('lidar_aug_matrix', None) for m in batch_input_metas_other]

        # LEVER 3: when the keypoint head has a dedicated camera descriptor sub-space,
        # also fetch the pre-fusion camera BEV so it can be threaded into the head.
        want_cam_bev = (self.with_keypoint_head and
                        getattr(self.keypoint_head, 'camera_descriptor_bits', 0) > 0)
        camera_bev_current = camera_bev_other = None
        if want_cam_bev:
            feats, camera_bev_current = self.extract_feat(
                batch_inputs_dict["current"], batch_input_metas_current,
                drop_cam=current_cam_drop, drop_lidar=current_lidar_drop, return_camera_bev=True)
            feats_other, camera_bev_other = self.extract_feat(
                batch_inputs_dict["other"], batch_input_metas_other,
                drop_cam=other_cam_drop, drop_lidar=other_lidar_drop, return_camera_bev=True)
        else:
            feats = self.extract_feat(batch_inputs_dict["current"], batch_input_metas_current,
                                      drop_cam=current_cam_drop, drop_lidar=current_lidar_drop)

            feats_other = self.extract_feat(batch_inputs_dict["other"], batch_input_metas_other,
                                            drop_cam=other_cam_drop, drop_lidar=other_lidar_drop)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples["current"])
            losses.update(bbox_loss)
        if self.with_seg_head:
            #print min-max of gt_masks_bev
            seg_loss = self.seg_head.loss(feats, batch_input_metas_current)
            losses.update(seg_loss)
        if self.with_obstacle_head:
            # Binary obstacle-occupancy loss (gt_obstacle_bev in the CURRENT frame metainfo),
            # symmetric with the seg head. 127-'unknown' cells are masked inside the head.
            obstacle_loss = self.obstacle_head.loss(feats, batch_input_metas_current)
            losses.update(obstacle_loss)
        if self.with_keypoint_head:
            if current_cam_drop:
                batch_inputs_dict["current"]['imgs'] = batch_inputs_dict["current"]['imgs'] * 0
            if current_lidar_drop:
                batch_inputs_dict["current"]['points'] = [torch.zeros_like(p) for p in
                                                          batch_inputs_dict["current"]['points']]

            if other_lidar_drop:
                batch_inputs_dict["other"]['points'] = [torch.zeros_like(p) for p in
                                                        batch_inputs_dict["other"]['points']]
            if other_cam_drop:
                batch_inputs_dict["other"]['imgs'] = batch_inputs_dict["other"]['imgs'] * 0

            # Supervised-mode gt: teacher landmark xy (BEV metres, lidar frame) for the
            # CURRENT frame, per sample. Reaches metainfo via the Pack transform's
            # meta_keys (config wiring is Task 10). Pass-through is harmless when the
            # head is self-supervised (supervised=False ignores it).
            gt_keypoint_xy_current = [m.get('gt_keypoint_xy') for m in batch_input_metas_current]

            # Binary-descriptor (Task 9) correspondence gt: per-sample track ids for the
            # current frame and the OTHER frame's keypoint xy + track ids, so the head can
            # intersect shared track_ids to form positive descriptor pairs. Harmless when
            # the head is not in binary mode (it ignores these); entries may be None/empty.
            gt_track_id_current = [m.get('gt_track_id') for m in batch_input_metas_current]
            gt_keypoint_xy_other = [m.get('gt_keypoint_xy') for m in batch_input_metas_other]
            gt_track_id_other = [m.get('gt_track_id') for m in batch_input_metas_other]

            keypoint_loss = self.keypoint_head.loss(feats, batch_input_poses_current,
                                                    feats_other, batch_input_poses_other,
                                                    batch_inputs_dict['current']['points'],
                                                    batch_inputs_dict['other']['points'],
                                                    batch_inputs_dict['current']['imgs'],
                                                    batch_inputs_dict['other']['imgs'],
                                                    aug_current=aug_current,
                                                    aug_other=aug_other,
                                                    gt_keypoint_xy_current=gt_keypoint_xy_current,
                                                    gt_track_id_current=gt_track_id_current,
                                                    gt_keypoint_xy_other=gt_keypoint_xy_other,
                                                    gt_track_id_other=gt_track_id_other,
                                                    camera_bev_current=camera_bev_current,
                                                    camera_bev_other=camera_bev_other)

            losses.update(keypoint_loss)

            # --- CROSS-MODAL descriptor consistency (fixes C2L): re-extract the CURRENT frame under
            # fused / camera-only / lidar-only, and align each detected keypoint's descriptor across
            # the three via InfoNCE. Only active when keypoint_head.cross_modal_weight > 0. ---
            _xw = getattr(self.keypoint_head, 'cross_modal_weight', 0.0)
            if _xw and _xw > 0.0:
                _ci = batch_inputs_dict["current"]
                _f_fused = self.extract_feat(_ci, batch_input_metas_current, drop_cam=False, drop_lidar=False)
                _f_cam = self.extract_feat(_ci, batch_input_metas_current, drop_cam=False, drop_lidar=True)
                _f_lid = self.extract_feat(_ci, batch_input_metas_current, drop_cam=True, drop_lidar=False)
                _kp, _d_fused, _ = self.keypoint_head.forward(_f_fused[0], None)
                _d_cam = self.keypoint_head.descriptor_head(_f_cam[0])
                _d_lid = self.keypoint_head.descriptor_head(_f_lid[0])
                losses['keypoints_cross_modal_loss'] = _xw * self.keypoint_head.cross_modal_consistency_loss(
                    _d_fused, _d_cam, _d_lid, _kp[0])

        return losses

    # ... rest of the methods remain the same ...
    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        pass

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)

        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    @property
    def with_keypoint_head(self):
        return hasattr(self, 'keypoint_head') and self.keypoint_head is not None

    @property
    def with_obstacle_head(self):
        return hasattr(self, 'obstacle_head') and self.obstacle_head is not None

    def extract_img_feat(self, x, points, lidar2image, camera_intrinsics, camera2lidar,
                         img_aug_matrix, lidar_aug_matrix, img_metas) -> torch.Tensor:
        B, N, C, H, W = x.size()
        original_images = x.clone()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(x, points, lidar2image, camera_intrinsics,
                                    camera2lidar, img_aug_matrix, lidar_aug_matrix, img_metas, original_images=original_images)

        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes


    @torch.no_grad()
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> Dict:
        #get keys
        keys = batch_inputs_dict.keys()
        #remove "other" and "current" from keys if present


        if "current" in keys:
            keys = list(keys)
            keys.remove("current")
            keys.remove("other")
        print(keys)
        all_ranks = True
        nth_processing = 1  #process every nth

        if self.val_iteration % nth_processing == 0 and (get_rank() == 0 or all_ranks):
            with torch.no_grad():

                batch_input_metas_current = [item.metainfo for item in batch_data_samples["current"]]
                batch_input_poses_current = [self.lidar2global(item) for item in batch_input_metas_current]

                # === EVAL SPEEDUP: cache encoder features by modality-drop combo. The CURRENT frame's
                # feature depends only on its (cam_drop, lidar_drop) and is INDEPENDENT of the range
                # (other_key), so the <=3 distinct current features are computed ONCE here (vs 3 ranges
                # x 4 modes = 12 naively). OTHER features (range-dependent) are cached per range below.
                # Only the expensive extract_feat is cached; the cheap zeroed input tensors are still
                # rebuilt per mode so keypoint_head.predict sees exactly the naive inputs. Bit-identical:
                # VERIFY_OPT=1 re-extracts naively per mode and asserts torch.equal on the features.
                _verify_opt = os.environ.get('VERIFY_OPT') == '1'

                def _drop_inputs(src_key, cam_drop, lidar_drop):
                    inp = {
                        "imgs": batch_inputs_dict[src_key]['imgs'].clone(),
                        "points": [p.clone() for p in batch_inputs_dict[src_key]['points']],
                        "valid": batch_inputs_dict[src_key]['valid'],
                    }
                    if cam_drop:
                        inp['imgs'] = torch.zeros_like(inp['imgs'])
                    if lidar_drop:
                        inp['points'] = [torch.zeros_like(p) for p in inp['points']]
                    return inp

                _cur_feat_cache = {}
                if not self.only_reference_implementation and (self.with_bbox_head or self.with_keypoint_head):
                    for _cam_d, _lid_d in {(d[0], d[1]) for d in self.set_of_dropouts}:
                        _cur_feat_cache[(_cam_d, _lid_d)] = self.extract_feat(
                            _drop_inputs("current", _cam_d, _lid_d),
                            batch_input_metas_current, drop_cam=_cam_d, drop_lidar=_lid_d)

                for other_key in keys:

                    print("Processing other_key:", other_key)
                    #res = batch_data_samples["current"]
                    #for it in batch_data_samples[other_key]:
                    #    print(it)


                    batch_input_metas_other = [item.metainfo for item in batch_data_samples[other_key]]
                    batch_input_poses_other = [self.lidar2global(item) for item in batch_input_metas_other]



                    #return empty list if no heads
                    if not (self.with_bbox_head or self.with_keypoint_head):
                        return batch_data_samples


                    if self.only_reference_implementation:
                        print("Executing only reference implementation")

                        #no dropouts anymore.
                        #loop through all batches, no feature extraction
                        R_batch_gt, t_batch_gt, scale_batch = self.keypoint_head.getGTTransformBatch(
                            batch_input_poses_current,
                            batch_input_poses_other
                        )

                        batch_results = {}

                        for batch_idx in range(len(batch_data_samples["current"])):
                            # --- make deep copies so we don't mutate the originals ---
                            current_inputs = {
                                "imgs": batch_inputs_dict["current"]['imgs'][batch_idx:batch_idx+1].clone(),
                                "points": [p.clone() for p in batch_inputs_dict["current"]['points'][batch_idx:batch_idx+1]],
                                "valid": [batch_inputs_dict["current"]['valid'][batch_idx]]
                            }

                            other_inputs = {
                                "imgs": batch_inputs_dict[other_key]['imgs'][batch_idx:batch_idx+1].clone(),
                                "points": [p.clone() for p in batch_inputs_dict[other_key]['points'][batch_idx:batch_idx+1]],
                                "valid": [batch_inputs_dict[other_key]['valid'][batch_idx]]
                            }

                            #get relative pose gt


                            #scale is a tensor, get scalar value

                            results = self.evaluator.all(scale_batch[batch_idx].item(),
                                                         points_current=current_inputs['points'][0],
                                                         points_other=other_inputs['points'][0],
                                                         images_current=current_inputs['imgs'],
                                                         images_other=other_inputs['imgs'],
                                                         camera_matrix= batch_input_metas_current[batch_idx]['cam2img'][0],
                                                         cam2lidar=batch_input_metas_current[batch_idx]['cam2lidar'][0])
                            print(results)
                            #iterathe through each key, if exists in batch_results, append, else create new entry.
                            # each entry has R and t, such that batch_results[key]['R'] is a list of R matrices
                            for key in results:
                                if key not in batch_results:
                                    batch_results[key] = {'R': [], 't': []}
                                batch_results[key]['R'].append(results[key]['R'])
                                batch_results[key]['t'].append(results[key]['t'])


                        #write to file, filepath based on rank and method
                        base_path = os.environ.get("VAL_CSV_DIR", "./validation_reference/")
                        for method in batch_results:
                            filepath = f"val_results_rank{get_rank()}_reference_{other_key}_{method}.csv"
                            filepath = os.path.join(base_path, filepath)
                            R_list = batch_results[method]['R']
                            t_list = batch_results[method]['t']
                            self.writeResultToFile(filepath, R_list, t_list, R_batch_gt, t_batch_gt, batch_inputs_dict["current"]['valid'])

                    else:
                        # OTHER features are range-dependent: cache the <=3 distinct for THIS range,
                        # reused across the modes that share an other-drop combo (l2l & c2l share T,F).
                        _oth_feat_cache = {}
                        for _cam_d, _lid_d in {(d[2], d[3]) for d in self.set_of_dropouts}:
                            _oth_feat_cache[(_cam_d, _lid_d)] = self.extract_feat(
                                _drop_inputs(other_key, _cam_d, _lid_d),
                                batch_input_metas_other, drop_cam=_cam_d, drop_lidar=_lid_d)

                        for current_cam_drop, current_lidar_drop, other_cam_drop, other_lidar_drop, mode in self.set_of_dropouts:
                            # cheap zeroed inputs rebuilt per mode (identical to naive); keypoint_head.predict
                            # reads these raw point/img tensors for its geometric evaluator.
                            current_inputs = _drop_inputs("current", current_cam_drop, current_lidar_drop)
                            other_inputs = _drop_inputs(other_key, other_cam_drop, other_lidar_drop)

                            # expensive encoder features from cache (current is range-independent)
                            feat_current = _cur_feat_cache[(current_cam_drop, current_lidar_drop)]
                            feat_other = _oth_feat_cache[(other_cam_drop, other_lidar_drop)]

                            if _verify_opt:
                                _fc = self.extract_feat(
                                    _drop_inputs("current", current_cam_drop, current_lidar_drop),
                                    batch_input_metas_current, drop_cam=current_cam_drop, drop_lidar=current_lidar_drop)
                                _fo = self.extract_feat(
                                    _drop_inputs(other_key, other_cam_drop, other_lidar_drop),
                                    batch_input_metas_other, drop_cam=other_cam_drop, drop_lidar=other_lidar_drop)
                                assert all(torch.equal(a, b) for a, b in zip(feat_current, _fc)), f"cur feat mismatch {mode}"
                                assert all(torch.equal(a, b) for a, b in zip(feat_other, _fo)), f"oth feat mismatch {mode}"

                            # --- predictions ---
                            if self.with_bbox_head:
                                outputs = self.bbox_head.predict(feat_current, batch_input_metas_current)
                                res = self.add_pred_to_datasample(batch_data_samples, outputs)

                            if self.with_keypoint_head:
                                R_batch,t_batch = self.keypoint_head.predict(
                                    feat_current, batch_input_poses_current,
                                    feat_other, batch_input_poses_other,
                                    current_inputs['points'],
                                    other_inputs['points'],
                                    current_inputs['imgs'],
                                    other_inputs['imgs']
                                )
                                #get GT transformations
                                R_batch_gt, t_batch_gt, _ = self.keypoint_head.getGTTransformBatch(batch_input_poses_current, batch_input_poses_other)

                            #write to file, filepath based on rank and dropout setting
                                base_path = os.environ.get("VAL_CSV_DIR", "./validation_reference/")
                                os.makedirs(base_path, exist_ok=True)
                                filepath = f"val_results_rank{get_rank()}_{mode}_{other_key}.csv"
                                filepath = os.path.join(base_path, filepath)
                                self.writeResultToFile(filepath, R_batch, t_batch, R_batch_gt, t_batch_gt, other_inputs['valid'])

                            #if self.with_seg_head:
                            #    with torch.no_grad():
                            #        self.seg_head.predict(feat_current, batch_input_metas_current)

        self.val_iteration += 1

        return "Hello"

    def writeResultToFile(self, filepath, R_batch, t_batch, R_batch_gt, t_batch_gt, valid_items):
        # first: convert R to angle around z
        angle_batch = []
        for R in R_batch:
            if R is None:
                angle_batch.append(None)
                continue
            angle = np.arctan2(R[1, 0], R[0, 0])
            angle_batch.append(angle)

        angle_batch_gt = []
        for R in R_batch_gt:
            if R is None:
                angle_batch_gt.append(None)
                continue
            angle = np.arctan2(R[1, 0], R[0, 0])
            angle_batch_gt.append(angle)

        # if file exists, remove file once
        if not self.files_removed:
            if os.path.exists(filepath):
                os.remove(filepath)
            self.files_removed = True

        # if file does not exist, write header
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write("pred_angle,pred_tx,pred_ty,gt_angle,gt_tx,gt_ty,translation_error,angle_error\n")

        with open(filepath, 'a') as f:
            for angle, t, angle_gt, t_gt, valid in zip(angle_batch, t_batch, angle_batch_gt, t_batch_gt, valid_items):
                # convert to numpy if tensor
                if hasattr(t_gt, 'cpu'):
                    t_gt = t_gt.cpu().numpy()
                if hasattr(t, 'cpu'):
                    t = t.cpu().numpy()

                # handle invalids
                if t_gt is None or angle_gt is None or t is None or angle is None:
                    angle_diff = float('nan')
                    t_diff = float('nan')
                    t_gt = [float('nan'), float('nan')]
                    t = [float('nan'), float('nan')]
                    angle_gt = float('nan')
                    angle = float('nan')
                else:
                    # --- automatische Richtungs-Korrektur ---
                    dot = np.dot(t[:2], t_gt[:2])  # nur x,y vergleichen
                    if dot < 0:  # falsche Richtung erkannt
                        t = -t
                        # Optional: yaw um 180° drehen, wenn Bewegungsrichtung relevant ist
                        angle = (angle + np.pi) % (2 * np.pi)
                        if angle > np.pi:
                            angle -= 2 * np.pi
                    # ------------------------------------------

                    # Fehler berechnen
                    angle_diff = angle - angle_gt
                    t_diff = np.linalg.norm(t[:2] - t_gt[:2])

                if valid:
                    f.write(f"{angle},{t[0]},{t[1]},{angle_gt},{t_gt[0]},{t_gt[1]},{t_diff},{angle_diff}\n")
                else:
                    print("Skipping invalid item in writeResultToFile")

    def extract_feat(self, batch_inputs_dict, batch_input_metas, drop_cam=False, drop_lidar=False,
                     extract_student=False, **kwargs):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []

        # === Get current batch size ===
        if imgs is not None:
            current_batch_size = imgs.size(0)
        elif points is not None:
            current_batch_size = len(points)
        else:
            raise ValueError("Neither images nor points provided!")

        # === Initialize feature dimension cache (only once, without batch dim) ===
        if not hasattr(self, '_img_channels'):
            self._img_channels = None
            self._img_height = None
            self._img_width = None

        if not hasattr(self, '_pts_channels'):
            self._pts_channels = None
            self._pts_height = None
            self._pts_width = None

        # === Camera Features ===
        if imgs is not None and not drop_cam:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix = \
                self.setupTransformsCamera(batch_input_metas, imgs)

            img_feature = self.extract_img_feat(imgs, points, lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix, lidar_aug_matrix,
                                                batch_input_metas)

            # Cache dimensions (without batch)
            _, self._img_channels, self._img_height, self._img_width = img_feature.shape
            features.append(img_feature)

        elif imgs is not None and drop_cam:
            # Need to initialize cache first
            if self._img_channels is None:
                # Extract once to get dimensions
                imgs_temp = imgs.contiguous()
                lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix = \
                    self.setupTransformsCamera(batch_input_metas, imgs_temp)

                temp_feature = self.extract_img_feat(imgs_temp, points, lidar2image, camera_intrinsics,
                                                     camera2lidar, img_aug_matrix, lidar_aug_matrix,
                                                     batch_input_metas)
                _, self._img_channels, self._img_height, self._img_width = temp_feature.shape

            # Create zeros with CURRENT batch size
            zero_img_feature = torch.zeros(
                current_batch_size,
                self._img_channels,
                self._img_height,
                self._img_width,
                device=imgs.device,
                dtype=imgs.dtype
            )
            features.append(zero_img_feature)

        # === LiDAR Features ===
        if points is not None and not drop_lidar:
            pts_feature = self.extract_pts_feat(batch_inputs_dict)

            # Cache dimensions (without batch)
            _, self._pts_channels, self._pts_height, self._pts_width = pts_feature.shape
            features.append(pts_feature)

        elif drop_lidar and len(features) > 0:
            # Need to initialize cache first
            if self._pts_channels is None:
                temp_feature = self.extract_pts_feat(batch_inputs_dict)
                _, self._pts_channels, self._pts_height, self._pts_width = temp_feature.shape

            # Create zeros with CURRENT batch size
            zero_pts_feature = torch.zeros(
                current_batch_size,
                self._pts_channels,
                self._pts_height,
                self._pts_width,
                device=features[0].device,
                dtype=features[0].dtype
            )
            features.append(zero_pts_feature)

        # LEVER 3: the pre-fusion camera BEV (view_transform output) is features[0] when
        # the camera branch ran; capture it before fusion so the keypoint head's dedicated
        # camera descriptor sub-space can read it. None when camera is absent/dropped.
        camera_bev = features[0] if (imgs is not None and not drop_cam and len(features) > 0) else None

        # === Fusion ===
        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, f"Expected 1 feature without fusion, got {len(features)}"
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        if kwargs.get('return_camera_bev', False):
            return x, camera_bev
        return x

    def getDrop(self, iterator):


        if not self.use_dropout and not self.use_camera or not self.use_lidar:

            drop_cam = not self.use_camera
            drop_lidar = not self.use_lidar
            return drop_cam, drop_lidar


        max_prob = 0.7
        warmup_iters = 500
        ramp_iters = 2000

        if iterator <= warmup_iters:
            prob = 0.0
        else:
            t = min(1.0, (iterator - warmup_iters) / (ramp_iters - warmup_iters))
            prob = max_prob * t

        drop_cam = np.random.rand() < prob
        drop_lidar = np.random.rand() < prob

        if drop_cam and drop_lidar:
            if np.random.rand() < 0.5:
                drop_cam = False
            else:
                drop_lidar = False

        return drop_cam, drop_lidar

    def lidar2global(self, meta):
        return np.array(meta['ego2global'], dtype=np.float32) @ \
            np.array(meta['lidar_points']['lidar2ego'], dtype=np.float32)