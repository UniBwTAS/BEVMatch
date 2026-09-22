# Sparse-Detector-Refinement auf NUSCENES (Revision, Plan-Item A6/B1-nuScenes): gleiche Rezeptur
# wie der in-house-Port — eingefrorenes nuScenes-Papier-Modell (epoch_8), NUR der Score-Head
# trainiert mit den geometrischen KITTI-Losses (pose-only, label-frei):
#   detach_score_head=True + descriptor_loss_weight=0 + seg/bbox aus + Dropout aus + Warmup 0/500
#   (alles vom Basis-Config geerbt). Loader auf echtes nuScenes umgestellt; LoadBEVSegmentation
#   entfernt (Seg-Head weg; Speicher/IO). Pipeline exakt aus dem Trainings-Dump des Papier-Modells
#   uebernommen (resize_lim 0.48, final_dim 256x704, load_dim/use_dim 5, PointsRangeFilter +-54m),
#   nur ohne Seg-Transform und ohne 'gt_masks_bev' meta_key. CBGS-Wrapper weggelassen
#   (Klassen-Resampling irrelevant fuer den Detektor).
_base_ = ['./bevmatch_nuscenes.py']

_nus_root = 'data/nuscenes/'
_pcr = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
_prefix = dict(
    CAM_BACK='samples/CAM_BACK', CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT', CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT', CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP')
_meta = dict(classes=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                      'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'])

_train_pipe = [
    dict(backend_args=None, color_type='color', to_float32=True,
         type='BEVLoadMultiViewImageFromFiles'),
    dict(backend_args=None, coord_type='LIDAR', load_dim=5, type='LoadPointsFromFile', use_dim=5),
    dict(bot_pct_lim=[0.0, 0.0], final_dim=[256, 704], is_train=True, rand_flip=False,
         resize_lim=[0.48, 0.48], rot_lim=[0.0, 0.0], type='ImageAug3D'),
    dict(type='LoadAnnotations3D', with_attr_label=False, with_bbox_3d=True, with_label_3d=True),
    dict(point_cloud_range=_pcr, type='PointsRangeFilter'),
    dict(keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
         meta_keys=['ego2global', 'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
                    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx', 'lidar_path',
                    'img_path', 'transformation_3d_flow', 'pcd_rotation', 'pcd_scale_factor',
                    'pcd_trans', 'img_aug_matrix', 'lidar_aug_matrix', 'lidar_points'],
         type='Pack3DDetInputs'),
]
_test_pipe = [
    dict(backend_args=None, color_type='color', to_float32=True,
         type='BEVLoadMultiViewImageFromFiles'),
    dict(backend_args=None, coord_type='LIDAR', load_dim=5, type='LoadPointsFromFile', use_dim=5),
    dict(bot_pct_lim=[0.0, 0.0], final_dim=[256, 704], is_train=False, rand_flip=False,
         resize_lim=[0.48, 0.48], rot_lim=[0.0, 0.0], type='ImageAug3D'),
    dict(point_cloud_range=_pcr, type='PointsRangeFilter'),
    dict(keys=['points', 'img'],
         meta_keys=['ego2global', 'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
                    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx', 'lidar_path',
                    'img_path', 'transformation_3d_flow', 'pcd_rotation', 'pcd_scale_factor',
                    'pcd_trans', 'img_aug_matrix', 'lidar_aug_matrix', 'lidar_points'],
         type='Pack3DDetInputs'),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=5,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'),
    dataset=dict(
        type='NuScenesDatasetKeypoints',
        data_root=_nus_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=_train_pipe,
        metainfo=_meta,
        modality=dict(use_camera=True, use_lidar=True),
        data_prefix=_prefix,
        box_type_3d='LiDAR',
        test_mode=False,
        use_valid_flag=True))

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        type='NuScenesDatasetKeypoints',
        data_root=_nus_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=_test_pipe,
        metainfo=_meta,
        modality=dict(use_camera=True, use_lidar=True),
        data_prefix=_prefix,
        box_type_3d='LiDAR',
        test_mode=True))
test_dataloader = val_dataloader

# Mid-Epoch-Snapshots fuer die Knee-Suche (Papier-Protokoll-Evals pro Snapshot).
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500, max_keep_ckpts=16))
