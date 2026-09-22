"""BEVMatch — keypoint detection and matching in a shared BEV latent space.

Modules for the two released settings: nuScenes (the paper's main model, with
cross-attention fusion and the auxiliary segmentation head) and KITTI odometry.
Importing this package registers all components with the MMDetection3D registry,
so the config file can refer to them by name.
"""
from .attention_fusion import HardGateDownsampleAttentionFuser
from .bevfusion import BEVFusionKeypoints
from .bevfusion_necks import GeneralizedLSSFPN
from .data_preprocessor_keypoints import Det3DDataPreprocessorKeypoints
from .depth_lss import DepthLSSTransform, LSSTransform
from .keypoint_head import KeypointHead
from .kitti_odometry_keypoints import KittiOdometryKeypoints, WrapSingleImageAsMultiView
from .loading import BEVLoadMultiViewImageFromFiles
from .nuscenes_dataset_keypoints import NuScenesDatasetKeypoints
from .nuscenes_metric_keypoints import NuScenesMetricKeypoints
from .sparse_encoder import BEVFusionSparseEncoder
from .transfusion_head import ConvFuser
from .transforms_3d import (BEVFusionGlobalRotScaleTrans, BEVFusionRandomFlip3D,
                            ImageAug3D, KeepBeamsByElevation)
from .vanilla import BEVSegmentationHeadVanilla

__all__ = [
    'BEVFusionKeypoints', 'KeypointHead', 'Det3DDataPreprocessorKeypoints',
    'KittiOdometryKeypoints', 'WrapSingleImageAsMultiView',
    'NuScenesDatasetKeypoints', 'NuScenesMetricKeypoints',
    'HardGateDownsampleAttentionFuser',
    'BEVSegmentationHeadVanilla',
    'BEVFusionSparseEncoder', 'GeneralizedLSSFPN', 'DepthLSSTransform', 'LSSTransform',
    'ConvFuser', 'BEVLoadMultiViewImageFromFiles', 'ImageAug3D',
    'BEVFusionGlobalRotScaleTrans', 'BEVFusionRandomFlip3D', 'KeepBeamsByElevation',
]
