# modify from https://github.com/mit-han-lab/bevfusion
import copy
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from mmdet3d.registry import MODELS
from torch.utils.tensorboard import SummaryWriter
import cv2
import math
from mmdet.models.losses import FocalLoss
import torch.distributed as dist
from mmengine.dist import get_rank  # 0 without a process group, e.g. single-GPU training
from mmcv.cnn import build_norm_layer, build_activation_layer
import time
from datetime import datetime
from info_nce import InfoNCE
from sklearn.decomposition import PCA
from .keypoint_ops import compose_aug_warp, generate_correspondence as gen_corr, build_keypoint_targets, extract_keypoints, keypoint_valid_mask, build_geometric_targets_pair, xy_to_cells, render_gaussian_heatmap, build_superpoint_targets, superpoint_logits_to_heatmap
from .bev_warp_ops import warp_feature_map, random_bev_warp
from .losses import repeatability_cosine_loss, recall_hinge_loss, peakiness_ce_loss, repulsion_loss_logits, supervised_focal_loss, binary_descriptor_metric_loss, binary_selfsup_descriptor_loss, superpoint_detection_loss


COLOR_T0 = (0, 255, 0)  # Green for time t0
COLOR_T1 = (255, 255, 0)  # Yellow for time t1


# Methode 6: Progressive Sparsity (während Training stärker werden)
class ProgressiveWeight:
    def __init__(self, initial_weight=0.01, final_weight=1.0, total_steps=10000):
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.total_steps = total_steps

    def get_weight(self, current_step):
        if current_step >= self.total_steps:
            return self.final_weight

        progress = min(current_step / self.total_steps, 1.0)
        # linear progression
        weight = self.initial_weight + progress * (self.final_weight - self.initial_weight)

        return weight

@MODELS.register_module()
class KeypointHead(nn.Module):
    def __init__(
            self,
            in_channels=512,
            nms_kernel_size=5,
            bn_momentum=0.1,
            hidden_channel=256,
            upsample_factor=2,
            #norm_cfg=dict(type='GN', num_groups=32),
            #batchnorm2d
            norm_cfg=dict(type='BN2d', momentum=0.1, affine=True),
            act_cfg=dict(type='ReLU', inplace=True),
            log_dir=None,
            dim_grid: int = 180,
            tau: float = 0.5,
            kpt_warmup_phase: int = 5000,
            kpt_warmup_steps: int = 5000,
            kpt_artifact_mask: bool = True,
            kpt_mask_border: int = 8,
            kpt_mask_center_halfwidth: int = 4,
            repulsion_distance: float = 10.0,
            repulsion_weight: float = 1.5,
            l_sparsity: float = 0.0002,
            use_geometric_targets: bool = False,
            covariance_weight: float = 0.0,
            covariance_max_rot_deg: float = 20.0,
            covariance_max_trans_cells: float = 20.0,
            geometric_min_score: float = 0.0,
            geometric_max_targets: int = 0,
            reliability_weight: float = 0.0,
            detach_score_head: bool = False,
            supervised: bool = False,
            supervised_sigma: float = 2.0,
            binary_descriptor: bool = False,
            descriptor_bits: int = 256,
            descriptor_dilations=None,
            local_hard_neg: bool = False,
            local_neg_radius: int = 10,
            local_neg_max: int = 512,
            sparse_keypoint_descriptor: bool = False,
            sparse_dense_blend: float = 0.0,
            camera_descriptor_bits: int = 0,
            camera_bev_channels: int = 80,
            superpoint_detector: bool = False,
            sp_cell: int = 4,
            sp_dustbin_weight: float = 1.0,
            info_nce_negative_mode: str = 'unpaired',
    repeatability_weight: float = 0.0,
    repeatability_k: int = 5,
    density_mask_train: bool = False,
    density_min_nbr_points: int = 8,
    density_nbr_r: int = 1,
    density_neg_weight: float = 0.5,
    descriptor_loss_weight: float = 1.0,
    cross_modal_weight: float = 0.0,
    cross_modal_temperature: float = 0.15,
    ):
        super(KeypointHead, self).__init__()
        # CROSS-MODAL descriptor consistency (fixes C2L): when >0, an InfoNCE term aligns the
        # descriptor of each detected keypoint across fused/camera-only/lidar-only feature extractions
        # (same point = positive across modalities, other points = negatives). See
        # cross_modal_consistency_loss + BEVFusionKeypoints.loss.
        self.cross_modal_weight = float(cross_modal_weight)
        self.cross_modal_temperature = float(cross_modal_temperature)
        self.bn_momentum = bn_momentum
        self.in_channels = in_channels
        self.nms_kernel_size = nms_kernel_size
        self.hidden_channel = hidden_channel
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # Loss weights
        self.l_kpts = 2.0
        self.l_desc = 1.0
        # KITTI DETECTOR-ONLY KNOB (Task 4): multiplies the finalized float/self-sup
        # descriptor_loss (compute_losses_infoNCE's AVERAGE + WEIGHT step) alongside
        # l_desc, without replacing it. 0.0 -> descriptor_loss is exactly 0.0 so the
        # descriptor cannot backprop into the shared detector (KITTI has no
        # descriptor supervision signal). Default 1.0 leaves every prior config
        # byte-for-byte unchanged.
        self.descriptor_loss_weight = descriptor_loss_weight
        # SPARSITY/REPULSION (the "too dense" fix). The original used distance=2.0 cells
        # (1.2m), weight=0.3, l_sparsity=1e-5 -> far too weak -> dense blobs. Adapted to
        # enforce sparse, well-separated landmarks: wider repulsion radius, stronger
        # repulsion weight, and a meaningfully-weighted total-activation sparsity term.
        # Config-settable so the strength can be swept.
        self.repulsion_distance = repulsion_distance
        self.repulsion_weight = repulsion_weight
        self.l_sparsity = l_sparsity
        # REPEATABILITY levers (both off by default -> config C behavior unchanged):
        #  - use_geometric_targets: replace descriptor-correlation targets (saliency leak)
        #    with mutual-local-max-under-warp targets on the score maps.
        #  - covariance_weight: R2D2 synthetic-homography term. Warp the BEV feature by a
        #    random affine, re-run the score head, penalize score maps that don't warp
        #    identically -> rewards content-based detection, kills fixed-frame/border firing.
        self.use_geometric_targets = use_geometric_targets
        self.covariance_weight = covariance_weight
        self.covariance_max_rot_deg = covariance_max_rot_deg
        self.covariance_max_trans_cells = covariance_max_trans_cells
        # over-fire guard for geometric targets (config D lesson)
        self.geometric_min_score = geometric_min_score
        self.geometric_max_targets = geometric_max_targets
        # descriptor-reliability weighting for detection targets (default 0.0 = off)
        self.reliability_weight = reliability_weight
        # decouple detector from encoder (protect descriptors from keypoint-loss grad)
        self.detach_score_head = detach_score_head
        # SUPERVISED DETECTION (Phase 2 stripe-artifact kill): when True, the detection
        # loss is a CenterNet penalty-reduced focal loss against the teacher's landmark
        # heatmap (rendered from gt_keypoint_xy on the current frame), replacing the
        # self-supervised geometric/descriptor-correlation detection terms. The
        # descriptor InfoNCE path stays unchanged (Phase 3 redesigns it). Default False
        # leaves every prior config byte-for-byte unchanged.
        self.supervised = supervised
        self.supervised_sigma = supervised_sigma

        # BINARY DESCRIPTOR (Task 9): when set, the descriptor head emits
        # descriptor_bits tanh-domain logits instead of the 128-D InfoNCE field.
        # Training learns them by a metric loss over the teacher's persistence
        # correspondences (shared gt_track_id across the current/other frame pair);
        # inference sign-packs them to uint8 codes (descriptor_bits//8 per cell).
        # Default False leaves the 128-D InfoNCE path byte-for-byte unchanged.
        self.binary_descriptor = binary_descriptor
        self.descriptor_bits = descriptor_bits
        # WIDER-CONTEXT DESCRIPTOR (flatfix #2, gated): when descriptor_dilations is a
        # list (e.g. [1,2,4]) the descriptor head is built from stacked dilated 3x3
        # convs -> a receptive field of tens of metres so flat cells get identity from
        # distant structure. None (default) -> the legacy single-3x3 descriptor head,
        # byte-identical to prior configs. Changing this arch means a warm-start will
        # NOT load descriptor_head.* (shape mismatch) -> descriptor head trains fresh.
        self.descriptor_dilations = descriptor_dilations
        # LOCAL/FLAT HARD-NEGATIVE MINING (flatfix #1, gated): when local_hard_neg=True
        # the self-sup binary descriptor loss additionally mines negatives from cells
        # within local_neg_radius (Chebyshev, cells) of the correspondences (excluding
        # the correspondences themselves), capped at local_neg_max. Default OFF ->
        # byte-identical to prior configs.
        self.local_hard_neg = local_hard_neg
        self.local_neg_radius = local_neg_radius
        self.local_neg_max = local_neg_max
        # SPARSE KEYPOINT DESCRIPTOR (gated): train the binary descriptor ONLY at DETECTED
        # keypoint cells (top-K heatmap peaks) rather than at every geometric-correspondence
        # cell -> aligns training with inference (descriptor read only at keypoints) and, with
        # local_hard_neg, teaches the code to disambiguate a keypoint from its spatial
        # confusers (the ties a pose-guided geometric matcher must resolve). Anchors restricted
        # to detected keypoints; positives stay the geometric-warp partners. Default OFF.
        self.sparse_keypoint_descriptor = sparse_keypoint_descriptor
        # SPARSE v2 stabilizer: when >0, additionally keep a random subset of NON-keypoint
        # geometric correspondences equal to sparse_dense_blend x (#keypoint anchors), so each
        # step has more (but still keypoint-dominated) anchors -> smoother gradients than the
        # pure-sparse loss, which oscillates because it has few anchors. 0.0 = pure sparse.
        self.sparse_dense_blend = sparse_dense_blend
        if binary_descriptor:
            assert descriptor_bits % 8 == 0, (
                f"descriptor_bits ({descriptor_bits}) must be a multiple of 8 for uint8 packing")
        descriptor_out_channels = descriptor_bits if binary_descriptor else 128

        # SUPERPOINT DETECTOR (gated, default-OFF): when True, replaces the dense
        # per-cell sigmoid heatmap + focal-loss formulation with a SuperPoint-style
        # softmax-over-region + dustbin head. Detection becomes a per-region
        # classification (no positive/negative dilution). Default False leaves the
        # entire dense path byte-for-byte unchanged. Mirror of supervised/binary_descriptor
        # gating pattern.
        self.superpoint_detector = superpoint_detector
        self.sp_cell = sp_cell
        self.sp_dustbin_weight = sp_dustbin_weight

        # HARD-NEGATIVE InfoNCE mode (gated, default 'unpaired' = byte-identical to prior).
        # When 'hard': per-positive negatives are drawn from a window around the QUERY's BEV
        # position (guaranteeing the same-position cell is a negative), plus random, excluding
        # cells near the TRUE positive (false-negative avoidance). Breaks descriptor
        # position-locking without changing any other training path.
        self.info_nce_negative_mode = info_nce_negative_mode
        self.info_nce_hard_window = 6   # Chebyshev radius in cells for the window draw
        self.info_nce_hard_num = 64     # total negatives per positive (Nwin=32 + Nrand=32)

        # CROSS-FRAME DETECTION REPEATABILITY (warp-consistency on the SP heatmap).
        # When repeatability_weight > 0 and supervised=True, a gated warp-consistency
        # term is added: 1 - mean cosine between score-map windows of the CURRENT frame
        # at corr0_valid and the OTHER frame at corr1_valid (the real pose correspondences
        # already used for InfoNCE). This directly optimises detection warp-equivariance
        # -> lifts repeatability without any extra GT. Default 0.0 = byte-identical.
        self.repeatability_weight = repeatability_weight
        self.repeatability_k = repeatability_k

        # DENSITY-MASK TRAINING (gated, default OFF -> byte-identical): when True,
        # suppress the detector score map on low-density (empty-space) BEV cells
        # DURING training so the top-K budget and all downstream losses focus on
        # real LiDAR structure. density_nbr_r controls the (2r+1)^2 neighbourhood
        # cell count; density_min_nbr_points is the threshold. Deploy forward() and
        # ONNX export are unaffected (no points available there).
        self.density_mask_train = density_mask_train
        self.density_min_nbr_points = density_min_nbr_points
        self.density_nbr_r = density_nbr_r
        self.density_neg_weight = density_neg_weight

        # ONNX export mode: when True, emit raw float32 sign-bits instead of packed
        # uint8 codes (ORT doesn't implement Mul for uint8). Default False -> normal
        # inference behavior (uint8 packing). This attribute must persist through
        # serialization/deepcopy; initialized here so getattr(self, 'onnx_export', False)
        # survives a round-trip.
        self.onnx_export = False

        # Spatial parameters
        self.dim_spatial = 108.0
        self.dim_grid = dim_grid  # eager: from config, no lazy first-forward init
        self.bev_resolution = self.dim_spatial / self.dim_grid
        self.tau = tau  # calibrated inference threshold; overridden by config / τ-sweep
        # Keypoint-loss warmup (config-surfaced): kpt losses are zero for the first
        # kpt_warmup_phase steps, then ramp over kpt_warmup_steps. Set small for the
        # overfit gate; default 5000/5000 for full training.
        self.kpt_warmup_phase = kpt_warmup_phase
        self.kpt_warmup_steps = kpt_warmup_steps

        # Artifact mask: suppress keypoints on the fixed BEV border + center axis
        # (ego/sensor-origin column). These cells are identical every frame, so
        # the repeatability/recall objective collapses onto them; masking forces
        # the head onto real interior structure. Applied to logits in forward()
        # AND to the supervision targets in compute_losses_infoNCE. See
        # keypoint_ops.keypoint_valid_mask.
        self.kpt_artifact_mask = kpt_artifact_mask
        self.kpt_mask_border = kpt_mask_border
        self.kpt_mask_center_halfwidth = kpt_mask_center_halfwidth
        self.kpt_mask_fill = -1e4  # additive logit bias -> sigmoid ~0 at masked cells
        self._kpt_valid_mask_cache = None  # lazily built [H,W] bool, cached per device

        # Build keypoint / SuperPoint detector head with configurable norm.
        # When superpoint_detector=True: build self.superpoint_head (sp_cell**2+1 channels)
        # and do NOT build self.keypoint_head, avoiding DDP unused-param with
        # find_unused_parameters=False. When False: build self.keypoint_head (1 channel)
        # as today and do NOT build self.superpoint_head.
        if superpoint_detector:
            assert dim_grid % sp_cell == 0, (
                f"dim_grid ({dim_grid}) must be divisible by sp_cell ({sp_cell})")
            # The SP head must output [B, sp_cell**2+1, g, g] where g = dim_grid//sp_cell.
            # Use a stride-sp_cell first conv to downsample the BEV feature grid to the
            # coarse region grid in one step (matches the SuperPoint detector architecture
            # where the detector head operates on coarse features). padding=(sp_cell//2)
            # keeps the output grid exactly g = dim_grid // sp_cell.
            _sp_pad = sp_cell // 2
            layers = []
            layers.append(nn.Conv2d(in_channels, hidden_channel,
                                    kernel_size=sp_cell, stride=sp_cell, padding=0))
            if act_cfg is not None:
                layers.append(build_activation_layer(act_cfg))
            if norm_cfg is not None:
                _, norm_layer = build_norm_layer(norm_cfg, hidden_channel)
                layers.append(norm_layer)
            layers.append(nn.Conv2d(hidden_channel, sp_cell ** 2 + 1, kernel_size=1))
            self.superpoint_head = nn.Sequential(*layers)
        else:
            self.keypoint_head = self._make_head(
                in_channels=in_channels,
                hidden_channel=hidden_channel,
                out_channels=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

        # Build descriptor head with configurable norm.
        # out_channels = descriptor_bits in binary mode, else the legacy 128-D field.
        self.descriptor_head = self._make_head(
            in_channels=in_channels,
            hidden_channel=hidden_channel,
            out_channels=descriptor_out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dilations=descriptor_dilations,
        )

        # LEVER 3 (dedicated camera descriptor sub-space, gated): the modality ablation
        # showed the fused descriptor == lidar-only (camera ignored at deployment) because
        # a single shared code defaults to the sharper LiDAR signal. When
        # camera_descriptor_bits>0, an ADDITIONAL small head reads the PRE-fusion camera
        # BEV (view_transform out, camera_bev_channels ch) and emits camera_descriptor_bits
        # tanh-domain logits that are CONCATENATED after the geometry bits -> a
        # (descriptor_bits + camera_descriptor_bits) code. Camera AUGMENTS geometry instead
        # of competing for the same bits; the geometry descriptor_head keeps descriptor_bits
        # so it warm-loads 1:1 and only the camera head trains fresh. Default 0 = off
        # (existing configs byte-identical). Requires binary_descriptor.
        self.camera_descriptor_bits = camera_descriptor_bits
        self.camera_bev_channels = camera_bev_channels
        if camera_descriptor_bits > 0:
            assert binary_descriptor, "camera_descriptor_bits requires binary_descriptor=True"
            assert camera_descriptor_bits % 8 == 0, (
                f"camera_descriptor_bits ({camera_descriptor_bits}) must be a multiple of 8")
            self.camera_descriptor_head = nn.Sequential(
                nn.Conv2d(camera_bev_channels, hidden_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channel, camera_descriptor_bits, 1),
            )

        # Loss function
        """self.kpt_loss_fn = FocalLoss(
            use_sigmoid=True,  # works with raw logits
            gamma=2.0, # 1
            alpha=0.25, # 0.5
            reduction='mean',
            loss_weight=1.0
        )"""
        # 2.0 0.25 original, no keypoints detected
        # 1.0, 0.75 keypoints2, too dense
        # 1.5, 0.5 keypoints3, no keypoints detected
        # 1.2, 0.6 keypoints3, good
        self.kpt_loss_fn = FocalLoss(
            use_sigmoid=True,
            gamma=1.2,  # GEÄNDERT: von 2.0 auf 1.0
            alpha=0.4,  # Task 10: tuned for geometry-target regime
            reduction='mean',
            loss_weight=1.0
        )

        # Setup tensorboard writers
        #get current_date+time as string, format YYYYMMDD_HHMM
        datestring = datetime.now().strftime("%Y%m%d_%H%M")
        if log_dir is None:
            log_dir = os.environ.get("KPT_LOG_DIR", "./work_dirs/keypoint_debug")

        log_dir_train = f"{log_dir}/training{datestring}"
        log_dir_val = f"{log_dir}/validation{datestring}"
        #only for rank 0
        if get_rank() == 0:
            print(f"Logging to {log_dir_train} and {log_dir_val}")
            self.writer_train = SummaryWriter(log_dir=log_dir_train)
            self.writer_val = SummaryWriter(log_dir=log_dir_val)
        self.step = 0
        self.step_val = 0

    def _make_head(self, in_channels, hidden_channel, out_channels, norm_cfg, act_cfg,
                   dilations=None):
        """Build a head with conv-act-norm-conv structure.

        Args:
            in_channels (int): Input channels.
            hidden_channel (int): Hidden layer channels.
            out_channels (int): Output channels.
            norm_cfg (dict): Normalization config.
            act_cfg (dict): Activation config.
            dilations (list[int] | None): When None (default) the head is a single
                3x3(pad1)-act-norm-conv1x1 stack — byte-identical to the legacy head.
                When a list (e.g. [1,2,4]) one 3x3 conv-act-norm block is stacked per
                dilation (padding=dilation keeps the spatial size), giving each output
                cell a much larger receptive field (wider context) before the final
                1x1 projection. Used for the wider-context descriptor head so flat
                points can borrow identity from distant structure.

        Returns:
            nn.Sequential: The head module.
        """
        layers = []

        # dilations=None -> [1] reproduces the original single-3x3 head exactly.
        _dilations = [1] if dilations is None else list(dilations)
        ch_in = in_channels
        for d in _dilations:
            # padding=d keeps H,W fixed for a 3x3 kernel at dilation d.
            layers.append(
                nn.Conv2d(ch_in, hidden_channel,
                          kernel_size=3, stride=1, padding=d, dilation=d)
            )
            if act_cfg is not None:
                layers.append(build_activation_layer(act_cfg))
            if norm_cfg is not None:
                # build_norm_layer returns (name, layer)
                _, norm_layer = build_norm_layer(norm_cfg, hidden_channel)
                layers.append(norm_layer)
            ch_in = hidden_channel

        # Output conv
        layers.append(
            nn.Conv2d(hidden_channel, out_channels, kernel_size=1)
        )

        return nn.Sequential(*layers)

    def normalize_keypoints(self, logits):
        """Apply softmax to keypoint probabilities across spatial cells."""
        B, C, H, W = logits.shape
        logits = logits.view(B, C, -1)  # Flatten spatial dimensions
        logits = F.softmax(logits, dim=-1)  # Normalize probabilities
        return logits.view(B, C, H, W)

    def _make_upsample_layers(self, channels, factor):
        """
        Create ConvTranspose2d layers to upsample by `factor`.
        For factor=2 -> one layer, factor=4 -> two layers, etc.
        """
        layers = []
        num_layers = int(torch.log2(torch.tensor(factor)))
        for _ in range(num_layers):
            layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(channels))
        return layers


    @staticmethod
    def soft_nms_3d(heat, kernel=31):
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    ############# Keypoint Target Generation, radius around ground-truth #############
    def generate_keypoint_targets_with_radius(self, corr0, corr1, dim, radius=2):
        """
        Generiert erweiterte Keypoint-Targets durch Dilatation um MNN-Punkte
        """
        device = corr0.device
        y_target = torch.zeros(dim * dim, device=device)

        # Basis MNN targets
        y_target[corr0] = 1.0
        y_target[corr1] = 1.0

        # Erweitere um Nachbarn (optional)
        if radius > 0:
            y_target_2d = y_target.view(dim, dim)
            kernel = torch.ones(2 * radius + 1, 2 * radius + 1, device=device)
            y_target_2d = F.conv2d(
                y_target_2d.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=radius
            ).squeeze().clamp(0, 1)
            y_target = y_target_2d.view(-1)

        return y_target

    def get_kpt_valid_mask(self, dim, device):
        """Cached [dim,dim] bool artifact mask (True == interior/allowed).

        Returns None when masking is disabled. Cache is rebuilt if device or dim
        changes; the mask is a fixed function of geometry so this is a one-time cost.
        """
        if not getattr(self, 'kpt_artifact_mask', False):
            return None
        m = self._kpt_valid_mask_cache
        if m is None or m.shape[-1] != dim or m.device != device:
            m = keypoint_valid_mask(
                dim, border=self.kpt_mask_border,
                center_halfwidth=self.kpt_mask_center_halfwidth, device=device)
            self._kpt_valid_mask_cache = m
        return m

    def get_kpt_valid_mask_eroded(self, dim, device, k=5):
        """Valid mask eroded by k//2, for the repeatability windows.

        The rep cosine gathers k*k LOGIT windows; a valid cell within k//2 of a
        masked region has masked (~-inf) cells in its window, corrupting the
        cosine. Eroding keeps only cells whose FULL k*k window is unmasked.
        Returns None when masking is disabled.
        """
        base = self.get_kpt_valid_mask(dim, device)
        if base is None:
            return None
        e = self._kpt_eroded_mask_cache if hasattr(self, '_kpt_eroded_mask_cache') else None
        if e is None or e.shape[-1] != dim or e.device != device:
            inv = (~base).float().view(1, 1, dim, dim)
            pad = k // 2
            dil = F.max_pool2d(inv, k, stride=1, padding=pad)  # dilate masked region
            e = (dil.view(dim, dim) == 0)                       # erode valid region
            self._kpt_eroded_mask_cache = e
        return e

    @staticmethod
    def pack_binary_descriptor(logits):
        """Sign-pack tanh-domain descriptor logits to uint8 codes.

        logits: [B, D, H, W] (D = descriptor_bits, a multiple of 8). Each channel
        becomes one bit (1 if logit > 0 else 0); 8 consecutive channels are packed
        MSB-first into one uint8, giving [B, D//8, H, W] uint8. Inverse of the
        sign(tanh(z)) the metric loss optimizes (tanh is monotone, so the bit is
        just sign(z)).
        """
        B, D, H, W = logits.shape
        assert D % 8 == 0, f"descriptor_bits {D} must be a multiple of 8 for packing"
        bits = (logits > 0).to(torch.uint8)                    # [B, D, H, W] in {0,1}
        bits = bits.view(B, D // 8, 8, H, W)                    # group into bytes
        weights = (1 << torch.arange(7, -1, -1, device=logits.device, dtype=torch.uint8))
        packed = (bits * weights.view(1, 1, 8, 1, 1)).sum(dim=2)  # MSB-first
        return packed.to(torch.uint8)                          # [B, D//8, H, W]

    def forward(self, feats, metas, camera_bev=None):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
            camera_bev (torch.Tensor | None): pre-fusion camera BEV feature
                [B, camera_bev_channels, h, w] for the LEVER 3 dedicated camera
                descriptor sub-space. Only used when camera_descriptor_bits>0.
        Returns:
            tuple(list[dict]): Output results. first index by level, second
            index by layer
        """

        # keypoint head. DECOUPLE detector from encoder: when detach_score_head is set,
        # the score head trains on DETACHED features so the keypoint losses (focal/recall/
        # peak/repulsion/covariance) NEVER backprop into the shared encoder. The encoder
        # (and thus the descriptor field) is then trained ONLY by the InfoNCE descriptor
        # loss -> descriptors stay stable while detection improves on top of fixed features
        # (SuperPoint-style detector-on-frozen-backbone). Fixes the match_precision crash
        # 0.36->0.00 that the geometric+covariance keypoint losses caused via the encoder.
        kp_in = feats
        if self.training and getattr(self, 'detach_score_head', False):
            kp_in = feats.detach()

        if getattr(self, 'superpoint_detector', False):
            # SUPERPOINT DETECTOR: predict softmax-over-region + dustbin logits, then
            # PixelShuffle back to the dense [B,1,dim,dim] logit heatmap so all
            # downstream consumers (extract_keypoints, viz, eval) are unaffected.
            from .keypoint_ops import superpoint_logits_to_heatmap
            sp_logits = self.superpoint_head(kp_in)
            keypoint = superpoint_logits_to_heatmap(sp_logits, self.sp_cell)
        else:
            keypoint = self.keypoint_head(kp_in)
        #keypoint = self.soft_nms_3d(keypoint, self.nms_kernel_size)

        # Artifact mask: drive border + center-axis logits to ~-inf so they can
        # never be selected (inference) or rewarded (loss). Applied here so EVERY
        # consumer — both loss paths, repulsion, viz, eval — sees masked logits.
        vmask = self.get_kpt_valid_mask(keypoint.shape[-1], keypoint.device)
        if vmask is not None:
            keypoint = keypoint + (~vmask).to(keypoint.dtype) * self.kpt_mask_fill

        # descriptor head (geometry bits from the fused feature)
        descriptor = self.descriptor_head(feats)

        # LEVER 3: append the dedicated camera-appearance bits read from the PRE-fusion
        # camera BEV. Resized to the descriptor grid if the camera BEV is at a different
        # resolution. If camera_bev is missing at inference, append zero bits so the code
        # width (and thus Hamming/packing) stays constant. Concatenated BEFORE sign/pack.
        if getattr(self, 'camera_descriptor_bits', 0) > 0:
            if camera_bev is not None:
                cam_desc = self.camera_descriptor_head(camera_bev)
                if cam_desc.shape[-2:] != descriptor.shape[-2:]:
                    cam_desc = F.interpolate(cam_desc, size=descriptor.shape[-2:],
                                             mode='bilinear', align_corners=False)
            else:
                cam_desc = descriptor.new_zeros(
                    descriptor.shape[0], self.camera_descriptor_bits,
                    descriptor.shape[2], descriptor.shape[3])
            descriptor = torch.cat([descriptor, cam_desc], dim=1)

        # BINARY DESCRIPTOR inference: in eval mode emit sign-packed uint8 codes
        # ([B, descriptor_bits//8, H, W]) instead of the raw tanh logits. Training
        # keeps the raw logits so the metric loss (tanh-domain) backprops normally.
        #
        # ONNX-EXPORT MODE: ORT does not implement Mul for uint8 operands, so the
        # packed-uint8 path is not ORT-runnable. When self.onnx_export is True the
        # head emits raw per-bit float32 sign-bits ([B, descriptor_bits, H, W]) with
        # values in {0.0, 1.0} instead. The uint8 packing is then done on the HOST
        # after inference (post-processing, not inside the ONNX graph).
        # Normal eval (onnx_export not set / False) is byte-for-byte unchanged.
        if getattr(self, 'binary_descriptor', False) and not self.training:
            if getattr(self, 'onnx_export', False):
                descriptor = (descriptor > 0).float()
            else:
                descriptor = self.pack_binary_descriptor(descriptor)

        """
        keypoint = self.keypoint_head(feats)
        keypoint = self.soft_nms_3d(keypoint, self.nms_kernel_size)
        descriptor = self.descriptor_head(feats)
        intermediate_feats = feats
        """

        assert keypoint.shape[2] == keypoint.shape[3]

        assert keypoint.shape[2] == self.dim_grid, (
            f"feature grid {keypoint.shape[2]} != configured dim_grid {self.dim_grid}")
        return keypoint, descriptor, feats

    """
    def predict(self, feat0, position_current, feat1, position_other, input_points_current=None, input_points_other=None, input_images_current=None, input_images_other=None):
        with torch.no_grad():
            kpts0, desc0, _ = self.forward(feat0[0], None)
            kpts1, desc1, _ = self.forward(feat1[0], None)

            R, t = self.get_transformation_matrix(position_current[0], position_other[0])
            corr0, corr1 = self.generate_correspondence(dim=self.dim_grid, R=R, t=t)

            if( get_rank() == 0):
                #also calculate losses
                self.step_val += 1
                self.visualize_val(R, t, dim=self.dim_grid, corr_tensor=torch.stack([corr0, corr1], dim=1),
                               kpts0=kpts0, kpts1=kpts1, desc0=desc0, desc1=desc1, writer = self.writer_val, step=self.step_val,
                                 images_current=input_images_current, images_other=input_images_other,
                                    points=input_points_current, points_other=input_points_other,
                                   input_feats0=feat0[0], input_feats1=feat1[0])


            return kpts0, desc0, kpts1, desc1
    """



    def predict(self, feat0, position_current, feat1, position_other, input_points_current=None,
                input_points_other=None, input_images_current=None, input_images_other=None):
        """
        Prediction method - now batch-compatible.

        Args:
            feat0: List containing [B, C, H, W] feature tensor
            position_current: List of length B with [4, 4] matrices
            feat1: List containing [B, C, H, W] feature tensor
            position_other: List of length B with [4, 4] matrices
        """
        #print feat shapes
        print("Predicting keypoints and descriptors...")
        print("feat0 shape:", feat0[0].shape)
        print("feat1 shape:", feat1[0].shape)
        onlyFirst = False
        visualize = False
        R_batch = []
        t_batch = []
        with torch.no_grad():
            if onlyFirst:
                ids = [0]
            else:
                ids = range(len(position_current))
            for id in ids:
                if feat0 is None or feat1 is None:
                    R_batch.append(None)
                    t_batch.append(None)
                    continue

                # Forward pass expects [B, C, H, W], so keep batch dimension
                kpts0, desc0, _ = self.forward(feat0[0][id:id+1], None)
                kpts1, desc1, _ = self.forward(feat1[0][id:id+1], None)
                R_estimate, t_estimate = self.calcTransform(kpts0, kpts1, desc0, desc1, dim=self.dim_grid, sim_thresh=None, topk=None)
                R_batch.append(R_estimate)
                t_batch.append(t_estimate)


                if visualize:
                    if get_rank() == 0:
                        # Calculate losses for validation
                        self.step_val += 1
                        # Process only first batch element for visualization
                        R, t = self.get_transformation_matrix(position_current[id], position_other[id])

                        corr0, corr1 = self.generate_correspondence(dim=self.dim_grid, R=R, t=t)

                        # Prepare data for visualization (first batch element only)
                        kpts0_vis = kpts0[id:id+1]
                        kpts1_vis = kpts1[id:id+1]
                        desc0_vis = desc0[id:id+1]
                        desc1_vis = desc1[id:id+1]
                        feat0_vis = feat0[0][id:id+1]
                        feat1_vis = feat1[0][id:id+1]

                        # Handle images and points
                        if input_images_current is not None:
                            if isinstance(input_images_current, list):
                                images_current_vis = [input_images_current[id]] if len(input_images_current) > 0 else None
                            else:
                                images_current_vis = input_images_current[id:id+1]
                        else:
                            images_current_vis = None

                        if input_images_other is not None:
                            if isinstance(input_images_other, list):
                                images_other_vis = [input_images_other[id]] if len(input_images_other) > 0 else None
                            else:
                                images_other_vis = input_images_other[id:id+1]
                        else:
                            images_other_vis = None

                        if input_points_current is not None:
                            if isinstance(input_points_current, list):
                                points_current_vis = [input_points_current[id]] if len(input_points_current) > 0 else None
                            else:
                                points_current_vis = input_points_current[id:id+1]
                        else:
                            points_current_vis = None

                        if input_points_other is not None:
                            if isinstance(input_points_other, list):
                                points_other_vis = [input_points_other[id]] if len(input_points_other) > 0 else None
                            else:
                                points_other_vis = input_points_other[id:id+1]
                        else:
                            points_other_vis = None

                        self.visualize_val(
                            R, t, dim=self.dim_grid,
                            corr_tensor=torch.stack([corr0, corr1], dim=1),
                            kpts0=kpts0_vis, kpts1=kpts1_vis,
                            desc0=desc0_vis, desc1=desc1_vis,
                            writer=self.writer_val, step=self.step_val,
                            images_current=images_current_vis,
                            images_other=images_other_vis,
                            points=points_current_vis,
                            points_other=points_other_vis,
                            input_feats0=feat0_vis,
                            input_feats1=feat1_vis
                        )


            return R_batch, t_batch


    def generate_correspondence(self, dim, R: torch.Tensor, t: torch.Tensor):
        device = R.device if isinstance(R, torch.Tensor) else torch.device('cpu')
        dim = int(dim)

        # Use 'ij' indexing: grid[i,j] where i=row, j=col
        grid_row, grid_col = torch.meshgrid(
            torch.arange(dim, device=device),
            torch.arange(dim, device=device),
            indexing='ij'
        )
        # points0[k] = [row, col] = [i, j]
        points0 = torch.stack([grid_row.flatten(), grid_col.flatten()], dim=1).float()

        # SE2 operates on [row, col] vectors
        SE2 = torch.eye(3, device=device, dtype=torch.float32)
        SE2[:2, :2] = R[:2, :2].to(device).float()
        SE2[:2, 2] = t[:2].to(device).float()
        invSE2 = torch.inverse(SE2)

        center = dim / 2.0
        pts_centered = points0 - center

        warped = self.warp_points(pts_centered + 0.5, invSE2)
        warped = torch.round(warped - 0.5)
        pts_rec = self.warp_points(warped + 0.5, SE2)
        pts_rec = torch.round(pts_rec - 0.5)

        warped = warped + center
        pts_rec = pts_rec + center
        points0 = points0.long()
        pts_rec = pts_rec.long()
        warped = warped.long()

        # Bijective: [row, col] must match
        bij_mask = (points0[:, 0] == pts_rec[:, 0]) & (points0[:, 1] == pts_rec[:, 1])
        src = points0[bij_mask]
        tgt = warped[bij_mask]

        in_bounds = (tgt[:, 0] >= 0) & (tgt[:, 1] >= 0) & (tgt[:, 0] < dim) & (tgt[:, 1] < dim)
        src = src[in_bounds]
        tgt = tgt[in_bounds]

        # Flatten: idx = row * dim + col
        corr0 = (src[:, 0] * dim + src[:, 1]).to(torch.long)
        corr1 = (tgt[:, 0] * dim + tgt[:, 1]).to(torch.long)

        return corr0, corr1

    def warp_points(self, points: torch.Tensor, H: torch.Tensor):
        """warp points with homography

        Args:
            points (torch.Tensor): [N, 2], [[y, x]] or [[row, col]]
            H (torch.Tensor): [3, 3] homography matrix
        """
        N = points.shape[0]
        points = torch.concat([points, torch.ones((N, 1))], dim=1)
        points = points.permute((1, 0))
        points = torch.matmul(H, points)
        points = points[0:2, :] / points[2:, :]
        points = points.permute((1, 0))
        return points


    def get_transformation_matrix(self, ego2global_A, ego2global_B,
                                  aug_current=None, aug_other=None):
        eye = torch.eye(4, dtype=torch.float64)
        augc = eye if aug_current is None else torch.as_tensor(aug_current, dtype=torch.float64)
        augo = eye if aug_other is None else torch.as_tensor(aug_other, dtype=torch.float64)
        R_grid, t_grid = compose_aug_warp(
            ego2global_A, ego2global_B, augc, augo, self.bev_resolution)
        return R_grid, t_grid


    def _compute_losses_infoNCE_loop(self, kpts0, kpts1, desc0, desc1, R, t,
                                     info_nce_temperature=0.15,
                                     info_nce_num_negatives=1024,
                                     info_nce_negative_mode='unpaired',
                                     validity_threshold=0.1,
                                     use_repulsion=True,
                                     repulsion_distance=2.0,
                                     batch_position_current=None,
                                     batch_position_other=None,
                                     aug_current=None,
                                     aug_other=None):
        """
        Batch-compatible loss: aug-aware geometry targets + InfoNCE + new keypoint terms.

        Args:
            kpts0, kpts1: [B, 1, H, W] keypoint logits
            desc0, desc1: [B, D, H, W] raw descriptors
            R, t: ignored (kept for API compat); warp is derived per-frame from poses+aug
            batch_position_current/other: list of B ego2global [4,4] matrices
            aug_current/other: list of B lidar_aug_matrix [4,4] (or None -> identity)
        """
        # Let the head-level attr override the function-arg default; config drives behavior.
        info_nce_negative_mode = getattr(self, 'info_nce_negative_mode', info_nce_negative_mode)

        device = kpts0.device
        batch_size = kpts0.shape[0]

        # ========== WARMUP WEIGHT (computed once, before the loop) ==========
        # During kpt_warmup_phase the 5 keypoint loss terms are zero (weight=0).
        # We skip building keypoint targets entirely in that case — numerically
        # identical to the old ×0 path, but avoids ~1.23 s/iter of wasted work.
        warmup_phase = getattr(self, 'kpt_warmup_phase', 5000)
        progressive_kpt_weight = ProgressiveWeight(
            initial_weight=0.1,
            final_weight=1.0,
            total_steps=getattr(self, 'kpt_warmup_steps', 5000)
        )
        if self.step < warmup_phase:
            current_kpt_weight = 0.0
        else:
            current_kpt_weight = progressive_kpt_weight.get_weight(self.step - warmup_phase)

        skip_kpt = (current_kpt_weight == 0.0)

        # Tensor accumulators (so all-skip batches still have live grad)
        total_loss_desc = torch.zeros((), device=device)
        total_loss_focal = torch.zeros((), device=device)
        total_loss_rep = torch.zeros((), device=device)
        total_loss_recall = torch.zeros((), device=device)
        total_loss_peak = torch.zeros((), device=device)
        total_loss_repulsion = torch.zeros((), device=device)
        valid_samples = 0

        # Store for visualization (first sample only)
        first_corr0, first_corr1, first_y_target = None, None, None

        # Process each sample in batch
        for b in range(batch_size):
            # ========== AUG-AWARE WARP ==========
            R_b, t_b = self.get_transformation_matrix(
                batch_position_current[b], batch_position_other[b],
                aug_current[b] if aug_current is not None else None,
                aug_other[b] if aug_other is not None else None,
            )
            corr0_geo, corr1_geo = gen_corr(self.dim_grid, R_b, t_b)

            # Gate on geometric correspondences (not desc validity) so blank frames
            # still reach focal loss
            if len(corr0_geo) < 10:
                continue

            valid_samples += 1

            # Store first sample for visualization
            if b == 0:
                first_corr0, first_corr1 = corr0_geo, corr1_geo

            # ========== VALIDITY MASKING for descriptor path ==========
            feature_norm0 = desc0[b].norm(dim=0, keepdim=True)  # [1, H, W]
            feature_norm1 = desc1[b].norm(dim=0, keepdim=True)

            valid_mask0_flat = (feature_norm0 > validity_threshold).view(-1)
            valid_mask1_flat = (feature_norm1 > validity_threshold).view(-1)

            corr0_geo_dev = corr0_geo.to(device).long()
            corr1_geo_dev = corr1_geo.to(device).long()

            corr_valid_mask = valid_mask0_flat[corr0_geo_dev] & valid_mask1_flat[corr1_geo_dev]
            corr0_valid = corr0_geo_dev[corr_valid_mask]
            corr1_valid = corr1_geo_dev[corr_valid_mask]

            # ========== DESCRIPTOR LOSS (InfoNCE) — always runs ==========
            D = desc0.shape[1]
            desc0_flat = F.normalize(desc0[b].view(D, -1).T, dim=1)  # [HW, D]
            desc1_flat = F.normalize(desc1[b].view(D, -1).T, dim=1)

            if len(corr0_valid) >= 2:
                query = desc0_flat[corr0_valid]
                positive_key = desc1_flat[corr1_valid]
                num_corr = len(corr0_valid)
                if info_nce_negative_mode == 'unpaired':
                    all_indices = torch.arange(self.dim_grid ** 2, device=device)
                    negative_pool = all_indices[~torch.isin(all_indices, corr1_valid)]
                    num_negatives = min(info_nce_num_negatives, len(negative_pool))
                    neg_indices = negative_pool[torch.randperm(len(negative_pool), device=device)[:num_negatives]]
                    negative_keys = desc1_flat[neg_indices]
                elif info_nce_negative_mode == 'paired':
                    all_indices = torch.arange(self.dim_grid ** 2, device=device)
                    num_negatives = min(info_nce_num_negatives, self.dim_grid ** 2 - 1)
                    negative_keys = torch.zeros(num_corr, num_negatives, D, device=device)
                    for i, pos_idx in enumerate(corr1_valid):
                        negative_pool = all_indices[all_indices != pos_idx]
                        neg_indices = negative_pool[torch.randperm(len(negative_pool), device=device)[:num_negatives]]
                        negative_keys[i] = desc1_flat[neg_indices]
                elif info_nce_negative_mode == 'hard':
                    # Hard-negative mode: per-positive negatives from a window around the
                    # QUERY's own BEV position (same-position cell guaranteed) + random,
                    # excluding cells near the TRUE positive (false-negative avoidance).
                    negative_keys, query, positive_key, _ = self._hard_negative_keys(
                        corr0_valid, corr1_valid, desc1_flat, query, positive_key
                    )
                    num_negatives = self.info_nce_hard_num
                else:
                    raise ValueError(f"Unknown negative_mode: {info_nce_negative_mode}")

                nce_mode = 'paired' if info_nce_negative_mode == 'hard' else info_nce_negative_mode
                cache_key = f'_info_nce_loss_{nce_mode}'
                if not hasattr(self, cache_key):
                    setattr(self, cache_key, InfoNCE(
                        temperature=info_nce_temperature,
                        reduction='mean',
                        negative_mode=nce_mode,
                    ))
                loss_desc = getattr(self, cache_key)(query, positive_key, negative_keys)
            else:
                # Keep the descriptor head in the autograd graph (zero-valued) so
                # DDP never sees its params as unused -> lets us run with
                # find_unused_parameters=False (the find-unused graph traversal is a
                # major per-iter cost on this model's large dynamic loss graph).
                loss_desc = (desc0[b].sum() + desc1[b].sum()) * 0.0

            total_loss_desc += loss_desc

            # ========== KEYPOINT TARGETS + 5 LOSSES — skipped during warmup ==========
            # When current_kpt_weight==0 the 5 terms are multiplied by 0 post-loop
            # anyway, so we skip build_keypoint_targets (the dominant cost) and add
            # a grad-guard zero to each accumulator so the keypoint-head params remain
            # in the autograd graph (identical behavior: 0 loss, 0 grad on kpt losses).
            if skip_kpt:
                kpt_guard = (kpts0[b].sum() + kpts1[b].sum()) * 0.0
                total_loss_focal += kpt_guard
                total_loss_rep += kpt_guard
                total_loss_recall += kpt_guard
                total_loss_peak += kpt_guard
                total_loss_repulsion += kpt_guard
                continue

            # ========== GEOMETRY TARGETS (windowed, no dense OOM matmul) ==========
            d0n = F.normalize(desc0[b], dim=0)  # [D, H, W] L2-normalized
            d1n = F.normalize(desc1[b], dim=0)
            vmask = self.get_kpt_valid_mask(self.dim_grid, kpts0.device)
            valid_flat = vmask.view(-1) if vmask is not None else None
            y0, y1 = build_keypoint_targets(
                d0n, d1n, R_b, t_b,
                dim=self.dim_grid, k=5, rho=0.0, k_target=64, valid_flat=valid_flat)

            if b == 0:
                first_y_target = y0[corr0_geo_dev] if corr0_geo_dev.numel() > 0 else None

            # ========== FOCAL LOSS on keypoint logits ==========
            loss_focal = self.kpt_loss_fn(kpts0[b].view(-1), y0) + \
                         self.kpt_loss_fn(kpts1[b].view(-1), y1)

            # ========== REPEATABILITY (window cosine between frames) ==========
            # ERODED mask: rep gathers k*k logit windows, so a valid cell within
            # k//2 of a masked region would pull masked (~-inf) logits into the
            # cosine; keep only cells whose full window is unmasked.
            erode_mask = self.get_kpt_valid_mask_eroded(self.dim_grid, kpts0.device)
            erode_flat = erode_mask.view(-1) if erode_mask is not None else None
            corr0_rep, corr1_rep = corr0_geo_dev, corr1_geo_dev
            if erode_flat is not None and corr0_geo_dev.numel() > 0:
                rep_keep = erode_flat[corr0_geo_dev] & erode_flat[corr1_geo_dev]
                corr0_rep, corr1_rep = corr0_geo_dev[rep_keep], corr1_geo_dev[rep_keep]
            loss_rep = repeatability_cosine_loss(
                kpts0[b, 0], kpts1[b, 0], corr0_rep, corr1_rep,
                self.dim_grid, k=5)

            # ========== RECALL HINGE (encourage targets to fire) ==========
            loss_recall = recall_hinge_loss(kpts0[b].view(-1), y0, n=64) + \
                          recall_hinge_loss(kpts1[b].view(-1), y1, n=64)

            # ========== PEAKINESS CE (center must be argmax in window) ==========
            loss_peak = peakiness_ce_loss(kpts0[b, 0], y0, self.dim_grid, k=5) + \
                        peakiness_ce_loss(kpts1[b, 0], y1, self.dim_grid, k=5)

            # ========== REPULSION (logit-space) ==========
            # Peak selection (sigmoid->NMS->topk) is non-differentiable; pick coords
            # under no_grad to avoid building an autograd graph for it. The repulsion
            # loss itself still backprops through the logits gathered at those coords.
            with torch.no_grad():
                coords0, _ = extract_keypoints(kpts0[b, 0], tau=0.0, cap=80, k=5)
                coords1, _ = extract_keypoints(kpts1[b, 0], tau=0.0, cap=80, k=5)
            loss_repulsion = repulsion_loss_logits(kpts0[b, 0], coords0) + \
                             repulsion_loss_logits(kpts1[b, 0], coords1)

            total_loss_focal += loss_focal
            total_loss_rep += loss_rep
            total_loss_recall += loss_recall
            total_loss_peak += loss_peak
            total_loss_repulsion += loss_repulsion

        # All-skip guard: if no frame passed the geometric gate, return zeros
        # that are still connected to the graph so .backward() does not raise.
        if valid_samples == 0:
            zero = (kpts0.sum() + kpts1.sum() + desc0.sum() + desc1.sum()) * 0.0
            loss_dict = {
                "descriptor_loss": zero,
                "keypoints_focal_loss": zero,
                "keypoints_rep_loss": zero,
                "keypoints_recall_loss": zero,
                "keypoints_peak_loss": zero,
                "keypoints_repulsion_loss": zero,
            }
            return loss_dict, first_corr0, first_corr1, first_y_target

        # ========== AVERAGE + WEIGHT ==========
        n = max(valid_samples, 1)
        total_loss_desc = (total_loss_desc / n) * self.l_desc
        total_loss_focal = (total_loss_focal / n) * 1.0 * current_kpt_weight
        total_loss_rep = (total_loss_rep / n) * 1.0 * current_kpt_weight
        total_loss_recall = (total_loss_recall / n) * 1.0 * current_kpt_weight
        total_loss_peak = (total_loss_peak / n) * 1.0 * current_kpt_weight
        total_loss_repulsion = (total_loss_repulsion / n) * 0.3 * current_kpt_weight

        loss_dict = {
            "descriptor_loss": total_loss_desc,
            "keypoints_focal_loss": total_loss_focal,
            "keypoints_rep_loss": total_loss_rep,
            "keypoints_recall_loss": total_loss_recall,
            "keypoints_peak_loss": total_loss_peak,
            "keypoints_repulsion_loss": total_loss_repulsion,
        }

        return loss_dict, first_corr0, first_corr1, first_y_target

    def _gather_binary_corr(self, desc_cur, desc_oth, b,
                            gt_xy_cur, gt_id_cur, gt_xy_oth, gt_id_oth, device):
        """Gather binary-descriptor logits at matched-track cells for one batch elem.

        Intersects the current/other frame gt_track_id sets; for each shared track,
        reads the keypoint's BEV cell (xy_to_cells) in both frames and gathers the
        descriptor logits there.

        Args:
            desc_cur, desc_oth: [D, H, W] raw descriptor logits for this batch elem.
            b: batch index into the per-sample gt lists.
            gt_xy_*, gt_id_*: lists (len B) of [N,2] keypoint xy (BEV metres) and
                [N] track ids; entries may be None / empty.

        Returns:
            (z_a, z_b): [K, D] logit tensors in the same matched-track order, or
            (None, None) if there is no shared track for this element.
        """
        def _get(lst):
            if lst is None or b >= len(lst):
                return None
            return lst[b]

        xy_c, id_c = _get(gt_xy_cur), _get(gt_id_cur)
        xy_o, id_o = _get(gt_xy_oth), _get(gt_id_oth)
        if any(v is None for v in (xy_c, id_c, xy_o, id_o)):
            return None, None

        def _to_long(t):
            if not torch.is_tensor(t):
                t = torch.as_tensor(np.asarray(t))
            return t.to(device=device).long().view(-1)

        def _to_xy(t):
            if not torch.is_tensor(t):
                t = torch.as_tensor(np.asarray(t), dtype=torch.float32)
            return t.to(device=device, dtype=torch.float32).view(-1, 2)

        id_c = _to_long(id_c)
        id_o = _to_long(id_o)
        if id_c.numel() == 0 or id_o.numel() == 0:
            return None, None
        xy_c = _to_xy(xy_c)
        xy_o = _to_xy(xy_o)

        # Shared track ids (deterministic order from the current frame).
        in_other = torch.isin(id_c, id_o)
        shared = id_c[in_other]
        # de-dup while preserving order (a track appears once per frame in practice)
        seen = set()
        order = []
        for tid in shared.tolist():
            if tid not in seen:
                seen.add(tid)
                order.append(tid)
        if len(order) == 0:
            return None, None

        rows_c, cols_c, rows_o, cols_o = [], [], [], []
        cells_c = xy_to_cells(xy_c, self.dim_grid, self.bev_resolution)
        cells_o = xy_to_cells(xy_o, self.dim_grid, self.bev_resolution)
        for tid in order:
            ic = (id_c == tid).nonzero(as_tuple=True)[0][0]
            io = (id_o == tid).nonzero(as_tuple=True)[0][0]
            rows_c.append(cells_c[ic, 0]); cols_c.append(cells_c[ic, 1])
            rows_o.append(cells_o[io, 0]); cols_o.append(cells_o[io, 1])

        rc = torch.stack(rows_c); cc = torch.stack(cols_c)
        ro = torch.stack(rows_o); co = torch.stack(cols_o)
        z_a = desc_cur[:, rc, cc].T  # [K, D]
        z_b = desc_oth[:, ro, co].T  # [K, D]
        return z_a, z_b

    @staticmethod
    def _hard_negative_indices(corr0_valid, corr1_valid, dim, window, K):
        """Compute hard-negative INDEX tensor [n, K] (long) without any descriptor lookup.

        Per positive i, draws K negatives:
          - K//2 from a Chebyshev window of radius `window` around the QUERY position
            (corr0_valid[i]); offset[0] is forced to (0,0) so the exact same-position cell
            is always included.
          - K - K//2 uniformly random from the full grid.
        Any negative within Chebyshev radius 2 of the TRUE positive (corr1_valid[i]) is
        replaced by an independent random draw (false-negative exclusion).

        Args:
            corr0_valid: [n] LongTensor of flat QUERY indices (row*dim+col) on the device.
            corr1_valid: [n] LongTensor of flat TRUE-POSITIVE indices on the same device.
            dim: int, BEV grid side length (grid is dim×dim).
            window: int, Chebyshev half-radius for the window draw.
            K: int, total negatives per positive.

        Returns:
            neg: [n, K] LongTensor of flat negative indices (values in [0, dim*dim)).
        """
        device = corr0_valid.device
        n = corr0_valid.shape[0]
        Nwin = K // 2
        Nrand = K - Nwin

        # Decode query positions
        r0 = corr0_valid // dim   # [n]
        c0 = corr0_valid % dim    # [n]
        # Decode true-positive positions (for false-negative exclusion)
        r1 = corr1_valid // dim   # [n]
        c1 = corr1_valid % dim    # [n]

        # Window offsets: [n, Nwin, 2] random in [-window, window]
        off = torch.randint(-window, window + 1, (n, Nwin, 2), device=device)
        # Force offset index 0 to (0,0) -> SAME-POSITION cell is always a negative candidate
        off[:, 0, :] = 0

        wr = (r0[:, None] + off[:, :, 0]).clamp(0, dim - 1)  # [n, Nwin]
        wc = (c0[:, None] + off[:, :, 1]).clamp(0, dim - 1)  # [n, Nwin]
        win = wr * dim + wc                                    # [n, Nwin] flat indices

        # Random negatives from the full grid
        rnd = torch.randint(0, dim * dim, (n, Nrand), device=device)  # [n, Nrand]

        neg = torch.cat([win, rnd], dim=1)  # [n, K]

        # False-negative exclusion: any cell within Chebyshev radius 2 of the TRUE positive
        # -> replace with an independent random draw.
        nr = neg // dim   # [n, K]
        nc = neg % dim    # [n, K]
        bad = ((nr - r1[:, None]).abs() <= 2) & ((nc - c1[:, None]).abs() <= 2)
        replacement = torch.randint(0, dim * dim, neg.shape, device=device)
        neg = torch.where(bad, replacement, neg)

        return neg  # [n, K]

    def _hard_negative_keys(self, corr0_valid, corr1_valid, desc1_flat, query, positive_key):
        """Build hard negative descriptor keys and (optionally) subsample query/positive.

        Subsamples correspondences to at most 512 before computing hard negatives so the
        [n, K, D] negative_keys tensor stays tractable (O(n*K*D) memory).

        Args:
            corr0_valid: [N] LongTensor of flat QUERY BEV indices.
            corr1_valid: [N] LongTensor of flat TRUE-POSITIVE BEV indices.
            desc1_flat: [dim*dim, D] L2-normalised descriptor field of frame 1.
            query: [N, D] query descriptors (already gathered from desc0_flat).
            positive_key: [N, D] positive descriptors (already gathered from desc1_flat).

        Returns:
            negative_keys: [n, K, D] where n = min(N, 512)
            query: [n, D] (subsampled if N > 512, else the original tensor)
            positive_key: [n, D] (subsampled if N > 512, else the original tensor)
            sub: [n] LongTensor of kept indices, or None if no subsampling.
        """
        device = corr0_valid.device
        N = corr0_valid.shape[0]
        Nmax = 512

        if N > Nmax:
            sub = torch.randperm(N, device=device)[:Nmax]
            c0 = corr0_valid[sub]
            c1 = corr1_valid[sub]
            query = query[sub]
            positive_key = positive_key[sub]
        else:
            sub = None
            c0 = corr0_valid
            c1 = corr1_valid

        K = self.info_nce_hard_num
        neg = self._hard_negative_indices(c0, c1, self.dim_grid, self.info_nce_hard_window, K)
        negative_keys = desc1_flat[neg]  # [n, K, D]

        return negative_keys, query, positive_key, sub

    def _detect_keypoint_flat(self, heatmap):
        """Top-K detected-keypoint flat indices (row*dim+col) from a heatmap [1,H,W] or [H,W].
        Detached — detection is non-differentiable; the descriptor loss backprops only into the
        descriptor field gathered at these cells (SuperPoint-style sparse training). Matches the
        inference detector: plateau_safe_nms + interior valid mask + top-K."""
        from .keypoint_ops import plateau_safe_nms, keypoint_valid_mask
        h = heatmap[0] if heatmap.dim() == 3 else heatmap
        dim = h.shape[-1]
        with torch.no_grad():
            prob = torch.sigmoid(h.float())
            vmask = keypoint_valid_mask(
                dim, border=getattr(self, 'kpt_mask_border', 8),
                center_halfwidth=getattr(self, 'kpt_mask_center_halfwidth', 4)).to(h.device)
            nms = plateau_safe_nms(prob, k=5) & vmask
            ys, xs = torch.nonzero(nms, as_tuple=True)
            if ys.numel() == 0:
                return torch.zeros(0, dtype=torch.long, device=h.device)
            k = min(int(getattr(self, 'descriptor_topk', 256)), ys.numel())
            top = torch.topk(prob[ys, xs], k=k).indices
            return (ys[top] * dim + xs[top]).long()

    def cross_modal_consistency_loss(self, desc_fused, desc_cam, desc_lidar, heatmap_fused):
        """Cross-modal InfoNCE at detected keypoints. desc_* are [1,D,H,W] descriptor fields from the
        SAME frame extracted fused / camera-only / lidar-only. At each detected keypoint, that point's
        descriptor in one modality is the POSITIVE for the same point in another modality, all other
        keypoints are NEGATIVES. Pulls same-point descriptors together across modalities (so C2L can
        match) while staying discriminative (so it can't collapse). Returns a scalar loss."""
        cells = self._detect_keypoint_flat(heatmap_fused)          # [K] flat row*dim+col, detached
        if cells.numel() < 4:
            return desc_fused.sum() * 0.0
        Dd = desc_fused.shape[1]
        temp = getattr(self, 'cross_modal_temperature', 0.15)

        def _g(d):
            return F.normalize(d.reshape(1, Dd, -1)[0, :, cells].T.float(), dim=1)  # [K, D]
        A, B, C = _g(desc_fused), _g(desc_cam), _g(desc_lidar)
        labels = torch.arange(A.shape[0], device=A.device)

        def _nce(X, Y):
            logits = (X @ Y.T) / temp
            return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return (_nce(A, B) + _nce(A, C) + _nce(B, C)) / 3.0

    def compute_losses_infoNCE(self, kpts0, kpts1, desc0, desc1, R, t,
                               info_nce_temperature=0.15,
                               info_nce_num_negatives=1024,
                               info_nce_negative_mode='unpaired',
                               validity_threshold=0.1,
                               use_repulsion=True,
                               repulsion_distance=2.0,
                               batch_position_current=None,
                               batch_position_other=None,
                               aug_current=None,
                               aug_other=None,
                               featA=None, featB=None,
                               gt_keypoint_xy_current=None,
                               gt_track_id_current=None,
                               gt_keypoint_xy_other=None,
                               gt_track_id_other=None,
                               density_masks_current=None,
                               density_masks_other=None):
        """
        Batch-compatible loss: aug-aware geometry targets + InfoNCE + new keypoint terms.
        Vectorized: per-b expensive work (build_keypoint_targets, 5 kpt losses) is
        restructured out of the main loop; only the RNG-sensitive randperm calls stay
        per-b to preserve bit-identical results.

        Args:
            kpts0, kpts1: [B, 1, H, W] keypoint logits
            desc0, desc1: [B, D, H, W] raw descriptors
            R, t: ignored (kept for API compat); warp is derived per-frame from poses+aug
            batch_position_current/other: list of B ego2global [4,4] matrices
            aug_current/other: list of B lidar_aug_matrix [4,4] (or None -> identity)
        """
        # Let the head-level attr override the function-arg default; config drives behavior.
        info_nce_negative_mode = getattr(self, 'info_nce_negative_mode', info_nce_negative_mode)

        device = kpts0.device
        batch_size = kpts0.shape[0]
        D = desc0.shape[1]

        # ===== WARMUP WEIGHT =====
        warmup_phase = getattr(self, 'kpt_warmup_phase', 5000)
        progressive_kpt_weight = ProgressiveWeight(
            initial_weight=0.1,
            final_weight=1.0,
            total_steps=getattr(self, 'kpt_warmup_steps', 5000)
        )
        if self.step < warmup_phase:
            current_kpt_weight = 0.0
        else:
            current_kpt_weight = progressive_kpt_weight.get_weight(self.step - warmup_phase)
        skip_kpt = (current_kpt_weight == 0.0)

        # ===== PHASE 1: deterministic per-b preparation (no RNG) =====
        # Compute warps, gen_corr, gate check, validity masks, desc normalization.
        # Results collected into lists indexed over ALL b, with None for gated-out b's.
        prep = []  # per-b: None if gated, else dict of precomputed data
        first_corr0, first_corr1, first_y_target = None, None, None

        for b in range(batch_size):
            R_b, t_b = self.get_transformation_matrix(
                batch_position_current[b], batch_position_other[b],
                aug_current[b] if aug_current is not None else None,
                aug_other[b] if aug_other is not None else None,
            )
            corr0_geo, corr1_geo = gen_corr(self.dim_grid, R_b, t_b)

            if len(corr0_geo) < 10:
                prep.append(None)
                continue

            # Validity masking
            feature_norm0 = desc0[b].norm(dim=0, keepdim=True)  # [1, H, W]
            feature_norm1 = desc1[b].norm(dim=0, keepdim=True)
            valid_mask0_flat = (feature_norm0 > validity_threshold).view(-1)
            valid_mask1_flat = (feature_norm1 > validity_threshold).view(-1)
            corr0_geo_dev = corr0_geo.to(device).long()
            corr1_geo_dev = corr1_geo.to(device).long()
            corr_valid_mask = valid_mask0_flat[corr0_geo_dev] & valid_mask1_flat[corr1_geo_dev]
            corr0_valid = corr0_geo_dev[corr_valid_mask]
            corr1_valid = corr1_geo_dev[corr_valid_mask]

            # Normalize descriptors for InfoNCE
            desc0_flat = F.normalize(desc0[b].view(D, -1).T, dim=1)  # [HW, D]
            desc1_flat = F.normalize(desc1[b].view(D, -1).T, dim=1)

            prep.append({
                'b': b,
                'R_b': R_b, 't_b': t_b,
                'corr0_geo': corr0_geo,
                'corr1_geo': corr1_geo,
                'corr0_geo_dev': corr0_geo_dev,
                'corr1_geo_dev': corr1_geo_dev,
                'corr0_valid': corr0_valid,
                'corr1_valid': corr1_valid,
                'desc0_flat': desc0_flat,
                'desc1_flat': desc1_flat,
            })

            if b == 0:
                first_corr0, first_corr1 = corr0_geo, corr1_geo

        valid_preps = [p for p in prep if p is not None]
        valid_samples = len(valid_preps)

        # ===== ALL-SKIP GUARD =====
        if valid_samples == 0:
            zero = (kpts0.sum() + kpts1.sum() + desc0.sum() + desc1.sum()) * 0.0
            # When superpoint_detector=True, kpts0 was produced by superpoint_head
            # (via forward()), so kpts0.sum()*0.0 already anchors its params. Add an
            # extra explicit anchor via featA to be safe for DDP find_unused_parameters=False.
            if getattr(self, 'superpoint_detector', False) and featA is not None:
                zero = zero + self.superpoint_head(featA).sum() * 0.0
            if self.supervised:
                # Supervised mode emits a single detection-loss key. Anchor every
                # param (score + descriptor heads) at zero so DDP never marks them
                # unused.
                loss_dict = {
                    "descriptor_loss": zero,
                    "keypoints_supervised_loss": zero,
                }
                if self.repeatability_weight > 0.0:
                    loss_dict["keypoints_repeatability_loss"] = zero
            else:
                loss_dict = {
                    "descriptor_loss": zero,
                    "keypoints_focal_loss": zero,
                    "keypoints_rep_loss": zero,
                    "keypoints_recall_loss": zero,
                    "keypoints_peak_loss": zero,
                    "keypoints_repulsion_loss": zero,
                    "keypoints_covariance_loss": zero,
                }
                if self.density_mask_train:
                    loss_dict["keypoints_density_neg_loss"] = zero
            return loss_dict, first_corr0, first_corr1, first_y_target

        # ===== PHASE 2: InfoNCE — mini per-valid-b loop (preserves RNG stream) =====
        total_loss_desc = torch.zeros((), device=device)
        total_loss_focal = torch.zeros((), device=device)
        total_loss_rep = torch.zeros((), device=device)
        total_loss_recall = torch.zeros((), device=device)
        total_loss_peak = torch.zeros((), device=device)
        total_loss_repulsion = torch.zeros((), device=device)
        total_loss_cov = torch.zeros((), device=device)
        total_loss_supervised = torch.zeros((), device=device)
        total_loss_repeatability = torch.zeros((), device=device)
        total_loss_density_neg = torch.zeros((), device=device)

        binary_desc = getattr(self, 'binary_descriptor', False)

        for p in valid_preps:
            b = p['b']
            corr0_valid = p['corr0_valid']
            corr1_valid = p['corr1_valid']
            desc0_flat = p['desc0_flat']
            desc1_flat = p['desc1_flat']

            # SPARSE keypoint descriptor (float/InfoNCE path): restrict the descriptor anchors to
            # DETECTED keypoints (+ optional dense blend). These local corr0/corr1_valid are used
            # ONLY by the InfoNCE descriptor block below; the binary path re-reads p['corr0_valid']
            # fresh and gates itself, so this does not affect it or the detection targets.
            if getattr(self, 'sparse_keypoint_descriptor', False) and not binary_desc:
                kpt0_flat = self._detect_keypoint_flat(kpts0[b])
                keep = torch.isin(corr0_valid, kpt0_flat)
                blend = float(getattr(self, 'sparse_dense_blend', 0.0))
                if blend > 0.0:
                    n_add = int(blend * int(keep.sum()))
                    non_kpt = torch.nonzero(~keep, as_tuple=False).squeeze(1)
                    if n_add > 0 and non_kpt.numel() > 0:
                        sel = non_kpt[torch.randperm(non_kpt.numel(), device=keep.device)[:n_add]]
                        keep[sel] = True
                corr0_valid = corr0_valid[keep]
                corr1_valid = corr1_valid[keep]

            # BINARY DESCRIPTOR: the InfoNCE descriptor loss is replaced by a metric
            # loss over the teacher's persistence correspondences (computed below in a
            # dedicated block). Skip the (expensive, unused) InfoNCE work here; the
            # kpt-warmup guard still runs so the score head stays anchored.
            if binary_desc:
                if skip_kpt:
                    kpt_guard = (kpts0[b].sum() + kpts1[b].sum()) * 0.0
                    if self.supervised:
                        total_loss_supervised += kpt_guard
                    else:
                        total_loss_focal += kpt_guard
                        total_loss_rep += kpt_guard
                        total_loss_recall += kpt_guard
                        total_loss_peak += kpt_guard
                        total_loss_repulsion += kpt_guard
                        total_loss_cov += kpt_guard
                continue

            if len(corr0_valid) >= 2:
                query = desc0_flat[corr0_valid]
                positive_key = desc1_flat[corr1_valid]
                num_corr = len(corr0_valid)
                if info_nce_negative_mode == 'unpaired':
                    all_indices = torch.arange(self.dim_grid ** 2, device=device)
                    negative_pool = all_indices[~torch.isin(all_indices, corr1_valid)]
                    num_negatives = min(info_nce_num_negatives, len(negative_pool))
                    neg_indices = negative_pool[torch.randperm(len(negative_pool), device=device)[:num_negatives]]
                    negative_keys = desc1_flat[neg_indices]
                elif info_nce_negative_mode == 'paired':
                    all_indices = torch.arange(self.dim_grid ** 2, device=device)
                    num_negatives = min(info_nce_num_negatives, self.dim_grid ** 2 - 1)
                    negative_keys = torch.zeros(num_corr, num_negatives, D, device=device)
                    for i, pos_idx in enumerate(corr1_valid):
                        negative_pool = all_indices[all_indices != pos_idx]
                        neg_indices = negative_pool[torch.randperm(len(negative_pool), device=device)[:num_negatives]]
                        negative_keys[i] = desc1_flat[neg_indices]
                elif info_nce_negative_mode == 'hard':
                    # Hard-negative mode: per-positive negatives from a window around the
                    # QUERY's own BEV position (same-position cell guaranteed) + random,
                    # excluding cells near the TRUE positive (false-negative avoidance).
                    # Returns [n, K, D] 'paired'-shaped negatives; subsamples to <=512.
                    negative_keys, query, positive_key, _ = self._hard_negative_keys(
                        corr0_valid, corr1_valid, desc1_flat, query, positive_key
                    )
                    num_negatives = self.info_nce_hard_num
                else:
                    raise ValueError(f"Unknown negative_mode: {info_nce_negative_mode}")

                # The InfoNCE library only understands 'unpaired' / 'paired'.
                # 'hard' produces [n, K, D] paired-shaped negatives, so pass 'paired'.
                nce_mode = 'paired' if info_nce_negative_mode == 'hard' else info_nce_negative_mode
                # Cache the InfoNCE loss object; key includes the effective nce_mode so a
                # mode switch (e.g. from 'unpaired' to 'hard') rebuilds with the right shape.
                cache_key = f'_info_nce_loss_{nce_mode}'
                if not hasattr(self, cache_key):
                    setattr(self, cache_key, InfoNCE(
                        temperature=info_nce_temperature,
                        reduction='mean',
                        negative_mode=nce_mode,
                    ))
                loss_desc = getattr(self, cache_key)(query, positive_key, negative_keys)
            else:
                # Keep the descriptor head in the autograd graph (zero-valued) so
                # DDP never sees its params as unused -> lets us run with
                # find_unused_parameters=False (the find-unused graph traversal is a
                # major per-iter cost on this model's large dynamic loss graph).
                loss_desc = (desc0[b].sum() + desc1[b].sum()) * 0.0

            total_loss_desc += loss_desc

            # Warmup guard for kpt terms
            if skip_kpt:
                kpt_guard = (kpts0[b].sum() + kpts1[b].sum()) * 0.0
                if self.supervised:
                    # Only the supervised detection key is emitted in supervised mode;
                    # anchor it (still touches the score head) during warmup.
                    total_loss_supervised += kpt_guard
                else:
                    total_loss_focal += kpt_guard
                    total_loss_rep += kpt_guard
                    total_loss_recall += kpt_guard
                    total_loss_peak += kpt_guard
                    total_loss_repulsion += kpt_guard
                    total_loss_cov += kpt_guard

        # ===== BINARY DESCRIPTOR metric loss (correspondence sampling) =====
        # Replaces the InfoNCE descriptor_loss.  Two paths:
        #   supervised=True : intersect gt_track_id sets across frames, gather
        #       descriptor logits at matched-track BEV cells (_gather_binary_corr).
        #   supervised=False (self-sup, config I): use the geometric pose
        #       correspondences (corr0/1_valid) that InfoNCE uses, gathering RAW
        #       tanh-domain logits — no F.normalize, as binary_descriptor_metric_loss
        #       applies tanh internally.
        # Either way, if no element yields >=1 pair, emit a grad-anchored zero so
        # DDP find_unused_parameters=False never marks the descriptor head unused.
        if binary_desc:
            matched_any = False
            if self.supervised:
                # Supervised path: teacher track-id correspondences.
                for p in valid_preps:
                    b = p['b']
                    z_a, z_b = self._gather_binary_corr(
                        desc0[b], desc1[b], b,
                        gt_keypoint_xy_current, gt_track_id_current,
                        gt_keypoint_xy_other, gt_track_id_other, device)
                    if z_a is None or z_a.shape[0] < 1:
                        continue
                    matched_any = True
                    K = z_a.shape[0]
                    match = torch.arange(K, device=device).unsqueeze(1).repeat(1, 2)
                    total_loss_desc = total_loss_desc + binary_descriptor_metric_loss(z_a, z_b, match)
            else:
                # Self-supervised path: geometric pose correspondences — real far-cell
                # hard negatives + confidence via binary_selfsup_descriptor_loss.
                # Mirrors the InfoNCE 'unpaired' negative pool: all cells NOT in
                # corr1_valid become candidate negatives; sample info_nce_num_negatives.
                D_bin = desc0.shape[1]
                for p in valid_preps:
                    b = p['b']
                    corr0_valid = p['corr0_valid']
                    corr1_valid = p['corr1_valid']
                    # SPARSE KEYPOINT DESCRIPTOR: keep only correspondences whose frame-0 anchor
                    # is a DETECTED keypoint, so the descriptor is trained where it is actually
                    # read at inference (positives stay the geometric-warp partners; local_hard_neg
                    # supplies the spatial confusers).
                    if getattr(self, 'sparse_keypoint_descriptor', False):
                        kpt0_flat = self._detect_keypoint_flat(kpts0[b])
                        keep = torch.isin(corr0_valid, kpt0_flat)
                        # v2 stabilizer: also keep blend x (#kpt anchors) random non-keypoint corrs
                        blend = float(getattr(self, 'sparse_dense_blend', 0.0))
                        if blend > 0.0:
                            n_add = int(blend * int(keep.sum()))
                            non_kpt = torch.nonzero(~keep, as_tuple=False).squeeze(1)
                            if n_add > 0 and non_kpt.numel() > 0:
                                sel = non_kpt[torch.randperm(non_kpt.numel(), device=keep.device)[:n_add]]
                                keep[sel] = True
                        corr0_valid = corr0_valid[keep]
                        corr1_valid = corr1_valid[keep]
                    if len(corr0_valid) < 2:
                        continue
                    desc1_flat_raw = desc1[b].view(D_bin, -1).T          # [HW, D] RAW logits (no normalize)
                    z_a = desc0[b].view(D_bin, -1).T[corr0_valid]        # [K, D] raw logits, frame-A corr cells
                    z_b_pos = desc1_flat_raw[corr1_valid]                 # [K, D] raw logits, frame-B matched cells
                    all_idx = torch.arange(self.dim_grid ** 2, device=device)
                    neg_pool = all_idx[~torch.isin(all_idx, corr1_valid)]
                    num_neg = min(getattr(self, 'info_nce_num_negatives', 256), len(neg_pool))
                    neg_idx = neg_pool[torch.randperm(len(neg_pool), device=device)[:num_neg]]
                    z_b_neg = desc1_flat_raw[neg_idx]                     # [M, D] far-cell negatives
                    # LOCAL/FLAT HARD NEGATIVES (flatfix #1): gather frame-B cells within
                    # local_neg_radius (Chebyshev) of the correspondences, minus the corr
                    # cells themselves, as extra (spatially-near, geometrically similar)
                    # negatives. Gated behind self.local_hard_neg (default OFF).
                    local_neg_cells = None
                    if getattr(self, 'local_hard_neg', False):
                        R = int(getattr(self, 'local_neg_radius', 10))
                        G = self.dim_grid
                        rows = torch.div(corr1_valid, G, rounding_mode='floor')  # [K]
                        cols = corr1_valid % G                                    # [K]
                        off = torch.arange(-R, R + 1, device=device)
                        dr, dc = torch.meshgrid(off, off, indexing='ij')
                        dr = dr.reshape(-1); dc = dc.reshape(-1)                  # [(2R+1)^2]
                        nr = rows.unsqueeze(1) + dr.unsqueeze(0)                  # [K, P]
                        nc = cols.unsqueeze(1) + dc.unsqueeze(0)
                        inb = (nr >= 0) & (nr < G) & (nc >= 0) & (nc < G)
                        local_flat = torch.unique((nr * G + nc)[inb])            # [L0]
                        # drop the true correspondences from the local negative pool
                        local_flat = local_flat[~torch.isin(local_flat, corr1_valid)]
                        cap = int(getattr(self, 'local_neg_max', 512))
                        if local_flat.numel() > cap:
                            sel = torch.randperm(local_flat.numel(), device=device)[:cap]
                            local_flat = local_flat[sel]
                        if local_flat.numel() > 0:
                            local_neg_cells = desc1_flat_raw[local_flat]         # [L, D]
                    total_loss_desc = total_loss_desc + binary_selfsup_descriptor_loss(
                        z_a, z_b_pos, z_b_neg, local_neg_cells=local_neg_cells)
                    matched_any = True
            if not matched_any:
                # Grad-anchored zero: touch the descriptor head params via the full field.
                total_loss_desc = total_loss_desc + (desc0.sum() + desc1.sum()) * 0.0

        # ===== PHASE 3 (SUPERVISED): teacher-heatmap focal loss OR SP cross-entropy =====
        # When self.supervised and NOT superpoint_detector: CenterNet penalty-reduced focal
        # loss against the teacher's landmark heatmap for the CURRENT frame (kpts0).
        # When self.supervised and superpoint_detector=True: per-region cross-entropy loss
        # (SuperPoint-style softmax+dustbin) against the teacher landmark targets.
        # The self-supervised geometric/descriptor-correlation detection terms below are
        # skipped; the descriptor InfoNCE path (Phase 2) is kept unchanged.
        sp_det = getattr(self, 'superpoint_detector', False)
        if self.supervised and not skip_kpt:
            if sp_det:
                # Recompute the CURRENT-frame SP logits for the loss (fresh forward through
                # superpoint_head so the loss grad flows through the head, not through kpts0
                # which was already mask-processed in forward()). This mirrors how the dense
                # path re-derives heatmap-level supervision.
                sp_in = featA
                if self.training and getattr(self, 'detach_score_head', False):
                    sp_in = sp_in.detach()
                sp_logits0 = self.superpoint_head(sp_in)  # [B, C, g, g]
            for p in valid_preps:
                b = p['b']
                # gt for this sample: BEV metres (x,y, lidar frame), [N,2] np/tensor or
                # empty/None.
                gt_xy = None
                if gt_keypoint_xy_current is not None and b < len(gt_keypoint_xy_current):
                    gt_xy = gt_keypoint_xy_current[b]
                if gt_xy is None or (hasattr(gt_xy, '__len__') and len(gt_xy) == 0):
                    cells = torch.zeros((0, 2), dtype=torch.long, device=device)
                else:
                    if not torch.is_tensor(gt_xy):
                        gt_xy = torch.as_tensor(np.asarray(gt_xy), dtype=torch.float32)
                    gt_xy = gt_xy.to(device=device, dtype=torch.float32)
                    cells = xy_to_cells(gt_xy, self.dim_grid, self.bev_resolution)

                if sp_det:
                    # SuperPoint cross-entropy: build per-region class labels and compute CE.
                    tgt = build_superpoint_targets(cells, self.dim_grid, self.sp_cell).to(device)
                    total_loss_supervised += superpoint_detection_loss(
                        sp_logits0[b], tgt, dustbin_weight=self.sp_dustbin_weight)
                else:
                    heatmap = render_gaussian_heatmap(cells, self.dim_grid, sigma=self.supervised_sigma)
                    # supervised_focal_loss expects {0,1} positives; the rendered map peaks at
                    # exactly 1.0 at gt cells, with penalty-reduced negatives elsewhere.
                    total_loss_supervised += supervised_focal_loss(kpts0[b, 0], heatmap)

                # CROSS-FRAME DETECTION REPEATABILITY: gated by repeatability_weight > 0.
                # Uses the REAL pose correspondences (corr0_valid/corr1_valid, already used
                # for InfoNCE) to measure score-map window cosine agreement across frames.
                # This enforces warp-equivariance on the detection head: a point that scores
                # high on the current frame should also score high at its matched cell on the
                # other frame -> directly optimises the repeatability metric.
                corr0_valid = p['corr0_valid']
                corr1_valid = p['corr1_valid']
                if self.repeatability_weight > 0.0 and corr0_valid.numel() > 0:
                    total_loss_repeatability = total_loss_repeatability + repeatability_cosine_loss(
                        kpts0[b, 0], kpts1[b, 0], corr0_valid, corr1_valid,
                        self.dim_grid, self.repeatability_k)


        # ===== PHASE 3: keypoint targets + 5 losses (per-valid-b mini loop) =====
        if not self.supervised and not skip_kpt:
            for p in valid_preps:
                b = p['b']
                R_b, t_b = p['R_b'], p['t_b']
                corr0_geo_dev = p['corr0_geo_dev']
                corr1_geo_dev = p['corr1_geo_dev']

                # NEGATIVE-SUPERVISION DENSITY PENALTY: mean sigmoid score on empty
                # (LiDAR-unsupported) cells. Uses UNMASKED kpts0/kpts1 so autograd flows
                # back to the detector head. In the SELF-SUPERVISED branch (this run's path);
                # gated by density_mask_train (via density_masks_current being set).
                if density_masks_current is not None and b < len(density_masks_current):
                    empty0 = ~density_masks_current[b]
                    empty1 = ~density_masks_other[b]
                    if empty0.any():
                        total_loss_density_neg = total_loss_density_neg + torch.sigmoid(kpts0[b, 0])[empty0].mean()
                    if empty1.any():
                        total_loss_density_neg = total_loss_density_neg + torch.sigmoid(kpts1[b, 0])[empty1].mean()

                vmask = self.get_kpt_valid_mask(self.dim_grid, kpts0.device)
                valid_flat = vmask.view(-1) if vmask is not None else None
                if self.use_geometric_targets:
                    # REPEATABILITY targets: mutual-local-max-under-warp on the score maps.
                    # No descriptor consulted -> no saliency leak. Detached (targets carry
                    # no grad); the focal/recall/peak losses pull the logits onto them.
                    # DENSITY-TARGET RESTRICTION: AND per-sample density masks into vmask so
                    # that only LiDAR-supported cells can become positive geometric targets.
                    # This prevents reinforcing any residual grid-firing in empty space while
                    # still letting the focal loss on the UNMASKED map suppress it.
                    vmask_b = vmask
                    if density_masks_current is not None and b < len(density_masks_current):
                        dm = density_masks_current[b] & density_masks_other[b]
                        vmask_b = (vmask & dm) if vmask is not None else dm
                    with torch.no_grad():
                        s0 = torch.sigmoid(kpts0[b, 0]).detach()
                        s1 = torch.sigmoid(kpts1[b, 0]).detach()
                        y0, y1 = build_geometric_targets_pair(
                            s0, s1, R_b, t_b, dim=self.dim_grid, k=5, valid_mask=vmask_b,
                            min_score=self.geometric_min_score,
                            max_targets=(self.geometric_max_targets or None),
                            desc_a=desc0[b], desc_b=desc1[b],
                            reliability_weight=self.reliability_weight)
                else:
                    d0n = F.normalize(desc0[b], dim=0)  # [D, H, W] L2-normalized
                    d1n = F.normalize(desc1[b], dim=0)
                    y0, y1 = build_keypoint_targets(
                        d0n, d1n, R_b, t_b,
                        dim=self.dim_grid, k=5, rho=0.0, k_target=64, valid_flat=valid_flat)

                if b == 0:
                    first_y_target = y0[corr0_geo_dev] if corr0_geo_dev.numel() > 0 else None

                loss_focal = self.kpt_loss_fn(kpts0[b].view(-1), y0) + \
                             self.kpt_loss_fn(kpts1[b].view(-1), y1)

                # ERODED mask (see loop ref): rep gathers k*k logit windows, so
                # keep only corr cells whose full window is unmasked.
                erode_mask = self.get_kpt_valid_mask_eroded(self.dim_grid, kpts0.device)
                erode_flat = erode_mask.view(-1) if erode_mask is not None else None
                corr0_rep, corr1_rep = corr0_geo_dev, corr1_geo_dev
                if erode_flat is not None and corr0_geo_dev.numel() > 0:
                    rep_keep = erode_flat[corr0_geo_dev] & erode_flat[corr1_geo_dev]
                    corr0_rep, corr1_rep = corr0_geo_dev[rep_keep], corr1_geo_dev[rep_keep]
                loss_rep = repeatability_cosine_loss(
                    kpts0[b, 0], kpts1[b, 0], corr0_rep, corr1_rep,
                    self.dim_grid, k=5)

                loss_recall = recall_hinge_loss(kpts0[b].view(-1), y0, n=64) + \
                              recall_hinge_loss(kpts1[b].view(-1), y1, n=64)

                loss_peak = peakiness_ce_loss(kpts0[b, 0], y0, self.dim_grid, k=5) + \
                            peakiness_ce_loss(kpts1[b, 0], y1, self.dim_grid, k=5)

                with torch.no_grad():
                    coords0, _ = extract_keypoints(kpts0[b, 0], tau=0.0, cap=80, k=5)
                    coords1, _ = extract_keypoints(kpts1[b, 0], tau=0.0, cap=80, k=5)
                # SPARSER: wider repulsion radius (self.repulsion_distance) so keypoints
                # must be well-separated (the "too dense" fix).
                loss_repulsion = repulsion_loss_logits(kpts0[b, 0], coords0, self.repulsion_distance) + \
                                 repulsion_loss_logits(kpts1[b, 0], coords1, self.repulsion_distance)
                # total-activation sparsity: penalize broad firing -> fewer, peakier cells.
                loss_repulsion = loss_repulsion + self.l_sparsity * (
                    torch.sigmoid(kpts0[b, 0]).sum() + torch.sigmoid(kpts1[b, 0]).sum())

                # R2D2 SYNTHETIC-HOMOGRAPHY COVARIANCE: warp the BEV feature by a random
                # affine, re-run the score head, and require the score map to warp the same
                # way. Content-based detections survive the warp; fixed-frame/border/conv
                # artifacts do NOT -> directly penalizes the non-repeatable firing.
                # ISOLATED TO THE SCORE HEAD: both branches use DETACHED features, so the
                # covariance gradient flows ONLY through self.keypoint_head (the detector
                # conv) and never into the shared encoder. Earlier (encoder-coupled)
                # covariance corrupted the descriptor field -> match_precision crashed
                # 0.33->0.01. Detaching protects the descriptors; covariance shapes only
                # the detector's warp-equivariance.
                if self.covariance_weight > 0 and featA is not None:
                    Rs, ts = random_bev_warp(
                        max_rot_deg=self.covariance_max_rot_deg,
                        max_trans_cells=self.covariance_max_trans_cells,
                        device=device)
                    feat_d = featA[b].detach()                                  # [C,H,W] no encoder grad
                    vmask_w = self.get_kpt_valid_mask(self.dim_grid, device)
                    def _score(f):
                        s = self.keypoint_head(f[None])                         # score-head grad ONLY
                        if vmask_w is not None:
                            s = s + (~vmask_w).to(s.dtype) * self.kpt_mask_fill
                        return s
                    k_ref = _score(feat_d)                                      # reference score
                    k_w = _score(warp_feature_map(feat_d, Rs, ts, self.dim_grid))  # warped-input score
                    corr0_s, corr1_s = gen_corr(self.dim_grid, Rs, ts)
                    corr0_s = corr0_s.to(device).long(); corr1_s = corr1_s.to(device).long()
                    if vmask is not None:
                        vf = vmask.view(-1)
                        sk = vf[corr0_s] & vf[corr1_s]
                        corr0_s, corr1_s = corr0_s[sk], corr1_s[sk]
                    if corr0_s.numel() >= 2:
                        loss_cov = repeatability_cosine_loss(
                            k_ref[0, 0], k_w[0, 0], corr0_s, corr1_s, self.dim_grid, k=5)
                    else:
                        loss_cov = (k_w.sum()) * 0.0
                    total_loss_cov += loss_cov

                total_loss_focal += loss_focal
                total_loss_rep += loss_rep
                total_loss_recall += loss_recall
                total_loss_peak += loss_peak
                total_loss_repulsion += loss_repulsion

        # ===== AVERAGE + WEIGHT =====
        n = max(valid_samples, 1)
        # descriptor_loss_weight (Task 4, KITTI detector-only): single chokepoint for
        # the live float/self-sup descriptor path (this method, called via loss() ->
        # loss_new()). getattr fallback keeps bare-head test doubles (that bypass
        # __init__, e.g. test_loss_vectorized_equiv.py) working unchanged.
        _wdesc = getattr(self, 'descriptor_loss_weight', 1.0)
        total_loss_desc = (total_loss_desc / n) * self.l_desc * _wdesc
        if _wdesc == 0.0:
            total_loss_desc = torch.nan_to_num(total_loss_desc)

        if self.supervised:
            # Supervised detection loss replaces the self-supervised detection terms.
            # Warmup-weighted like the other kpt terms so resume/warmup behaves the same.
            total_loss_supervised = (total_loss_supervised / n) * 1.0 * current_kpt_weight
            loss_dict = {
                "descriptor_loss": total_loss_desc,
                "keypoints_supervised_loss": total_loss_supervised,
            }
            if self.repeatability_weight > 0.0:
                total_loss_repeatability = (total_loss_repeatability / n) * self.repeatability_weight
                loss_dict["keypoints_repeatability_loss"] = total_loss_repeatability
            return loss_dict, first_corr0, first_corr1, first_y_target

        total_loss_focal = (total_loss_focal / n) * 1.0 * current_kpt_weight
        total_loss_rep = (total_loss_rep / n) * 1.0 * current_kpt_weight
        total_loss_recall = (total_loss_recall / n) * 1.0 * current_kpt_weight
        total_loss_peak = (total_loss_peak / n) * 1.0 * current_kpt_weight
        # SPARSER: stronger repulsion weight (self.repulsion_weight, was hardcoded 0.3).
        total_loss_repulsion = (total_loss_repulsion / n) * self.repulsion_weight * current_kpt_weight
        total_loss_cov = (total_loss_cov / n) * self.covariance_weight * current_kpt_weight

        loss_dict = {
            "descriptor_loss": total_loss_desc,
            "keypoints_focal_loss": total_loss_focal,
            "keypoints_rep_loss": total_loss_rep,
            "keypoints_recall_loss": total_loss_recall,
            "keypoints_peak_loss": total_loss_peak,
            "keypoints_repulsion_loss": total_loss_repulsion,
            "keypoints_covariance_loss": total_loss_cov,
        }

        # NEGATIVE-SUPERVISION DENSITY LOSS: emitted only when density_mask_train=True AND
        # density_neg_weight > 0, so the key is never present in the default-off path
        # (byte-identical to prior behavior).
        if self.density_mask_train and getattr(self, 'density_neg_weight', 0.0) > 0:
            loss_dict["keypoints_density_neg_loss"] = (total_loss_density_neg / n) * self.density_neg_weight

        return loss_dict, first_corr0, first_corr1, first_y_target

    def repulsion_loss(self, kpts, min_distance=5.0):
        """
        Bestraft Keypoints, die zu nah beieinander liegen.

        Args:
            kpts: [B, 1, H, W] keypoint logits
            min_distance: Mindestabstand in Grid-Zellen

        Returns:
            Repulsion loss (scalar)
        """
        B, _, H, W = kpts.shape
        device = kpts.device

        # Sigmoid probabilities
        prob = torch.sigmoid(kpts).view(B, H, W)

        total_loss = 0.0
        for b in range(B):
            prob_b = prob[b]  # [H, W]

            # Top-K Keypoints (nur die stärksten betrachten)
            k = 50  # Anzahl der zu betrachtenden Top-Keypoints
            prob_flat = prob_b.view(-1)

            if prob_flat.sum() < 1e-6:  # Skip if no keypoints
                continue

            topk_vals, topk_idx = torch.topk(prob_flat, k=min(k, len(prob_flat)))

            # Filter out very weak keypoints
            valid_mask = topk_vals > 0.1
            if valid_mask.sum() < 2:  # Need at least 2 keypoints
                continue

            topk_vals = topk_vals[valid_mask]
            topk_idx = topk_idx[valid_mask]

            # Koordinaten der Top-Keypoints
            topk_h = (topk_idx // W).float()
            topk_w = (topk_idx % W).float()
            coords = torch.stack([topk_h, topk_w], dim=1)  # [K, 2]

            # Pairwise distances
            dist_matrix = torch.cdist(coords, coords, p=2)  # [K, K]

            # Mask Diagonale (Abstand zu sich selbst)
            k_actual = len(topk_vals)
            mask = ~torch.eye(k_actual, device=device, dtype=torch.bool)

            # Soft penalty: ReLU für Distanzen < min_distance
            penalties = F.relu(min_distance - dist_matrix) * mask.float()

            # Gewichte mit Probabilities (wichtiger für starke Keypoints)
            weight_matrix = topk_vals.unsqueeze(1) * topk_vals.unsqueeze(0)

            # Normalisierung
            if k_actual > 1:
                loss_b = (penalties * weight_matrix).sum() / (k_actual * (k_actual - 1))
                total_loss += loss_b

        return total_loss / B if B > 0 else torch.tensor(0.0, device=device)

    def getGTTransformBatch(self, batch_position_current, batch_position_other):
        R_list, t_list, scale_list = [], [], []
        batch_size = len(batch_position_current)
        for b in range(batch_size):
            relative_transformation = torch.linalg.inv(torch.tensor(batch_position_current[b])).matmul(torch.tensor(batch_position_other[b]))
            orthogonal_projection = torch.eye(4)
            orthogonal_projection[2, 2] = 0
            relative_transformation = orthogonal_projection.matmul(relative_transformation)

            R = relative_transformation[:3, :3]
            t = relative_transformation[:3, 3]
            #get scale for relatve cameras (for eval) from t
            scale = torch.norm(t)
            t = t[:2]
            R = R[:2, :2]

            R_list.append(R)
            t_list.append(t)
            scale_list.append(scale)
        return R_list, t_list, scale_list


    def compute_losses(self, kpts0, kpts1, desc0, desc1, R, t):
        """
        Batch-compatible version of compute_losses.

        Args:
            kpts0, kpts1: [B, 1, H, W] keypoint logits
            desc0, desc1: [B, D, H, W] descriptors
            R: [B, 2, 2] or single [2, 2] rotation matrices
            t: [B, 2] or single [2] translation vectors
        """
        device = kpts0.device
        batch_size = kpts0.shape[0]

        # Handle single R, t for batch
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(batch_size, -1, -1)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(batch_size, -1)

        total_loss_desc = 0.0
        total_loss_kpts = 0.0

        # Store for visualization (first sample only)
        first_corr0, first_corr1, first_y_target = None, None, None

        # Process each sample in batch
        for b in range(batch_size):
            corr0, corr1 = self.generate_correspondence(
                dim=self.dim_grid, R=R[b], t=t[b]
            )

            if b == 0:
                first_corr0, first_corr1 = corr0, corr1

            tau = 0.05
            D = desc0.shape[1]

            # Prepare descriptors for current batch sample
            desc0_flat = F.normalize(desc0[b].view(D, -1).T, dim=1)  # [HW, D]
            desc1_flat = F.normalize(desc1[b].view(D, -1).T, dim=1)

            # Compute similarity matrix
            sim = torch.matmul(desc0_flat, desc1_flat.T)  # [HW, HW]

            # Descriptor Loss
            log_prob_0to1 = F.log_softmax(sim / tau, dim=1)
            log_prob_1to0 = F.log_softmax(sim / tau, dim=0)

            loss_desc = -(log_prob_0to1[corr0, corr1] + log_prob_1to0[corr0, corr1]).mean()

            # Keypoint Targets (MNN)
            idx0_to_1 = sim.argmax(dim=1)
            idx1_to_0 = sim.argmax(dim=0)
            is_mnn = (torch.arange(self.dim_grid ** 2, device=device) == idx1_to_0[idx0_to_1])

            y_target = is_mnn.float()

            if b == 0:
                first_y_target = y_target[corr0]

            # Keypoint Loss
            logits0 = kpts0[b].view(-1)
            logits1 = kpts1[b].view(-1)

            loss_kpts = self.kpt_loss_fn(logits0, y_target) + \
                        self.kpt_loss_fn(logits1, y_target)

            total_loss_desc += loss_desc
            total_loss_kpts += loss_kpts

        # Average over batch
        total_loss_desc = (total_loss_desc / batch_size) * self.l_desc
        total_loss_kpts = (total_loss_kpts / batch_size) * self.l_kpts

        loss_dict = dict()
        loss_dict["descriptor_loss"] = total_loss_desc
        loss_dict["keypoints_loss"] = total_loss_kpts

        return loss_dict, first_corr0, first_corr1, first_y_target

    def _density_mask(self, points, device):
        """Build a [dim_grid, dim_grid] bool mask where True = the (2*r+1)^2 BEV-cell
        neighbourhood has >= density_min_nbr_points LiDAR points.

        Args:
            points: [N, >=2] LiDAR tensor in the lidar frame (metres), or None.
            device: target torch device.
        Returns:
            [dim_grid, dim_grid] bool tensor, True where density is sufficient.
        Convention: row <- x (forward), col <- y (left), matching self.bev_resolution
        and the voxelizer used for BEV feature extraction.
        """
        import torch.nn.functional as F
        D = self.dim_grid
        res = self.bev_resolution
        half = D / 2.0
        if points is None or points.numel() == 0:
            return torch.zeros((D, D), dtype=torch.bool, device=device)
        xy = points[:, :2].detach().float()
        r = torch.round(xy[:, 0] / res + half).long()
        c = torch.round(xy[:, 1] / res + half).long()
        ok = (r >= 0) & (r < D) & (c >= 0) & (c < D)
        cnt = torch.zeros((D, D), device=device)
        idx = r[ok] * D + c[ok]
        cnt.view(-1).index_add_(0, idx, torch.ones(int(ok.sum()), device=device))
        rr = self.density_nbr_r
        win = F.avg_pool2d(cnt[None, None], 2 * rr + 1, 1, rr)[0, 0] * (2 * rr + 1) ** 2
        return win >= self.density_min_nbr_points

    def loss_new(self, batch_feats_current, batch_position_current,
                 batch_feats_other, batch_position_other,
                 points_current=None, points_other=None,
                 aug_current=None, aug_other=None,
                 gt_keypoint_xy_current=None,
                 gt_track_id_current=None,
                 gt_keypoint_xy_other=None,
                 gt_track_id_other=None,
                 camera_bev_current=None, camera_bev_other=None):
        """
        Batch-compatible loss computation.

        Args:
            batch_feats_current: List of [B, C, H, W] tensors
            batch_position_current: List of length B with [4, 4] matrices
            batch_feats_other: List of [B, C, H, W] tensors
            batch_position_other: List of length B with [4, 4] matrices
            aug_current: List of length B with [4, 4] aug matrices (or None)
            aug_other: List of length B with [4, 4] aug matrices (or None)
        """
        # RESUME-ROBUST WARMUP: self.step is a plain int that resets to 0 when the head is
        # rebuilt on a Slurm-requeue resume, which would re-trigger the keypoint-loss warmup
        # every requeue (freezing the score head for 2000+ iters, since detach_score_head
        # means it only gets keypoint-loss gradient). Sync from mmengine's global iter so
        # warmup is based on true training progress, not the per-process counter.
        try:
            from mmengine.logging import MessageHub
            _gi = MessageHub.get_current_instance().get_info('iter')
            if isinstance(_gi, int) and _gi > self.step:
                self.step = _gi
        except Exception:
            pass  # eval / no runner -> keep local step

        featA = batch_feats_current[0]
        featB = batch_feats_other[0]

        kpts0, desc0, _ = self.forward(featA, None, camera_bev=camera_bev_current)
        kpts1, desc1, _ = self.forward(featB, None, camera_bev=camera_bev_other)

        # DENSITY-MASK TRAINING: instead of suppressing the score map (which gave
        # 0 loss / no gradient), pass per-sample density masks to compute_losses_infoNCE
        # so that (a) geometric targets are restricted to LiDAR-supported cells and
        # (b) an explicit penalty on empty-cell scores drives the detector to avoid
        # empty space (negative supervision). kpts0/kpts1 are NOT modified here so
        # the focal loss on the UNMASKED map provides the base negative supervision.
        # Gated by density_mask_train (default False -> byte-identical).
        density_masks_current = None
        density_masks_other = None
        if self.density_mask_train and points_current is not None and points_other is not None:
            B = kpts0.shape[0]
            density_masks_current = [
                self._density_mask(points_current[b] if b < len(points_current) else None, kpts0.device)
                for b in range(B)
            ]
            density_masks_other = [
                self._density_mask(points_other[b] if b < len(points_other) else None, kpts1.device)
                for b in range(B)
            ]

        loss_dict, corr0, corr1, y = self.compute_losses_infoNCE(
            kpts0, kpts1, desc0, desc1, None, None,
            batch_position_current=batch_position_current,
            batch_position_other=batch_position_other,
            aug_current=aug_current,
            aug_other=aug_other,
            featA=featA, featB=featB,
            gt_keypoint_xy_current=gt_keypoint_xy_current,
            gt_track_id_current=gt_track_id_current,
            gt_keypoint_xy_other=gt_keypoint_xy_other,
            gt_track_id_other=gt_track_id_other,
            density_masks_current=density_masks_current,
            density_masks_other=density_masks_other,
        )

        # Build R,t for visualization (first batch element only, no aug for compat)
        R_b, t_b = self.get_transformation_matrix(
            batch_position_current[0], batch_position_other[0],
            aug_current[0] if aug_current is not None else None,
            aug_other[0] if aug_other is not None else None,
        )
        R = R_b.unsqueeze(0)
        t = t_b.unsqueeze(0)

        return loss_dict, R, t, corr0, corr1, y, kpts0, kpts1, desc0, desc1


    def loss(self, batch_feats_current, batch_position_current,
             batch_feats_other, batch_position_other,
             points_current=None, points_other=None,
             images_current=None, images_other=None,
             aug_current=None, aug_other=None,
             gt_keypoint_xy_current=None,
             gt_track_id_current=None,
             gt_keypoint_xy_other=None,
             gt_track_id_other=None,
             camera_bev_current=None, camera_bev_other=None):
        losses, R, t, corr0, corr1, y_target, kpts0, kpts1, desc0, desc1 = self.loss_new(batch_feats_current,
                                                                                         batch_position_current,
                                                                                         batch_feats_other,
                                                                                         batch_position_other,
                                                                                         points_current=points_current,
                                                                                         points_other=points_other,
                                                                                         aug_current=aug_current,
                                                                                         aug_other=aug_other,
                                                                                         gt_keypoint_xy_current=gt_keypoint_xy_current,
                                                                                         gt_track_id_current=gt_track_id_current,
                                                                                         gt_keypoint_xy_other=gt_keypoint_xy_other,
                                                                                         gt_track_id_other=gt_track_id_other,
                                                                                         camera_bev_current=camera_bev_current,
                                                                                         camera_bev_other=camera_bev_other)
        if get_rank() == 0:
            if self.step % 1000 == 0:
                # Use only first batch element for visualization (every 1000 iters:
                # the rank-0 render stalls all DDP ranks at the next allreduce).
                R_vis = R[0] if R.dim() == 3 else R
                t_vis = t[0] if t.dim() == 2 else t
                kpts0_vis = kpts0[0:1]  # Keep batch dim
                kpts1_vis = kpts1[0:1]
                desc0_vis = desc0[0:1]
                desc1_vis = desc1[0:1]
                feats0_vis = batch_feats_current[0][0:1]
                feats1_vis = batch_feats_other[0][0:1]

                # Handle images and points - they need to keep batch structure for visualization
                # The visualization functions expect shape [1, ...] not just [...]
                if images_current is not None:
                    if isinstance(images_current, list):
                        images_current_vis = [images_current[0]] if len(images_current) > 0 else None
                    else:
                        images_current_vis = images_current[0:1]  # Keep batch dim
                else:
                    images_current_vis = None

                if images_other is not None:
                    if isinstance(images_other, list):
                        images_other_vis = [images_other[0]] if len(images_other) > 0 else None
                    else:
                        images_other_vis = images_other[0:1]  # Keep batch dim
                else:
                    images_other_vis = None

                if points_current is not None:
                    if isinstance(points_current, list):
                        points_current_vis = [points_current[0]] if len(points_current) > 0 else None
                    else:
                        points_current_vis = points_current[0:1]  # Keep batch dim
                else:
                    points_current_vis = None

                if points_other is not None:
                    if isinstance(points_other, list):
                        points_other_vis = [points_other[0]] if len(points_other) > 0 else None
                    else:
                        points_other_vis = points_other[0:1]  # Keep batch dim
                else:
                    points_other_vis = None

                # Viz is non-essential TensorBoard monitoring; never let it kill a
                # long training run. During warmup the kpt targets are skipped
                # (y_target is None), which some viz scalars don't expect.
                try:
                    self.visualize(R_vis, t_vis, dim=self.dim_grid,
                                   corr_tensor=torch.stack([corr0, corr1], dim=1),
                                   y_success=y_target,
                                   kpts0=kpts0_vis, kpts1=kpts1_vis,
                                   desc0=desc0_vis, desc1=desc1_vis,
                                   loss_dict=losses, writer=self.writer_train, step=self.step,
                                   images_current=images_current_vis, images_other=images_other_vis,
                                   points=points_current_vis, points_other=points_other_vis,
                                   input_feats0=feats0_vis, input_feats1=feats1_vis)
                except Exception as _viz_err:
                    print(f"[keypoint_head] visualize() skipped at step {self.step}: "
                          f"{type(_viz_err).__name__}: {_viz_err}", flush=True)

        # CRITICAL (DDP): advance the step counter on EVERY rank, not only rank 0.
        # warmup/skip_kpt are derived from self.step; if only rank 0 advanced,
        # ranks 1-7 stayed at step 0 forever -> after warmup rank 0 builds the kpt
        # loss graph while the others skip it -> divergent DDP graphs -> hang.
        self.step += 1

        return losses


    #### VISUALIZATIONS #####

    def plotInputImage(self, images_current, images_other):
        with torch.no_grad():
            img = images_current[0][0]
            # convert to numpy
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (img + 1.0) / 2.0

            img = cv2.resize(img, (2 * self.dim_grid, self.dim_grid))
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

            img_other = images_other[0][0]
            # convert to numpy
            img_other = img_other.permute(1, 2, 0).cpu().numpy()
            img_other = (img_other + 1.0) / 2.0
            img_other = cv2.resize(img_other, (2 * self.dim_grid, self.dim_grid))
            img_other = torch.tensor(img_other, dtype=torch.float32).permute(2, 0, 1)

            # both images side by side
            img_combined = torch.zeros((3, self.dim_grid, 4 * self.dim_grid), dtype=torch.float32)
            img_combined[:, :, :2 * self.dim_grid] = img
            img_combined[:, :, 2 * self.dim_grid:] = img_other

            return img_combined

    # returns image
    def plotPointCloudImage(self, points, bev_dim=180):
        with torch.no_grad():
            points = points[0]  # shape [N, 3]
            sampled = points[::5]

            # Correct: row = X (forward), col = Y (left) (thats for BEV, but lidar is rotated)

            row = (sampled[:, 0] // self.bev_resolution).to(torch.int64) + bev_dim // 2
            col = (sampled[:, 1] // self.bev_resolution).to(torch.int64) + bev_dim // 2
            z = sampled[:, 2]
            mask = (row >= 0) & (row < bev_dim) & (col >= 0) & (col < bev_dim)
            row, col, z = row[mask], col[mask], z[mask]

            z_buffer = torch.full((bev_dim, bev_dim), float("-inf"), dtype=torch.float32, device=points.device)

            # Assign max z per pixel
            for r, c, zi in zip(row, col, z):
                z_buffer[r, c] = torch.maximum(z_buffer[r, c], zi)

            valid_mask = z_buffer > float("-inf")
            z_valid = z_buffer[valid_mask]
            r = (255 * (z_valid + 5) / 10).clamp(0, 255).to(torch.uint8)
            b = (255 * (10 - (z_valid + 5)) / 10).clamp(0, 255).to(torch.uint8)
            g = torch.zeros_like(r, dtype=torch.uint8)

            img = torch.zeros((bev_dim, bev_dim, 3), dtype=torch.uint8, device=points.device)
            img[valid_mask] = torch.stack([r, g, b], dim=1)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

            return img

    def getKeypoints(self, kpts):
        if kpts is None:
            return None

        with torch.no_grad():
            coords, _ = extract_keypoints(kpts[0, 0], tau=self.tau, cap=80, k=5)
            return coords  # [N,2] (row,col); may be empty (N=0)

    def drawKeypoints(self, img, keypoints):
        if keypoints is None or keypoints.shape[0] == 0:
            return img

        # Ensure integer pixel coordinates
        y = keypoints[:, 0].long()  # row
        x = keypoints[:, 1].long()  # col

        # Filter only valid keypoints inside image bounds
        H, W = img.shape[1], img.shape[2]
        mask = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        y, x = y[mask], x[mask]

        if x.numel() == 0:
            return img  # nothing to draw

        # Assign green color [0, 1, 0] at all valid (y, x)
        img[0, y, x] = 0.0  # Red channel
        img[1, y, x] = 1.0  # Green channel
        img[2, y, x] = 0.0  # Blue channel

        return img

    # function in order to plot the input points
    def plotInputPoints(self, points, points_other, kpts0=None, kpts1=None):
        if points is None:
            return None, None

        with torch.no_grad():
            # fast implementation
            img1 = self.plotPointCloudImage(points, self.dim_grid)
            img2 = self.plotPointCloudImage(points_other, self.dim_grid)

            img = torch.zeros((3, self.dim_grid, 2 * self.dim_grid), dtype=torch.float32)
            img[:, :, :self.dim_grid] = img1
            img[:, :, self.dim_grid:] = img2

            # scale image up (3x)
            img_points = F.interpolate(img.unsqueeze(0), scale_factor=4, mode='nearest')[0]

            # Create keypoints version
            if kpts1 is not None:
                keypoints1 = self.getKeypoints(kpts1)
                img2 = self.drawKeypoints(img2, keypoints1)

            if kpts0 is not None:
                keypoints0 = self.getKeypoints(kpts0)
                img1 = self.drawKeypoints(img1, keypoints0)

            img[:, :, :self.dim_grid] = img1
            img[:, :, self.dim_grid:] = img2
            img_kpts = F.interpolate(img.unsqueeze(0), scale_factor=4, mode='nearest')[0]

            return img_points, img_kpts

    def resize_to_common_size(self, images, target_height=None, target_width=None):
        """Resize all images to a common size (max dimensions by default)"""
        with torch.no_grad():
            if target_height is None or target_width is None:
                # Find max dimensions
                max_h = max(img.shape[1] for img in images)
                max_w = max(img.shape[2] for img in images)
                target_height = max_h
                target_width = max_w

            resized_images = []
            for img in images:
                if img.shape[1] != target_height or img.shape[2] != target_width:
                    # Resize using interpolate
                    img_resized = F.interpolate(
                        img.unsqueeze(0),
                        size=(target_height, target_width),
                        mode='bilinear',
                        align_corners=False
                    )[0]
                    resized_images.append(img_resized)
                else:
                    resized_images.append(img)

            return resized_images

    def resize_to_common_height(self, images, target_height=None):
        """Resize all images to a common height (preserving width ratio)."""
        with torch.no_grad():
            # Find target height automatically if not provided
            if target_height is None:
                target_height = max(img.shape[1] for img in images)

            resized_images = []
            for img in images:
                c, h, w = img.shape
                if h != target_height:
                    # Compute new width while preserving aspect ratio
                    new_width = int(w * (target_height / h))
                    img_resized = F.interpolate(
                        img.unsqueeze(0),
                        size=(target_height, new_width),
                        mode='bilinear',
                        align_corners=False
                    )[0]
                else:
                    img_resized = img
                resized_images.append(img_resized)

            return resized_images

    def visualize_val(self, R, t, dim, corr_tensor, kpts0, kpts1, desc0, desc1,
                  images_current, images_other, points, points_other, writer, step, input_feats0=None, input_feats1=None):
        with torch.no_grad():
            # Log scalar metrics
            writer.add_image('Keypoints/kpts0_sigmoid', torch.sigmoid(kpts0)[0], step)
            writer.add_image('Keypoints/kpts1_sigmoid', torch.sigmoid(kpts1)[0], step)

            # Generate consolidated image batch
            image_batch = []

            # Add input images
            img_combined = self.plotInputImage(images_current, images_other)
            image_batch.append(img_combined.cpu())

            # Add point cloud visualizations
            img_points, img_kpts = self.plotInputPoints(points, points_other, kpts0, kpts1)
            if img_points is not None:
                image_batch.append(img_points.cpu())
                image_batch.append(img_kpts.cpu())

            # Add correspondence visualizations
            image_batch.append(self.draw_matches(kpts0, kpts1, desc0, desc1, dim=dim, prob_thresh=0.5).cpu())
            image_batch.append(self.draw_matches(kpts0, kpts1, desc0, desc1, topk=1000, dim=dim).cpu())
            image_batch.append(self.draw_matches(kpts0, kpts1, desc0, desc1, dim=dim, prob_thresh=0.0).cpu())
            image_batch.append(self.draw_maches_correct(desc0, desc1, corr_tensor, dim=dim).cpu())
            image_batch.append(self.plot_ground_truth_correspondence(corr_tensor, dim=dim).cpu())
            image_batch.append(self.draw_vehicle_movement(R, t, dim=dim).cpu())

            # Optionally add input feature maps if available, but flatten to 3 channels first
            if input_feats0 is not None and input_feats1 is not None:
                # Normalize feature maps to [0, 1] for visualization
                feat0 = input_feats0[0].cpu()
                feat1 = input_feats1[0].cpu()

                # Reduce to 3 channels using PCA or just take first 3 channels
                if feat0.shape[0] >= 3:
                    def pca_to_rgb(feat):
                        C, H, W = feat.shape
                        feat_np = feat.cpu().numpy().reshape(C, -1).T  # shape: [H*W, C]
                        pca = PCA(n_components=3)
                        feat_pca = pca.fit_transform(feat_np)  # [H*W, 3]
                        feat_pca = feat_pca.reshape(H, W, 3)
                        # normalize to [0, 1]
                        feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min())
                        return torch.from_numpy(feat_pca).permute(2, 0, 1)

                    feat0_vis = pca_to_rgb(feat0)
                    feat1_vis = pca_to_rgb(feat1)

                # Normalize each feature map to [0, 1], based on min-max of BOTH

                #feat0_vis = (feat0_vis - feat0_vis.min()) / (feat0_vis.max() - feat0_vis.min() + 1e-8)
                #feat1_vis = (feat1_vis - feat1_vis.min()) / (feat1_vis.max() - feat1_vis.min() + 1e-8)

                def percentile_normalize(feat, low=0, high=90):
                    lo = torch.quantile(feat, low / 100.0)
                    hi = torch.quantile(feat, high / 100.0)
                    feat = torch.clamp(feat, lo, hi)
                    feat = (feat - lo) / (hi - lo + 1e-8)
                    return feat
                feat0_vis = percentile_normalize(feat0_vis)
                feat1_vis = percentile_normalize(feat1_vis)
                #print(feat0_vis.shape, feat1_vis.shape)
                image_batch.append(feat0_vis.cpu())
                image_batch.append(feat1_vis.cpu())


            # Resize all images to common dimensions before stacking
            image_batch = self.resize_to_common_height(image_batch, target_height=256)
            grid = self.make_image_grid_dynamic(image_batch, max_width=1080, padding=10)

            writer.add_image('Paper/Matches', self.draw_matches_paper(kpts0,kpts1, desc0, desc1, dim, prob_thresh=0.5, latent_space0=feat0_vis.permute(1,2,0).cpu().numpy(), latent_space1=feat1_vis.permute(1,2,0).cpu().numpy()), step)
            #add all camera images separately

            writer.add_image('Paper/Camera_Image_Current_front', (images_current[0][0]+1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Other_front', (images_other[0][0] + 1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Current_front_left', (images_current[0][1] + 1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Other_front_left', (images_other[0][1] + 1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Current_front_right', (images_current[0][2] + 1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Other_front_right', (images_other[0][2] + 1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Current_rear_left', (images_current[0][3] + 1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Other_rear_left', (images_other[0][3] + 1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Current_rear_right', (images_current[0][4] + 1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Other_rear_right', (images_other[0][4] + 1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Current_rear', (images_current[0][5] + 1.0) / 2.0, step)
            writer.add_image('Paper/Camera_Image_Other_rear', (images_other[0][5] + 1.0) / 2.0, step)
            writer.add_image("Paper/PointCloud_Current", self.plotPointCloudImage(points, 180), step)
            writer.add_image("Paper/PointCloud_Other", self.plotPointCloudImage(points_other, 180), step)


            # Log to TensorBoard
            writer.add_image('All_Visualizations/Grid', grid, step)

    def visualize(self, R, t, dim, corr_tensor, y_success, kpts0, kpts1, desc0, desc1,
                  loss_dict, images_current, images_other, points, points_other, writer, step, input_feats0=None, input_feats1=None):
        with torch.no_grad():
            # Log scalar metrics
            writer.add_scalar('Loss/Descriptor', loss_dict["descriptor_loss"], step)
            # New geometry-target loss split into 5 terms (Task 10) — log each +
            # a local aggregate. Do NOT add an aggregate back into loss_dict
            # (mmengine sums every tensor value -> would double-count).
            kpt_keys = ['keypoints_focal_loss', 'keypoints_rep_loss',
                        'keypoints_recall_loss', 'keypoints_peak_loss',
                        'keypoints_repulsion_loss']
            kpt_total = sum(loss_dict[k] for k in kpt_keys if k in loss_dict)
            writer.add_scalar('Loss/Keypoints', kpt_total, step)
            for k in kpt_keys:
                if k in loss_dict:
                    writer.add_scalar('Loss/' + k, loss_dict[k], step)
            writer.add_scalar('Loss/Total', loss_dict["descriptor_loss"] + kpt_total, step)


            writer.add_scalar('Metrics/Successful_Matches (GT)', torch.sum(y_success).item(), step)

            # Log individual keypoint heatmaps
            writer.add_image('Keypoints/kpts0_sigmoid', torch.sigmoid(kpts0)[0], step)
            writer.add_image('Keypoints/kpts1_sigmoid', torch.sigmoid(kpts1)[0], step)

            # Generate consolidated image batch
            image_batch = []

            # Add input images
            img_combined = self.plotInputImage(images_current, images_other)
            image_batch.append(img_combined.cpu())

            # Add point cloud visualizations
            img_points, img_kpts = self.plotInputPoints(points, points_other, kpts0, kpts1)
            if img_points is not None:
                image_batch.append(img_points.cpu())
                image_batch.append(img_kpts.cpu())

            # Add correspondence visualizations
            image_batch.append(self.draw_matches(kpts0, kpts1, desc0, desc1, dim=dim, prob_thresh=0.5).cpu())
            image_batch.append(self.draw_matches(kpts0, kpts1, desc0, desc1, topk=1000, dim=dim).cpu())
            image_batch.append(self.draw_matches(kpts0, kpts1, desc0, desc1, dim=dim, prob_thresh=0.0).cpu())
            image_batch.append(self.draw_maches_correct(desc0, desc1, corr_tensor, dim=dim).cpu())
            image_batch.append(self.plot_ground_truth_correspondence(corr_tensor, dim=dim).cpu())
            image_batch.append(self.draw_vehicle_movement(R, t, dim=dim).cpu())

            # Optionally add input feature maps if available, but flatten to 3 channels first
            if input_feats0 is not None and input_feats1 is not None:
                # Normalize feature maps to [0, 1] for visualization
                feat0 = input_feats0[0].cpu()
                feat1 = input_feats1[0].cpu()

                # Reduce to 3 channels using PCA or just take first 3 channels
                if feat0.shape[0] >= 3:
                    feat0_vis = feat0[:3]
                    feat1_vis = feat1[:3]
                else:
                    # Pad with zeros if less than 3 channels
                    pad_size = 3 - feat0.shape[0]
                    feat0_vis = F.pad(feat0, (0, 0, 0, 0, 0, pad_size))
                    feat1_vis = F.pad(feat1, (0, 0, 0, 0, 0, pad_size))

                # Normalize each feature map to [0, 1]
                feat0_vis = (feat0_vis - feat0_vis.min()) / (feat0_vis.max() - feat0_vis.min() + 1e-8)
                feat1_vis = (feat1_vis - feat1_vis.min()) / (feat1_vis.max() - feat1_vis.min() + 1e-8)
                print(feat0_vis.shape, feat1_vis.shape)
                image_batch.append(feat0_vis.cpu())
                image_batch.append(feat1_vis.cpu())

            #image_batch.append(torch.sigmoid(kpts0)[0].cpu())
            #image_batch.append(torch.sigmoid(kpts1)[0].cpu())

            # Resize all images to common dimensions before stacking
            image_batch = self.resize_to_common_height(image_batch, target_height=256)
            grid = self.make_image_grid_dynamic(image_batch, max_width=1080, padding=10)

            # Log to TensorBoard
            writer.add_image('All_Visualizations/Grid', grid, step)

            # Write all visualizations as one batch
            #writer.add_images('All_Visualizations', torch.stack(image_batch), step)
    ############# IMAGE GRID #############
    def make_image_grid_dynamic(self, image_batch, max_width=900, padding=10, bg_color=0.0):
        """
        Combines a list of (C,H,W) tensors into a single grid image.
        Starts a new row whenever the total width would exceed max_width.
        """
        if not image_batch:
            return None

        # Ensure all images have same number of channels
        c = image_batch[0].shape[0]

        # Convert all to same height (largest one)
        heights = [img.shape[1] for img in image_batch]
        widths = [img.shape[2] for img in image_batch]
        max_h = max(heights)

        rows = []
        current_row = []
        current_width = 0
        row_heights = []

        for img, w, h in zip(image_batch, widths, heights):
            # Resize smaller images to match max height
            if h != max_h:
                img = \
                F.interpolate(img.unsqueeze(0), size=(max_h, int(w * max_h / h)), mode='bilinear', align_corners=False)[
                    0]
                w = img.shape[2]

            # Start a new row if adding this image would exceed max_width
            if current_width + w + padding > max_width and current_row:
                rows.append((current_row, current_width, max_h))
                current_row = []
                current_width = 0

            current_row.append(img)
            current_width += w + padding

        # Add last row
        if current_row:
            rows.append((current_row, current_width, max_h))

        # Now combine rows vertically
        row_images = []
        for current_row, current_width, _ in rows:
            # Pad all images in the row to the same height
            row_imgs_padded = []
            for img in current_row:
                pad_right = 0
                pad_bottom = max_h - img.shape[1]
                row_imgs_padded.append(F.pad(img, (0, pad_right, 0, pad_bottom), value=bg_color))
            row_cat = torch.cat(row_imgs_padded, dim=2)
            row_images.append(row_cat)

        # Combine all rows vertically with padding
        total_height = sum([r.shape[1] + padding for r in row_images]) - padding
        total_width = max([r.shape[2] for r in row_images])

        # Create background
        grid = torch.full((c, total_height, total_width), bg_color, dtype=row_images[0].dtype)

        y = 0
        for r in row_images:
            h = r.shape[1]
            grid[:, y:y + h, :r.shape[2]] = r
            y += h + padding

        return grid

    ############# ESTIMATION METHODS #############

    def estimateTransform(self, pts0, pts1):
        c0 = np.mean(pts0, axis=0)
        c1 = np.mean(pts1, axis=0)
        X0 = pts0 - c0
        X1 = pts1 - c1

        # solve rotation with SVD
        U, _, Vt = np.linalg.svd(X0.T @ X1)
        R = U @ Vt

        # handle reflection case
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # translation
        t = c1 - R @ c0

        # build 2x3 matrix
        M = np.hstack([R, t.reshape(2, 1)])
        H = np.vstack([M, [0, 0, 1]])

        #dummy mask, say all inliers
        mask = np.ones((pts0.shape[0],), dtype=np.uint8)
        #in order to recover the movement (how the vehicle moved, not how the world moved), invert the transformation
        H = np.linalg.inv(H)
        return H, mask

    def estimate_2d_transform(self, P, Q):
        """
        Estimate 2D rotation (around Z) and translation between
        two sets of corresponding 2D points.

        Parameters
        ----------
        P : (N,2) ndarray - source points
        Q : (N,2) ndarray - target points

        Returns
        -------
        R : (2,2) ndarray - rotation matrix
        t : (2,) ndarray  - translation vector
        theta : float     - rotation angle in radians
        """

        # 1️⃣ Compute centroids
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)

        # 2️⃣ Center the points
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # 3️⃣ Compute covariance matrix
        H = P_centered.T @ Q_centered

        # 4️⃣ SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = +1)
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        # 5️⃣ Compute translation
        t = centroid_Q - R @ centroid_P

        # 6️⃣ Extract rotation angle
        theta = np.arctan2(R[1, 0], R[0, 0])

        return R, t, theta

    def estimateHomography(self, pts0, pts1):
        H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 1.0)
        if H is None:
            H = np.eye(3)
            mask = np.zeros((pts0.shape[0],), dtype=np.uint8)
        H = np.linalg.inv(H)

        return H, mask
    def estimateRigidTransform(self, pts0, pts1):
        H, mask = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            H = np.eye(3)
            mask = np.zeros((pts0.shape[0],), dtype=np.uint8)
        else:
            H = np.vstack([H, [0, 0, 1]])
        H = np.linalg.inv(H)

        return H, mask

    def add_text_to_image(self, img, R_grid, t_grid):

        R, t = self.grid_to_metric(R_grid, t_grid)

        text = f"dx: {t[0]:.2f}m, dy: {t[1]:.2f}m"
        angle = math.atan2(R[1, 0], R[0, 0]) * 180.0 / math.pi
        angle_text = f"rot: {angle:.2f}deg"
        x = 5
        y = 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.3
        thickness = 1
        margin = 3
        (text_w1, text_h1), _ = cv2.getTextSize(text, font, scale, thickness)
        (text_w2, text_h2), _ = cv2.getTextSize(angle_text, font, scale, thickness)
        panel_width = max(text_w1, text_w2)

        cv2.rectangle(
            img,
            (x - margin, y - text_h1 - margin),
            (x - margin + panel_width, y + text_h2 + margin),
            (30, 30, 30),  # black
            -1  # filled
        )

        # Then put the text on top
        cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(img, angle_text, (x, y + text_h2 + margin), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


    def calcTransform(self, kpts0, kpts1, desc0, desc1, dim, sim_thresh=None, prob_thresh=None, topk=None):
        """
        New logic: Find ALL MNN matches first, then select top-k by similarity.
        """
        H, W = dim, dim

        # --- 1. Select keypoints via calibrated extract_keypoints (tau+NMS+cap) ---
        # KPT_EVAL_CAP env: keypoint budget for eval extraction (KITTI-style top-K after NMS).
        # Default 80 = historical behavior, unchanged unless explicitly set.
        import os as _os
        _cap = int(_os.environ.get('KPT_EVAL_CAP', '80'))
        with torch.no_grad():
            coords0, _ = extract_keypoints(kpts0[0, 0], tau=self.tau, cap=_cap, k=5)
            coords1, _ = extract_keypoints(kpts1[0, 0], tau=self.tau, cap=_cap, k=5)

        if coords0.shape[0] == 0 or coords1.shape[0] == 0:
            return None, None

        y0, x0 = coords0[:, 0], coords0[:, 1]  # [N0], row, col
        y1, x1 = coords1[:, 0], coords1[:, 1]  # [N1], row, col

        # --- 2. Gather descriptors ---
        desc0_sel = desc0[0, :, y0, x0].T  # [N0, D]
        desc1_sel = desc1[0, :, y1, x1].T  # [N1, D]
        desc0_sel = F.normalize(desc0_sel, p=2, dim=1)
        desc1_sel = F.normalize(desc1_sel, p=2, dim=1)

        # --- 3. Similarity + MNN filtering (ALL matches first) ---
        sim = torch.matmul(desc0_sel, desc1_sel.T)  # [N0, N1]
        nn12 = sim.argmax(dim=1)  # best match in desc1 for each desc0
        nn21 = sim.argmax(dim=0)  # best match in desc0 for each desc1

        # Collect ALL MNN matches with their similarities
        mnn_candidates = []
        for i, j in enumerate(nn12):
            if nn21[j] == i:  # mutual nearest neighbor
                similarity = sim[i, j].item()

                # Apply sim_thresh filter if provided
                if sim_thresh is not None and similarity < sim_thresh:
                    continue

                # Distance to center filter
                dx0 = abs(x0[i].item() - W / 2)
                dy0 = abs(y0[i].item() - H / 2)
                dist0 = math.sqrt(dx0 * dx0 + dy0 * dy0)
                dx1 = abs(x1[j].item() - W / 2)
                dy1 = abs(y1[j].item() - H / 2)
                dist1 = math.sqrt(dx1 * dx1 + dy1 * dy1)

                if dist0 > 100 or dist1 > 100 or dist0 < 5 or dist1 < 5:
                    continue

                mnn_candidates.append({
                    'similarity': similarity,
                    'x0': x0[i].item(),
                    'y0': y0[i].item(),
                    'x1': x1[j].item(),
                    'y1': y1[j].item()
                })

        if len(mnn_candidates) == 0:
            return None, None

        # --- 4. Select top-k matches by similarity ---
        if topk is not None:
            # Sort by similarity (descending) and take top-k
            mnn_candidates.sort(key=lambda x: x['similarity'], reverse=True)  # keep BEST topk matches (spec §4.3)
            mnn_candidates = mnn_candidates[:topk]

        # --- 5. Prepare data for transformation estimation ---
        matches = []
        pts0, pts1 = [], []
        for match in mnn_candidates:
            matches.append((match['x0'], match['y0'], match['x1'], match['y1']))
            pts0.append((match['y0'], match['x0']))  # [row, col]
            pts1.append((match['y1'], match['x1']))  # [row, col]

        pts0 = np.asarray(pts0, np.float64)
        pts1 = np.asarray(pts1, np.float64)
        # SUB-CELL (env KPT_SUBCELL=1): 3x3-Sigmoid-CoM-Verfeinerung vor der Posen-Schaetzung.
        import os as _os
        if _os.environ.get('KPT_SUBCELL', '0') == '1' and len(pts0):
            _p0 = torch.sigmoid(kpts0[0, 0].float()).detach().cpu().numpy()
            _p1 = torch.sigmoid(kpts1[0, 0].float()).detach().cpu().numpy()
            _or0, _oc0 = _subcell_offsets(_p0, pts0[:, 0], pts0[:, 1])
            _or1, _oc1 = _subcell_offsets(_p1, pts1[:, 0], pts1[:, 1])
            pts0 = pts0 + np.stack([_or0, _oc0], 1)
            pts1 = pts1 + np.stack([_or1, _oc1], 1)
        pts0 = pts0 - np.array([H / 2, W / 2])
        pts1 = pts1 - np.array([H / 2, W / 2])
        pts0 = pts0.astype(np.float32)
        pts1 = pts1.astype(np.float32)

        #from grid to metric
        scale = self.bev_resolution
        pts0 = pts0 * scale
        pts1 = pts1 * scale


        H_mat, mask = self.estimateRigidTransform(pts0, pts1)
        t_rec = None
        R_rec = None
        if H_mat is not None:
            H_mat = H_mat / H_mat[2, 2]
            R_rec = H_mat[0:2, 0:2]
            t_rec = H_mat[0:2, 2]

        return R_rec, t_rec



    def draw_matches_topkfirst(self, kpts0, kpts1, desc0, desc1, dim, sim_thresh=None, prob_thresh=None, topk=None):
        """
        Draw matches between two BEV maps using *all* keypoints
        above a probability threshold (no top-k selection).
        """
        H, W = dim, dim
        kpts0_probs = torch.sigmoid(kpts0)[0, 0]  # [H, W]
        kpts1_probs = torch.sigmoid(kpts1)[0, 0]

        y0, x0 = None, None
        y1, x1 = None, None

        # --- 1. Keep all keypoints above threshold ---
        if prob_thresh is not None:
            mask0 = kpts0_probs >= prob_thresh
            mask1 = kpts1_probs >= prob_thresh
            y0, x0 = mask0.nonzero(as_tuple=True)  # [N0], row, col
            y1, x1 = mask1.nonzero(as_tuple=True)  # [N1], row, col
        elif topk is not None:
            # --- 1. Top-k keypoints ---
            _, idx0 = torch.topk(kpts0_probs.view(-1), topk)
            _, idx1 = torch.topk(kpts1_probs.view(-1), topk)
            y0, x0 = idx0 // W, idx0 % W  # [k], row, col
            y1, x1 = idx1 // W, idx1 % W  # [k], row, col


        if len(y0) == 0 or len(y1) == 0:
            # nothing above threshold
            return torch.zeros((3, H, W), dtype=torch.float32)

        # --- 2. Gather descriptors ---
        desc0_sel = desc0[0, :, y0, x0].T  # [N0, D]
        desc1_sel = desc1[0, :, y1, x1].T  # [N1, D]
        desc0_sel = F.normalize(desc0_sel, p=2, dim=1)
        desc1_sel = F.normalize(desc1_sel, p=2, dim=1)

        # --- 3. Similarity + MNN filtering ---
        sim = torch.matmul(desc0_sel, desc1_sel.T)  # [N0, N1]
        nn12 = sim.argmax(dim=1)  # best match in desc1 for each desc0
        nn21 = sim.argmax(dim=0)  # best match in desc0 for each desc1

        matches = []
        pts0, pts1 = [], []
        for i, j in enumerate(nn12):
            if nn21[j] == i:  # mutual nearest neighbor
                if sim_thresh is None or sim[i, j] >= sim_thresh:
                    # only use, if distance to center is higher 90, continue
                    dx0 = abs(x0[i].item() - W / 2)
                    dy0 = abs(y0[i].item() - H / 2)
                    dist0 = math.sqrt(dx0 * dx0 + dy0 * dy0)
                    dx1 = abs(x1[j].item() - W / 2)
                    dy1 = abs(y1[j].item() - H / 2)
                    dist1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
                    if dist0 > 100 or dist1 > 100 or dist0 < 5 or dist1 < 5:
                        continue



                    matches.append((x0[i].item(), y0[i].item(),
                                    x1[j].item(), y1[j].item()))
                    pts0.append((y0[i].item(), x0[i].item()))  # [row, col]
                    pts1.append((y1[j].item(), x1[j].item()))  # [row, col]
        if len(matches) == 0:
            return torch.zeros((3, H, W), dtype=torch.float32)

        pts0 = np.array(pts0) - np.array([H / 2, W / 2])
        pts1 = np.array(pts1) - np.array([H / 2, W / 2])

        pts0 = pts0.astype(np.float32)
        pts1 = pts1.astype(np.float32)

        H_mat, mask = self.estimateRigidTransform(pts0, pts1)
        t_rec = np.array([9999.0, 9999.0])
        R_rec = np.eye(2)
        if H_mat is not None:
            H_mat = H_mat / H_mat[2, 2]
            R_rec = H_mat[0:2, 0:2]
            t_rec = H_mat[0:2, 2]

        # --- 4. Visualization ---
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for (x0i, y0i, x1i, y1i) in matches:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(img, (x0i, y0i), (x1i, y1i), color, 1)
            img[y0i, x0i, :] = COLOR_T0  # [row, col]
            img[y1i, x1i, :] = COLOR_T1

        self.add_text_to_image(img, R_rec, t_rec)

        return (torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0).cpu()

    def draw_matches(self, kpts0, kpts1, desc0, desc1, dim, sim_thresh=None, prob_thresh=None, topk=None):
        """
        Draw matches between two BEV maps.
        New logic: Find ALL MNN matches first, then select top-k by similarity.
        """
        H, W = dim, dim

        # --- 1. Select keypoints via calibrated extract_keypoints (tau+NMS+cap) ---
        # KPT_EVAL_CAP env: keypoint budget for eval extraction (KITTI-style top-K after NMS).
        # Default 80 = historical behavior, unchanged unless explicitly set.
        import os as _os
        _cap = int(_os.environ.get('KPT_EVAL_CAP', '80'))
        with torch.no_grad():
            coords0, _ = extract_keypoints(kpts0[0, 0], tau=self.tau, cap=_cap, k=5)
            coords1, _ = extract_keypoints(kpts1[0, 0], tau=self.tau, cap=_cap, k=5)

        if coords0.shape[0] == 0 or coords1.shape[0] == 0:
            return torch.zeros((3, H, W), dtype=torch.float32)

        y0, x0 = coords0[:, 0], coords0[:, 1]  # [N0], row, col
        y1, x1 = coords1[:, 0], coords1[:, 1]  # [N1], row, col

        # --- 2. Gather descriptors ---
        desc0_sel = desc0[0, :, y0, x0].T  # [N0, D]
        desc1_sel = desc1[0, :, y1, x1].T  # [N1, D]
        desc0_sel = F.normalize(desc0_sel, p=2, dim=1)
        desc1_sel = F.normalize(desc1_sel, p=2, dim=1)

        # --- 3. Similarity + MNN filtering (ALL matches first) ---
        sim = torch.matmul(desc0_sel, desc1_sel.T)  # [N0, N1]
        nn12 = sim.argmax(dim=1)  # best match in desc1 for each desc0
        nn21 = sim.argmax(dim=0)  # best match in desc0 for each desc1

        # Collect ALL MNN matches with their similarities
        mnn_candidates = []
        for i, j in enumerate(nn12):
            if nn21[j] == i:  # mutual nearest neighbor
                similarity = sim[i, j].item()

                # Apply sim_thresh filter if provided
                if sim_thresh is not None and similarity < sim_thresh:
                    continue

                # Distance to center filter
                dx0 = abs(x0[i].item() - W / 2)
                dy0 = abs(y0[i].item() - H / 2)
                dist0 = math.sqrt(dx0 * dx0 + dy0 * dy0)
                dx1 = abs(x1[j].item() - W / 2)
                dy1 = abs(y1[j].item() - H / 2)
                dist1 = math.sqrt(dx1 * dx1 + dy1 * dy1)

                if dist0 > 100 or dist1 > 100 or dist0 < 5 or dist1 < 5:
                    continue

                mnn_candidates.append({
                    'similarity': similarity,
                    'x0': x0[i].item(),
                    'y0': y0[i].item(),
                    'x1': x1[j].item(),
                    'y1': y1[j].item()
                })

        if len(mnn_candidates) == 0:
            return torch.zeros((3, H, W), dtype=torch.float32)

        # --- 4. Select top-k matches by similarity ---
        if topk is not None:
            # Sort by similarity (descending) and take top-k
            mnn_candidates.sort(key=lambda x: x['similarity'], reverse=True)  # keep BEST topk matches (spec §4.3)
            mnn_candidates = mnn_candidates[:topk]

        # --- 5. Prepare data for transformation estimation ---
        matches = []
        pts0, pts1 = [], []
        for match in mnn_candidates:
            matches.append((match['x0'], match['y0'], match['x1'], match['y1']))
            pts0.append((match['y0'], match['x0']))  # [row, col]
            pts1.append((match['y1'], match['x1']))  # [row, col]

        pts0 = np.asarray(pts0, np.float64)
        pts1 = np.asarray(pts1, np.float64)
        # SUB-CELL (env KPT_SUBCELL=1): 3x3-Sigmoid-CoM-Verfeinerung vor der Posen-Schaetzung.
        import os as _os
        if _os.environ.get('KPT_SUBCELL', '0') == '1' and len(pts0):
            _p0 = torch.sigmoid(kpts0[0, 0].float()).detach().cpu().numpy()
            _p1 = torch.sigmoid(kpts1[0, 0].float()).detach().cpu().numpy()
            _or0, _oc0 = _subcell_offsets(_p0, pts0[:, 0], pts0[:, 1])
            _or1, _oc1 = _subcell_offsets(_p1, pts1[:, 0], pts1[:, 1])
            pts0 = pts0 + np.stack([_or0, _oc0], 1)
            pts1 = pts1 + np.stack([_or1, _oc1], 1)
        pts0 = pts0 - np.array([H / 2, W / 2])
        pts1 = pts1 - np.array([H / 2, W / 2])
        pts0 = pts0.astype(np.float32)
        pts1 = pts1.astype(np.float32)

        H_mat, mask = self.estimateRigidTransform(pts0, pts1)
        t_rec = np.array([9999.0, 9999.0])
        R_rec = np.eye(2)
        if H_mat is not None:
            H_mat = H_mat / H_mat[2, 2]
            R_rec = H_mat[0:2, 0:2]
            t_rec = H_mat[0:2, 2]

        # --- 6. Visualization ---
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for (x0i, y0i, x1i, y1i) in matches:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(img, (x0i, y0i), (x1i, y1i), color, 1)
            img[y0i, x0i, :] = COLOR_T0  # [row, col]
            img[y1i, x1i, :] = COLOR_T1

        self.add_text_to_image(img, R_rec, t_rec)
        return (torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0).cpu()


    def draw_matches_paper(self, kpts0, kpts1, desc0, desc1, dim, sim_thresh=None, prob_thresh=None, topk=None, latent_space0=None, latent_space1=None):
        """
        Draw matches between two BEV maps.
        New logic: Find ALL MNN matches first, then select top-k by similarity.
        """
        H, W = dim, dim
        kpts0_probs = torch.sigmoid(kpts0)[0, 0]  # [H, W]
        kpts1_probs = torch.sigmoid(kpts1)[0, 0]

        # --- 1. Select keypoints (using prob_thresh OR all keypoints if topk is set) ---
        if prob_thresh is not None:
            mask0 = kpts0_probs >= prob_thresh
            mask1 = kpts1_probs >= prob_thresh
            y0, x0 = mask0.nonzero(as_tuple=True)  # [N0], row, col
            y1, x1 = mask1.nonzero(as_tuple=True)  # [N1], row, col
        else:
            # Use ALL keypoints when topk is set (we'll filter later)
            y0, x0 = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            y0, x0 = y0.reshape(-1), x0.reshape(-1)
            y1, x1 = y0.clone(), x0.clone()

        if len(y0) == 0 or len(y1) == 0:
            return torch.zeros((3, H, W), dtype=torch.float32)

        # --- 2. Gather descriptors ---
        desc0_sel = desc0[0, :, y0, x0].T  # [N0, D]
        desc1_sel = desc1[0, :, y1, x1].T  # [N1, D]
        desc0_sel = F.normalize(desc0_sel, p=2, dim=1)
        desc1_sel = F.normalize(desc1_sel, p=2, dim=1)

        # --- 3. Similarity + MNN filtering (ALL matches first) ---
        sim = torch.matmul(desc0_sel, desc1_sel.T)  # [N0, N1]
        nn12 = sim.argmax(dim=1)  # best match in desc1 for each desc0
        nn21 = sim.argmax(dim=0)  # best match in desc0 for each desc1

        # Collect ALL MNN matches with their similarities
        mnn_candidates = []
        for i, j in enumerate(nn12):
            if nn21[j] == i:  # mutual nearest neighbor
                similarity = sim[i, j].item()

                # Apply sim_thresh filter if provided
                if sim_thresh is not None and similarity < sim_thresh:
                    continue

                mnn_candidates.append({
                    'similarity': similarity,
                    'x0': x0[i].item(),
                    'y0': y0[i].item(),
                    'x1': x1[j].item(),
                    'y1': y1[j].item()
                })

        if len(mnn_candidates) == 0:
            return torch.zeros((3, H, W), dtype=torch.float32)

        # --- 4. Select top-k matches by similarity ---
        if topk is not None:
            # Sort by similarity (descending) and take top-k
            mnn_candidates.sort(key=lambda x: x['similarity'], reverse=True)  # keep BEST topk matches (spec §4.3)
            mnn_candidates = mnn_candidates[:topk]

        # --- 5. Prepare data for transformation estimation ---
        matches = []
        pts0, pts1 = [], []
        for match in mnn_candidates:
            matches.append((match['x0'], match['y0'], match['x1'], match['y1']))
            pts0.append((match['y0'], match['x0']))  # [row, col]
            pts1.append((match['y1'], match['x1']))  # [row, col]

        pts0 = np.asarray(pts0, np.float64)
        pts1 = np.asarray(pts1, np.float64)
        # SUB-CELL (env KPT_SUBCELL=1): 3x3-Sigmoid-CoM-Verfeinerung vor der Posen-Schaetzung.
        import os as _os
        if _os.environ.get('KPT_SUBCELL', '0') == '1' and len(pts0):
            _p0 = torch.sigmoid(kpts0[0, 0].float()).detach().cpu().numpy()
            _p1 = torch.sigmoid(kpts1[0, 0].float()).detach().cpu().numpy()
            _or0, _oc0 = _subcell_offsets(_p0, pts0[:, 0], pts0[:, 1])
            _or1, _oc1 = _subcell_offsets(_p1, pts1[:, 0], pts1[:, 1])
            pts0 = pts0 + np.stack([_or0, _oc0], 1)
            pts1 = pts1 + np.stack([_or1, _oc1], 1)
        pts0 = pts0 - np.array([H / 2, W / 2])
        pts1 = pts1 - np.array([H / 2, W / 2])
        pts0 = pts0.astype(np.float32)
        pts1 = pts1.astype(np.float32)
        H_mat, mask = self.estimateRigidTransform(pts0, pts1)

        # --- 6. Visualization: Two separate grids with connecting lines ---

        def prepare_latent(latent):
            if isinstance(latent, torch.Tensor):
                latent = latent.permute(1, 2, 0).cpu().numpy()
            latent = (latent * 255).astype(np.uint8)
            latent = cv2.rotate(latent, cv2.ROTATE_90_CLOCKWISE)
            return latent

        rotated_matches = []
        for (x0i, y0i, x1i, y1i) in matches:
            # (x, y) → (y, W - 1 - x)
            rx0, ry0 = y0i, W - 1 - x0i
            rx1, ry1 = y1i, W - 1 - x1i
            rotated_matches.append((rx0, ry0, rx1, ry1))

        latent_space0 = prepare_latent(latent_space0)
        latent_space1 = prepare_latent(latent_space1)

        sep = 10
        combined = np.zeros((H, W * 2 + sep, 3), dtype=np.uint8)
        combined[:, :W, :] = latent_space0
        combined[:, W + sep:, :] = latent_space1

        # mask: 1 = inlier, 0 = outlier
        if mask is not None:
            mask = mask.flatten().astype(bool)
        else:
            mask = np.zeros(len(matches), dtype=bool)

        #filter out: only random 200 matches
        random_indices = np.random.choice(len(matches), size=min(100, len(matches)), replace=False)
        rotated_matches = [rotated_matches[i] for i in random_indices]
        mask = [mask[i] for i in random_indices]

        for idx, (x0i, y0i, x1i, y1i) in enumerate(rotated_matches):
            is_inlier = mask[idx] if idx < len(mask) else False
            if not is_inlier:
                continue
            color = (0, 255, 0) if is_inlier else (255, 0, 0)  # green = inlier, red = outlier

            pt1 = (int(x0i), int(y0i))
            pt2 = (int(x1i + W + sep), int(y1i))
            cv2.line(combined, pt1, pt2, color, 1, cv2.LINE_AA)

        for idx, (x0i, y0i, x1i, y1i) in enumerate(rotated_matches):
            is_inlier = mask[idx] if idx < len(mask) else False
            if is_inlier:
                continue
            color = (0, 255, 0) if is_inlier else (255, 0, 0)  # green = inlier, red = outlier

            pt1 = (int(x0i), int(y0i))
            pt2 = (int(x1i + W + sep), int(y1i))
            cv2.line(combined, pt1, pt2, color, 1, cv2.LINE_AA)

        return (torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1) / 255.0).cpu()




    def grid_to_metric(self, R_grid, t_grid):
        """
        Convert a relative transformation in BEV-Grid coordinates
        back to relative, metrical movement in real-world coordinates.

        Args:
            R_grid: 2x2 rotation matrix in grid coordinates (numpy array or tensor).
            t_grid: 2-element translation vector in grid coordinates ([row, col]).

        Returns:
            R: 3x3 rotation matrix in real-world coordinates (torch tensor).
            t: 3-element translation vector in meters (torch tensor).
        """
        # Ensure R_grid is a torch tensor
        if isinstance(R_grid, np.ndarray):
            R_grid = torch.tensor(R_grid, dtype=torch.float32)
        else:
            R_grid = R_grid.float()

        # Convert grid translation back to metric coordinates
        t = t_grid * self.bev_resolution
        t = torch.tensor([t[0], t[1], 0.0], dtype=torch.float32)  # add z=0

        # Convert rotation back to 3x3 in metric coordinates
        R = torch.eye(3, dtype=torch.float32)
        R[:2, :2] = R_grid

        return R, t


    def plot_ground_truth_correspondence(self, corr_tensor, dim, max_lines=150):
        if corr_tensor.numel() == 0:
            return torch.zeros((3, dim, dim))

        img = np.zeros((dim, dim, 3), dtype=np.uint8)
        step = max(1, corr_tensor.shape[0] // max_lines)
        pts0 = []
        pts1 = []
        for idx in range(0, step):
            i = np.random.randint(0, corr_tensor.shape[0])
            y0, x0 = int(corr_tensor[i, 0] // dim), int(corr_tensor[i, 0] % dim)
            y1, x1 = int(corr_tensor[i, 1] // dim), int(corr_tensor[i, 1] % dim)
            pts0.append([y0, x0])
            pts1.append([y1, x1])
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(img, (x0, y0), (x1, y1), color, 1)
            img[y0, x0, :] = COLOR_T0
            img[y1, x1, :] = COLOR_T1

        pts0 = np.array(pts0) - np.array([dim / 2, dim / 2])
        pts1 = np.array(pts1) - np.array([dim / 2, dim / 2])
        H_mat, mask = self.estimateRigidTransform(pts0, pts1)
        t_rec = np.array([9999.0, 9999.0])
        R_rec = np.eye(2)
        if H_mat is not None:
            H_mat = H_mat / H_mat[2, 2]
            R_rec = H_mat[0:2, 0:2]
            t_rec = H_mat[0:2, 2]

        self.add_text_to_image(img, R_rec, t_rec)

        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

    def draw_maches_correct(self, desc0, desc1, corr_tensor, dim):
        """
        Draw matches between two BEV maps using ground-truth correspondences
        to filter only correct matches.

        FIXED VERSION with proper device handling.
        """
        device = desc0.device  # Get the device from desc0

        # Validate dim_grid is set
        if self.dim_grid is None or self.dim_grid == 0:
            print(f"ERROR: self.dim_grid is not properly initialized: {self.dim_grid}")
            return torch.zeros((3, dim, dim), dtype=torch.float32, device=device)

        desc0_flat = desc0[0].view(desc0.shape[1], -1).T  # [H*W, D]
        desc1_flat = desc1[0].view(desc1.shape[1], -1).T  # [H*W, D]
        desc0_flat = F.normalize(desc0_flat, p=2, dim=1)
        desc1_flat = F.normalize(desc1_flat, p=2, dim=1)

        # FIX: Ensure corr_tensor is on the same device
        corr_tensor = corr_tensor.to(device)

        corr0 = corr_tensor[:, 0]
        corr1 = corr_tensor[:, 1]
        distance_based = True

        if corr0.numel() == 0:
            return torch.zeros((3, dim, dim), dtype=torch.float32, device=device)

        # ========== DESCRIPTOR MATCHES ==========
        sim = torch.matmul(desc0_flat, desc1_flat.T)
        idx0_to_1 = sim.argmax(dim=1)

        if distance_based:
            # ========== MIT MAX. DISTANZ ==========
            # Add bounds checking for indices
            if torch.any(corr0 >= desc0_flat.shape[0]) or torch.any(corr0 < 0):
                print(
                    f"ERROR: Invalid corr0 indices. Max: {corr0.max()}, Min: {corr0.min()}, desc0_flat size: {desc0_flat.shape[0]}")
                return torch.zeros((3, dim, dim), dtype=torch.float32, device=device)

            if torch.any(corr1 >= self.dim_grid * self.dim_grid) or torch.any(corr1 < 0):
                print(
                    f"ERROR: Invalid corr1 indices. Max: {corr1.max()}, Min: {corr1.min()}, grid size: {self.dim_grid * self.dim_grid}")
                return torch.zeros((3, dim, dim), dtype=torch.float32, device=device)

            # FIX: Ensure all operations stay on the same device
            # Convert to long type for indexing operations
            corr0 = corr0.long()
            corr1 = corr1.long()

            # Flattened Indizes -> 2D-Koordinaten (keep on device)
            y0 = corr1 // self.dim_grid
            x0 = corr1 % self.dim_grid

            # Ensure idx0_to_1[corr0] produces valid indices
            pred_indices = idx0_to_1[corr0]

            # Clamp indices to valid range
            pred_indices = torch.clamp(pred_indices, 0, self.dim_grid * self.dim_grid - 1)

            y_pred = pred_indices // self.dim_grid
            x_pred = pred_indices % self.dim_grid

            # Ensure all tensors are float for distance calculation
            x_pred = x_pred.float()
            y_pred = y_pred.float()
            x0 = x0.float()
            y0 = y0.float()

            # Calculate distance (all operations on same device)
            dist_sq = (x_pred - x0) ** 2 + (y_pred - y0) ** 2
            dist_sq = torch.clamp(dist_sq, min=0.0)  # Ensure non-negative
            dist = torch.sqrt(dist_sq)

            radius = 1.5  # z. B. 1–2 Pixel
            is_correct = (dist <= radius)
        else:
            # ========== RICHTIGE MATCHES ==========
            y_target = torch.zeros(self.dim_grid ** 2, device=device)
            is_correct = (idx0_to_1[corr0] == corr1)

        # Nur korrekte Matches
        corr0_correct = corr0[is_correct]
        corr1_correct = corr1[is_correct]

        # FIX: Only convert to CPU when needed for numpy operations
        # 2D-Koordinaten berechnen
        y0 = (corr0_correct // dim).cpu().numpy().astype(int)
        x0 = (corr0_correct % dim).cpu().numpy().astype(int)
        y1 = (corr1_correct // dim).cpu().numpy().astype(int)
        x1 = (corr1_correct % dim).cpu().numpy().astype(int)

        if len(y0) == 0 or len(y1) == 0:
            # nothing above threshold
            return torch.zeros((3, dim, dim), dtype=torch.float32, device=device)

        img = np.zeros((dim, dim, 3), dtype=np.uint8)
        pts0, pts1 = [], []

        for i in range(len(x0)):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(img, (x0[i], y0[i]), (x1[i], y1[i]), color, 1)
            img[y0[i], x0[i], :] = COLOR_T0
            img[y1[i], x1[i], :] = COLOR_T1
            pts0.append([y0[i], x0[i]])
            pts1.append([y1[i], x1[i]])

        pts0 = np.array(pts0) - np.array([dim / 2, dim / 2])
        pts1 = np.array(pts1) - np.array([dim / 2, dim / 2])

        # Transformationsschätzung
        H_mat, mask = self.estimateRigidTransform(pts0, pts1)
        t_rec = np.array([9999.0, 9999.0])
        R_rec = np.eye(2)
        if H_mat is not None:
            H_mat = H_mat / H_mat[2, 2]
            R_rec = H_mat[0:2, 0:2]
            t_rec = H_mat[0:2, 2]

        self.add_text_to_image(img, R_rec, t_rec)

        # FIX: Ensure output is on CPU since it's being converted to numpy
        return (torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0).cpu()

    def draw_vehicle_movement(self, R: torch.Tensor, t: torch.Tensor, dim=180):
        ego_center_img = np.array([dim // 2, dim // 2])
        vehicle_corners_ego = np.array([
            [2.5, 1.0],
            [2.5, -1.0],
            [-2.5, -1.0],
            [-2.5, 1.0],
        ], dtype=np.float32)

        vehicle_corners_lidar = np.array([
            [1.0, 2.5],
            [-1.0, 2.5],
            [-1.0, -2.5],
            [1.0, -2.5],
        ], dtype=np.float32)

        vehicle_corners = vehicle_corners_lidar  # BEV frame matches LiDAR frame

        def bev_to_img_coords(points_bev):
            # BEV (x, y) → image (u, v): u = col = left, v = row = forward
            # Ego at center, forward is up
            u_img = ego_center_img[0] + points_bev[:, 1] / self.bev_resolution
            v_img = ego_center_img[1] + points_bev[:, 0] / self.bev_resolution
            return np.stack([v_img, u_img], axis=1).astype(np.int32)

        img = np.zeros((dim, dim, 3), dtype=np.uint8)
        current_vehicle_img_pts = bev_to_img_coords(vehicle_corners)
        cv2.polylines(img, [current_vehicle_img_pts], isClosed=True, color=COLOR_T0, thickness=1)
        t_bev = t.cpu().numpy()
        R_bev = R[:2, :2].cpu().numpy()

        R_vehicle, t_vehicle = self.grid_to_metric(R_bev, t_bev)
        R_vehicle = R_vehicle[:2, :2].cpu().numpy()
        t_vehicle = t_vehicle.cpu().numpy()



        other_vehicle_in_current_frame_bev = (R_vehicle @ vehicle_corners.T).T + t_vehicle[:2]
        other_vehicle_img_pts = bev_to_img_coords(other_vehicle_in_current_frame_bev)
        cv2.polylines(img, [other_vehicle_img_pts], isClosed=True, color=COLOR_T1, thickness=1)
        # No negation or transpose needed if transformation matches grid convention



        self.add_text_to_image(img, R_bev, t_bev)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        ### SPARSITY ####
        # Methode 1: Stärkere L1 Regularization mit adaptivem Weight
        def sparsity_regularization(self, kpts0, kpts1, num_correspondences):
            """
            Dynamische Sparsity basierend auf Anzahl gefundener Korrespondenzen
            """
            # Bei wenigen Korrespondenzen: weniger Sparsity-Druck
            # Bei vielen Korrespondenzen: starker Sparsity-Druck
            """
            if num_correspondences < 50:
                sparsity_weight = 0.01  # Sehr schwach
            elif num_correspondences < 200:
                sparsity_weight = 0.05  # Schwach
            elif num_correspondences < 500:
                sparsity_weight = 0.2  # Mittel
            else:
                sparsity_weight = 0.5  # Stark
            """
            sparsity_refularization = (kpts0.sigmoid().mean() + kpts1.sigmoid().mean())
            return sparsity_refularization

        # Methode 2: Top-K Sparsity (nur die K besten Keypoints erlauben)
        def topk_sparsity_loss(self, kpts, target_k=150):
            """
            Erlaube nur die top-K keypoints, bestrafe den Rest stark
            """
            kpts_flat = kpts.view(-1)  # [H*W]
            kpts_probs = torch.sigmoid(kpts_flat)

            # Finde die K-th höchste Wahrscheinlichkeit
            if len(kpts_probs) > target_k:
                kth_value, _ = torch.kthvalue(kpts_probs, len(kpts_probs) - target_k + 1)

                # Alle Keypoints unter dem K-th Wert stark bestrafen
                below_threshold_mask = kpts_probs < kth_value
                penalty = torch.sum(kpts_probs[below_threshold_mask])

                return penalty
            else:
                return torch.tensor(0.0, device=kpts.device)

        # Methode 3: Competitive Suppression (Winner-Takes-All in lokalen Patches)
        def local_winner_takes_all_loss(self, kpts, patch_size=8):
            """
            In jedem patch_size x patch_size Bereich darf nur 1 Keypoint aktiv sein
            """
            B, C, H, W = kpts.shape
            kpts_sigmoid = torch.sigmoid(kpts)

            # Unfold zu patches
            patches = F.unfold(kpts_sigmoid, kernel_size=patch_size, stride=patch_size // 2, padding=patch_size // 4)
            patches = patches.view(B, C, patch_size * patch_size, -1)  # [B, C, patch_pixels, num_patches]

            # In jedem Patch: nur das Maximum behalten, Rest bestrafen
            max_vals, max_indices = torch.max(patches, dim=2, keepdim=True)  # [B, C, 1, num_patches]

            # Alle nicht-maximalen Werte bestrafen
            suppression_loss = torch.sum(patches - max_vals.detach()) / (B * C)
            return suppression_loss.clamp(min=0)

        # Methode 4: Entropy-basierte Sparsity
        def entropy_sparsity_loss(self, kpts, target_entropy=0.1):
            """
            Minimiere die Entropie der Keypoint-Verteilung
            """
            kpts_probs = torch.sigmoid(kpts).view(-1)
            kpts_probs = kpts_probs.clamp(1e-8, 1 - 1e-8)  # Numerical stability

            entropy = -torch.mean(kpts_probs * torch.log(kpts_probs) +
                                  (1 - kpts_probs) * torch.log(1 - kpts_probs))

            # Wir wollen niedrige Entropie (wenige, aber sichere Keypoints)
            entropy_loss = torch.relu(entropy - target_entropy)
            return entropy_loss

        # Methode 5: Differenzierbares NMS während Training
        def differentiable_nms_loss(self, kpts, min_distance=5):
            """
            Bestrafe nahe beieinander liegende Keypoints
            """
            B, C, H, W = kpts.shape
            kpts_sigmoid = torch.sigmoid(kpts)

            # Finde lokale Maxima
            max_pool = F.max_pool2d(kpts_sigmoid, kernel_size=min_distance * 2 + 1,
                                    stride=1, padding=min_distance)

            # Punkte, die keine lokalen Maxima sind, bestrafen
            non_maxima_penalty = torch.sum(torch.relu(kpts_sigmoid - max_pool))

            return non_maxima_penalty / (B * C * H * W)

        # Methode 7: Kombinierte Sparsity-Strategie
        def advanced_sparsity_loss(self, kpts0, kpts1, num_correspondences):
            """
            Kombiniert mehrere Sparsity-Techniken
            """
            # Basis L1 Sparsity
            l1_sparsity = self.sparsity_regularization(kpts0, kpts1, num_correspondences)

            # Top-K Constraint (nur 500 beste Keypoints)
            topk_loss0 = self.topk_sparsity_loss(kpts0, target_k=1000)
            topk_loss1 = self.topk_sparsity_loss(kpts1, target_k=1000)

            # Lokale Suppression
            local_suppression0 = self.local_winner_takes_all_loss(kpts0, patch_size=5)
            local_suppression1 = self.local_winner_takes_all_loss(kpts1, patch_size=5)

            # Progressive Gewichtung
            # if not hasattr(self, 'progressive_sparsity'):
            #    self.progressive_sparsity = ProgressiveWeight(0.00, 0.8, 1000)

            prog_weight = 1.0  # self.progressive_sparsity.get_weight(self.step)

            total_sparsity = (
                    l1_sparsity +
                    prog_weight * (topk_loss0 + topk_loss1) +
                    prog_weight * (local_suppression0 + local_suppression1)
            )
            # using: sparsity_loss = total_sparsity
            return total_sparsity




def _subcell_offsets(prob_np, rr, cc):
    """Vektorisierter 3x3-Sigmoid-CoM-Offset (in Zellen) fuer Peak-Koordinaten (rr, cc).
    Sub-Zell-Lokalisierung (R8: "no sub-cell keypoint localization") — Inferenz-only,
    env-gated via KPT_SUBCELL=1. Plateau-Zellen (flaches Fenster) erhalten ~0-Offset."""
    import numpy as np
    p = np.pad(prob_np, 1)
    rr = np.asarray(rr, np.int64); cc = np.asarray(cc, np.int64)
    num_r = np.zeros(len(rr)); num_c = np.zeros(len(rr)); den = np.zeros(len(rr))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            w = p[rr + 1 + dr, cc + 1 + dc]
            num_r += dr * w; num_c += dc * w; den += w
    ok = den > 1e-6
    off_r = np.where(ok, num_r / np.maximum(den, 1e-9), 0.0)
    off_c = np.where(ok, num_c / np.maximum(den, 1e-9), 0.0)
    return off_r, off_c

