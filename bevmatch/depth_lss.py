# modify from https://github.com/mit-han-lab/bevfusion
from typing import Tuple

import torch
from torch import nn

from mmdet3d.registry import MODELS
from .ops import bev_pool

# Standalone LSS Transform variants with integrated depth models
from typing import Tuple
import torch
from torch import nn
from .ops import bev_pool
from torch.nn import functional as F
# Import depth models from transformers
try:
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
except ImportError:
    AutoModelForDepthEstimation = None
    AutoImageProcessor = None
    print("Warning: transformers not found. Install with: pip install transformers")

# Import base transforms (assuming they exist in your codebase)


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                           for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class BaseViewTransform(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound,
                         dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW))
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW,
                           dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        ys = (
            torch.linspace(0, iH - 1, fH,
                           dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots).view(B, N, 1, 1, 1, 3,
                                          3).matmul(points.unsqueeze(-1)))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if 'extra_rots' in kwargs:
            extra_rots = kwargs['extra_rots']
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3,
                                3).repeat(1, N, 1, 1, 1, 1, 1).matmul(
                                    points.unsqueeze(-1)).squeeze(-1))
        if 'extra_trans' in kwargs:
            extra_trans = kwargs['extra_trans']
            points += extra_trans.view(B, 1, 1, 1, 1,
                                       3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) /
                      self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = ((geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(
        self,
        img,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class LSSTransform(BaseViewTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


class BaseDepthTransform(BaseViewTransform):

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1,
                            *self.image_size).to(points[0].device)

        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0))
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = ((cur_coords[..., 0] < self.image_size[0])
                      & (cur_coords[..., 0] >= 0)
                      & (cur_coords[..., 1] < self.image_size[1])
                      & (cur_coords[..., 1] >= 0))
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth = depth.to(masked_dist.dtype)
                depth[b, c, 0, masked_coords[:, 0],
                      masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class DepthLSSTransform(BaseDepthTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


@MODELS.register_module()
class LSSTransformDepthPro(BaseViewTransform):
    """LSS Transform using DepthPro's metric depth directly.

    Unlike standard LSS which predicts depth distributions, this uses DepthPro's
    metric depth values directly to create the depth distribution. This is more
    accurate because DepthPro provides actual metric depth in meters.

    Key difference from DepthLSS:
    - DepthLSS: Uses sparse LiDAR + learns to densify
    - This: Uses dense metric depth + converts to distribution
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 image_size: Tuple[int, int],
                 feature_size: Tuple[int, int],
                 xbound: Tuple[float, float, float],
                 ybound: Tuple[float, float, float],
                 zbound: Tuple[float, float, float],
                 dbound: Tuple[float, float, float],
                 downsample: int = 1,
                 depthpro_model: str = "apple/DepthPro-hf",
                 max_depth: float = 60.0,
                 use_fp16: bool = True,
                 depth_sigma: float = 1.0) -> None:
        """Initialize LSSTransformDepthProMetric.

        Args:
            depth_sigma: Standard deviation for converting metric depth to distribution
                        (in units of depth bins). Higher = softer distribution.
            hybrid_mode: If True, also learn depth distribution and blend with metric depth.
                        If False, use only metric depth (no learning).
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound
        )

        self.max_depth = max_depth
        self.use_fp16 = use_fp16
        self.image_size = image_size
        self.depth_sigma = depth_sigma

        # Load DepthPro model
        try:
            print(f"Loading DepthPro model: {depthpro_model}")
            dtype = torch.float16 if use_fp16 else torch.float32
            self.depth_pro = AutoModelForDepthEstimation.from_pretrained(
                depthpro_model,
                torch_dtype=dtype,
                use_fov_model=False
            )
            self.depthpro_processor = AutoImageProcessor.from_pretrained(
                depthpro_model,
                use_fast=True
            )

            # Always freeze DepthPro
            for param in self.depth_pro.parameters():
                param.requires_grad = False
            self.depth_pro.eval()

            print(f"DepthPro loaded (max depth: {max_depth}m")
        except ImportError:
            raise ImportError("transformers library required for DepthPro")


        # Pure metric: Only transform features to output channels
        # This is simpler than standard LSS since we don't predict depth
        self.depthnet = nn.Conv2d(in_channels, self.C, 1)

        # Downsampling layer
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_metric_depth(self, images):
        """Get metric depth predictions from DepthPro.

        Args:
            images: Tensor of shape (B*N, 3, H, W) - original RGB images

        Returns:
            depth_batch: Tensor of shape (B*N, 1, H, W) - metric depth in meters
        """
        B, C, H, W = images.shape
        device = images.device

        depth_maps = []

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                for i in range(B):
                    img = images[i:i + 1]
                    img = images[i:i + 1]

                    # Forward through DepthPro
                    dtype = torch.float16 if self.use_fp16 else torch.float32
                    # === PREPARE INPUT ===
                    inputs = self.depthpro_processor(images=img, return_tensors="pt").to(device)
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
                    outputs = self.depth_pro(**inputs)
                    # === POSTPROCESS DEPTH ===

                    # Post-process to get depth at original size
                    post_processed = self.depthpro_processor.post_process_depth_estimation(
                        outputs, target_sizes=[(H, W)]
                    )

                    metric_depth = post_processed[0]["predicted_depth"]

                    # Ensure it's a tensor on correct device
                    if not isinstance(metric_depth, torch.Tensor):
                        metric_depth = torch.from_numpy(metric_depth).to(device)

                    # Ensure correct shape [1, 1, H, W]
                    if metric_depth.dim() == 2:
                        metric_depth = metric_depth.unsqueeze(0).unsqueeze(0)
                    elif metric_depth.dim() == 3:
                        metric_depth = metric_depth.unsqueeze(0)

                    # Clamp to max depth
                    metric_depth = torch.clamp(metric_depth, 0, self.max_depth)
                    depth_maps.append(metric_depth)

        depth_batch = torch.cat(depth_maps, dim=0).to(device).float()
        return depth_batch

    def metric_depth_to_distribution(self, metric_depth, fH, fW):
        """Convert metric depth (meters) to soft distribution over depth bins.

        This is the KEY function - it uses DepthPro's metric predictions directly!

        Args:
            metric_depth: [B*N, 1, H, W] metric depth in meters
            fH, fW: Target feature height and width

        Returns:
            depth_distribution: [B*N, D, fH, fW] soft distribution over depth bins
        """
        device = metric_depth.device
        BN = metric_depth.shape[0]

        # Resize to feature resolution
        if metric_depth.shape[-2:] != (fH, fW):
            metric_depth = F.interpolate(
                metric_depth, size=(fH, fW), mode='bilinear', align_corners=False
            )

        # Create depth bins [dmin, dmin+dstep, ..., dmax]
        d_min, d_max, d_step = self.dbound
        depth_bins = torch.arange(d_min, d_max, d_step, device=device)
        D = len(depth_bins)

        # Reshape for broadcasting
        # depth_bins: [D] -> [1, D, 1, 1]
        # metric_depth: [BN, 1, fH, fW]
        depth_bins = depth_bins.view(1, D, 1, 1)

        # Create Gaussian distribution centered at metric depth
        sigma = self.depth_sigma * d_step  # Scale sigma by bin width

        # Compute Gaussian: exp(-((bin - depth)^2) / (2*sigma^2))
        # Result shape: [BN, D, fH, fW]
        depth_distribution = torch.exp(
            -((depth_bins - metric_depth) ** 2) / (2 * sigma ** 2)
        )

        # Normalize to sum to 1 across depth dimension
        depth_distribution = depth_distribution / (
                depth_distribution.sum(dim=1, keepdim=True) + 1e-6
        )

        return depth_distribution

    def get_cam_feats(self, x):
        """Extract camera features using metric depth from DepthPro.

        Key difference: We use DepthPro's metric depth to CREATE the depth distribution,
        rather than learning to predict it.

        Args:
            x: Image features of shape (B, N, C, fH, fW)

        Returns:
            Camera features with depth information
        """
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)

        # Get original images from stored variable
        # These were passed in forward() and stored in self._current_images
        images = self._current_images.view(B * N, 3, self.image_size[0], self.image_size[1])

        # Get metric depth from DepthPro on ORIGINAL images
        metric_depth = self.get_metric_depth(images)

        # Convert metric depth to depth distribution
        metric_depth_dist = self.metric_depth_to_distribution(metric_depth, fH, fW)

        # Process through network
        x = self.depthnet(x)


        # Pure metric: Use ONLY DepthPro's depth
        depth = metric_depth_dist
        features = x  # All channels are features

        # Apply depth distribution to features
        x = depth.unsqueeze(1) * features.unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, original_images, **kwargs):
        """Forward pass with original images for DepthPro.

        Args:
            images: Original RGB images (B, N, 3, H, W)
            *args, **kwargs: Additional arguments for parent forward
        """
        # Store original images for get_cam_feats
        self._current_images = original_images

        # Call parent forward (which will call get_cam_feats)
        x = super().forward(*args, **kwargs)

        # Apply downsampling
        x = self.downsample(x)

        # Clean up
        self._current_images = None

        return x


@MODELS.register_module()
class LSSTransformDepthProRelative(BaseViewTransform):
    """LSS Transform with integrated Apple DepthPro for metric depth estimation.

    DepthPro provides metric depth directly in meters, which can improve
    the depth estimation in the LSS transform without requiring finetuning.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 image_size: Tuple[int, int],
                 feature_size: Tuple[int, int],
                 xbound: Tuple[float, float, float],
                 ybound: Tuple[float, float, float],
                 zbound: Tuple[float, float, float],
                 dbound: Tuple[float, float, float],
                 downsample: int = 1,
                 depthpro_model: str = "apple/DepthPro-hf",
                 max_depth: float = 60.0,
                 use_fp16: bool = True) -> None:
        """Initialize LSSTransformDepthPro.

        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output BEV channels
            image_size: Original image size (H, W) for DepthPro
            feature_size: Feature map size (fH, fW)
            xbound: X-axis bounds (min, max, resolution)
            ybound: Y-axis bounds (min, max, resolution)
            zbound: Z-axis bounds (min, max, resolution)
            dbound: Depth bounds (min, max, resolution)
            downsample: Downsampling factor for output
            depthpro_model: HuggingFace model name for DepthPro
            max_depth: Maximum depth in meters to clamp predictions
            use_fp16: Whether to use float16 for DepthPro inference
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound
        )

        self.max_depth = max_depth
        self.use_fp16 = use_fp16
        self.image_size = image_size

        # Load DepthPro model
        try:
            print(f"Loading DepthPro model: {depthpro_model}")

            dtype = torch.float16 if use_fp16 else torch.float32
            self.depth_pro = AutoModelForDepthEstimation.from_pretrained(
                depthpro_model,
                torch_dtype=dtype,
                use_fov_model=False  # We don't need FOV estimation
            )

            self.depthpro_processor = AutoImageProcessor.from_pretrained(
                depthpro_model,
                use_fast=True
            )

            # Freeze DepthPro - it's already pretrained for metric depth
            for param in self.depth_pro.parameters():
                param.requires_grad = False
            self.depth_pro.eval()

            print(f"DepthPro loaded and frozen (max depth: {max_depth}m)")
        except ImportError:
            raise ImportError("transformers library required for DepthPro. Install with: pip install transformers")

        # Feature projection for metric depth
        self.depth_feature_proj = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1),  # Changed: removed stride=2
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        # Modified depth network to incorporate metric depth features
        self.depthnet = nn.Conv2d(in_channels + 128, self.D + self.C, 1)

        # Downsampling layer
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_metric_depth(self, images):
        """Get metric depth predictions from DepthPro on original images.
        Args:
            images: Tensor of shape (B*N, 3, H, W) - original RGB images

        Returns:
            depth_batch: Tensor of shape (B*N, 1, H, W) - metric depth in meters
        """

        B, C, H, W = images.shape
        device = images.device

        depth_maps = []

        with torch.no_grad():
            # Use automatic mixed precision if fp16 is enabled
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                # TODO: DepthPro can do batch inference - optimize later
                for i in range(B):
                    img = images[i:i + 1]


                    # Forward through DepthPro
                    dtype = torch.float16 if self.use_fp16 else torch.float32
                    # === PREPARE INPUT ===
                    inputs = self.depthpro_processor(images=img, return_tensors="pt").to(device)
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
                    outputs = self.depth_pro(**inputs)

                    print(outputs)
                    # === POSTPROCESS DEPTH ===
                    post_processed = self.depthpro_processor.post_process_depth_estimation(
                        outputs, target_sizes=[(H, W)]
                    )
                    predicted_depth = post_processed[0]["predicted_depth"].squeeze().detach().cpu().numpy()
                    print(predicted_depth)

                    #print min max
                    print(f"DepthPro output min: {outputs.predicted_depth.min().item()}, max: {outputs.predicted_depth.max().item()}")
                    print(f"predicted_depth min: {predicted_depth.min()}, max: {predicted_depth.max()}")
                    #write outputs to file as image
                    # =====================
                                        #import torchvision
                    #torchvision.utils.save_image(outputs.predicted_depth, path)
                    #print(f"DepthPro output saved to {path}")


                    # Get metric depth in meters
                    metric_depth = outputs.predicted_depth

                    # Ensure correct shape [1, 1, H, W]
                    if metric_depth.dim() == 2:
                        metric_depth = metric_depth.unsqueeze(0).unsqueeze(0)
                    elif metric_depth.dim() == 3:
                        metric_depth = metric_depth.unsqueeze(1)

                    # Clamp to max depth
                    metric_depth = torch.clamp(metric_depth, 0, self.max_depth)
                    depth_maps.append(metric_depth)

        # Concatenate and convert to float32 for further processing
        depth_batch = torch.cat(depth_maps, dim=0).to(device).float()

        return depth_batch

    def get_cam_feats(self, x):
        """Extract camera features with metric depth from DepthPro.

        Args:
            x: Image features of shape (B, N, C, fH, fW)
            images: Original images of shape (B, N, 3, H, W) - REQUIRED for DepthPro

        Returns:
            Camera features with depth information
        """

        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)

        # Reshape original images for batch processing
        images = self._current_images.view(B * N, 3, self.image_size[0], self.image_size[1])

        # Get metric depth from DepthPro on ORIGINAL IMAGES
        metric_depth = self.get_metric_depth(images)

        # Resize depth to match feature size
        if metric_depth.shape[-2:] != (fH, fW):
            metric_depth = torch.nn.functional.interpolate(
                metric_depth, size=(fH, fW), mode='bilinear', align_corners=False
            )

        # Project metric depth to feature space
        depth_features = self.depth_feature_proj(metric_depth)

        # Concatenate with image features
        x = torch.cat([x, depth_features], dim=1)

        # Process through depth network
        x = self.depthnet(x)
        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, original_images, **kwargs):
        """Forward pass with original images for DepthPro.

        Args:
            images: Original RGB images (B, N, 3, H, W)
            *args, **kwargs: Additional arguments for parent forward
        """
        # Store original images for get_cam_feats
        self._current_images = original_images

        # Call parent forward (which will call get_cam_feats)
        x = super().forward(*args, **kwargs)

        # Apply downsampling
        x = self.downsample(x)

        # Clean up
        self._current_images = None

        return x



@MODELS.register_module()
class LSSTransformDepthAnything(BaseViewTransform):
    """LSS Transform with DepthAnything for relative depth estimation.

    DepthAnything has NO minimum size requirements and works well with small images.
    It provides high-quality relative depth that can be used directly or scaled.

    Advantages over DepthPro:
    - Works with ANY image size (no 1536x1536 minimum)
    - Faster inference
    - More flexible
    - Can be finetuned easily

    Structure matches LSSTransformDepthPro for easy replacement.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 image_size: Tuple[int, int],
                 feature_size: Tuple[int, int],
                 xbound: Tuple[float, float, float],
                 ybound: Tuple[float, float, float],
                 zbound: Tuple[float, float, float],
                 dbound: Tuple[float, float, float],
                 downsample: int = 1,
                 da_model: str = "LiheYoung/depth-anything-small-hf",
                 freeze_depth: bool = True,
                 depth_scale: float = 1.0) -> None:
        """Initialize LSSTransformDepthAnything.

        Args:
            da_model: HuggingFace model name for DepthAnything
                Options:
                - 'LiheYoung/depth-anything-small-hf' (fastest, 24.8M params)
                - 'LiheYoung/depth-anything-base-hf' (balanced, 97.5M params)
                - 'LiheYoung/depth-anything-large-hf' (best quality, 335.3M params)
            freeze_depth: Whether to freeze DepthAnything weights
            depth_scale: Scale factor for relative depth values
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound
        )

        self.freeze_depth = freeze_depth
        self.depth_scale = depth_scale
        self.image_size = image_size

        # Load DepthAnything model
        try:
            print(f"Loading DepthAnything model: {da_model}")
            self.depth_anything = AutoModelForDepthEstimation.from_pretrained(da_model)
            self.da_processor = AutoImageProcessor.from_pretrained(da_model)

            if self.freeze_depth:
                for param in self.depth_anything.parameters():
                    param.requires_grad = False
                self.depth_anything.eval()
                print("DepthAnything frozen for inference")
            else:
                print("DepthAnything unfrozen - will train with gradients")
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers")

        # Lightweight depth feature projection (no stride=2!)
        self.depth_feature_proj = nn.Sequential(
            nn.Conv2d(1, 32, 1),  # 1x1 conv for efficiency
            nn.ReLU(True),
        )

        # Modified depth network to incorporate depth features
        self.depthnet = nn.Conv2d(in_channels + 32, self.D + self.C, 1)

        # Downsampling layer
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_depth_predictions(self, images):
        """Get depth predictions from DepthAnything.

        Args:
            images: Tensor of shape (B*N, 3, H, W) - original RGB images

        Returns:
            depth: Tensor of shape (B*N, 1, H, W) - relative depth predictions
        """
        B, C, H, W = images.shape
        device = images.device

        # DepthAnything works with ANY size - no restrictions!

        # Normalize to [0, 1] if needed
        if images.max() > 1.0:
            images_norm = images / 255.0
        else:
            images_norm = images

        # Forward through DepthAnything
        if self.freeze_depth:
            with torch.no_grad():
                outputs = self.depth_anything(pixel_values=images_norm)
        else:
            outputs = self.depth_anything(pixel_values=images_norm)

        predicted_depth = outputs.predicted_depth

        # Ensure correct shape [B, 1, H, W]
        if predicted_depth.dim() == 3:
            predicted_depth = predicted_depth.unsqueeze(1)

        # Apply depth scale (to adjust relative depth range)
        predicted_depth = predicted_depth * self.depth_scale

        return predicted_depth

    def get_cam_feats(self, x):
        """Extract camera features with depth from DepthAnything.

        Args:
            x: Image features of shape (B, N, C, fH, fW)

        Returns:
            Camera features with depth information
        """
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)

        # Get original images from stored variable
        images = self._current_images.view(B * N, 3, self.image_size[0], self.image_size[1])

        # Get depth from DepthAnything on ORIGINAL images
        depth = self.get_depth_predictions(images)

        # Resize depth to match feature size
        if depth.shape[-2:] != (fH, fW):
            depth = torch.nn.functional.interpolate(
                depth, size=(fH, fW), mode='bilinear', align_corners=False
            )

        # Normalize depth to [0, 1] range for better feature interaction
        # DepthAnything outputs relative depth, so normalize by its range
        depth_min = depth.amin(dim=(2, 3), keepdim=True)
        depth_max = depth.amax(dim=(2, 3), keepdim=True)
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)

        # Project depth to feature space
        depth_features = self.depth_feature_proj(depth_normalized)

        # Concatenate with image features
        x = torch.cat([x, depth_features], dim=1)

        # Process through depth network - outputs depth distribution over bins
        x = self.depthnet(x)
        depth_dist = x[:, :self.D].softmax(dim=1)
        x = depth_dist.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, img, points, lidar2image, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, metas, original_images=None, **kwargs):
        """Forward pass - modified to use original images.

        Args:
            img: Extracted image features (NOT original images!)
            original_images: Original RGB images (B, N, 3, H, W) - REQUIRED
            ... (other standard LSS arguments)
        """
        if original_images is None:
            raise ValueError("original_images must be provided for DepthAnything depth estimation!")

        # Store original images for get_cam_feats to access
        self._current_images = original_images

        # Call parent forward - it will call our get_cam_feats()
        x = super().forward(
            img, points, lidar2image, camera_intrinsics, camera2lidar,
            img_aug_matrix, lidar_aug_matrix, metas, **kwargs
        )

        # Apply downsampling
        x = self.downsample(x)

        # Clean up
        self._current_images = None

        return x


@MODELS.register_module()
class DepthLSSTransformDepthPro(BaseViewTransform):
    """DepthLSS-style transform using DepthPro dense metric depth instead of LiDAR.

    This is the exact same architecture as DepthLSSTransform, but:
    - Instead of sparse LiDAR depth projections → uses dense DepthPro metric depth
    - DepthPro provides depth everywhere (not just sparse points)
    - Uses the same dtransform + depthnet architecture as DepthLSS

    Advantages:
    - Dense depth everywhere (no sparse LiDAR gaps)
    - Metric depth in meters (like LiDAR)
    - No LiDAR calibration issues

    Disadvantages:
    - DepthPro requires padding for small images (1536x1536 minimum)
    - Less accurate than real LiDAR
    - Adds computation overhead
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            image_size: Tuple[int, int],
            feature_size: Tuple[int, int],
            xbound: Tuple[float, float, float],
            ybound: Tuple[float, float, float],
            zbound: Tuple[float, float, float],
            dbound: Tuple[float, float, float],
            downsample: int = 1,
            depthpro_model: str = "apple/DepthPro-hf",
            max_depth: float = 60.0,
            use_fp16: bool = True,
    ) -> None:
        """Initialize DepthLSSTransformDepthPro.

        Args:
            depthpro_model: HuggingFace model for DepthPro
            max_depth: Maximum depth to clamp predictions
            use_fp16: Use float16 for efficiency
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )

        self.max_depth = max_depth
        self.use_fp16 = use_fp16
        self.image_size = image_size

        # Load DepthPro
        try:
            print(f"Loading DepthPro as dense depth source: {depthpro_model}")

            dtype = torch.float16 if use_fp16 else torch.float32
            self.depth_pro = AutoModelForDepthEstimation.from_pretrained(
                depthpro_model,
                torch_dtype=dtype,
                use_fov_model=False
            )

            self.depthpro_processor = AutoImageProcessor.from_pretrained(
                depthpro_model,
                use_fast=True
            )

            # Freeze DepthPro
            for param in self.depth_pro.parameters():
                param.requires_grad = False
            self.depth_pro.eval()

            print(f"DepthPro loaded as dense depth source (max: {max_depth}m)")
        except ImportError:
            raise ImportError("transformers required for DepthPro")

        # Depth transform network - SAME as DepthLSS
        # But now processing dense depth instead of sparse LiDAR
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # Combine depth features with image features - SAME as DepthLSS
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )

        # Downsampling
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_dense_metric_depth(self, images):
        """Get dense metric depth from DepthPro for all cameras.

        Args:
            images: [B, N, 3, H, W] original RGB images

        Returns:
            depth: [B, N, 1, H, W] dense metric depth in meters
        """
        B, N, C, H, W = images.shape
        device = images.device

        # Reshape for processing
        images = images.view(B * N, C, H, W)

        depth_maps = []

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                for i in range(B * N):
                    img = images[i:i + 1]

                    # Normalize to [0, 1]
                    if img.max() > 1.0:
                        img = img / 255.0

                    # Convert to numpy for processor (handles resizing automatically)
                    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    # Use processor - handles DepthPro size requirements
                    inputs = self.depthpro_processor(
                        images=img_np,
                        return_tensors="pt"
                    ).to(device)

                    dtype = torch.float16 if self.use_fp16 else torch.float32
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

                    # Get depth predictions
                    outputs = self.depth_pro(**inputs)

                    # Post-process to get depth at original size
                    post_processed = self.depthpro_processor.post_process_depth_estimation(
                        outputs, target_sizes=[(H, W)]
                    )
                    metric_depth = post_processed[0]["predicted_depth"]

                    # Ensure it's a tensor on correct device
                    if not isinstance(metric_depth, torch.Tensor):
                        metric_depth = torch.from_numpy(metric_depth).to(device)

                    # Ensure shape [1, 1, H, W]
                    if metric_depth.dim() == 2:
                        metric_depth = metric_depth.unsqueeze(0).unsqueeze(0)
                    elif metric_depth.dim() == 3:
                        metric_depth = metric_depth.unsqueeze(0)

                    # Apply scale and clamp
                    metric_depth = torch.clamp(metric_depth, 0, self.max_depth)

                    depth_maps.append(metric_depth)

        # Concatenate all depth maps
        full_depth = torch.cat(depth_maps, dim=0).to(device).float()

        # Reshape back to [B, N, 1, H, W]
        full_depth = full_depth.view(B, N, 1, H, W)

        return full_depth

    def get_cam_feats(self, x, d):
        """Process camera features with dense depth from DepthPro.

        This is IDENTICAL to DepthLSS's get_cam_feats, but:
        - d is now DENSE depth from DepthPro (not sparse LiDAR)

        Args:
            x: [B, N, C, fH, fW] image features
            d: [B, N, 1, H, W] dense depth from DepthPro

        Returns:
            Camera features with depth
        """
        B, N, C, fH, fW = x.shape

        # Reshape for processing
        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        # Transform depth through network (same as DepthLSS)
        d = self.dtransform(d)

        # Combine with image features (same as DepthLSS)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        # Depth distribution and features (same as DepthLSS)
        depth_dist = x[:, :self.D].softmax(dim=1)
        x = depth_dist.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        # Reshape back
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def forward(self, img, points, lidar2image, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, metas, original_images=None, **kwargs):
        """Forward pass using DepthPro instead of LiDAR projections.

        Args:
            img: Extracted image features
            original_images: [B, N, 3, H, W] - REQUIRED for DepthPro
            points: LiDAR points (IGNORED - we use DepthPro instead)
            ... (other standard LSS arguments)

        Note: The 'points' argument is ignored as we use DepthPro instead of LiDAR.
        """
        if original_images is None:
            raise ValueError("original_images required for DepthPro!")

        # Get dense depth from DepthPro instead of projecting LiDAR points
        dense_depth = self.get_dense_metric_depth(original_images)

        # Get geometry (same as BaseViewTransform)
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # Get camera features with dense depth (same as DepthLSS)
        x = self.get_cam_feats(img, dense_depth)

        # BEV pooling (same as BaseViewTransform)
        x = self.bev_pool(geom, x)

        # Downsampling
        x = self.downsample(x)

        return x