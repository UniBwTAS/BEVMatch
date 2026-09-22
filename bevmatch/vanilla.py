from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from mmdet3d.registry import MODELS
from datetime import datetime
import torch.distributed as dist
from mmengine.dist import get_rank  # 0 without a process group, e.g. single-GPU training


__all__ = ["BEVSegmentationHeadVanilla"]


def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


@MODELS.register_module()
class BEVSegmentationHeadVanilla(nn.Module):
    def __init__(
            self,
            in_channels: int,
            grid_transform: Dict[str, Any],
            classes: List[str],
            loss: str,
            vis_interval: int = 1000,  # Visualize every N iterations
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss_f = loss
        self.vis_interval = vis_interval
        self.iter_count = 0
        self.transform = BEVGridTransform(**grid_transform)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes), 1),
        )

        # Define distinct colors for each class (RGB format)
        self.class_colors = self._get_class_colors(len(classes))
        #get random number for logging path
        number = np.random.randint(0, 100000)
        datestring = datetime.now().strftime("%Y%m%d_%H%M")
        self.writer = SummaryWriter("./work_dirs/seg_debug")


    def _get_class_colors(self, num_classes: int) -> torch.Tensor:
        """Generate distinct colors for each class."""
        colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
            [0, 255, 128],  # Spring green
            [255, 128, 128],  # Light red
        ]
        # Extend if more classes
        while len(colors) < num_classes:
            colors.append([
                torch.randint(0, 256, (1,)).item(),
                torch.randint(0, 256, (1,)).item(),
                torch.randint(0, 256, (1,)).item()
            ])
        return torch.tensor(colors[:num_classes], dtype=torch.float32)

    def _visualize_segmentation(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            global_step: int,
    ) -> None:
        """Create colored segmentation visualization.

        Args:
            predictions: (B, C, H, W) prediction logits after sigmoid
            targets: (B, C, H, W) ground truth masks
            writer: TensorBoard writer
            global_step: Current training iteration
        """
        # Only visualize first sample in batch
        pred = predictions[0]  # (C, H, W)
        target = targets[0]  # (C, H, W)

        # Move to CPU and threshold predictions
        pred = (pred > 0.5).float().cpu()  # (C, H, W)
        target = target.float().cpu()  # (C, H, W)

        # Get colors
        colors = self.class_colors  # (C, 3)

        # Create RGB images: (H, W, 3)
        H, W = pred.shape[1], pred.shape[2]

        # Initialize with black background
        pred_rgb = torch.zeros(H, W, 3)
        target_rgb = torch.zeros(H, W, 3)

        # Overlay each class with its color
        for class_idx in range(len(self.classes)):
            class_mask_pred = pred[class_idx]  # (H, W)
            class_mask_target = target[class_idx]  # (H, W)

            # Add color where mask is active
            for c in range(3):  # RGB channels
                pred_rgb[:, :, c] += class_mask_pred * colors[class_idx, c]
                target_rgb[:, :, c] += class_mask_target * colors[class_idx, c]

        # Clip to valid range [0, 255] and normalize to [0, 1]
        pred_rgb = torch.clamp(pred_rgb, 0, 255) / 255.0
        target_rgb = torch.clamp(target_rgb, 0, 255) / 255.0

        # Convert to (3, H, W) for TensorBoard
        pred_rgb = pred_rgb.permute(2, 0, 1)  # (3, H, W)
        target_rgb = target_rgb.permute(2, 0, 1)  # (3, H, W)

        # Create side-by-side comparison
        comparison = torch.cat([target_rgb, pred_rgb], dim=2)  # (3, H, 2*W)

        # Add to TensorBoard
        self.writer.add_image(
            'segmentation/'+str(get_rank())+'ground_truth_vs_prediction',
            comparison,
            global_step=global_step
        )

        # Also log individual class predictions
        log_individual=False
        if log_individual:
            for class_idx, class_name in enumerate(self.classes):
                # Create 3-channel visualization for this class
                class_pred = pred[class_idx].unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
                class_target = target[class_idx].unsqueeze(0).repeat(3, 1, 1)

                class_comparison = torch.cat([class_target, class_pred], dim=2)
                self.writer.add_image(
                    f'segmentation/class_{class_name}',
                    class_comparison,
                    global_step=global_step
                )

    def _extract_feat(self, x):
        """Extract feature from input, handling both list and tensor."""
        if isinstance(x, (list, tuple)):
            # If it's a list of tensors, take the first one
            # or concatenate/process them as needed
            x = x[0]
        return x

    def forward(
            self,
            x: Union[torch.Tensor, List[torch.Tensor]],
            batch_input_metas: Optional[List[Dict]] = None,
            **kwargs,
    ) -> torch.Tensor:
        """Forward pass that returns predictions."""
        x = self._extract_feat(x)
        x = self.transform(x)
        x = self.classifier(x)
        return torch.sigmoid(x)

    def loss(
            self,
            x: Union[torch.Tensor, List[torch.Tensor]],
            batch_input_metas: List[Dict],
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Calculate loss given predictions and ground truth from batch_input_metas."""
        # Extract ground truth masks from batch_input_metas
        target_list = []
        for meta in batch_input_metas:
            gt_mask = meta['gt_masks_bev']  # Shape: (num_classes, H, W)
            # Convert to tensor if it's numpy
            if isinstance(gt_mask, np.ndarray):
                gt_mask = torch.from_numpy(gt_mask)
            target_list.append(gt_mask)

        # Stack into batch: (B, num_classes, H, W)
        target = torch.stack(target_list).to(
            x[0].device if isinstance(x, list) else x.device
        ).float()

        # Get predictions
        x = self._extract_feat(x)
        x = self.transform(x)
        x = self.classifier(x)
        predictions = torch.sigmoid(x)

        # Visualize periodically
        if self.training and self.iter_count % self.vis_interval == 0:
            self._visualize_segmentation(
                predictions.detach(),
                target,
                self.iter_count
            )

        self.iter_count += 1

        # Calculate losses per class
        losses = {}
        for index, name in enumerate(self.classes):
            if self.loss_f == "xent":
                loss = sigmoid_xent_loss(x[:, index], target[:, index])
            elif self.loss_f == "focal":
                loss = sigmoid_focal_loss(x[:, index], target[:, index])
            else:
                raise ValueError(f"unsupported loss: {self.loss_f}")
            losses[f"loss_{name}_{self.loss_f}"] = loss

        return losses

    def predict(
            self,
            x: Union[torch.Tensor, List[torch.Tensor]],
            batch_input_metas: List[Dict],
            **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """Prediction function for inference."""
        predictions = self.forward(x, batch_input_metas)

        self._visualize_segmentation(
            predictions.detach(),
            predictions.detach(),
            self.iter_count
        )

        self.iter_count += 1


        results = []
        for i in range(predictions.shape[0]):
            result = {
                'bev_seg': predictions[i],  # (num_classes, H, W)
            }
            results.append(result)

        return results