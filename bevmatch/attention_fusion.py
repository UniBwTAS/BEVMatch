import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from mmdet3d.registry import MODELS

@MODELS.register_module()
class HardGateDownsampleAttentionFuser(nn.Module):
    """
    SAFE VERSION: Downsample vor attention, standard PyTorch attention
    Keine externen dependencies, garantiert memory-efficient
    """

    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            num_heads: int = 8,
            attn_resolution: int = 32,  # Attention auf 32x32 (statt 180x180!)
            threshold: float = 0.01
    ):
        super().__init__()
        self.threshold = threshold
        self.out_channels = out_channels
        self.attn_resolution = attn_resolution

        # Projections
        self.cam_proj = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lidar_proj = nn.Conv2d(in_channels[1], out_channels, 1)

        # Downsample für attention (180x180 -> 32x32)
        # Das spart MASSIV memory!
        self.cam_down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # 180->90
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # 90->45
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # 45->22
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(attn_resolution)  # Force exact size
        )

        self.lidar_down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(attn_resolution)
        )

        # Standard multi-head attention (SAFE!)
        self.cam2lidar_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=False
        )

        self.lidar2cam_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=False
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        # NACHHER (keine Streifen):
        class UpsampleBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True)
                )

            def forward(self, x):
                # Erst interpolate, dann conv
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                return self.conv(x)

        self.cam_up = nn.Sequential(
            UpsampleBlock(out_channels),  # 32->64
            UpsampleBlock(out_channels),  # 64->128
            UpsampleBlock(out_channels),  # 128->256 (ungefähr 180)
        )
        self.lidar_up = nn.Sequential(
            UpsampleBlock(out_channels),
            UpsampleBlock(out_channels),
            UpsampleBlock(out_channels),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def compute_hard_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Hard gate based on feature magnitude"""
        magnitude = x.abs().mean(dim=(1, 2, 3))
        gate = (magnitude > self.threshold).float()
        return gate.view(-1, 1, 1, 1)

    #def forward(self,cam_bev, lidar_bev) -> torch.Tensor:
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: [cam_bev, lidar_bev]
                cam_bev: [B, C_cam, 180, 180]
                lidar_bev: [B, C_lidar, 180, 180]

        Returns:
            fused: [B, out_channels, 180, 180]
        """
        cam_bev, lidar_bev = inputs
        B, _, H, W = cam_bev.shape  # H=180, W=180

        # 1. Hard gates
        cam_gate = self.compute_hard_gate(cam_bev)
        lidar_gate = self.compute_hard_gate(lidar_bev)

        # 2. Project
        cam_feat = self.cam_proj(cam_bev)  # [B, 256, 180, 180]
        lidar_feat = self.lidar_proj(lidar_bev)

        # 3. Downsample (180x180 -> 32x32)
        cam_down = self.cam_down(cam_feat)  # [B, 256, 32, 32]
        lidar_down = self.lidar_down(lidar_feat)

        _, _, H_attn, W_attn = cam_down.shape  # 32, 32

        # 4. Flatten für attention
        cam_flat = cam_down.flatten(2).permute(2, 0, 1)  # [1024, B, 256]
        lidar_flat = lidar_down.flatten(2).permute(2, 0, 1)

        # 5. Cross-attention (SAFE - nur 1024x1024!)
        cam_attn_out, _ = self.cam2lidar_attn(
            query=self.norm1(cam_flat),
            key=self.norm2(lidar_flat),
            value=lidar_flat
        )
        cam_enhanced = cam_flat + cam_attn_out  # Residual

        lidar_attn_out, _ = self.lidar2cam_attn(
            query=self.norm2(lidar_flat),
            key=self.norm1(cam_flat),
            value=cam_flat
        )
        lidar_enhanced = lidar_flat + lidar_attn_out

        # 6. Reshape back
        cam_enhanced = cam_enhanced.permute(1, 2, 0).reshape(B, -1, H_attn, W_attn)
        lidar_enhanced = lidar_enhanced.permute(1, 2, 0).reshape(B, -1, H_attn, W_attn)

        # 7. Upsample zurück (32x32 -> 180x180)
        cam_enhanced = self.cam_up(cam_enhanced)
        lidar_enhanced = self.lidar_up(lidar_enhanced)

        # 8. Resize falls nicht exakt (wegen stride)
        if cam_enhanced.shape[2:] != (H, W):
            cam_enhanced = F.interpolate(cam_enhanced, size=(H, W), mode='bilinear', align_corners=False)
            lidar_enhanced = F.interpolate(lidar_enhanced, size=(H, W), mode='bilinear', align_corners=False)

        # 9. Apply hard gates
        cam_enhanced = cam_enhanced * cam_gate
        lidar_enhanced = lidar_enhanced * lidar_gate

        # 10. Fusion
        combined = torch.cat([cam_enhanced, lidar_enhanced], dim=1)
        fused = self.fusion(combined)

        return fused


@MODELS.register_module()
class HardGateDownsampleAttentionFuserV2(nn.Module):
    """
    SAFE VERSION: Downsample vor attention, standard PyTorch attention
    Keine externen dependencies, garantiert memory-efficient
    """

    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            num_heads: int = 8,
            threshold: float = 0.01
    ):
        super().__init__()
        self.threshold = threshold
        self.out_channels = out_channels

        # Projections
        self.cam_proj = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lidar_proj = nn.Conv2d(in_channels[1], out_channels, 1)

        # Downsample für attention (180x180 -> 32x32)
        # Das spart MASSIV memory!
        self.cam_down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # 180->90
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # 90->45
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

        self.lidar_down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        # Standard multi-head attention (SAFE!)
        self.cam2lidar_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=False
        )

        self.lidar2cam_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=False
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        # NACHHER (keine Streifen):
        class UpsampleBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True)
                )

            def forward(self, x):
                # Erst interpolate, dann conv
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                return self.conv(x)

        self.cam_up = nn.Sequential(
            UpsampleBlock(out_channels),  # 45->90
            UpsampleBlock(out_channels)  # 90->180
        )
        self.lidar_up = nn.Sequential(
            UpsampleBlock(out_channels),
            UpsampleBlock(out_channels)
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def compute_hard_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Hard gate based on feature magnitude"""
        magnitude = x.abs().mean(dim=(1, 2, 3))
        gate = (magnitude > self.threshold).float()
        return gate.view(-1, 1, 1, 1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: [cam_bev, lidar_bev]
                cam_bev: [B, C_cam, 180, 180]
                lidar_bev: [B, C_lidar, 180, 180]

        Returns:
            fused: [B, out_channels, 180, 180]
        """
        cam_bev, lidar_bev = inputs
        B, _, H, W = cam_bev.shape  # H=180, W=180

        # 1. Hard gates
        cam_gate = self.compute_hard_gate(cam_bev)
        lidar_gate = self.compute_hard_gate(lidar_bev)

        # 2. Project
        cam_feat = self.cam_proj(cam_bev)  # [B, 256, 180, 180]
        lidar_feat = self.lidar_proj(lidar_bev)

        # 3. Downsample (180x180 -> 32x32)
        cam_down = self.cam_down(cam_feat)  # [B, 256, 32, 32]
        lidar_down = self.lidar_down(lidar_feat)

        _, _, H_attn, W_attn = cam_down.shape  # 32, 32

        # 4. Flatten für attention
        cam_flat = cam_down.flatten(2).permute(2, 0, 1)  # [1024, B, 256]
        lidar_flat = lidar_down.flatten(2).permute(2, 0, 1)

        # 5. Cross-attention (SAFE - nur 1024x1024!)
        cam_attn_out, _ = self.cam2lidar_attn(
            query=self.norm1(cam_flat),
            key=self.norm2(lidar_flat),
            value=lidar_flat
        )
        cam_enhanced = cam_flat + cam_attn_out  # Residual

        lidar_attn_out, _ = self.lidar2cam_attn(
            query=self.norm2(lidar_flat),
            key=self.norm1(cam_flat),
            value=cam_flat
        )
        lidar_enhanced = lidar_flat + lidar_attn_out

        # 6. Reshape back
        cam_enhanced = cam_enhanced.permute(1, 2, 0).reshape(B, -1, H_attn, W_attn)
        lidar_enhanced = lidar_enhanced.permute(1, 2, 0).reshape(B, -1, H_attn, W_attn)

        # 7. Upsample zurück (32x32 -> 180x180)
        cam_enhanced = self.cam_up(cam_enhanced)
        lidar_enhanced = self.lidar_up(lidar_enhanced)

        # 8. Resize falls nicht exakt (wegen stride)
        if cam_enhanced.shape[2:] != (H, W):
            cam_enhanced = F.interpolate(cam_enhanced, size=(H, W), mode='bilinear', align_corners=False)
            lidar_enhanced = F.interpolate(lidar_enhanced, size=(H, W), mode='bilinear', align_corners=False)

        # 9. Apply hard gates
        cam_enhanced = cam_enhanced * cam_gate
        lidar_enhanced = lidar_enhanced * lidar_gate

        # 10. Fusion
        combined = torch.cat([cam_enhanced, lidar_enhanced], dim=1)
        fused = self.fusion(combined)

        return fused


class HardGateDownsampleAttentionFuserOld(nn.Module):
    """
    SAFE VERSION: Downsample vor attention, standard PyTorch attention
    Keine externen dependencies, garantiert memory-efficient
    """

    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            num_heads: int = 8,
            threshold: float = 0.01
    ):
        super().__init__()
        self.threshold = threshold
        self.out_channels = out_channels

        # Projections
        self.cam_proj = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lidar_proj = nn.Conv2d(in_channels[1], out_channels, 1)

        self.cam_down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # 180->90
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # 90->45
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.lidar_down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # 180->90
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # 90->45
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        # Standard multi-head attention (SAFE!)
        self.cam2lidar_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=False
        )

        self.lidar2cam_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=False
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        # NACHHER (keine Streifen):
        class UpsampleBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True)
                )

            def forward(self, x):
                # Erst interpolate, dann conv
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                return self.conv(x)

        self.cam_up = nn.Sequential(
            UpsampleBlock(out_channels),  # 45->90
            UpsampleBlock(out_channels),  # 90->180
        )
        self.lidar_up = nn.Sequential(
            UpsampleBlock(out_channels),
            UpsampleBlock(out_channels),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def compute_hard_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Hard gate based on feature magnitude"""
        magnitude = x.abs().mean(dim=(1, 2, 3))
        gate = (magnitude > self.threshold).float()
        return gate.view(-1, 1, 1, 1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: [cam_bev, lidar_bev]
                cam_bev: [B, C_cam, 180, 180]
                lidar_bev: [B, C_lidar, 180, 180]

        Returns:
            fused: [B, out_channels, 180, 180]
        """
        cam_bev, lidar_bev = inputs
        B, _, H, W = cam_bev.shape  # H=180, W=180

        # 1. Hard gates
        cam_gate = self.compute_hard_gate(cam_bev)
        lidar_gate = self.compute_hard_gate(lidar_bev)

        # 2. Project
        cam_feat = self.cam_proj(cam_bev)  # [B, 256, 180, 180]
        lidar_feat = self.lidar_proj(lidar_bev)

        # 3. Downsample (180x180 -> 32x32)
        cam_down = self.cam_down(cam_feat)  # [B, 256, 32, 32]
        lidar_down = self.lidar_down(lidar_feat)

        _, _, H_attn, W_attn = cam_down.shape  # 32, 32
        N_attn = H_attn * W_attn  # 1024 (viel kleiner!)

        # 4. Flatten für attention
        cam_flat = cam_down.flatten(2).permute(2, 0, 1)  # [1024, B, 256]
        lidar_flat = lidar_down.flatten(2).permute(2, 0, 1)

        # 5. Cross-attention (SAFE - nur 1024x1024!)
        cam_attn_out, _ = self.cam2lidar_attn(
            query=self.norm1(cam_flat),
            key=self.norm2(lidar_flat),
            value=lidar_flat
        )
        cam_enhanced = cam_flat + cam_attn_out  # Residual

        lidar_attn_out, _ = self.lidar2cam_attn(
            query=self.norm2(lidar_flat),
            key=self.norm1(cam_flat),
            value=cam_flat
        )
        lidar_enhanced = lidar_flat + lidar_attn_out

        # 6. Reshape back
        cam_enhanced = cam_enhanced.permute(1, 2, 0).reshape(B, -1, H_attn, W_attn)
        lidar_enhanced = lidar_enhanced.permute(1, 2, 0).reshape(B, -1, H_attn, W_attn)

        # 7. Upsample zurück (32x32 -> 180x180)
        cam_enhanced = self.cam_up(cam_enhanced)
        lidar_enhanced = self.lidar_up(lidar_enhanced)

        # 8. Resize falls nicht exakt (wegen stride)
        if cam_enhanced.shape[2:] != (H, W):
            cam_enhanced = F.interpolate(cam_enhanced, size=(H, W), mode='bilinear', align_corners=False)
            lidar_enhanced = F.interpolate(lidar_enhanced, size=(H, W), mode='bilinear', align_corners=False)

        # 9. Apply hard gates
        cam_enhanced = cam_enhanced * cam_gate
        lidar_enhanced = lidar_enhanced * lidar_gate

        # 10. Fusion
        combined = torch.cat([cam_enhanced, lidar_enhanced], dim=1)
        fused = self.fusion(combined)

        return fused


@MODELS.register_module()
class WindowedCrossAttentionFuser(nn.Module):
    """
    DROP-IN REPLACEMENT für HardGateDownsampleAttentionFuser
    Nutzt windowed attention statt downsampling für bessere spatial accuracy
    """

    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            num_heads: int = 8,
            window_size: int = 16,  # Statt attn_resolution
            threshold: float = 0.01
    ):
        super().__init__()
        self.threshold = threshold
        self.out_channels = out_channels
        self.window_size = window_size
        self.num_heads = num_heads

        # Projections (identisch)
        self.cam_proj = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lidar_proj = nn.Conv2d(in_channels[1], out_channels, 1)

        # Windowed attention modules
        self.cam2lidar_attn = WindowedCrossAttention(
            dim=out_channels,
            num_heads=num_heads,
            window_size=window_size
        )

        self.lidar2cam_attn = WindowedCrossAttention(
            dim=out_channels,
            num_heads=num_heads,
            window_size=window_size
        )

        # Final fusion (identisch)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def compute_hard_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Hard gate based on feature magnitude (identisch)"""
        magnitude = x.abs().mean(dim=(1, 2, 3))
        gate = (magnitude > self.threshold).float()
        return gate.view(-1, 1, 1, 1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: [cam_bev, lidar_bev]
                cam_bev: [B, C_cam, 180, 180]
                lidar_bev: [B, C_lidar, 180, 180]

        Returns:
            fused: [B, out_channels, 180, 180]
        """
        cam_bev, lidar_bev = inputs
        B, _, H, W = cam_bev.shape

        # 1. Hard gates (identisch)
        cam_gate = self.compute_hard_gate(cam_bev)
        lidar_gate = self.compute_hard_gate(lidar_bev)

        # 2. Project (identisch)
        cam_feat = self.cam_proj(cam_bev)  # [B, 256, 180, 180]
        lidar_feat = self.lidar_proj(lidar_bev)

        # 3. Windowed cross-attention (ERSETZT downsample+attention+upsample)
        cam_enhanced = self.cam2lidar_attn(
            query_feat=cam_feat,
            kv_feat=lidar_feat
        )  # [B, 256, 180, 180]

        lidar_enhanced = self.lidar2cam_attn(
            query_feat=lidar_feat,
            kv_feat=cam_feat
        )  # [B, 256, 180, 180]

        # 4. Residual connection (wie im original)
        cam_enhanced = cam_feat + cam_enhanced
        lidar_enhanced = lidar_feat + lidar_enhanced

        # 5. Apply hard gates (identisch)
        cam_enhanced = cam_enhanced * cam_gate
        lidar_enhanced = lidar_enhanced * lidar_gate

        # 6. Fusion (identisch)
        combined = torch.cat([cam_enhanced, lidar_enhanced], dim=1)
        fused = self.fusion(combined)

        return fused


class WindowedCrossAttention(nn.Module):
    """
    Optimized windowed cross-attention with fixed reshape logic
    """

    def __init__(self, dim, num_heads=8, window_size=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, query_feat: torch.Tensor, kv_feat: torch.Tensor):
        """
        Args:
            query_feat: [B, C, H, W]
            kv_feat:    [B, C, H, W]
        Returns:
            attn_output: [B, C, H, W]
        """
        B, C, H, W = query_feat.shape
        ws = self.window_size

        # Pad to make divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            query_feat = F.pad(query_feat, (0, pad_w, 0, pad_h))
            kv_feat = F.pad(kv_feat, (0, pad_w, 0, pad_h))

        H_pad, W_pad = query_feat.shape[2:]
        nH, nW = H_pad // ws, W_pad // ws
        num_windows = nH * nW

        # Reshape to windows: [B, C, H, W] -> [B*nH*nW, ws*ws, C]
        # Step 1: [B, C, nH, ws, nW, ws]
        query_win = query_feat.view(B, C, nH, ws, nW, ws)
        kv_win = kv_feat.view(B, C, nH, ws, nW, ws)

        # Step 2: [B, nH, nW, ws, ws, C]
        query_win = query_win.permute(0, 2, 4, 3, 5, 1).contiguous()
        kv_win = kv_win.permute(0, 2, 4, 3, 5, 1).contiguous()

        # Step 3: [B*nH*nW, ws*ws, C]
        query_win = query_win.view(B * num_windows, ws * ws, C)
        kv_win = kv_win.view(B * num_windows, ws * ws, C)

        # Cross-attention per window
        q = self.norm_q(query_win)
        kv = self.norm_kv(kv_win)
        attn_out, _ = self.attn(q, kv, kv)  # [B*nH*nW, ws*ws, C]

        # Reverse reshape: [B*nH*nW, ws*ws, C] -> [B, C, H_pad, W_pad]
        # Step 1: [B, nH, nW, ws, ws, C]
        attn_out = attn_out.view(B, nH, nW, ws, ws, C)

        # Step 2: [B, C, nH, ws, nW, ws]
        attn_out = attn_out.permute(0, 5, 1, 3, 2, 4).contiguous()

        # Step 3: [B, C, H_pad, W_pad]
        attn_out = attn_out.view(B, C, H_pad, W_pad)

        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            attn_out = attn_out[:, :, :H, :W]

        return attn_out