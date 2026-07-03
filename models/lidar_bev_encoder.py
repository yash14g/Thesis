"""
Lightweight LiDAR BEV Encoder
==============================
Your core contribution component 1 of 2.

Takes the raw BEV voxel map (4 channels: density, mean_z, max_z, intensity)
and produces a rich spatial feature map using ONLY shallow convolutions.

Design goals:
    - No transformers
    - No heavy backbone
    - Minimal parameter count
    - Fast inference on edge GPUs
    - Preserves spatial detail for later SE fusion

Architecture:
    Input  (4, H, W)
        → ConvBlock x3  (progressively richer channels)
        → Residual refinement
        → Output (C_out, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Basic building blocks
# ------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution.
    Replaces a 3x3 conv with a depthwise + pointwise.
    Roughly 8-9x fewer FLOPs for the same feature transformation.
    Used to keep the LiDAR encoder truly lightweight.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = ConvBNReLU(in_ch, in_ch,  3, stride=stride, groups=in_ch)
        self.pw = ConvBNReLU(in_ch, out_ch, 1, padding=0)

    def forward(self, x):
        return self.pw(self.dw(x))


class LightResBlock(nn.Module):
    """
    Lightweight residual block using depthwise separable convolutions.
    Adds identity skip connection to prevent feature degradation.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.bn    = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        return F.relu(out + residual, inplace=True)


# ------------------------------------------------------------------
# LiDAR BEV Encoder
# ------------------------------------------------------------------
class LiDARBEVEncoder(nn.Module):
    """
    Lightweight LiDAR BEV feature encoder.

    Input  : (B, 4, BH, BW)   — raw BEV voxel map
    Output : (B, C_out, BH, BW) — spatial LiDAR BEV features

    Parameter count target: < 500K
    """

    def __init__(self,
                 in_channels:  int = 4,
                 base_channels: int = 32,
                 out_channels:  int = 64):
        super().__init__()

        # Stage 1: initial feature extraction
        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, base_channels, kernel=3),
            DepthwiseSeparableConv(base_channels, base_channels),
        )

        # Stage 2: increase channel depth, maintain resolution
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels * 2),
            LightResBlock(base_channels * 2),
        )

        # Stage 3: further refinement
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 2, out_channels),
            LightResBlock(out_channels),
            LightResBlock(out_channels),
        )

        # Final projection
        self.final = ConvBNReLU(out_channels, out_channels, kernel=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (B, 4, BH, BW) — raw BEV voxel map
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final(x)
        return x  # (B, C_out, BH, BW)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------------
# Quick unit test
# ------------------------------------------------------------------
if __name__ == "__main__":
    encoder = LiDARBEVEncoder(in_channels=4, base_channels=32, out_channels=64)
    dummy   = torch.randn(2, 4, 200, 200)
    out     = encoder(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {encoder.count_parameters():,}")
    assert out.shape == (2, 64, 200, 200), "Shape mismatch!"
    print("LiDARBEVEncoder OK")
