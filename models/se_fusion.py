"""
Cross-Modal Squeeze-and-Excitation Fusion Module
=================================================
YOUR MAIN CONTRIBUTION.

Core idea:
    LiDAR BEV features → global context (squeeze)
                       → channel weights (excitation)
                       → recalibrate camera BEV features
                       → fuse recalibrated camera + LiDAR


    LiDAR has precise geometric structure — it knows where objects are
    Camera has rich semantics — but some channels are more relevant than
    others depending on the scene geometry.

    By letting LiDAR decide which camera channels to amplify or suppress,
    we get geometry guided semantic fusion without expensive cross attention.

Architecture:
    LiDAR BEV  (B, C, H, W)  ──→  Global Avg Pool  - (B, C)
                                   FC bottleneck     - (B, C)
                                   Sigmoid           - (B, C, 1, 1) weights
                                        ↓
    Camera BEV (B, C, H, W)  ──→  × weights  →  recalibrated camera
                                        ↓
    Fusion:  concat([recalibrated_camera, lidar])  →  conv  →  (B, C, H, W)

Variants implemented:
    1. CrossModalSEFusion      — LiDAR guides Camera (your primary proposal)
    2. BidirectionalSEFusion   — mutual recalibration (ablation variant)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# Core SE block (single direction)
class SqueezeExciteBlock(nn.Module):
    """
    SE block for cross modal guidance.

    source_feats - squeeze (global avg pool) → excitation (FC bottleneck)
                → channel weights for target_feats
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)

        self.squeeze = nn.AdaptiveAvgPool2d(1)  # global context [(B, C, H, W) --> (B, C, 1, 1)]

        self.excitation = nn.Sequential(
            nn.Flatten(), #(B, C)
            nn.Linear(channels, mid, bias=False), #reduce to (B, mid)
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False), # expand back to (B,C)
            nn.Sigmoid(),
        )
      

    def forward(self, source, target):
        """
        source : (B, C, H, W)  — provides the channel guidance
        target : (B, C, H, W)  — gets recalibrated

        Returns: recalibrated target (B, C, H, W)
        """
        # squeeze: distil global context from source
        gap     = self.squeeze(source)       # (B, C, 1, 1)
        weights = self.excitation(gap)       # (B, C)
        weights = weights.view(-1, weights.shape[1], 1, 1)  # (B, C, 1, 1)
        # helps to brodacst to spatial diomensions
        recalibrated = target * weights
        # excite- reweight target channels
        return target + recalibrated             # (B, C, H, W)



# Main contribution: CrossModalSEFusion

class CrossModalSEFusion(nn.Module):
    """
    LiDAR-guided camera feature recalibration + fusion.

    This is your primary proposed module.

    Steps:
        1. LiDAR features squeeze to global context vector
        2. FC bottleneck produces per-channel weights
        3. Camera features are recalibrated by those weights
        4. Recalibrated camera + LiDAR are concatenated
        5. 1x1 conv reduces back to original channel count

    Computational cost
        - 2 FC layers (tiny)
        - 1 conv layer (1x1,)
        - No attention maps, no cross-attention, no transformers
    """

    def __init__(self, channels: int, reduction: int = 4):
        """
        channels  : number of channels in both BEV feature maps 
        reduction : bottleneck ratio for SE (default 4 → C/4 hidden units)
        """
        super().__init__()
        self.lidar_to_camera_se = SqueezeExciteBlock(channels, reduction)
        self.camera_bn = nn.BatchNorm2d(channels)
        self.lidar_bn = nn.BatchNorm2d(channels)
        # fusion conv: concat(recalib_camera, lidar) - channels
        ## (B, 2C, H, W)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),# reduces 2c -> c
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, lidar_bev, camera_bev):
        """
        lidar_bev  : (B, C, H, W)  — LiDAR BEV features
        camera_bev : (B, C, H, W)  — Camera BEV features

        Returns    : (B, C, H, W)  — fused BEV features
        """
        # step 1+2+3: LiDAR recalibrates camera 
        ## camera is adaptively filtered according to LiDAR context
        lidar_bev = self.lidar_bn(lidar_bev)
        camera_bev = self.camera_bn(camera_bev)

        recalib_camera = self.lidar_to_camera_se(
        source=lidar_bev,
        target=camera_bev
        ) # (B, C, H, W)

        # step 4: concatenate
        fused = torch.cat([recalib_camera, lidar_bev], dim=1)  # (B, 2C, H, W)

        # step 5: project back to C channels
        fused = self.fusion_conv(fused)  # (B, C, H, W) -- final feature map for detection head

        return fused

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# Ablation variant: BidirectionalSEFusion

class BidirectionalSEFusion(nn.Module):
    """
    Mutual recalibration — both modalities guide each other.
    Use this as an ablation to compare against CrossModalSEFusion.

    Slightly heavier but may capture complementary context better.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.lidar_to_camera = SqueezeExciteBlock(channels, reduction)
        self.camera_to_lidar = SqueezeExciteBlock(channels, reduction)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, lidar_bev, camera_bev):
        recalib_camera = self.lidar_to_camera(source=lidar_bev,  target=camera_bev)
        recalib_lidar  = self.camera_to_lidar(source=camera_bev, target=lidar_bev)

        fused = torch.cat([recalib_camera, recalib_lidar], dim=1)
        return self.fusion_conv(fused)



# Ablation variant: SimpleConcatFusion (no attention baseline)

class SimpleConcatFusion(nn.Module):
    """
    Naive concatenation fusion — no attention.
    Use as the weakest baseline in your ablation study.

    If CrossModalSEFusion > SimpleConcatFusion → y SE module helps.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, lidar_bev, camera_bev):
        fused = torch.cat([camera_bev, lidar_bev], dim=1)
        return self.fusion_conv(fused)


#  unit test

if __name__ == "__main__":
    B, C, H, W = 2, 64, 200, 200

    lidar_bev  = torch.randn(B, C, H, W)
    camera_bev = torch.randn(B, C, H, W)

    print("=" * 50)
    print("Testing CrossModalSEFusion") # propsed
    se_fusion = CrossModalSEFusion(channels=C, reduction=4)
    out = se_fusion(lidar_bev, camera_bev)
    print(f"  Input  : lidar {lidar_bev.shape}, camera {camera_bev.shape}")
    print(f"  Output : {out.shape}")
    print(f"  Params : {se_fusion.count_parameters():,}")
    assert out.shape == (B, C, H, W)

    print("\nTesting BidirectionalSEFusion (ablation)")
    bi_fusion = BidirectionalSEFusion(channels=C)
    out2 = bi_fusion(lidar_bev, camera_bev)
    print(f"  Output : {out2.shape}")

    print("\nTesting SimpleConcatFusion (baseline)")
    cat_fusion = SimpleConcatFusion(channels=C)
    out3 = cat_fusion(lidar_bev, camera_bev)
    print(f"  Output : {out3.shape}")

    print("\nAll fusion modules OK")
