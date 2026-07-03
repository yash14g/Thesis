"""
LightFusion — Full Model
========================
Assembles all components into one end-to-end network.

Architecture summary:
    Camera Branch:
        6x images → ImageBackbone → DepthHead → FrustumPooler → CameraBEV (B,C,H,W)

    LiDAR Branch:
        BEV voxel map → LiDARBEVEncoder → LiDARBEV (B,C,H,W)

    Fusion (YOUR CONTRIBUTION):
        LiDARBEV + CameraBEV → CrossModalSEFusion → FusedBEV (B,C,H,W)

    Detection:
        FusedBEV → DetectionHead → {heatmap, offset, height, size, rotation}
"""

import torch
import torch.nn as nn

from models.camera_bev_encoder import CameraBEVEncoder
from models.lidar_bev_encoder  import LiDARBEVEncoder
from models.se_fusion          import (CrossModalSEFusion,
                                        BidirectionalSEFusion,
                                        SimpleConcatFusion)
from models.detection_head     import DetectionHead, DetectionLoss, decode_predictions


# ------------------------------------------------------------------
# Fusion mode options (for ablation studies)
# ------------------------------------------------------------------
FUSION_MODES = {
    "se_unidirectional": CrossModalSEFusion,    # your main contribution
    "se_bidirectional":  BidirectionalSEFusion, # ablation variant
    "concat":            SimpleConcatFusion,    # baseline
    "lidar_only":        None,                  # LiDAR-only baseline
    "camera_only":       None,                  # camera-only baseline
}


# ------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------
class LightFusionNet(nn.Module):
    """
    Lightweight camera-LiDAR BEV fusion network for 3D object detection.

    Args:
        feat_channels : shared channel dim across all modules
        depth_bins    : discrete depth bins for frustum lifting
        bev_h, bev_w  : BEV grid resolution
        num_classes   : number of detection categories
        fusion_mode   : one of FUSION_MODES keys
        se_reduction  : SE bottleneck ratio
    """

    def __init__(self,
                 feat_channels: int  = 64,
                 depth_bins:    int  = 64,
                 bev_h:         int  = 200,
                 bev_w:         int  = 200,
                 num_classes:   int  = 10,
                 fusion_mode:   str  = "se_unidirectional",
                 se_reduction:  int  = 4):
        super().__init__()

        self.fusion_mode   = fusion_mode
        self.feat_channels = feat_channels
        self.bev_h         = bev_h
        self.bev_w         = bev_w

        # ---- Camera branch ------------------------------------------
        if fusion_mode != "lidar_only":
            self.camera_encoder = CameraBEVEncoder(
                feat_channels=feat_channels,
                depth_bins=depth_bins,
                bev_h=bev_h,
                bev_w=bev_w,
            )

        # ---- LiDAR branch -------------------------------------------
        if fusion_mode != "camera_only":
            self.lidar_encoder = LiDARBEVEncoder(
                in_channels=4,
                base_channels=32,
                out_channels=feat_channels,
            )

        # ---- Fusion module -- proposed
        if fusion_mode == "se_unidirectional":
            self.fusion = CrossModalSEFusion(feat_channels, se_reduction)
        elif fusion_mode == "se_bidirectional":
            self.fusion = BidirectionalSEFusion(feat_channels, se_reduction)
        elif fusion_mode == "concat":
            self.fusion = SimpleConcatFusion(feat_channels)
        elif fusion_mode in ("lidar_only", "camera_only"):
            self.fusion = None
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        # ---- Detection head 
        self.det_head = DetectionHead(
            in_channels=feat_channels,
            num_classes=num_classes,
        )

        # ---- Loss
        self.criterion = DetectionLoss()

    # -----------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        """
        batch keys:
            "bev_lidar"  : (B, 4, BH, BW)
            "images"     : (B, 6, 3, H, W)      — optional
            "cam2ego"    : (B, 6, 4, 4)          — optional
            "intrinsics" : (B, 6, 3, 3)          — optional

        Returns dict with "preds" and optionally "loss"
        """

        # ---- Camera BEV features ------------------------------------
        if self.fusion_mode != "lidar_only":
            camera_bev = self.camera_encoder(
                batch["images"],
                batch["cam2ego"],
                batch["intrinsics"],
            )  # (B, C, H, W)

        # ---- LiDAR BEV features -------------------------------------
        if self.fusion_mode != "camera_only":
            lidar_bev = self.lidar_encoder(
                batch["bev_lidar"]
            )  # (B, C, H, W)

        # ---- Fusion -------------------------------------------------
        if self.fusion_mode == "lidar_only":
            fused = lidar_bev
        elif self.fusion_mode == "camera_only":
            fused = camera_bev
        else:
            fused = self.fusion(lidar_bev, camera_bev)  # (B, C, H, W)

        # ---- Detection ----------------------------------------------
        preds = self.det_head(fused)

        return preds

    # -----------------------------------------------------------------
    def compute_loss(self, preds: dict, targets: dict) -> dict:
        return self.criterion(preds, targets)

    # -----------------------------------------------------------------
    def count_parameters(self):
        total    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        sections = {}

        if hasattr(self, "camera_encoder"):
            sections["camera_encoder"] = sum(
                p.numel() for p in self.camera_encoder.parameters() if p.requires_grad)
        if hasattr(self, "lidar_encoder"):
            sections["lidar_encoder"] = sum(
                p.numel() for p in self.lidar_encoder.parameters() if p.requires_grad)
        if self.fusion is not None:
            sections["fusion_module"] = sum(
                p.numel() for p in self.fusion.parameters() if p.requires_grad)
        sections["det_head"] = sum(
            p.numel() for p in self.det_head.parameters() if p.requires_grad)

        return {"total": total, **sections}



# Convenience factory functions 

def build_model(fusion_mode: str = "se_unidirectional",
                feat_channels: int = 64,
                num_classes: int = 10) -> LightFusionNet:
    ## a LightFusionNet with sensible defaults
    return LightFusionNet(
        feat_channels=feat_channels,
        depth_bins=64,
        bev_h=200,
        bev_w=200,
        num_classes=num_classes,
        fusion_mode=fusion_mode,
    )



# Unit test

if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # dummy batch
    B = 2
    dummy_batch = {
        "bev_lidar":  torch.randn(B, 4, 200, 200).to(device),
        "images":     torch.randn(B, 6, 3, 224, 224).to(device),
        "cam2ego":    torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 6, 4, 4).to(device),
        "intrinsics": (torch.eye(3) * 500).unsqueeze(0).unsqueeze(0).expand(B, 6, 3, 3).to(device),
    }

    for mode in ["concat", "se_unidirectional", "lidar_only"]:
        print(f"--- fusion_mode: {mode} ---")
        model = build_model(fusion_mode=mode).to(device)

        t0    = time.time()
        with torch.no_grad():
            preds = model(dummy_batch)
        elapsed = (time.time() - t0) * 1000

        params = model.count_parameters()
        print(f"  Total params : {params['total']:,}")
        print(f"  Inference    : {elapsed:.1f} ms")
        print(f"  Heatmap shape: {preds['heatmap'].shape}")
        print()

    print("LightFusionNet OK")
