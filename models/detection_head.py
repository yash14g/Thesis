"""
3D Detection Head
=================
Takes fused BEV features → predicts 3D bounding boxes.

Anchor-free, centre-point-based detection head.
Inspired by CenterPoint — simplified for lightweight deployment.

Outputs per BEV cell:
    - heatmap   : (num_classes,) — object centre probability
    - offset    : (2,)           — sub-voxel xy offset
    - height    : (1,)           — z centre
    - size      : (3,)           — w, l, h
    - rotation  : (2,)           — sin(yaw), cos(yaw)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


NUM_CLASSES = 10  # nuScenes detection classes


# ------------------------------------------------------------------
# Shared BEV neck (lightweight FPN-style upsampling)
# ------------------------------------------------------------------
class BEVNeck(nn.Module):
    """
    Lightweight neck to upsample and refine BEV features before heads.
    No heavy FPN — just a simple 2-layer refinement.
    """
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.neck(x)


# ------------------------------------------------------------------
# Single task head
# ------------------------------------------------------------------
class TaskHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
        )

    def forward(self, x):
        return self.conv(x)


# ------------------------------------------------------------------
# Full detection head
# ------------------------------------------------------------------
class DetectionHead(nn.Module):
    """
    Anchor-free 3D detection head.

    Input : (B, C, BH, BW) — fused BEV features
    Output: dict of prediction maps
        "heatmap"  : (B, num_classes, BH, BW)
        "offset"   : (B, 2, BH, BW)
        "height"   : (B, 1, BH, BW)
        "size"     : (B, 3, BH, BW)
        "rotation" : (B, 2, BH, BW)
    """
    def __init__(self,
                 in_channels:  int = 64,
                 num_classes:  int = NUM_CLASSES):
        super().__init__()
        self.neck     = BEVNeck(in_channels, in_channels)
        self.heatmap  = TaskHead(in_channels, num_classes)
        self.offset   = TaskHead(in_channels, 2)
        self.height   = TaskHead(in_channels, 1)
        self.size     = TaskHead(in_channels, 3)
        self.rotation = TaskHead(in_channels, 2)

        # initialise heatmap bias for focal loss stability
        nn.init.constant_(self.heatmap.conv[-1].bias, -2.19)

    def forward(self, x):
        x = self.neck(x)
        return {
            "heatmap":  self.heatmap(x).sigmoid(),
            "offset":   self.offset(x),
            "height":   self.height(x),
            "size":     self.size(x).exp(),       # predict log(size)
            "rotation": self.rotation(x),
        }


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------
def gaussian_focal_loss(pred: torch.Tensor,
                         gt:   torch.Tensor,
                         alpha: float = 2.0,
                         beta:  float = 4.0) -> torch.Tensor:
    """
    Focal loss for heatmap prediction (CornerNet/CenterPoint style).
    pred, gt: (B, C, H, W) in [0, 1]
    """
    pos_mask  = gt.eq(1).float()
    neg_mask  = 1.0 - pos_mask
    neg_wt    = torch.pow(1.0 - gt, beta)

    pos_loss  = torch.log(pred + 1e-6) * torch.pow(1.0 - pred, alpha) * pos_mask
    neg_loss  = torch.log(1.0 - pred + 1e-6) * torch.pow(pred, alpha) * neg_wt * neg_mask

    num_pos   = pos_mask.sum().clamp(min=1)
    loss      = -(pos_loss + neg_loss).sum() / num_pos
    return loss


def reg_l1_loss(pred: torch.Tensor,
                target: torch.Tensor,
                mask:   torch.Tensor) -> torch.Tensor:
    """
    L1 regression loss applied only at positive (object centre) locations.
    pred, target: (B, C, H, W)
    mask:         (B, 1, H, W) binary
    """
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    num  = mask.sum().clamp(min=1)
    return loss / num


class DetectionLoss(nn.Module):
    """
    Combined loss for all detection head outputs.
    Weights from CenterPoint paper.
    """
    def __init__(self,
                 heatmap_weight: float = 1.0,
                 offset_weight:  float = 1.0,
                 height_weight:  float = 1.0,
                 size_weight:    float = 0.5,
                 rot_weight:     float = 1.0):
        super().__init__()
        self.w = {
            "heatmap":  heatmap_weight,
            "offset":   offset_weight,
            "height":   height_weight,
            "size":     size_weight,
            "rotation": rot_weight,
        }

    def forward(self, preds: dict, targets: dict) -> dict:
        """
        preds   : output dict from DetectionHead
        targets : dict with same keys + "pos_mask" (B,1,H,W)
        """
        mask = targets["pos_mask"]

        losses = {}
        losses["heatmap"]  = gaussian_focal_loss(preds["heatmap"],
                                                  targets["heatmap"])
        losses["offset"]   = reg_l1_loss(preds["offset"],
                                          targets["offset"], mask)
        losses["height"]   = reg_l1_loss(preds["height"],
                                          targets["height"], mask)
        losses["size"]     = reg_l1_loss(preds["size"],
                                          targets["size"],   mask)
        losses["rotation"] = reg_l1_loss(preds["rotation"],
                                          targets["rotation"], mask)

        total = sum(self.w[k] * v for k, v in losses.items())
        losses["total"] = total
        return losses


# ------------------------------------------------------------------
# Post-processing: decode predictions to 3D boxes
# ------------------------------------------------------------------
def decode_predictions(preds:       dict,
                        bev_config: dict,
                        score_thresh: float = 0.3,
                        max_preds:    int   = 500) -> list:
    """
    Decode heatmap + regression predictions into 3D bounding boxes.

    Returns list of dicts per batch item:
        {"boxes": (N,7), "scores": (N,), "labels": (N,)}
    """
    B          = preds["heatmap"].shape[0]
    x_min, x_max = bev_config["x_range"]
    y_min, y_max = bev_config["y_range"]
    vox          = bev_config["voxel_size"]

    results = []

    for b in range(B):
        hm     = preds["heatmap"][b]   # (C, H, W)
        offset = preds["offset"][b]    # (2, H, W)
        height = preds["height"][b]    # (1, H, W)
        size   = preds["size"][b]      # (3, H, W)
        rot    = preds["rotation"][b]  # (2, H, W)

        C, H, W = hm.shape
        scores, labels = hm.max(dim=0)  # (H, W) each

        # flatten and threshold
        scores_flat = scores.reshape(-1)
        labels_flat = labels.reshape(-1)

        keep = scores_flat > score_thresh
        if keep.sum() == 0:
            results.append({
                "boxes":  torch.zeros(0, 7),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
            })
            continue

        idx      = torch.where(keep)[0]
        # limit predictions
        if len(idx) > max_preds:
            topk = scores_flat[idx].topk(max_preds).indices
            idx  = idx[topk]

        yi = (idx // W).float()
        xi = (idx %  W).float()

        # convert grid coords to metric space
        bx = (xi + offset[0].reshape(-1)[idx]) * vox + x_min
        by = (yi + offset[1].reshape(-1)[idx]) * vox + y_min
        bz = height[0].reshape(-1)[idx]

        bw = size[0].reshape(-1)[idx]
        bl = size[1].reshape(-1)[idx]
        bh = size[2].reshape(-1)[idx]

        sin_yaw = rot[0].reshape(-1)[idx]
        cos_yaw = rot[1].reshape(-1)[idx]
        yaw     = torch.atan2(sin_yaw, cos_yaw)

        boxes = torch.stack([bx, by, bz, bw, bl, bh, yaw], dim=-1)

        results.append({
            "boxes":  boxes,
            "scores": scores_flat[idx],
            "labels": labels_flat[idx],
        })

    return results


# ------------------------------------------------------------------
# Unit test
# ------------------------------------------------------------------
if __name__ == "__main__":
    head  = DetectionHead(in_channels=64, num_classes=10)
    dummy = torch.randn(2, 64, 200, 200)
    preds = head(dummy)

    for k, v in preds.items():
        print(f"  {k:10s} : {v.shape}")

    params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\n  DetectionHead params: {params:,}")
    print("DetectionHead OK")
