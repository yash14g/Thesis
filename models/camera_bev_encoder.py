"""
Camera BEV Encoder
==================
Lifts 2D image features into 3D Bird's-Eye View space.
Inspired by Lift-Splat-Shoot (LSS) — simplified for lightweight deployment.

Pipeline:
    6x (3, H, W) images
        → shared CNN backbone  → (6, C, h, w) feature maps
        → depth prediction     → (6, D, h, w) depth distributions
        → frustum lifting      → (6, C, D, h, w) frustum features
        → voxel pooling (splat)→ (C, BH, BW) camera BEV features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0



# Lightweight image backbone (EfficientNet-B0 encoder)

class ImageBackbone(nn.Module):
    """
    Extracts feature maps from each camera image.
    Uses EfficientNet-B0 up to the second last layer.
    Output: (B*6, C_out, H/8, W/8)
    """
    def __init__(self, out_channels: int = 64):
        super().__init__()
        base       = efficientnet_b0(pretrained=True)
        # keep layers up to stage 5 (stride 16 total)
        self.stem  = base.features[:5]   # → 80 channels
        self.proj  = nn.Conv2d(80, out_channels, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B*6, 3, H, W)
        x = self.stem(x)
        x = F.relu(self.bn(self.proj(x)))
        return x  # (B*6, C_out, h, w)


# Depth prediction head

class DepthHead(nn.Module):
    """
    Predicts discrete depth distribution over D bins.
    Each pixel is associated with a probability over depth values.
    """
    def __init__(self, in_channels: int, depth_bins: int = 64):
        super().__init__()
        self.D    = depth_bins
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, depth_bins, 1),
        )

    def forward(self, x):
        return self.head(x).softmax(dim=1)  # (B*6, D, h, w)


# ------------------------------------------------------------------
# Frustum → BEV voxel pooling (simplified splat)
# ------------------------------------------------------------------
class FrustumPooler(nn.Module):
    """
    Projects image features through depth distribution into BEV.

    For each pixel (u,v) with depth probs p(d):
        3D point = back-project using camera intrinsics + extrinsics
        splat feature * p(d) into the BEV voxel it falls into

    Simplified: we use a learnable grid_sample-based approach
    instead of a custom CUDA kernel for portability.
    """
    def __init__(self,
                 feat_channels: int,
                 depth_bins:    int   = 64,
                 x_range:       tuple = (-50, 50),
                 y_range:       tuple = (-50, 50),
                 bev_h:         int   = 200,
                 bev_w:         int   = 200):
        super().__init__()
        self.C     = feat_channels
        self.D     = depth_bins
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

        # depth bin centres (metres)
        self.register_buffer(
            "depth_bins",
            torch.linspace(1.0, 60.0, depth_bins)
        )

    def forward(self, feats, depth_probs, cam2ego, intrinsics):
        """
        feats       : (B, 6, C, h, w)
        depth_probs : (B, 6, D, h, w)
        cam2ego     : (B, 6, 4, 4)   camera-to-ego extrinsics
        intrinsics  : (B, 6, 3, 3)   camera intrinsics

        Returns     : (B, C, bev_h, bev_w)
        """
        B, N, C, fh, fw = feats.shape
        device = feats.device

        bev_feats = torch.zeros(B, C, self.bev_h, self.bev_w, device=device)

        # build pixel grid in image space
        ys, xs = torch.meshgrid(
            torch.arange(fh, device=device, dtype=torch.float32),
            torch.arange(fw, device=device, dtype=torch.float32),
            indexing="ij"
        )
        # (h, w, 2) — homogeneous pixel coords
        ones   = torch.ones_like(xs)
        pixels = torch.stack([xs, ys, ones], dim=-1)  # (h, w, 3)

        for b in range(B):
            cam_bev = torch.zeros(C, self.bev_h, self.bev_w, device=device)
            cam_cnt = torch.zeros(1, self.bev_h, self.bev_w, device=device)

            for n in range(N):
                K_inv = torch.inverse(intrinsics[b, n])  # (3,3)
                E     = cam2ego[b, n]                    # (4,4)

                # un-project pixels to camera coords at unit depth
                # pixels: (h*w, 3) → (3, h*w)
                pix_flat = pixels.reshape(-1, 3).T  # (3, h*w)
                cam_pts  = K_inv @ pix_flat          # (3, h*w)

                # expand over depth bins
                depths   = self.depth_bins           # (D,)
                # (3, D, h*w) — 3D points at each depth
                cam_3d   = cam_pts.unsqueeze(1) * depths.view(1, -1, 1)
                ones_3d  = torch.ones(1, self.D, fh * fw, device=device)
                cam_3d_h = torch.cat([cam_3d, ones_3d], dim=0)  # (4, D, h*w)

                # transform to ego frame
                ego_3d   = (E @ cam_3d_h.reshape(4, -1)).reshape(4, self.D, fh * fw)

                # project onto BEV grid
                ex = ego_3d[0]  # (D, h*w)
                ey = ego_3d[1]  # (D, h*w)

                bev_x = ((ex - self.x_min) / (self.x_max - self.x_min) * self.bev_w).long()
                bev_y = ((ey - self.y_min) / (self.y_max - self.y_min) * self.bev_h).long()

                valid = (
                    (bev_x >= 0) & (bev_x < self.bev_w) &
                    (bev_y >= 0) & (bev_y < self.bev_h)
                )  # (D, h*w)

                # weighted feature accumulation
                # feat: (C, h, w) → (C, h*w)
                f_flat = feats[b, n].reshape(C, -1)  # (C, h*w)
                # depth weights: (D, h*w)
                dw     = depth_probs[b, n].reshape(self.D, -1)

                for d in range(self.D):
                    v_mask = valid[d]  # (h*w)
                    if v_mask.sum() == 0:
                        continue
                    bx = bev_x[d][v_mask]
                    by = bev_y[d][v_mask]
                    w  = dw[d][v_mask]               # (M,)
                    f  = f_flat[:, v_mask]           # (C, M)
                    weighted = f * w.unsqueeze(0)    # (C, M)

                    # scatter add into BEV grid
                    idx = by * self.bev_w + bx       # (M,)
                    bev_flat = cam_bev.reshape(C, -1)
                    bev_flat.scatter_add_(1, idx.unsqueeze(0).expand(C, -1), weighted)
                    cam_bev  = bev_flat.reshape(C, self.bev_h, self.bev_w)

                    cnt_flat = cam_cnt.reshape(1, -1)
                    cnt_flat.scatter_add_(1, idx.unsqueeze(0),
                                          w.unsqueeze(0))
                    cam_cnt  = cnt_flat.reshape(1, self.bev_h, self.bev_w)

            # normalise by count
            safe_cnt        = cam_cnt.clamp(min=1e-6)
            bev_feats[b]    = cam_bev / safe_cnt

        return bev_feats  # (B, C, bev_h, bev_w)


# ------------------------------------------------------------------
# Full Camera BEV Encoder
# ------------------------------------------------------------------
class CameraBEVEncoder(nn.Module):
    """
    End-to-end: (B, 6, 3, H, W) → (B, C, bev_h, bev_w)
    """
    def __init__(self,
                 feat_channels: int = 64,
                 depth_bins:    int = 64,
                 bev_h:         int = 200,
                 bev_w:         int = 200):
        super().__init__()
        self.backbone   = ImageBackbone(out_channels=feat_channels)
        self.depth_head = DepthHead(feat_channels, depth_bins)
        self.pooler     = FrustumPooler(feat_channels, depth_bins,
                                        bev_h=bev_h, bev_w=bev_w)

    def forward(self, images, cam2ego, intrinsics):
        """
        images     : (B, 6, 3, H, W)
        cam2ego    : (B, 6, 4, 4)
        intrinsics : (B, 6, 3, 3)
        """
        B, N, C, H, W = images.shape
        imgs_flat  = images.reshape(B * N, C, H, W)

        feats      = self.backbone(imgs_flat)              # (B*6, F, h, w)
        depth      = self.depth_head(feats)                # (B*6, D, h, w)

        _, F, fh, fw = feats.shape
        D            = depth.shape[1]

        feats = feats.reshape(B, N, F, fh, fw)
        depth = depth.reshape(B, N, D, fh, fw)

        bev = self.pooler(feats, depth, cam2ego, intrinsics)  # (B, F, bev_h, bev_w)
        return bev
