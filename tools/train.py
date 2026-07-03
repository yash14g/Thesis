"""
Training Script
===============
Trains LightFusionNet on nuScenes-mini.

Usage (Colab):
    from train import Trainer, TrainConfig
    cfg     = TrainConfig(dataroot="/content/drive/MyDrive/nuscenes")
    trainer = Trainer(cfg)
    trainer.train()
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from data.nuscenes_loader import NuScenesBEVDataset, BEV_CONFIG
from models.model import build_model, LightFusionNet
from models.detection_head import decode_predictions



# Config

@dataclass
class TrainConfig:
    # data
    dataroot:      str   = "/Users/yashgupta14/Downloads/bevfusion_ugv/Data_set/v1.0-mini"
    version:       str   = "v1.0-mini"
    val_fraction:  float = 0.2

    # model
    fusion_mode:   str   = "se_unidirectional"   
    feat_channels: int   = 64
    num_classes:   int   = 10
    bev_h:         int   = 200
    bev_w:         int   = 200

    # training
    epochs:        int   = 5
    batch_size:    int   = 2          # small for Colab GPU RAM
    lr:            float = 2e-4
    weight_decay:  float = 1e-4
    grad_clip:     float = 10.0

    # runtime
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers:   int   = 2
    save_dir:      str   = "checkpoints"
    log_every:     int   = 10         # log every N batches


# ------------------------------------------------------------------
# Target builder — converts GT boxes to dense BEV maps for loss
# ------------------------------------------------------------------
def build_targets(gt_boxes, gt_labels, bev_h, bev_w, bev_cfg, num_classes, device):
    """
    Converts list of GT boxes → dense BEV target maps for loss computation.

    Returns dict:
        heatmap  : (B, num_classes, BH, BW)
        offset   : (B, 2, BH, BW)
        height   : (B, 1, BH, BW)
        size     : (B, 3, BH, BW)
        rotation : (B, 2, BH, BW)
        pos_mask : (B, 1, BH, BW)
    """
    B         = len(gt_boxes)
    x_min, x_max = bev_cfg["x_range"]
    y_min, y_max = bev_cfg["y_range"]
    vox          = bev_cfg["voxel_size"]

    heatmap  = torch.zeros(B, num_classes, bev_h, bev_w, device=device)
    offset   = torch.zeros(B, 2, bev_h, bev_w, device=device)
    height_t = torch.zeros(B, 1, bev_h, bev_w, device=device)
    size_t   = torch.zeros(B, 3, bev_h, bev_w, device=device)
    rot_t    = torch.zeros(B, 2, bev_h, bev_w, device=device)
    pos_mask = torch.zeros(B, 1, bev_h, bev_w, device=device)

    for b in range(B):
        boxes  = gt_boxes[b].to(device)   # (N, 7)
        labels = gt_labels[b].to(device)  # (N,)

        if boxes.shape[0] == 0:
            continue

        for i in range(boxes.shape[0]):
            x, y, z, w, l, h, yaw = boxes[i]
            cls = labels[i].item()

            # convert to BEV grid coords
            xi = ((x - x_min) / vox).long()
            yi = ((y - y_min) / vox).long()

            if not (0 <= xi < bev_w and 0 <= yi < bev_h):
                continue

            # Gaussian heatmap splash (radius 2)
            radius = max(2, int((w.item() + l.item()) / (4 * vox)))
            _draw_gaussian(heatmap[b, cls], yi.item(), xi.item(), radius)

            # regression targets at centre cell
            x_off = (x / vox) - xi.float()
            y_off = (y / vox) - yi.float()
            offset[b, 0, yi, xi]   = x_off
            offset[b, 1, yi, xi]   = y_off
            height_t[b, 0, yi, xi] = z
            size_t[b, 0, yi, xi]   = w
            size_t[b, 1, yi, xi]   = l
            size_t[b, 2, yi, xi]   = h
            rot_t[b, 0, yi, xi]    = torch.sin(yaw)
            rot_t[b, 1, yi, xi]    = torch.cos(yaw)
            pos_mask[b, 0, yi, xi] = 1.0

    return {
        "heatmap":  heatmap,
        "offset":   offset,
        "height":   height_t,
        "size":     size_t,
        "rotation": rot_t,
        "pos_mask": pos_mask,
    }


def _draw_gaussian(heatmap, cy, cx, radius):
    """Draws a 2D Gaussian blob on a heatmap in-place."""
    H, W   = heatmap.shape
    sigma  = radius / 3.0
    x_grid = torch.arange(W, device=heatmap.device, dtype=torch.float32)
    y_grid = torch.arange(H, device=heatmap.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
    gauss  = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    torch.maximum(heatmap, gauss, out=heatmap)


# ------------------------------------------------------------------
# Collate function
# ------------------------------------------------------------------
def collate_fn(batch):
    images    = torch.stack([b["images"]    for b in batch])
    bev_lidar = torch.stack([b["bev_lidar"] for b in batch])
    gt_boxes  = [b["gt_boxes"]  for b in batch]
    gt_labels = [b["gt_labels"] for b in batch]
    tokens    = [b["token"]     for b in batch]

    # dummy calibration (real calibration loading is done in full pipeline)
    B = images.shape[0]
    cam2ego    = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 6, 4, 4)
    intrinsics = (torch.eye(3) * 500).unsqueeze(0).unsqueeze(0).expand(B, 6, 3, 3)

    return {
        "images":     images,
        "bev_lidar":  bev_lidar,
        "cam2ego":    cam2ego,
        "intrinsics": intrinsics,
        "gt_boxes":   gt_boxes,
        "gt_labels":  gt_labels,
        "tokens":     tokens,
    }



# Trainer

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)

        # ---- dataset 
        full_dataset = NuScenesBEVDataset(
            dataroot=cfg.dataroot,
            version=cfg.version,
        )
        n_val   = int(len(full_dataset) * cfg.val_fraction)
        n_train = len(full_dataset) - n_val
        self.train_ds, self.val_ds = random_split(full_dataset, [n_train, n_val])

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # ---- model 
        self.model = build_model(
            fusion_mode=cfg.fusion_mode,
            feat_channels=cfg.feat_channels,
            num_classes=cfg.num_classes,
        ).to(cfg.device)

        param_info = self.model.count_parameters()
        print(f"\nModel: LightFusionNet [{cfg.fusion_mode}]")
        for k, v in param_info.items():
            print(f"  {k:20s} : {v:,}")

        # ---- optimiser 
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=cfg.epochs
        )

        self.history = {"train_loss": [], "val_loss": []}

    # -----------------------------------------------------------------
    def train(self):
        print(f"\nTraining on {self.cfg.device} for {self.cfg.epochs} epochs")
        print(f"Train samples: {len(self.train_ds)} | Val: {len(self.val_ds)}\n")

        best_val = float("inf")

        for epoch in range(1, self.cfg.epochs + 1):
            t0         = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss   = self._val_epoch()
            self.scheduler.step()

            elapsed = time.time() - t0
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch:3d}/{self.cfg.epochs} | "
                  f"train: {train_loss:.4f} | "
                  f"val: {val_loss:.4f} | "
                  f"lr: {self.scheduler.get_last_lr()[0]:.2e} | "
                  f"time: {elapsed:.1f}s")

            # save best
            if val_loss < best_val:
                best_val = val_loss
                self._save_checkpoint(epoch, val_loss, tag="best")

        # save final
        self._save_checkpoint(self.cfg.epochs, val_loss, tag="final")
        print(f"\nTraining complete. Best val loss: {best_val:.4f}")
        return self.history

    # -----------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for i, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.cfg.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            preds   = self.model(batch)
            targets = build_targets(
                batch["gt_boxes"], batch["gt_labels"],
                self.cfg.bev_h, self.cfg.bev_w,
                BEV_CONFIG, self.cfg.num_classes, self.cfg.device,
            )

            losses = self.model.compute_loss(preds, targets)
            loss   = losses["total"]

            self.optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimiser.step()

            total_loss += loss.item()

            if (i + 1) % self.cfg.log_every == 0:
                print(f"  [{epoch}][{i+1}/{len(self.train_loader)}] "
                      f"loss={loss.item():.4f} "
                      f"(hm={losses['heatmap'].item():.3f} "
                      f"sz={losses['size'].item():.3f})")

        return total_loss / len(self.train_loader)

    # -----------------------------------------------------------------
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.cfg.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                preds   = self.model(batch)
                targets = build_targets(
                    batch["gt_boxes"], batch["gt_labels"],
                    self.cfg.bev_h, self.cfg.bev_w,
                    BEV_CONFIG, self.cfg.num_classes, self.cfg.device,
                )
                losses     = self.model.compute_loss(preds, targets)
                total_loss += losses["total"].item()

        return total_loss / max(len(self.val_loader), 1)

    ####
    def _save_checkpoint(self, epoch: int, loss: float, tag: str = ""):
        path = os.path.join(self.cfg.save_dir,
                            f"lightfusion_{self.cfg.fusion_mode}_{tag}.pth")
        torch.save({
            "epoch":       epoch,
            "val_loss":    loss,
            "fusion_mode": self.cfg.fusion_mode,
            "state_dict":  self.model.state_dict(),
            "optimiser":   self.optimiser.state_dict(),
        }, path)
        print(f"  → Saved checkpoint: {path}")
    
if __name__ == "__main__":
    cfg = TrainConfig()
    trainer = Trainer(cfg)
    trainer.train()
