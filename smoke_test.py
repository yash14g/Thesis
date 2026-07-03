
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

from data.nuscenes_loader import NuScenesBEVDataset
from tools.train import collate_fn
from models.model import build_model

DATA_ROOT = "/Users/yashgupta14/Downloads/bevfusion_ugv/Data_set/v1.0-mini"

dataset = NuScenesBEVDataset(dataroot=DATA_ROOT, version="v1.0-mini")
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

batch = next(iter(loader))

print("Images shape:", batch["images"].shape)
print("LiDAR BEV shape:", batch["bev_lidar"].shape)
print("GT boxes:", batch["gt_boxes"][0].shape)
print("GT labels:", batch["gt_labels"][0].shape)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

model = build_model(fusion_mode="se_unidirectional").to(device)

with torch.no_grad():
    out = model(batch_gpu)

print("\nModel outputs:")
for k, v in out.items():
    print(k, v.shape, "min:", float(v.min()), "max:", float(v.max()))

# -------------------------------
# Prepare camera image
# -------------------------------
img = batch["images"][0, 0].permute(1, 2, 0).numpy()
img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
img = img.clip(0, 1)

# -------------------------------
# Prepare BEV channels
# -------------------------------
bev_density = batch["bev_lidar"][0, 0].numpy()
bev_mean_h = batch["bev_lidar"][0, 1].numpy()
bev_max_h = batch["bev_lidar"][0, 2].numpy()
bev_intensity = batch["bev_lidar"][0, 3].numpy()

# -------------------------------
# Prepare predicted heatmap
# -------------------------------
heatmap = out["heatmap"][0, 0].detach().cpu().numpy()

# -------------------------------
# Show everything in one figure
# -------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img)
axes[0, 0].set_title("First camera image")
axes[0, 0].axis("off")

axes[0, 1].imshow(bev_density, cmap="gray")
axes[0, 1].set_title("LiDAR BEV - Density")
axes[0, 1].axis("off")

axes[0, 2].imshow(bev_mean_h, cmap="gray")
axes[0, 2].set_title("LiDAR BEV - Mean Height")
axes[0, 2].axis("off")

axes[1, 0].imshow(bev_max_h, cmap="gray")
axes[1, 0].set_title("LiDAR BEV - Max Height")
axes[1, 0].axis("off")

axes[1, 1].imshow(bev_intensity, cmap="gray")
axes[1, 1].set_title("LiDAR BEV - Mean Intensity")
axes[1, 1].axis("off")

axes[1, 2].imshow(heatmap, cmap="hot")
axes[1, 2].set_title("Predicted heatmap - Class 0")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()