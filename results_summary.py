from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

from data.nuscenes_loader import NuScenesBEVDataset
from tools.train import collate_fn
from models.model import build_model


# Paths

DATA_ROOT = "/Users/yashgupta14/Downloads/bevfusion_ugv/Data_set/v1.0-mini"
CKPT_PATH = "checkpoints/lightfusion_se_unidirectional_best.pth"

device = "cpu"


# Training curve data from your 5-epoch run

epochs = [1, 2, 3, 4, 5]
train_loss = [334.2662, 14.6491, 0.9096, 0.3834, 0.2994]
val_loss = [49.0891, 1.7094, 0.4276, 0.2913, 0.2646]


# Load one sample from dataset

dataset = NuScenesBEVDataset(dataroot=DATA_ROOT, version="v1.0-mini")
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

batch = next(iter(loader))
batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# Load trained model

model = build_model(fusion_mode="se_unidirectional").to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

with torch.no_grad():
    out = model(batch_gpu)

# Prepare camera image

img = batch["images"][0, 0].permute(1, 2, 0).numpy()
img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
img = img.clip(0, 1)


# Prepare BEV and trained heatmap

bev_density = batch["bev_lidar"][0, 0].numpy()
heatmap = out["heatmap"][0, 0].detach().cpu().numpy()


# Figure 1: qualitative results only

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

axes1[0].imshow(img)
axes1[0].set_title("Camera Image")
axes1[0].axis("off")

axes1[1].imshow(bev_density, cmap="gray")
axes1[1].set_title("LiDAR BEV Density")
axes1[1].axis("off")

axes1[2].imshow(heatmap, cmap="hot")
axes1[2].set_title("Trained Predicted Heatmap")
axes1[2].axis("off")

plt.tight_layout()
plt.show()


# Figure 2: training curve only

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over 5 Epochs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()