"""
nuScenes Data Loader
====================
Loads synchronized camera + LiDAR data from nuScenes.
BEV-ready tensors for the fusion pipeline.
"""

import os
from matplotlib.style import available
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import PIL
import torchvision.transforms as T


# ------------------------------------------------------------------
# BEV configuration
# ------------------------------------------------------------------
BEV_CONFIG = {
    "x_range": (-50.0, 50.0),   # metres forward/back
    "y_range": (-50.0, 50.0),   # metres left/right
    "z_range": (-5.0,  3.0),    # metres up/down
    "voxel_size": 0.5,          # metres per BEV cell
}

IMG_H, IMG_W = 224, 224        # resized camera image size
NUM_CAMERAS   = 6              # nuScenes surround cameras


# ------------------------------------------------------------------
# Helper: build BEV voxel map from raw point cloud
# ------------------------------------------------------------------
def pointcloud_to_bev(points: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Convert (N, 4) LiDAR point cloud → (C, H, W) BEV feature map.

    Channels:
        0 - point density  (normalised count per cell)
        1 - mean height
        2 - max height
        3 - mean intensity
    """
    x_min, x_max = cfg["x_range"]
    y_min, y_max = cfg["y_range"]
    z_min, z_max = cfg["z_range"]
    vox           = cfg["voxel_size"]

    W = int((x_max - x_min) / vox)
    H = int((y_max - y_min) / vox)

    density    = np.zeros((H, W), dtype=np.float32)
    height_sum = np.zeros((H, W), dtype=np.float32)
    height_max = np.full((H, W), z_min, dtype=np.float32)
    intensity  = np.zeros((H, W), dtype=np.float32)
    count      = np.zeros((H, W), dtype=np.float32)

    x, y, z, intens = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    # filter to BEV range
    mask = (
        (x >= x_min) & (x < x_max) &
        (y >= y_min) & (y < y_max) &
        (z >= z_min) & (z < z_max)
    )
    x, y, z, intens = x[mask], y[mask], z[mask], intens[mask]

    xi = ((x - x_min) / vox).astype(np.int32)
    yi = ((y - y_min) / vox).astype(np.int32)

    # clamp indices
    xi = np.clip(xi, 0, W - 1)
    yi = np.clip(yi, 0, H - 1)

    np.add.at(count,      (yi, xi), 1)
    np.add.at(height_sum, (yi, xi), z)
    np.add.at(intensity,  (yi, xi), intens)

    valid = count > 0
    height_max_vals = np.zeros_like(height_max)

    # max height per cell
    order = np.argsort(z)
    np.maximum.at(height_max_vals, (yi[order], xi[order]), z[order])

    density[valid]    = np.log1p(count[valid]) / np.log1p(count.max() + 1e-6)
    height_sum[valid] = height_sum[valid] / count[valid]
    height_max        = height_max_vals
    intensity[valid]  = intensity[valid] / count[valid]

    bev = np.stack([density, height_sum, height_max, intensity], axis=0)
    return bev  # (4, H, W)


# ------------------------------------------------------------------
# nuScenes Dataset class
# ------------------------------------------------------------------
class NuScenesBEVDataset(Dataset):
    """
    Returns one sample per keyframe:
        images   : (6, 3, H, W)   — 6 surround cameras
        bev_lidar: (4, BH, BW)    — LiDAR BEV feature map
        gt_boxes : (N, 7)         — [x, y, z, w, l, h, yaw] in EGO frame
        gt_labels: (N,)           — class indices
    """

    CAMERA_NAMES = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
        "CAM_BACK_LEFT",
    ]

    CLASS_NAMES = [
        "car", "truck", "bus", "trailer",
        "construction_vehicle", "pedestrian",
        "motorcycle", "bicycle",
        "traffic_cone", "barrier",
    ]

    def __init__(self, dataroot: str, version: str = "v1.0-mini", split: str = "train"):
        self.nusc      = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.bev_cfg   = BEV_CONFIG
        self.img_tf    = T.Compose([
            T.Resize((IMG_H, IMG_W)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        # collect all keyframe tokens
        self.samples = self._filter_available_samples(self.nusc.sample)
    
    def _filter_available_samples(self, all_samples):
        """
        Keep only samples for which every sensor required by the model
        is physically present on disk.

        The model consumes all six surround cameras plus LIDAR_TOP.
        Previously this check verified only CAM_FRONT, which could allow
        incomplete samples into the dataset and cause a later failure
        inside __getitem__ when another camera or LiDAR file was opened.

        A scene is excluded as soon as one required sensor file is missing.
        This preserves the existing scene-level filtering behaviour while
        making the availability check consistent with __getitem__.
        """
        available = []
        missing_scenes = set()

        # These are exactly the sensors loaded by __getitem__.
        required_sensors = self.CAMERA_NAMES + ["LIDAR_TOP"]

        for sample in all_samples:
            scene_token = sample["scene_token"]

            # Once a required file is missing from a scene, skip the
            # remaining samples from that scene.
            if scene_token in missing_scenes:
                continue

            sample_complete = True

            for sensor_name in required_sensors:
                sensor_token = sample["data"].get(sensor_name)

                # A missing sensor token means this sample is unusable.
                if sensor_token is None:
                    sample_complete = False
                    break

                sensor_data = self.nusc.get("sample_data", sensor_token)
                sensor_path = os.path.join(
                    self.nusc.dataroot,
                    sensor_data["filename"]
                )

                if not os.path.exists(sensor_path):
                    sample_complete = False
                    break

            if sample_complete:
                available.append(sample)
            else:
                missing_scenes.add(scene_token)

        n_total = len(all_samples)
        n_avail = len(available)

        print(
            f"[NuScenesBEVDataset] {n_avail}/{n_total} samples available "
            f"({len(missing_scenes)} scenes missing one or more required "
            f"camera/LiDAR files)"
        )

        return available

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Camera images + calibration
        images = []
        cam2ego_list = []
        intrinsics_list = []

        for cam in self.CAMERA_NAMES:
            cam_token = sample["data"][cam]
            cam_data = self.nusc.get("sample_data", cam_token)
            img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
            img = PIL.Image.open(img_path).convert("RGB")

            # Original image dimensions
            orig_w, orig_h = img.size

            # Camera calibration
            cs_record = self.nusc.get(
                "calibrated_sensor",
                cam_data["calibrated_sensor_token"]
            )

            K = np.array(cs_record["camera_intrinsic"], dtype=np.float32)

            # Camera -> ego transformation
            cam2ego = np.eye(4, dtype=np.float32)
            cam2ego[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
            cam2ego[:3, 3] = np.array(cs_record["translation"], dtype=np.float32)

            # Resize image
            img = self.img_tf(img)

            # Scale intrinsics to resized image
            sx = IMG_W / orig_w
            sy = IMG_H / orig_h
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy

            images.append(img)
            cam2ego_list.append(torch.from_numpy(cam2ego))
            intrinsics_list.append(torch.from_numpy(K))

        images = torch.stack(images, dim=0)  # (6, 3, H, W)
        cam2ego = torch.stack(cam2ego_list, dim=0)  # (6, 4, 4)
        intrinsics = torch.stack(intrinsics_list, dim=0)  # (6, 3, 3)

        # 2. LiDAR point cloud -> ego frame -> BEV
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = self.nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data["filename"])
        pc = LidarPointCloud.from_file(lidar_path)

        # sensor -> ego (calibrated_sensor extrinsics)
        cs_record = self.nusc.get(
            "calibrated_sensor",
            lidar_data["calibrated_sensor_token"]
        )
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pc.translate(np.array(cs_record["translation"]))

        bev_lidar = pointcloud_to_bev(pc.points.T, self.bev_cfg)
        bev_lidar = torch.from_numpy(bev_lidar)  # (4, H, W)

        # 3. Ground truth boxes (global -> ego, using the SAME ego_pose
        #    timestamp as the LiDAR sweep so boxes and points agree)
        gt_boxes, gt_labels = self._load_annotations(sample, lidar_data)

        return {
            "images": images,
            "bev_lidar": bev_lidar,
            "cam2ego": cam2ego,
            "intrinsics": intrinsics,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "token": sample["token"],
        }
    def _load_annotations(self, sample, lidar_data):
        """
        nuScenes stores `sample_annotation.translation` / `.rotation` in the
        GLOBAL map frame. Our LiDAR BEV lives in the EGO VEHICLE frame
        (see __getitem__: pc is rotated/translated by calibrated_sensor,
        i.e. sensor -> ego). If we don't also convert the boxes
        global -> ego, boxes and points live in different coordinate
        systems and every box lands far outside the BEV grid (this was the
        root cause of the all-zero heatmap / empty target bug).

        We use the ego_pose associated with the LIDAR_TOP sample_data,
        since that's the frame our BEV grid is actually built in.
        """
        boxes, labels = [], []

        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose["translation"])
        ego_rotation    = Quaternion(ego_pose["rotation"])

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            label = self._cat_to_label(ann["category_name"])
            if label < 0:
                continue

            # ---- global -> ego: translation ----
            center_global = np.array(ann["translation"])
            center_ego    = ego_rotation.inverse.rotate(center_global - ego_translation)
            x, y, z = center_ego

            # ---- global -> ego: rotation (yaw) ----
            rot_global = Quaternion(ann["rotation"])
            rot_ego    = ego_rotation.inverse * rot_global
            yaw        = rot_ego.yaw_pitch_roll[0]

            w, l, h = ann["size"]

            boxes.append([x, y, z, w, l, h, yaw])
            labels.append(label)

        if len(boxes) == 0:
            return torch.zeros((0, 7)), torch.zeros(0, dtype=torch.long)

        return (torch.tensor(boxes,  dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long))

    def _cat_to_label(self, category_name: str) -> int:
        mapping = {
            "vehicle.car":                  0,
            "vehicle.truck":                1,
            "vehicle.bus.rigid":            2,
            "vehicle.bus.bendy":            2,
            "vehicle.trailer":              3,
            "vehicle.construction":         4,
            "human.pedestrian.adult":       5,
            "human.pedestrian.child":       5,
            "human.pedestrian.wheelchair":  5,
            "vehicle.motorcycle":           6,
            "vehicle.bicycle":              7,
            "movable_object.trafficcone":   8,
            "movable_object.barrier":       9,
        }
        return mapping.get(category_name, -1)


# ====================================================================
# OLD / ORIGINAL CODE (pre-fix) — kept for reference only, DO NOT USE
# ====================================================================
# This is the original __init__ / __getitem__ / _load_annotations from
# before the global -> ego coordinate transform fix. Annotations here
# stay in nuScenes GLOBAL map frame while the LiDAR BEV is built in EGO
# frame, so gt_boxes and bev_lidar disagree and training produces
# all-zero heatmaps. Left here only so the diff against the fix above
# is easy to see.
# ====================================================================

# class NuScenesBEVDataset_OLD(Dataset):
#     """
#     Returns one sample per keyframe:
#         images  : (6, 3, H, W)   — 6 surround cameras
#         bev_lidar: (4, BH, BW)   — LiDAR BEV feature map
#         gt_boxes : (N, 7)        — [x, y, z, w, l, h, yaw] in ego frame
#         gt_labels: (N,)          — class indices
#     """
#
#     CAMERA_NAMES = [
#         "CAM_FRONT",
#         "CAM_FRONT_RIGHT",
#         "CAM_FRONT_LEFT",
#         "CAM_BACK",
#         "CAM_BACK_RIGHT",
#         "CAM_BACK_LEFT",
#     ]
#
#     CLASS_NAMES = [
#         "car", "truck", "bus", "trailer",
#         "construction_vehicle", "pedestrian",
#         "motorcycle", "bicycle",
#         "traffic_cone", "barrier",
#     ]
#
#     def __init__(self, dataroot: str, version: str = "v1.0-mini", split: str = "train"):
#         self.nusc      = NuScenes(version=version, dataroot=dataroot, verbose=False)
#         self.bev_cfg   = BEV_CONFIG
#         self.img_tf    = T.Compose([
#             T.Resize((IMG_H, IMG_W)),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406],
#                         std =[0.229, 0.224, 0.225]),
#         ])
#
#         # collect all keyframe tokens
#         self.samples = self.nusc.sample
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#
#         #  1. Camera images 
#         images = []
#         for cam in self.CAMERA_NAMES:
#             cam_token  = sample["data"][cam]
#             cam_data   = self.nusc.get("sample_data", cam_token)
#             img_path   = os.path.join(self.nusc.dataroot, cam_data["filename"])
#             img        = PIL.Image.open(img_path).convert("RGB")
#             images.append(self.img_tf(img))
#         images = torch.stack(images, dim=0)  # (6, 3, H, W)
#
#         #2. LiDAR point cloud → BEV 
#         lidar_token = sample["data"]["LIDAR_TOP"]
#         lidar_data  = self.nusc.get("sample_data", lidar_token)
#         lidar_path  = os.path.join(self.nusc.dataroot, lidar_data["filename"])
#         pc          = LidarPointCloud.from_file(lidar_path)
#
#         # transform to ego vehicle frame
#         cs_record = self.nusc.get("calibrated_sensor",
#                                   lidar_data["calibrated_sensor_token"])
#         pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
#         pc.translate(np.array(cs_record["translation"]))
#
#         bev_lidar = pointcloud_to_bev(pc.points.T, self.bev_cfg)
#         bev_lidar = torch.from_numpy(bev_lidar)  # (4, H, W)
#
#         #  3. Ground truth boxes 
#         gt_boxes, gt_labels = self._load_annotations(sample)
#
#         return {
#             "images":    images,
#             "bev_lidar": bev_lidar,
#             "gt_boxes":  gt_boxes,
#             "gt_labels": gt_labels,
#             "token":     sample["token"],
#         }
#
#     def _load_annotations(self, sample):
#         boxes, labels = [], []
#         for ann_token in sample["anns"]:
#             ann      = self.nusc.get("sample_annotation", ann_token)
#             cat_name = ann["category_name"].split(".")[0]  # e.g. "vehicle" → keep
#
#             # map to our class list
#             label = self._cat_to_label(ann["category_name"])
#             if label < 0:
#                 continue
#
#             x, y, z = ann["translation"]  # BUG: this is GLOBAL frame, not ego
#
#             w, l, h = ann["size"]
#             yaw     = Quaternion(ann["rotation"]).yaw_pitch_roll[0]
#
#             boxes.append([x, y, z, w, l, h, yaw])
#             labels.append(label)
#
#         if len(boxes) == 0:
#             return torch.zeros((0, 7)), torch.zeros(0, dtype=torch.long)
#
#         return (torch.tensor(boxes,  dtype=torch.float32),
#                 torch.tensor(labels, dtype=torch.long))
#
#     def _cat_to_label(self, category_name: str) -> int:
#         mapping = {
#             "vehicle.car":                  0,
#             "vehicle.truck":                1,
#             "vehicle.bus.rigid":            2,
#             "vehicle.bus.bendy":            2,
#             "vehicle.trailer":              3,
#             "vehicle.construction":         4,
#             "human.pedestrian.adult":       5,
#             "human.pedestrian.child":       5,
#             "human.pedestrian.wheelchair":  5,
#             "vehicle.motorcycle":           6,
#             "vehicle.bicycle":              7,
#             "movable_object.trafficcone":   8,
#             "movable_object.barrier":       9,
#         }
#         return mapping.get(category_name, -1)