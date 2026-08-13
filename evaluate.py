"""
Evaluation & Benchmarking
==========================

    - mAP (mean Average Precision)
    - FPS (inference speed)
    - Parameter count
    - FLOPs 


    For now, evaluate only the trained fused model
        se_unidirectional

"""


import os
import time
import json
import torch
import numpy as np
import argparse
from shapely.geometry import Polygon
from torch.utils.data import DataLoader

from data.nuscenes_loader import NuScenesBEVDataset, BEV_CONFIG
from models.model import build_model
from models.detection_head import decode_predictions
from tools.train import collate_fn
from fvcore.nn import FlopCountAnalysis



# IoU utilities

def box_iou_bev(box1, box2):
    """
    Simple BEV 2D IoU between two boxes.
    Boxes: [x, y, z, w, l, h, yaw]
    Approximated as axis-aligned for simplicity.
    """
    x1, y1, _, w1, l1 = box1[:5]
    x2, y2, _, w2, l2 = box2[:5]

    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
    y1_min, y1_max = y1 - l1 / 2, y1 + l1 / 2
    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
    y2_min, y2_max = y2 - l2 / 2, y2 + l2 / 2

    ix_min = max(x1_min, x2_min)
    ix_max = min(x1_max, x2_max)
    iy_min = max(y1_min, y2_min)
    iy_max = min(y1_max, y2_max)

    inter = max(0, ix_max - ix_min) * max(0, iy_max - iy_min)
    area1 = w1 * l1
    area2 = w2 * l2
    union = area1 + area2 - inter + 1e-6
    return inter / union



# mAP computation

def compute_map(all_preds, all_gts, num_classes=10, iou_thresh=0.5):
    """
    Compute mAP@0.5 over all samples.

    all_preds: list of {"boxes": (N,7), "scores": (N,), "labels": (N,)}
    all_gts  : list of {"boxes": (N,7), "labels": (N,)}
    """
    aps = []

    for cls in range(num_classes):
        det_scores = []
        det_tp = []
        n_gt = 0

        for pred, gt in zip(all_preds, all_gts):
            gt_boxes = gt["boxes"][gt["labels"] == cls]
            n_gt += len(gt_boxes)

            pred_mask = pred["labels"] == cls
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]

            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

            if len(pred_boxes) > 0:
                order = pred_scores.argsort(descending=True)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]

                for pb, ps in zip(pred_boxes, pred_scores):
                    best_iou = 0.0
                    best_j = -1

                    for j, gb in enumerate(gt_boxes):
                        if matched[j]:
                            continue
                        iou = box_iou_bev(pb.tolist(), gb.tolist())
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j

                    tp = (best_iou >= iou_thresh and best_j >= 0)
                    if tp and best_j >= 0:
                        matched[best_j] = True

                    det_scores.append(ps.item())
                    det_tp.append(float(tp))

        if n_gt == 0:
            continue

        if len(det_scores) == 0:
            aps.append(0.0)
            continue

        order = np.argsort(-np.array(det_scores))
        tp_sorted = np.array(det_tp)[order]
        cum_tp = np.cumsum(tp_sorted)
        cum_fp = np.cumsum(1 - tp_sorted)

        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        recall = cum_tp / (n_gt + 1e-6)

        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            p = precision[recall >= thr].max() if (recall >= thr).any() else 0.0
            ap += p / 11.0
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0



# FPS benchmark

def benchmark_fps(model, dataset, device, n_runs=30):
    """Measure inference FPS and latency using a real nuScenes sample."""
    model.eval()

    # Get one real sample
    sample = dataset[0]

    # Add batch dimension
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = v

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(batch)

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()

    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(batch)

            if device == "cuda":
                torch.cuda.synchronize()

    elapsed = time.time() - t0

    fps = n_runs / elapsed
    latency = (elapsed / n_runs) * 1000.0

    return fps, latency
# Optional FLOPs benchmark.
#
# Retained for a separate controlled FLOPs run. It is intentionally
# not called by the normal evaluation pipeline because fvcore tracing
# can require substantial temporary memory.
def compute_flops(model, device):

    model.eval()

    dummy = {
        "bev_lidar": torch.randn(1, 4, 200, 200).to(device),
        "images": torch.randn(1, 6, 3, 224, 224).to(device),
        "cam2ego": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1,6,4,4).to(device),
        "intrinsics": (torch.eye(3)*500).unsqueeze(0).unsqueeze(0).expand(1,6,3,3).to(device),
    }

    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy)

    return flops.total() / 1e9

# Full evaluation

def evaluate(
    fusion_mode: str,
    dataroot: str,
    checkpoint: str = None,
    device: str = "cpu",
    num_classes: int = 10,
):
    model = build_model(fusion_mode=fusion_mode, num_classes=num_classes).to(device)

    checkpoint_loaded = False
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        checkpoint_loaded = True
        print(f"  Loaded checkpoint: {checkpoint}")
    else:
        print(f"  No checkpoint found for {fusion_mode}; skipping mAP.")

    params = model.count_parameters()["total"]
    gflops = None
    mAP = None

    if checkpoint_loaded:

        # ---------------------------------------------------------
        # Load REAL nuScenes dataset
        # ---------------------------------------------------------
        dataset = NuScenesBEVDataset(
            dataroot=dataroot,
            version="v1.0-trainval"
        )

        # ---------------------------------------------------------
        # Benchmark FPS / latency using REAL sample
        # ---------------------------------------------------------
        fps, latency = benchmark_fps(
            model,
            dataset,
            device
        )

        # ---------------------------------------------------------
        # FLOPs
        # ---------------------------------------------------------
        # Disabled during the normal evaluation run. fvcore tracing can
        # create a large temporary memory footprint on the GPU.
        gflops = None

        # ---------------------------------------------------------
        # DataLoader for mAP evaluation
        # ---------------------------------------------------------
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        all_preds, all_gts = [], []
        model.eval()

        with torch.no_grad():
            for batch in loader:

                batch_gpu = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # -----------------------------------------------------
                # Forward pass
                # -----------------------------------------------------
                preds = model(batch_gpu)

                # -----------------------------------------------------
                # Decode predictions
                # -----------------------------------------------------
                dets = decode_predictions(
                    preds,
                    BEV_CONFIG,
                    score_thresh=0.3
                )

                # -----------------------------------------------------
                # Store predictions and ground truth
                # -----------------------------------------------------
                for i, det in enumerate(dets):

                    # Move predictions to CPU immediately so accumulated
                    # mAP data does not occupy GPU memory.
                    all_preds.append({
                        k: v.detach().cpu()
                        if isinstance(v, torch.Tensor)
                        else v
                        for k, v in det.items()
                    })

                    # Keep ground-truth tensors on CPU for mAP calculation.
                    all_gts.append({
                        "boxes": batch["gt_boxes"][i].detach().cpu(),
                        "labels": batch["gt_labels"][i].detach().cpu(),
                    })

                # Release batch/model outputs before loading the next sample.
                del dets
                del preds
                del batch_gpu

                if device == "cuda":
                    torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Compute mAP from the CPU-side prediction and ground-truth
        # collections accumulated during the dataset pass.
        # -------------------------------------------------------------
        mAP = compute_map(
            all_preds,
            all_gts,
            num_classes=num_classes
        )

    else:
        # Without a checkpoint there is no trained model to evaluate.
        # Therefore mAP/FPS/latency are not meaningful for this run.
        fps, latency = 0.0, 0.0

    return {
        "fusion_mode": fusion_mode,
        "mAP": None if mAP is None else round(mAP, 4),
        "FPS": round(fps, 1),
        "latency_ms": round(latency, 1),
        "params_M": round(params / 1e6, 2),
        "GFLOPs": None if gflops is None else round(gflops, 2),
        "checkpoint_loaded": checkpoint_loaded,
    }
def run_ablation_study(
    dataroot: str,
    device: str = "cpu",
    ckpt_dir: str = "checkpoints",
):

    experiments = [
        ("se_unidirectional", "SE Unidirectional ★ (proposed)"),
    ]

    results = []

    for mode, label in experiments:
        print(f"\nEvaluating: {label}")
        ckpt = os.path.join(ckpt_dir, f"lightfusion_{mode}_best.pth")
        res = evaluate(mode, dataroot, checkpoint=ckpt, device=device)
        res["label"] = label
        results.append(res)

        map_str = f"{res['mAP']:.4f}" if res["mAP"] is not None else "N/A"
        gflops_str = (
            f"{res['GFLOPs']:.2f}"
            if res["GFLOPs"] is not None
            else "N/A"
        )

        print(
            f"  mAP={map_str} | "
            f"FPS={res['FPS']:.1f} | "
            f"Latency={res['latency_ms']:.1f}ms | "
            f"Params={res['params_M']:.2f}M | "
            f"GFLOPs={gflops_str}"
        )

    print("\n" + "=" * 72)
    print(f"{'Model':<35} {'mAP':>8} {'FPS':>6} {'Latency':>10} {'Params':>8} {'GFLOPs':>10}")
    print("-" * 72)
    for r in results:
        map_str = f"{r['mAP']:.4f}" if r["mAP"] is not None else "N/A"
        gflops_str = (
            f"{r['GFLOPs']:.2f}"
            if r["GFLOPs"] is not None
            else "N/A"
        )

        print(
            f"{r['label']:<35} {map_str:>8} {r['FPS']:>6.1f} "
            f"{r['latency_ms']:>9.1f}ms {r['params_M']:>7.2f}M "
            f"{gflops_str:>9}"
        )
    print("=" * 72)

    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to ablation_results.json")

    return results




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataroot",
        required=True,
        help="Path to nuScenes root folder"
    )

    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints",
        help="Checkpoint directory"
    )

    args = parser.parse_args()

    run_ablation_study(
        dataroot=args.dataroot,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ckpt_dir=args.ckpt_dir,
    )