exp2 JUNE 1 -- residual se and batch  normalisation 
The residual connection allows the network to preserve the original semantic information while selectively enhancing important feature channels identified by LiDAR geometry.
  
  Independent Batch Normalization layers were introduced for both modalities before feature recalibration.
 Although the model converged successfully during training, the evaluation produced an mAP of 0.0. This indicates that while the proposed fusion module learned stable feature representations, additional improvements are required in the target generation, regression supervision, decoding pipeline, or longer training schedules before meaningful object detections can be achieved. 
  Loaded checkpoint: checkpoints/lightfusion_se_unidirectional_best.pth
  mAP=0.0000 | FPS=0.7 | Latency=1509.6ms | Params=0.67M

========================================================================
Model                                    mAP    FPS    Latency   Params
------------------------------------------------------------------------
SE Unidirectional ★ (proposed)        0.0000    0.7    1509.6ms    0.67M
========================================================================

e 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/Users/yashgupta14/Downloads/bevfusion_ugv/.venv/lib/python3.11/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=EfficientNet_B0_Weights.IMAGENET1K_V1`. You can also use `weights=EfficientNet_B0_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)

Model: LightFusionNet [se_unidirectional]
  total                : 670,150
  camera_encoder       : 355,124
  lidar_encoder        : 44,288
  fusion_module        : 10,624
  det_head             : 260,114

Training on cpu for 5 epochs
Train samples: 324 | Val: 80

/Users/yashgupta14/Downloads/bevfusion_ugv/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:1118: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  super().__init__(loader)
  [1][10/162] loss=961.5002 (hm=961.500 sz=0.000)
  [1][20/162] loss=671.7695 (hm=671.770 sz=0.000)
  [1][30/162] loss=476.0891 (hm=476.089 sz=0.000)
  [1][40/162] loss=381.6722 (hm=381.672 sz=0.000)
  [1][50/162] loss=311.1888 (hm=311.189 sz=0.000)
  [1][60/162] loss=263.5604 (hm=263.560 sz=0.000)
  [1][70/162] loss=220.5693 (hm=220.569 sz=0.000)
  [1][80/162] loss=192.2780 (hm=192.278 sz=0.000)
  [1][90/162] loss=158.7221 (hm=158.722 sz=0.000)
  [1][100/162] loss=130.5988 (hm=130.599 sz=0.000)
  [1][110/162] loss=114.4809 (hm=114.481 sz=0.000)
  [1][120/162] loss=93.7733 (hm=93.773 sz=0.000)
  [1][130/162] loss=78.9886 (hm=78.989 sz=0.000)
  [1][140/162] loss=66.7476 (hm=66.748 sz=0.000)
  [1][150/162] loss=55.9831 (hm=55.983 sz=0.000)
  [1][160/162] loss=51.4778 (hm=51.478 sz=0.000)
Epoch   1/5 | train: 337.3360 | val: 50.3708 | lr: 1.81e-04 | time: 666.9s
  → Saved checkpoint: checkpoints/lightfusion_se_unidirectional_best.pth
  [2][10/162] loss=39.3089 (hm=39.309 sz=0.000)
  [2][20/162] loss=33.1894 (hm=33.189 sz=0.000)
  [2][30/162] loss=28.7289 (hm=28.729 sz=0.000)
  [2][40/162] loss=26.2867 (hm=26.287 sz=0.000)
  [2][50/162] loss=20.0453 (hm=20.045 sz=0.000)
  [2][60/162] loss=16.8927 (hm=16.893 sz=0.000)
  [2][70/162] loss=14.7372 (hm=14.737 sz=0.000)
  [2][80/162] loss=12.5079 (hm=12.508 sz=0.000)
  [2][90/162] loss=10.6069 (hm=10.607 sz=0.000)
  [2][100/162] loss=9.1180 (hm=9.118 sz=0.000)
  [2][110/162] loss=7.5972 (hm=7.597 sz=0.000)
  [2][120/162] loss=6.4049 (hm=6.405 sz=0.000)
  [2][130/162] loss=5.3426 (hm=5.343 sz=0.000)
  [2][140/162] loss=4.6864 (hm=4.686 sz=0.000)
  [2][150/162] loss=4.4399 (hm=4.440 sz=0.000)
  [2][160/162] loss=3.3577 (hm=3.358 sz=0.000)
Epoch   2/5 | train: 16.1462 | val: 2.6588 | lr: 1.31e-04 | time: 645.4s
  → Saved checkpoint: checkpoints/lightfusion_se_unidirectional_best.pth
  [3][10/162] loss=2.7849 (hm=2.785 sz=0.000)
  [3][20/162] loss=2.4301 (hm=2.430 sz=0.000)
  [3][30/162] loss=2.1401 (hm=2.140 sz=0.000)
  [3][40/162] loss=1.8820 (hm=1.882 sz=0.000)
  [3][50/162] loss=1.6486 (hm=1.649 sz=0.000)
  [3][60/162] loss=1.5456 (hm=1.546 sz=0.000)
  [3][70/162] loss=1.3048 (hm=1.305 sz=0.000)
  [3][80/162] loss=1.1341 (hm=1.134 sz=0.000)
  [3][90/162] loss=1.0483 (hm=1.048 sz=0.000)
  [3][100/162] loss=0.9645 (hm=0.964 sz=0.000)
  [3][110/162] loss=0.8806 (hm=0.881 sz=0.000)
  [3][120/162] loss=0.8336 (hm=0.834 sz=0.000)
  [3][130/162] loss=0.7629 (hm=0.763 sz=0.000)
  [3][140/162] loss=0.7296 (hm=0.730 sz=0.000)
  [3][150/162] loss=0.6601 (hm=0.660 sz=0.000)
  [3][160/162] loss=0.6305 (hm=0.630 sz=0.000)
Epoch   3/5 | train: 1.4058 | val: 0.6850 | lr: 6.91e-05 | time: 676.1s
  → Saved checkpoint: checkpoints/lightfusion_se_unidirectional_best.pth
  [4][10/162] loss=0.5951 (hm=0.595 sz=0.000)
  [4][20/162] loss=0.5937 (hm=0.594 sz=0.000)
  [4][30/162] loss=0.5776 (hm=0.578 sz=0.000)
  [4][40/162] loss=0.5549 (hm=0.555 sz=0.000)
  [4][50/162] loss=0.5343 (hm=0.534 sz=0.000)
  [4][60/162] loss=0.5189 (hm=0.519 sz=0.000)
  [4][70/162] loss=0.5060 (hm=0.506 sz=0.000)
  [4][80/162] loss=0.4908 (hm=0.491 sz=0.000)
  [4][90/162] loss=0.4845 (hm=0.484 sz=0.000)
  [4][100/162] loss=0.4795 (hm=0.480 sz=0.000)
  [4][110/162] loss=0.4545 (hm=0.455 sz=0.000)
  [4][120/162] loss=0.4361 (hm=0.436 sz=0.000)
  [4][130/162] loss=0.4430 (hm=0.443 sz=0.000)
  [4][140/162] loss=0.4258 (hm=0.426 sz=0.000)
  [4][150/162] loss=0.4095 (hm=0.410 sz=0.000)
  [4][160/162] loss=0.4001 (hm=0.400 sz=0.000)
Epoch   4/5 | train: 0.4985 | val: 0.4124 | lr: 1.91e-05 | time: 647.6s
  → Saved checkpoint: checkpoints/lightfusion_se_unidirectional_best.pth
  [5][10/162] loss=0.3997 (hm=0.400 sz=0.000)
  [5][20/162] loss=0.3914 (hm=0.391 sz=0.000)
  [5][30/162] loss=0.3954 (hm=0.395 sz=0.000)
  [5][40/162] loss=0.3923 (hm=0.392 sz=0.000)
  [5][50/162] loss=0.3827 (hm=0.383 sz=0.000)
  [5][60/162] loss=0.3793 (hm=0.379 sz=0.000)
  [5][70/162] loss=0.3749 (hm=0.375 sz=0.000)
  [5][80/162] loss=0.3781 (hm=0.378 sz=0.000)
  [5][90/162] loss=0.3704 (hm=0.370 sz=0.000)
  [5][100/162] loss=0.3768 (hm=0.377 sz=0.000)
  [5][110/162] loss=0.3676 (hm=0.368 sz=0.000)
  [5][120/162] loss=0.3611 (hm=0.361 sz=0.000)
  [5][130/162] loss=0.3615 (hm=0.362 sz=0.000)
  [5][140/162] loss=0.3661 (hm=0.366 sz=0.000)
  [5][150/162] loss=0.3588 (hm=0.359 sz=0.000)
  [5][160/162] loss=0.3565 (hm=0.357 sz=0.000)
Epoch   5/5 | train: 0.3777 | val: 0.2984 | lr: 0.00e+00 | time: 656.7s
  → Saved checkpoint: checkpoints/lightfusion_se_unidirectional_best.pth
  → Saved checkpoint: checkpoints/lightfusion_se_unidirectional_final.pth

Training complete. Best val loss: 0.2984
