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


After debuuging the  loader 
Instead of using the global coordinates directly, now transform each annotation into the ego vehicle coordinate frame before creating the training targets.
Global annotation
        ↓
Subtract ego vehicle position
        ↓
Rotate by inverse ego orientation
        ↓
Ego coordinates
============================================================
TARGET DEBUG
============================================================
Positive pixels : 93.0
Heatmap max     : 1.0000
Size sum        : 571.1430
Height sum      : 106.6543
Offset sum      : -18505.2363
Rotation sum    : -5.2475
============================================================
Object 0: x=37.03, y=3.61, grid=(174,107)
Object 1: x=-26.48, y=0.64, grid=(47,101)
Object 2: x=-13.92, y=-2.81, grid=(72,94)
Object 0: x=-68.16, y=-11.37, grid=(-36,77)
Object 1: x=-9.98, y=3.61, grid=(80,107)
Object 2: x=83.75, y=-10.36, grid=(267,79)
Assigned object: class=6, cell=(65,69), size=(1.41, 2.11, 1.78)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 25.0
Heatmap max     : 1.0000
Size sum        : 212.7340
Height sum      : 23.7718
Offset sum      : -4975.8257
Rotation sum    : 0.5387
============================================================
Object 0: x=-16.45, y=5.81, grid=(67,111)
Object 1: x=-60.69, y=7.36, grid=(-21,114)
Object 2: x=-86.82, y=7.82, grid=(-73,115)
Object 0: x=-27.31, y=6.87, grid=(45,113)
Object 1: x=-59.92, y=-0.02, grid=(-19,99)
Object 2: x=-16.75, y=-11.29, grid=(66,77)
Assigned object: class=5, cell=(163,110), size=(0.64, 0.60, 1.99)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 66.0
Heatmap max     : 1.0000
Size sum        : 449.6480
Height sum      : 56.5781
Offset sum      : -13131.3906
Rotation sum    : 4.5058
============================================================
Object 0: x=-28.20, y=11.93, grid=(43,123)
Object 1: x=-46.52, y=-6.82, grid=(6,86)
Object 2: x=54.73, y=-19.15, grid=(209,61)
Object 0: x=4.32, y=-3.29, grid=(108,93)
Object 1: x=59.26, y=-0.70, grid=(218,98)
Object 2: x=-85.28, y=-12.13, grid=(-70,75)
Assigned object: class=0, cell=(-69,93), size=(2.07, 4.91, 2.00)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 46.0
Heatmap max     : 1.0000
Size sum        : 225.7430
Height sum      : 44.0091
Offset sum      : -9153.7930
Rotation sum    : 0.7224
============================================================
Object 0: x=-20.77, y=-13.68, grid=(58,72)
Object 1: x=-54.80, y=8.80, grid=(-9,117)
Object 2: x=-59.37, y=-13.75, grid=(-18,72)
Object 0: x=-77.95, y=-14.67, grid=(-55,70)
Object 1: x=24.56, y=-20.59, grid=(149,58)
Object 2: x=6.38, y=-10.42, grid=(112,79)
Assigned object: class=5, cell=(190,112), size=(0.65, 0.66, 1.67)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 26.0
Heatmap max     : 1.0000
Size sum        : 173.4190
Height sum      : 25.8433
Offset sum      : -5174.0557
Rotation sum    : -4.3733
============================================================
Object 0: x=-6.08, y=-6.78, grid=(87,86)
Object 1: x=-1.74, y=3.00, grid=(96,105)
Object 2: x=20.00, y=12.62, grid=(139,125)
Object 0: x=4.97, y=-13.77, grid=(109,72)
Object 1: x=5.63, y=6.42, grid=(111,112)
Object 2: x=7.44, y=-14.28, grid=(114,71)
Assigned object: class=0, cell=(142,42), size=(1.80, 4.66, 1.50)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 75.0
Heatmap max     : 1.0000
Size sum        : 515.2740
Height sum      : 82.3461
Offset sum      : -14921.4980
Rotation sum    : 28.6356
============================================================
  [1][20/162] loss=210.6801 (hm=9.961 sz=3.989)
Object 0: x=39.76, y=-18.79, grid=(179,62)
Object 1: x=-6.67, y=28.52, grid=(86,157)
Object 2: x=50.99, y=11.31, grid=(201,122)
Object 0: x=-54.85, y=-18.19, grid=(-9,63)
Object 1: x=64.24, y=-14.14, grid=(228,71)
Object 2: x=2.22, y=-11.84, grid=(104,76)
Assigned object: class=0, cell=(184,56), size=(2.00, 4.73, 1.92)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 63.0
Heatmap max     : 1.0000
Size sum        : 421.3550
Height sum      : 62.6813
Offset sum      : -12538.9492
Rotation sum    : -4.1892
============================================================
Object 0: x=55.89, y=-18.61, grid=(211,62)
Object 1: x=33.05, y=-21.13, grid=(166,57)
Object 2: x=60.69, y=-37.58, grid=(221,24)
Object 0: x=-30.68, y=-14.42, grid=(38,71)
Object 1: x=50.81, y=-20.66, grid=(201,58)
Object 2: x=58.84, y=-11.80, grid=(217,76)
Assigned object: class=0, cell=(252,91), size=(2.08, 4.58, 1.79)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 70.0
Heatmap max     : 1.0000
Size sum        : 312.7450
Height sum      : 53.7003
Offset sum      : -13925.2480
Rotation sum    : 49.3466
============================================================
Object 0: x=21.16, y=-6.29, grid=(142,87)
Object 1: x=16.92, y=-5.95, grid=(133,88)
Object 2: x=-5.45, y=-6.62, grid=(89,86)
Object 0: x=9.32, y=8.59, grid=(118,117)
Object 1: x=8.17, y=5.64, grid=(116,111)
Object 2: x=15.38, y=-40.89, grid=(130,18)
Assigned object: class=0, cell=(175,208), size=(2.01, 4.49, 1.49)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 47.0
Heatmap max     : 1.0000
Size sum        : 348.9110
Height sum      : 46.8857
Offset sum      : -9352.5244
Rotation sum    : 16.5648
============================================================
Object 0: x=34.77, y=-18.51, grid=(169,62)
Object 1: x=-12.01, y=28.50, grid=(75,156)
Object 2: x=-1.17, y=-30.59, grid=(97,38)
Object 0: x=25.23, y=15.10, grid=(150,130)
Object 1: x=-6.11, y=-6.79, grid=(87,86)
Object 2: x=-7.62, y=2.37, grid=(84,104)
Assigned object: class=0, cell=(136,109), size=(2.56, 5.64, 2.23)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 69.0
Heatmap max     : 1.0000
Size sum        : 440.4180
Height sum      : 67.1763
Offset sum      : -13732.5195
Rotation sum    : 11.8562
============================================================
Object 0: x=-8.38, y=3.55, grid=(83,107)
Object 1: x=-22.04, y=-0.08, grid=(55,99)
Object 2: x=-63.71, y=-36.91, grid=(-27,26)
Object 0: x=10.71, y=15.48, grid=(121,130)
Object 1: x=9.76, y=11.61, grid=(119,123)
Object 2: x=14.32, y=-23.91, grid=(128,52)
Assigned object: class=0, cell=(102,106), size=(1.86, 4.55, 1.47)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 56.0
Heatmap max     : 1.0000
Size sum        : 349.2090
Height sum      : 54.4962
Offset sum      : -11139.2295
Rotation sum    : 7.9947
============================================================
Object 0: x=-6.59, y=4.87, grid=(86,109)
Object 1: x=-36.06, y=0.59, grid=(27,101)
Object 2: x=32.97, y=7.51, grid=(165,115)
Object 0: x=-48.39, y=-8.54, grid=(3,82)
Object 1: x=-33.33, y=7.98, grid=(33,115)
Object 2: x=-27.79, y=8.00, grid=(44,115)
Assigned object: class=0, cell=(166,71), size=(2.03, 4.84, 1.68)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 65.0
Heatmap max     : 1.0000
Size sum        : 461.3660
Height sum      : 56.8042
Offset sum      : -12931.5137
Rotation sum    : -7.3106
============================================================
Object 0: x=17.73, y=-18.90, grid=(135,62)
Object 1: x=-6.05, y=-6.77, grid=(87,86)
Object 2: x=1.67, y=3.36, grid=(103,106)
Object 0: x=10.05, y=12.98, grid=(120,125)
Object 1: x=8.74, y=9.34, grid=(117,118)
Object 2: x=14.43, y=-30.48, grid=(128,39)
Assigned object: class=0, cell=(175,208), size=(2.01, 4.49, 1.49)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 51.0
Heatmap max     : 1.0000
Size sum        : 356.0920
Height sum      : 49.4620
Offset sum      : -10146.4824
Rotation sum    : 14.9578
============================================================
Object 0: x=-5.10, y=-18.58, grid=(89,62)
Object 1: x=57.70, y=-16.70, grid=(215,66)
Object 2: x=-10.09, y=10.73, grid=(79,121)
Object 0: x=-36.79, y=13.91, grid=(26,127)
Object 1: x=6.09, y=-27.90, grid=(112,44)
Object 2: x=-0.71, y=-6.30, grid=(98,87)
Assigned object: class=0, cell=(51,97), size=(1.66, 3.83, 1.55)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 67.0
Heatmap max     : 1.0000
Size sum        : 507.4680
Height sum      : 71.7217
Offset sum      : -13333.4590
Rotation sum    : 17.3237
============================================================
Object 0: x=-6.79, y=4.84, grid=(86,109)
Object 1: x=-35.40, y=0.45, grid=(29,100)
Object 2: x=28.74, y=5.52, grid=(157,111)
Object 0: x=12.31, y=21.62, grid=(124,143)
Object 1: x=11.99, y=18.49, grid=(123,136)
Object 2: x=14.85, y=-5.57, grid=(129,88)
Assigned object: class=0, cell=(102,106), size=(1.86, 4.55, 1.47)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 55.0
Heatmap max     : 1.0000
Size sum        : 343.4920
Height sum      : 46.3423
Offset sum      : -10940.4316
Rotation sum    : -0.4845
============================================================
Object 0: x=37.23, y=-18.63, grid=(174,62)
Object 1: x=-9.39, y=28.51, grid=(81,157)
Object 2: x=1.24, y=-30.60, grid=(102,38)
Object 0: x=-3.02, y=-13.17, grid=(93,73)
Object 1: x=-5.17, y=6.87, grid=(89,113)
Object 2: x=-0.42, y=-13.31, grid=(99,73)
Assigned object: class=0, cell=(119,87), size=(1.66, 3.83, 1.55)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 114.0
Heatmap max     : 1.0000
Size sum        : 768.5641
Height sum      : 108.6485
Offset sum      : -22692.1250
Rotation sum    : 15.7452
============================================================
  [1][30/162] loss=205.9424 (hm=6.030 sz=3.572)
Object 0: x=9.59, y=-18.98, grid=(119,62)
Object 1: x=-2.00, y=10.67, grid=(95,121)
Object 2: x=29.45, y=8.51, grid=(158,117)
Object 0: x=-46.73, y=4.21, grid=(6,108)
Object 1: x=-65.22, y=-1.85, grid=(-30,96)
Object 2: x=-46.70, y=3.10, grid=(6,106)
Assigned object: class=5, cell=(130,138), size=(0.89, 0.99, 1.95)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 123.0
Heatmap max     : 1.0000
Size sum        : 520.1710
Height sum      : 127.2106
Offset sum      : -24482.8086
Rotation sum    : -28.8143
============================================================
Object 0: x=-2.81, y=-8.61, grid=(94,82)
Object 1: x=16.87, y=9.92, grid=(133,119)
Object 2: x=24.66, y=24.11, grid=(149,148)
Object 0: x=14.00, y=-17.96, grid=(128,64)
Object 1: x=-21.90, y=-30.51, grid=(56,38)
Object 2: x=30.36, y=12.20, grid=(160,124)
Assigned object: class=0, cell=(78,52), size=(2.00, 4.93, 2.00)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 107.0
Heatmap max     : 1.0000
Size sum        : 605.9930
Height sum      : 115.7708
Offset sum      : -21293.1309
Rotation sum    : 8.4154
============================================================
Object 0: x=-6.05, y=-6.77, grid=(87,86)
Object 1: x=1.67, y=3.35, grid=(103,106)
Object 2: x=13.30, y=-21.25, grid=(126,57)
Object 0: x=23.37, y=-21.70, grid=(146,56)
Object 1: x=28.59, y=11.39, grid=(157,122)
Object 2: x=22.96, y=-9.64, grid=(145,80)
Assigned object: class=0, cell=(26,110), size=(2.09, 5.57, 2.00)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 29.0
Heatmap max     : 1.0000
Size sum        : 226.9990
Height sum      : 25.2698
Offset sum      : -5770.4180
Rotation sum    : 15.5993
============================================================
Object 0: x=0.48, y=-8.72, grid=(100,82)
Object 1: x=18.84, y=10.98, grid=(137,121)
Object 2: x=25.86, y=25.63, grid=(151,151)
Object 0: x=1.85, y=-37.75, grid=(103,24)
Object 1: x=2.83, y=-14.03, grid=(105,71)
Object 2: x=-6.47, y=-3.24, grid=(87,93)
Assigned object: class=0, cell=(105,8), size=(2.08, 4.47, 1.64)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 71.0
Heatmap max     : 1.0000
Size sum        : 365.6150
Height sum      : 42.0833
Offset sum      : -14128.5234
Rotation sum    : 19.8194
============================================================
Object 0: x=-26.12, y=-14.46, grid=(47,71)
Object 1: x=-36.10, y=-7.28, grid=(27,85)
Object 2: x=-38.76, y=-28.99, grid=(22,42)
Object 0: x=-27.09, y=-20.03, grid=(45,59)
Object 1: x=-31.06, y=7.41, grid=(37,114)
Object 2: x=41.24, y=-7.11, grid=(182,85)
Assigned object: class=5, cell=(85,114), size=(0.65, 0.66, 1.67)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 152.0
Heatmap max     : 1.0000
Size sum        : 630.1720
Height sum      : 118.9403
Offset sum      : -30242.6738
Rotation sum    : -17.1081
============================================================
Object 0: x=8.95, y=-16.82, grid=(117,66)
Object 1: x=28.99, y=-19.64, grid=(157,60)
Object 2: x=8.14, y=-19.07, grid=(116,61)
Object 0: x=-2.32, y=-17.83, grid=(95,64)
Object 1: x=31.62, y=-5.48, grid=(163,89)
Object 2: x=17.11, y=12.18, grid=(134,124)
Assigned object: class=0, cell=(46,53), size=(2.00, 4.93, 2.00)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 100.0
Heatmap max     : 1.0000
Size sum        : 627.7060
Height sum      : 95.9184
Offset sum      : -19898.5762
Rotation sum    : -1.9329
============================================================
Object 0: x=-9.33, y=-37.84, grid=(81,24)
Object 1: x=-6.77, y=-20.42, grid=(86,59)
Object 2: x=-3.55, y=-3.27, grid=(92,93)
Object 0: x=5.38, y=-20.10, grid=(110,59)
Object 1: x=-34.34, y=-10.33, grid=(31,79)
Object 2: x=-1.64, y=7.45, grid=(96,114)
Assigned object: class=5, cell=(151,114), size=(0.65, 0.66, 1.67)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 25.0
Heatmap max     : 1.0000
Size sum        : 179.9110
Height sum      : 16.1668
Offset sum      : -4973.5801
Rotation sum    : 1.9132
============================================================
Object 0: x=8.80, y=-17.97, grid=(117,64)
Object 1: x=28.34, y=-23.33, grid=(156,53)
Object 2: x=7.77, y=-20.06, grid=(115,59)
Object 0: x=-6.41, y=4.80, grid=(87,109)
Object 1: x=37.65, y=9.82, grid=(175,119)
Object 2: x=24.17, y=15.52, grid=(148,131)
Assigned object: class=0, cell=(31,68), size=(1.84, 4.42, 1.43)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 79.0
Heatmap max     : 1.0000
Size sum        : 465.8060
Height sum      : 75.4217
Offset sum      : -15720.1309
Rotation sum    : -8.3573
============================================================
Object 0: x=4.09, y=-13.33, grid=(108,73)
Object 1: x=2.62, y=6.84, grid=(105,113)
Object 2: x=6.67, y=-13.48, grid=(113,73)
Object 0: x=9.41, y=9.37, grid=(118,118)
Object 1: x=8.19, y=6.22, grid=(116,112)
Object 2: x=15.42, y=-39.25, grid=(130,21)
Assigned object: class=0, cell=(175,208), size=(2.01, 4.49, 1.49)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 92.0
Heatmap max     : 1.0000
Size sum        : 632.4050
Height sum      : 95.0951
Offset sum      : -18311.8477
Rotation sum    : 25.6639
============================================================
Object 0: x=-20.62, y=5.83, grid=(58,111)
Object 1: x=-64.86, y=7.41, grid=(-29,114)
Object 2: x=-90.99, y=7.85, grid=(-81,115)
Object 0: x=-8.41, y=3.54, grid=(83,107)
Object 1: x=-25.69, y=-0.77, grid=(48,98)
Object 2: x=-41.43, y=-26.65, grid=(17,46)
Assigned object: class=5, cell=(79,108), size=(0.79, 0.76, 1.79)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 64.0
Heatmap max     : 1.0000
Size sum        : 407.1940
Height sum      : 55.5739
Offset sum      : -12735.4414
Rotation sum    : 3.3727
============================================================
  [1][40/162] loss=207.4177 (hm=8.056 sz=3.653)
Object 0: x=16.29, y=-8.03, grid=(132,83)
Object 1: x=-24.69, y=-8.39, grid=(50,83)
Object 2: x=31.45, y=8.47, grid=(162,116)
Object 0: x=4.46, y=-22.89, grid=(108,54)
Object 1: x=-11.26, y=-22.13, grid=(77,55)
Object 2: x=4.44, y=-42.32, grid=(108,15)
Assigned object: class=0, cell=(104,2), size=(2.14, 4.73, 2.00)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 156.0
Heatmap max     : 1.0000
Size sum        : 786.7640
Height sum      : 103.4972
Offset sum      : -31045.0254
Rotation sum    : 21.6044
============================================================
Object 0: x=-41.87, y=6.32, grid=(16,112)
Object 1: x=-86.11, y=8.27, grid=(-72,116)
Object 2: x=-112.22, y=8.96, grid=(-124,117)
Object 0: x=22.99, y=-19.27, grid=(145,61)
Object 1: x=5.31, y=10.61, grid=(110,121)
Object 2: x=36.77, y=8.60, grid=(173,117)
Assigned object: class=9, cell=(93,118), size=(2.11, 0.45, 1.02)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 66.0
Heatmap max     : 1.0000
Size sum        : 436.5550
Height sum      : 54.4497
Offset sum      : -13135.6045
Rotation sum    : 3.9425
============================================================
Object 0: x=24.62, y=-8.05, grid=(149,83)
Object 1: x=-16.36, y=-8.53, grid=(67,82)
Object 2: x=39.80, y=8.49, grid=(179,116)
Object 0: x=4.68, y=-8.00, grid=(109,84)
Object 1: x=-36.30, y=-7.96, grid=(27,84)
Object 2: x=19.86, y=8.32, grid=(139,116)
Assigned object: class=0, cell=(109,60), size=(2.07, 4.59, 1.71)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 74.0
Heatmap max     : 1.0000
Size sum        : 616.9310
Height sum      : 63.6836
Offset sum      : -14718.0938
Rotation sum    : -19.2712
============================================================
Object 0: x=22.00, y=-17.21, grid=(143,65)
Object 1: x=23.98, y=6.03, grid=(147,112)
Object 2: x=18.25, y=-17.36, grid=(136,65)
Object 0: x=18.99, y=-12.62, grid=(137,74)
Object 1: x=-6.05, y=-6.77, grid=(87,86)
Object 2: x=1.68, y=3.38, grid=(103,106)
Assigned object: class=0, cell=(10,105), size=(2.09, 5.57, 2.00)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 27.0
Heatmap max     : 1.0000
Size sum        : 216.1080
Height sum      : 22.0647
Offset sum      : -5372.2734
Rotation sum    : 15.6370
============================================================
Object 0: x=-28.71, y=6.38, grid=(42,112)
Object 1: x=22.09, y=-25.43, grid=(144,49)
Object 2: x=10.83, y=-5.80, grid=(121,88)
Object 0: x=-7.27, y=4.28, grid=(85,108)
Object 1: x=-28.32, y=0.79, grid=(43,101)
Object 2: x=-18.83, y=-14.51, grid=(62,70)
Assigned object: class=5, cell=(81,110), size=(0.79, 0.76, 1.79)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 71.0
Heatmap max     : 1.0000
Size sum        : 508.6030
Height sum      : 79.4419
Offset sum      : -14122.2129
Rotation sum    : 18.1194
============================================================
Object 0: x=10.69, y=-5.26, grid=(121,89)
Object 1: x=7.96, y=-5.16, grid=(115,89)
Object 2: x=38.20, y=61.51, grid=(176,223)
Object 0: x=-8.43, y=3.52, grid=(83,107)
Object 1: x=-17.48, y=0.15, grid=(65,100)
Object 2: x=-15.95, y=15.79, grid=(68,131)
Assigned object: class=5, cell=(80,108), size=(0.79, 0.76, 1.79)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 49.0
Heatmap max     : 1.0000
Size sum        : 283.1420
Height sum      : 45.9372
Offset sum      : -9751.0996
Rotation sum    : 9.2855
============================================================
Object 0: x=10.18, y=13.54, grid=(120,127)
Object 1: x=8.87, y=9.83, grid=(117,119)
Object 2: x=14.32, y=-29.09, grid=(128,41)
Object 0: x=30.83, y=-20.30, grid=(161,59)
Object 1: x=11.41, y=-22.36, grid=(122,55)
Object 2: x=34.77, y=-39.50, grid=(169,20)
Assigned object: class=9, cell=(181,106), size=(2.06, 0.61, 1.18)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 131.0
Heatmap max     : 1.0000
Size sum        : 620.1620
Height sum      : 114.3191
Offset sum      : -26055.1992
Rotation sum    : 59.3535
============================================================
Object 0: x=8.13, y=-8.01, grid=(116,83)
Object 1: x=-32.85, y=-8.13, grid=(34,83)
Object 2: x=23.29, y=8.37, grid=(146,116)
Object 0: x=-35.74, y=11.56, grid=(28,123)
Object 1: x=-54.73, y=-7.31, grid=(-9,85)
Object 2: x=50.39, y=-18.98, grid=(200,62)
Assigned object: class=0, cell=(186,87), size=(1.99, 4.93, 1.86)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 88.0
Heatmap max     : 1.0000
Size sum        : 596.5870
Height sum      : 83.4782
Offset sum      : -17507.4922
Rotation sum    : -5.9661
============================================================
Object 0: x=-15.58, y=-6.15, grid=(68,87)
Object 1: x=0.20, y=-9.48, grid=(100,81)
Object 2: x=37.89, y=1.18, grid=(175,102)
Object 0: x=-5.73, y=4.45, grid=(88,108)
Object 1: x=69.55, y=25.79, grid=(239,151)
Object 2: x=-11.22, y=5.97, grid=(77,111)
Assigned object: class=0, cell=(102,92), size=(1.84, 4.42, 1.43)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 51.0
Heatmap max     : 1.0000
Size sum        : 298.9830
Height sum      : 59.3755
Offset sum      : -10144.5811
Rotation sum    : -11.9675
============================================================
Object 0: x=-17.13, y=3.33, grid=(65,106)
Object 1: x=8.41, y=-6.05, grid=(116,87)
Object 2: x=24.57, y=-6.88, grid=(149,86)
Object 0: x=27.16, y=-20.57, grid=(154,58)
Object 1: x=8.36, y=-22.55, grid=(116,54)
Object 2: x=30.96, y=-39.72, grid=(161,20)
Assigned object: class=8, cell=(173,35), size=(0.44, 0.44, 0.77)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 167.0
Heatmap max     : 1.0000
Size sum        : 733.9210
Height sum      : 170.5975
Offset sum      : -33231.5781
Rotation sum    : 41.1152
============================================================
  [1][50/162] loss=202.5654 (hm=4.177 sz=2.300)
Object 0: x=-8.40, y=3.55, grid=(83,107)
Object 1: x=-20.72, y=-0.05, grid=(58,99)
Object 2: x=-72.57, y=-41.11, grid=(-45,17)
Object 0: x=25.57, y=-6.04, grid=(151,87)
Object 1: x=2.91, y=-6.64, grid=(105,86)
Object 2: x=-24.19, y=-0.83, grid=(51,98)
Assigned object: class=0, cell=(29,110), size=(2.09, 5.57, 2.00)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 35.0
Heatmap max     : 1.0000
Size sum        : 222.8780
Height sum      : 34.9287
Offset sum      : -6961.3574
Rotation sum    : 9.1988
============================================================
Object 0: x=-11.28, y=-12.27, grid=(77,75)
Object 1: x=-19.02, y=-20.63, grid=(61,58)
Object 2: x=-77.06, y=-10.75, grid=(-54,78)
Object 0: x=12.18, y=21.16, grid=(124,142)
Object 1: x=11.96, y=17.80, grid=(123,135)
Object 2: x=14.90, y=-7.28, grid=(129,85)
Assigned object: class=0, cell=(102,106), size=(1.86, 4.55, 1.47)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 44.0
Heatmap max     : 1.0000
Size sum        : 311.8780
Height sum      : 33.5132
Offset sum      : -8754.6777
Rotation sum    : -4.7863
============================================================
Object 0: x=6.71, y=-15.68, grid=(113,68)
Object 1: x=27.03, y=-15.73, grid=(154,68)
Object 2: x=6.18, y=-18.05, grid=(112,63)
Object 0: x=-24.80, y=5.84, grid=(50,111)
Object 1: x=-69.05, y=7.38, grid=(-38,114)
Object 2: x=-95.18, y=7.77, grid=(-90,115)
Assigned object: class=0, cell=(129,152), size=(1.85, 4.21, 1.32)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 108.0
Heatmap max     : 1.0000
Size sum        : 654.8521
Height sum      : 95.6228
Offset sum      : -21488.7891
Rotation sum    : -0.7462
============================================================
Object 0: x=3.94, y=-8.92, grid=(107,82)
Object 1: x=22.58, y=10.80, grid=(145,121)
Object 2: x=29.80, y=25.37, grid=(159,150)
Object 0: x=-33.17, y=-6.30, grid=(33,87)
Object 1: x=-17.40, y=-9.51, grid=(65,80)
Object 2: x=22.28, y=2.85, grid=(144,105)
Assigned object: class=0, cell=(94,81), size=(1.79, 4.38, 1.66)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 84.0
Heatmap max     : 1.0000
Size sum        : 441.5470
Height sum      : 20.0872
Offset sum      : -16714.8965
Rotation sum    : 2.7198
============================================================
Object 0: x=50.97, y=-18.92, grid=(201,62)
Object 1: x=28.74, y=-21.32, grid=(157,57)
Object 2: x=55.65, y=-37.94, grid=(211,24)
Object 0: x=-1.61, y=-15.70, grid=(96,68)
Object 1: x=16.48, y=-6.82, grid=(132,86)
Object 2: x=-1.01, y=-18.12, grid=(97,63)
Assigned object: class=0, cell=(84,25), size=(1.80, 4.66, 1.50)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 122.0
Heatmap max     : 1.0000
Size sum        : 632.8210
Height sum      : 109.3416
Offset sum      : -24277.2812
Rotation sum    : 56.6051
============================================================
Object 0: x=-40.05, y=-8.59, grid=(19,82)
Object 1: x=-25.08, y=7.99, grid=(49,115)
Object 2: x=-19.46, y=7.98, grid=(61,115)
Object 0: x=-19.53, y=4.45, grid=(60,108)
Object 1: x=5.45, y=-6.34, grid=(110,87)
Object 2: x=21.56, y=-8.14, grid=(143,83)
Assigned object: class=0, cell=(175,86), size=(1.79, 4.38, 1.66)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 107.0
Heatmap max     : 1.0000
Size sum        : 661.6420
Height sum      : 113.1440
Offset sum      : -21286.0977
Rotation sum    : -20.7719
============================================================
Object 0: x=-15.12, y=0.59, grid=(69,101)
Object 1: x=11.64, y=-4.35, grid=(123,91)
Object 2: x=27.69, y=-2.37, grid=(155,95)
Object 0: x=-40.95, y=1.54, grid=(18,103)
Object 1: x=-47.31, y=9.05, grid=(5,118)
Object 2: x=-58.92, y=-5.70, grid=(-17,88)
Assigned object: class=8, cell=(-31,78), size=(0.59, 0.61, 0.86)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 174.0
Heatmap max     : 1.0000
Size sum        : 728.2800
Height sum      : 220.9567
Offset sum      : -34624.5078
Rotation sum    : -37.4089
============================================================
Object 0: x=-33.96, y=7.79, grid=(32,115)
Object 1: x=15.43, y=-26.12, grid=(130,47)
Object 2: x=5.04, y=-6.04, grid=(110,87)
Object 0: x=-32.53, y=11.64, grid=(34,123)
Object 1: x=-51.26, y=-7.12, grid=(-2,85)
Object 2: x=52.33, y=-19.00, grid=(204,61)
Assigned object: class=0, cell=(190,87), size=(1.99, 4.93, 1.86)

============================================================
TARGET DEBUG
============================================================
Positive pixels : 90.0
Heatmap max     : 1.0000
Size sum        : 597.3100
Height sum      : 97.9882
Offset sum      : -17914.0664
Rotation sum    : 23.1747
============================================================
Object 0: x=10.69, y=-4.72, grid=(121,90)
Object 1: x=7.98, y=-4.59, grid=(115,90)
Object 2: x=38.21, y=61.53, grid=(176,223)
Object 0: x=-32.41, y=-8.68, grid=(35,82)
Object 1: x=-40.77, y=-0.77, grid=(18,98)
Object 2: x=-47.62, y=-20.60, grid=(4,58)
Assigned object: class=8, cell=(-6,45), size=(0.59, 0.61, 0.86)

============================================================
