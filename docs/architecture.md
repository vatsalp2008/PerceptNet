# PerceptNet Architecture

End-to-end data flow. (A rendered `architecture.png` can be exported from this
diagram; the ASCII version is kept here as the diffable source of truth.)

```
                         ┌─────────────── KITTI frame ───────────────┐
                         │  image_2/<id>.png      velodyne/<id>.bin   │
                         │  calib/<id>.txt        label_2/<id>.txt    │
                         └───────────────┬───────────────┬───────────┘
                                         │               │
                  RGB image (3×H×W)      │               │  point cloud (N×4)
                                         ▼               ▼
                         ┌───────────────────┐   ┌────────────────────────────┐
                         │  Camera Branch     │   │  LiDAR Branch (PointPillars)│
                         │  YOLO v8 + FPN tap │   │  pillarize → PFN → scatter  │
                         │  → 2D dets + P3/4/5│   │  → 2D backbone → anchor head│
                         └─────────┬──────────┘   └──────────────┬──────────────┘
                                   │ FPN features                │ 3D proposals + BEV feats
                                   │                             │
                                   │     ┌───────────────────────▼───────────────┐
                                   │     │  decode proposals + gather BEV feature │
                                   │     └───────────────────────┬───────────────┘
                                   │                             │ proposals (K×7)
                  ┌────────────────▼─────────────────────────────▼───────────────┐
                  │              Sensor Fusion (ROI-based late fusion)            │
                  │  project box corners → 2D ROI (calibration P2·R0·Tr)          │
                  │  ROIAlign image features @ ROI → [LiDAR 128 | image 64] → MLP │
                  │  modality dropout: zero-fill image branch if camera missing   │
                  └───────────────────────────────┬──────────────────────────────┘
                                                   │ refined 3D detections
                                                   ▼
                  ┌────────────────────────────────────────────────────────────┐
                  │  AB3DMOT Tracker  (3D Kalman + Hungarian on 3D IoU)          │
                  │  state [x y z dx dy dz heading vx vy vz] → IDs + velocities  │
                  └───────────────────────────────┬──────────────────────────────┘
                                                   │
            ┌──────────────────┬──────────────────┼───────────────────┬─────────────────┐
            ▼                  ▼                  ▼                   ▼                 ▼
   ROS 2 Detection3DArray   RViz markers   Open3D / image viz   KITTI mAP / MOTA   TensorRT engine
   /perception/objects                                          evaluation         (FP16 / INT8)
```

## Coordinate frames
- **LiDAR (Velodyne):** x-forward, y-left, z-up. The canonical box format
  `[x, y, z, dx, dy, dz, heading]` lives here.
- **Camera (rectified):** x-right, y-down, z-forward. KITTI labels (`rotation_y`,
  bottom-center location) are in this frame; `perceptnet.data.calibration` converts
  them to the LiDAR frame.
- **Image:** pixels via `P2 · R0_rect(4×4) · Tr_velo_to_cam(4×4) · X`, divide by depth.

## Module → code map
| Stage | Code |
|---|---|
| Calibration / data | `perceptnet/data/{calibration,kitti_dataset,augmentation}.py` |
| Geometry primitives | `perceptnet/geometry/{boxes,heading,iou}.py` |
| Camera branch | `perceptnet/models/camera_branch.py` |
| LiDAR branch | `perceptnet/models/lidar_branch.py` |
| Fusion + wrapper | `perceptnet/models/{fusion,perceptnet}.py` |
| Tracking | `perceptnet/tracking/{kalman_filter_3d,hungarian,tracker}.py` |
| Evaluation | `perceptnet/evaluation/{kitti_eval,tracking_eval,robustness_study}.py` |
| Optimization | `perceptnet/optimization/{export_onnx,build_trt,benchmark}.py` |
| ROS 2 | `perceptnet/ros2/{perceptnet_node,conversions}.py` |
