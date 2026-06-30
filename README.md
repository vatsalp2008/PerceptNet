# PerceptNet

**Production-grade multi-modal perception pipeline fusing camera and LiDAR for real-time 3D object detection and tracking.**

PerceptNet fuses RGB camera and LiDAR point clouds to perform 3D object detection, classification, and
multi-object tracking — mirroring the perception stack used in production autonomous-driving systems.
Each modality is processed independently (YOLO v8 for the camera, a PointPillars BEV backbone for
LiDAR), fused at the feature level via ROI-based late fusion, and the resulting detections are tracked
across time with a 3D Kalman filter + Hungarian association (AB3DMOT).

`PyTorch · PointPillars · YOLO v8 · ROI Fusion · AB3DMOT · TensorRT · ROS 2 · KITTI`

> ### ⚠️ Status & Reproducibility — read this first
>
> This repository is a **breadth-first scaffold**: the full architecture is in place and the
> CPU-runnable components (calibration, tracking, evaluation, fusion geometry, pretrained-YOLO
> inference) are **implemented and unit-tested**. The components that require an NVIDIA GPU
> (PointPillars/fusion **training**, **TensorRT** optimization) or **Linux + ROS 2** (the node) are
> implemented to run on that hardware but have **not been trained/benchmarked yet**.
>
> **Every number in the results tables below is a TARGET, not a measured result.** Targets come from
> the project specification and published PointPillars/AB3DMOT baselines. Cells are filled in only
> after a real run on the documented hardware. See [Component status](#component-status) for exactly
> what runs where.

---

## Architecture

```
[KITTI / nuScenes]
   ├── RGB frames ──► Camera Branch: YOLO v8 + FPN ──► 2D dets + image feature maps
   └── Point cloud ─► LiDAR Branch: Pillarize → PointPillars BEV → 3D proposals
                                  │
                                  ▼
                       Sensor Fusion (ROI-based late fusion)
                       project 3D proposals → image ROI → ROIAlign
                       → concat [LiDAR | image] features → MLP refine
                                  │
                                  ▼
                       3D detections (class, box, heading, conf)
                                  │
                                  ▼
                       AB3DMOT tracker (3D Kalman + Hungarian)
                                  │
            ┌─────────────────────┼───────────────────────┐
            ▼                     ▼                         ▼
   ROS 2 /perception/objects   Open3D / image viz    KITTI mAP + MOTA eval
                                                      TensorRT optimized inference
```

Full specification: [`Project_Perception_MultiModal_Fusion.md`](Project_Perception_MultiModal_Fusion.md).

---

## Results

### Detection — KITTI val, Moderate difficulty (3D AP) — *TARGET, untrained*

| Setting        | Car @0.7 | Pedestrian @0.5 | Cyclist @0.5 |
|----------------|----------|-----------------|--------------|
| Camera only (2D AP) | — | — | — |
| LiDAR only (3D AP)  | _target ~75%_ | _target ~55%_ | _target ~60%_ |
| **Fusion (3D AP)**  | _target ~77%_ | _target ~59%_ | _target ~63%_ |
| Δ fusion − LiDAR    | — | _target +4%_ | _target +3%_ |

### Inference benchmark (single frame) — *TARGET, not measured*

| Metric          | PyTorch CPU | PyTorch GPU | TRT FP16 | TRT INT8 |
|-----------------|-------------|-------------|----------|----------|
| Latency p50 (ms)| —           | —           | —        | _target ~22_ |
| Latency p99 (ms)| —           | —           | —        | —        |
| FPS             | —           | —           | —        | _target >40_ |
| mAP (Car Mod.)  | —           | —           | —        | —        |
| mAP drop vs FP32| —           | —           | —        | _target <1%_ |

### Modality robustness — *TARGET, not measured*

| Scenario               | Car mAP (Mod.) | Δ vs full fusion |
|------------------------|----------------|------------------|
| Full fusion (baseline) | _target ~77%_  | —                |
| Camera dropout         | —              | —                |
| LiDAR 50% dropout      | —              | _target <8% drop_ |
| LiDAR 90% dropout      | —              | —                |
| LiDAR-only @ 50% drop  | —              | _target ~31% drop_ |

---

## Quick start

```bash
# 1. Environment (macOS arm64 / Linux, Python 3.10–3.12; CPU is fine for the core)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -r requirements.txt

# 2. Sanity-check the install
python -c "import perceptnet; print(perceptnet.__version__)"

# 3. Run the CPU test suite (calibration, tracking, fusion geometry, eval)
make test

# 4. Dataset (manual — KITTI requires a free account; see Dataset setup)
make data        # verifies layout if present; prints download steps if absent

# 5. Train on a CUDA box (see Component status)
make train-lidar
make train-fusion
make evaluate
```

See the [Makefile](Makefile) for the full command list (`make help`).

---

## Component status

| Component | Runs on | Status |
|---|---|---|
| Data loader + **calibration** (Module 1) | macOS/Linux CPU | ✅ implemented + tested |
| Camera branch — YOLO v8 inference (Module 2) | CPU/MPS/GPU | ✅ pretrained inference works; KITTI fine-tune → GPU |
| LiDAR branch — PointPillars (Module 3) | CPU shape-test / GPU train | ✅ forward pass tested; ⏳ training needs GPU |
| Fusion — ROI late fusion (Module 4) | CPU geometry / GPU train | ✅ projection + dropout tested; ⏳ MLP training needs GPU |
| Tracker — AB3DMOT (Module 5) | CPU | ✅ implemented + tested |
| TensorRT / ONNX (Module 6) | ONNX on CPU / TRT on GPU | ✅ ONNX export; ⏳ TensorRT engine needs NVIDIA GPU |
| ROS 2 node (Module 7) | Linux + ROS 2 Humble | 🔧 implemented; runs only under ROS 2 |
| Evaluation — mAP / MOTA (Module —) | CPU | ✅ implemented + tested |

Legend: ✅ runs & tested here · ⏳ needs cloud NVIDIA GPU · 🔧 needs Linux + ROS 2.

---

## Project structure

```
perceptnet/        data · models · tracking · fusion · evaluation · optimization · visualization · ros2
scripts/           train_lidar · train_fusion · evaluate · infer_ros
configs/           kitti_pointpillars.yaml · fusion.yaml · ros2.yaml
tests/             unit tests + tests/fixtures/mini_kitti (committed, runs with zero download)
docs/              ADR-001 … ADR-007, architecture
docker/            Dockerfile.cuda (training) · Dockerfile.ros2 (node)
```

---

## Architecture Decision Records

1. [ADR-001 — ROI-based late fusion](docs/ADR-001-roi-fusion.md)
2. [ADR-002 — Pure-PyTorch PointPillars](docs/ADR-002-pointpillars.md)
3. [ADR-003 — AB3DMOT for 3D tracking](docs/ADR-003-ab3dmot.md)
4. [ADR-004 — TensorRT INT8 + ONNX subgraph strategy](docs/ADR-004-tensorrt-int8.md)
5. [ADR-005 — Temporal synchronization](docs/ADR-005-temporal-sync.md)
6. [ADR-006 — Modality dropout design](docs/ADR-006-modality-dropout.md)
7. [ADR-007 — ROS 2 over ROS 1](docs/ADR-007-ros2.md)

---

## Dataset setup

PerceptNet uses the **KITTI 3D Object Detection** benchmark. KITTI requires a free account, so the
data cannot be auto-downloaded.

1. Register at [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
2. Download: *left color images* (`image_2`), *Velodyne point clouds* (`velodyne`), *camera calibration*
   (`calib`), and *training labels* (`label_2`).
3. Unzip into:
   ```
   data/kitti/training/{image_2, velodyne, label_2, calib}
   data/kitti/testing/{image_2, velodyne, calib}
   ```
4. Run `make data` to verify the layout and frame counts.

The committed `tests/fixtures/mini_kitti/` lets the test suite and demos run **without** downloading
anything.

---

*Built by Vatsal Patel · [github.com/vatsalp2008/PerceptNet](https://github.com/vatsalp2008/PerceptNet)*
