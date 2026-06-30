# ADR-002: Pure-PyTorch PointPillars over PointNet++ / spconv stacks

**Status:** Accepted

## Context
The LiDAR branch needs a 3D detector. Options:
- **PointNet++ / point-based** — operates on raw points; accurate but ~10× slower, with gather/sample ops that are awkward to deploy.
- **Voxel + sparse-conv (SECOND, PV-RCNN)** — strong accuracy but depends on `spconv`, which requires CUDA to even *import* on many builds and complicates portability.
- **PointPillars** — collapses the z-axis into vertical "pillars" and runs a **dense 2D CNN** on the resulting BEV pseudo-image. Production-proven, fast, hardware-friendly.

We also have a hard constraint: the package must import and shape-test on a CPU/MPS Mac, and train on a cloud GPU, with one codebase.

## Decision
Implement PointPillars in **pure PyTorch** — no `spconv`, `mmdet3d`, or OpenPCDet. PointPillars uses *dense* 2D convolution on a scattered pseudo-image, so sparse-conv libraries buy nothing here. Pillarization is `unique`/scatter indexing; the backbone is ordinary `Conv2d`/`ConvTranspose2d`.

## Consequences
- **+** `import perceptnet.models.lidar_branch` works on CPU/MPS; the forward pass is shape-testable locally (`tests/test_pointpillars.py`).
- **+** No fragile pinned `mmcv`/`spconv`/CUDA-compile step; trains unchanged on the GPU box.
- **+** ONNX-exportable dense graph (ADR-004).
- **−** The reference **BEV-IoU anchor assigner is pure NumPy** (correctness-first) and too slow for the full KITTI anchor grid; full-scale training should swap in a vectorized/CUDA BEV-IoU (e.g. `mmcv.ops.boxes_iou_bev`). Documented in `scripts/train_lidar.py`.
- **−** Pillarization runs on CPU (MPS `unique`/scatter are unreliable) before features move to the compute device — a minor host↔device copy.
