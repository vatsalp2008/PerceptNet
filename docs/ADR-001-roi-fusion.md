# ADR-001: ROI-based late fusion over early fusion

**Status:** Accepted

## Context
Camera and LiDAR must be combined for 3D detection. Three strategies exist:
- **Early fusion** — paint LiDAR points with image features (or vice-versa) before the backbone. Tightly coupled, but projecting one modality into the other's frame discards information (LiDAR painted onto the image loses 3D structure; image features lifted to 3D are depth-ambiguous), and a single sensor failure breaks the whole input.
- **Late fusion** — run two independent detectors and merge boxes. Robust and simple, but it merges *decisions*, not features, so it misses cross-modal context (e.g. a sparse LiDAR cluster confirmed as a pedestrian by image texture).
- **ROI / middle fusion** — generate proposals in one modality and fuse per-proposal features from both.

## Decision
Use **ROI-based late fusion**: LiDAR (PointPillars) generates 3D proposals; each proposal's box is projected to the image, ROIAlign extracts camera FPN features there, and an MLP fuses `[LiDAR | image]` features into a refined class + box. This is the Frustum/MV3D-style approach used in production stacks.

## Consequences
- **+** Keeps 3D spatial reasoning in the LiDAR branch while adding camera texture exactly where it helps (small/distant Pedestrian/Cyclist).
- **+** Interpretable and modular; supports modality dropout (ADR-006) by zero-filling the image branch.
- **+** Each branch is independently trainable and benchmarkable.
- **−** Proposal-bound: objects the LiDAR branch never proposes can't be recovered by the camera (mitigated by a low proposal score threshold).
- **−** Requires accurate extrinsic calibration to place ROIs correctly (handled in `perceptnet.data.calibration`).
