# ADR-003: AB3DMOT (3D Kalman + Hungarian) over DeepSORT

**Status:** Accepted

## Context
Detections must be linked across frames into tracks with stable IDs and velocities. Candidates:
- **DeepSORT** — Kalman in 2D image space + a learned appearance (re-ID) embedding for association. Designed for 2D image tracking.
- **AB3DMOT** — a 3D Kalman filter with a constant-velocity model over the full 3D box state, associated by 3D IoU via the Hungarian algorithm. No appearance network.

## Decision
Use **AB3DMOT**. Our detections are already 3D boxes; a 3D-native state vector `[x, y, z, dx, dy, dz, heading, vx, vy, vz]` tracks position *and* velocity directly, and 3D-IoU association needs no re-ID embedding.

## Consequences
- **+** No appearance model → lower latency, no extra training, no camera dependency for tracking.
- **+** Velocity vectors fall out of the Kalman state, which downstream planning needs.
- **+** Pure NumPy/SciPy → runs anywhere, fully unit-tested (`tests/test_tracker.py`).
- **+** Heading handled with explicit orientation correction so a π-flipped detection never snaps the track 180°.
- **−** Pure-motion association struggles with long occlusions / crossing objects where appearance would disambiguate (acceptable for the AV baseline; re-ID is a stretch goal).
