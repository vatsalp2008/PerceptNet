# ADR-005: Temporal synchronization of camera and LiDAR

**Status:** Accepted

## Context
Fusion assumes the camera frame and LiDAR sweep describe the same instant. KITTI ships pre-synchronized data, so the offline pipeline can pair files by index. A real vehicle does not: the camera (~10–30 Hz, global/rolling shutter) and a spinning LiDAR (10 Hz, points stamped *across* the ~100 ms sweep) arrive on independent clocks, and a 50 ms skew at 15 m/s is ~0.75 m of misalignment — enough to push ROIs off their objects.

## Decision
- **Offline (KITTI):** rely on the dataset's index-level synchronization; pair `image_2/<id>` with `velodyne/<id>`.
- **Online (ROS 2):** synchronize by timestamp. Buffer the most recent image and, on each LiDAR sweep, fuse against the image whose stamp is nearest the sweep's reference time (`message_filters.ApproximateTimeSynchronizer` in production). Reject pairs whose stamp gap exceeds a threshold (default 50 ms). Motion-compensate the sweep to a single reference time before projection when ego-motion is available.

## Consequences
- **+** Clean separation: the model is sync-agnostic; only the ROS node owns timing.
- **+** Explicit max-skew gate prevents silently fusing stale frames.
- **−** Approximate sync adds latency (waiting for the matching image) and drops frames under large skew.
- **−** Full per-point motion compensation needs an ego-motion source (IMU/odometry) not present in the basic KITTI object benchmark; deferred until the tracking/odometry input exists.
