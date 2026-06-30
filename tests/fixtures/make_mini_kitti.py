"""Generate the committed mini-KITTI fixture (2 frames) used by tests and demos.

Deterministic (fixed seed). Produces a valid KITTI directory layout small enough to
commit, so the test suite and visualization demos run with zero dataset download.

Run:  python tests/fixtures/make_mini_kitti.py
"""

from pathlib import Path

import cv2
import numpy as np

from perceptnet.data.calibration import Calibration, boxes_camera_to_lidar

FIXTURE = Path(__file__).parent / "mini_kitti" / "training"
IMG_W, IMG_H = 1242, 375

# A real KITTI training/calib record (000000.txt).
CALIB_TXT = (
    "P0: 7.215377e+02 0.0 6.095593e+02 0.0 0.0 7.215377e+02 1.728540e+02 0.0 0.0 0.0 1.0 0.0\n"
    "P1: 7.215377e+02 0.0 6.095593e+02 -3.875744e+02 0.0 7.215377e+02 1.728540e+02 0.0 0.0 0.0 1.0 0.0\n"
    "P2: 7.215377e+02 0.0 6.095593e+02 4.485728e+01 0.0 7.215377e+02 1.728540e+02 2.163791e-01 0.0 0.0 1.0 2.745884e-03\n"
    "P3: 7.215377e+02 0.0 6.095593e+02 -3.395242e+02 0.0 7.215377e+02 1.728540e+02 2.199936e+00 0.0 0.0 1.0 2.729905e-03\n"
    "R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01\n"
    "Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01\n"
    "Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01\n"
)

# label_2 rows: type trunc occ alpha x1 y1 x2 y2  h w l  x y z  rotation_y
LABELS = {
    "000000": [
        "Car 0.00 0 -1.57 600.0 150.0 660.0 230.0 1.50 1.60 4.00 1.50 1.65 18.00 -1.57",
        "Car 0.00 1 -1.20 700.0 160.0 740.0 200.0 1.45 1.55 3.80 6.00 1.70 28.00 1.30",
        "Pedestrian 0.00 0 -0.50 500.0 140.0 520.0 220.0 1.75 0.60 0.80 -2.00 1.60 12.00 0.20",
        "DontCare -1 -1 -10 0.0 0.0 0.0 0.0 -1 -1 -1 -1000 -1000 -1000 -10",
    ],
    "000001": [
        "Car 0.00 0 -1.57 610.0 150.0 670.0 235.0 1.50 1.62 4.10 0.50 1.66 16.00 -1.55",
        "Cyclist 0.00 0 -1.00 540.0 145.0 565.0 215.0 1.70 0.55 1.75 -1.50 1.60 14.00 0.10",
    ],
}


def parse_box_cam(row: str) -> np.ndarray:
    p = row.split()
    h, w, l = float(p[8]), float(p[9]), float(p[10])
    x, y, z = float(p[11]), float(p[12]), float(p[13])
    ry = float(p[14])
    return np.array([x, y, z, l, w, h, ry])


def make_points(frame_id: str, calib: Calibration, rng) -> np.ndarray:
    # Background: random points in a forward frustum.
    n_bg = 1800
    x = rng.uniform(2.0, 45.0, n_bg)
    y = rng.uniform(-18.0, 18.0, n_bg)
    z = rng.uniform(-2.0, 0.5, n_bg)
    inten = rng.uniform(0.0, 1.0, n_bg)
    bg = np.stack([x, y, z, inten], axis=1)

    # Object clusters: points sprinkled around each GT box center (in LiDAR frame).
    clusters = []
    for row in LABELS[frame_id]:
        if row.startswith("DontCare"):
            continue
        box_lidar = boxes_camera_to_lidar(parse_box_cam(row)[None, :], calib)[0]
        center = box_lidar[:3]
        dims = box_lidar[3:6]
        n = 120
        offs = (rng.random((n, 3)) - 0.5) * dims[None, :]
        pts = center[None, :] + offs
        clusters.append(np.concatenate([pts, rng.uniform(0.2, 1.0, (n, 1))], axis=1))

    allpts = np.concatenate([bg] + clusters, axis=0) if clusters else bg
    return allpts.astype(np.float32)


def make_image(frame_id: str) -> np.ndarray:
    # A cheap deterministic gradient so the PNG is tiny but non-trivial.
    grad = np.linspace(40, 200, IMG_W, dtype=np.uint8)
    img = np.tile(grad[None, :], (IMG_H, 1))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main():
    for sub in ("calib", "label_2", "velodyne", "image_2"):
        (FIXTURE / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for frame_id in ("000000", "000001"):
        (FIXTURE / "calib" / f"{frame_id}.txt").write_text(CALIB_TXT)
        (FIXTURE / "label_2" / f"{frame_id}.txt").write_text("\n".join(LABELS[frame_id]) + "\n")
        calib = Calibration.from_file(FIXTURE / "calib" / f"{frame_id}.txt")
        pts = make_points(frame_id, calib, rng)
        pts.tofile(FIXTURE / "velodyne" / f"{frame_id}.bin")
        cv2.imwrite(str(FIXTURE / "image_2" / f"{frame_id}.png"), make_image(frame_id))
        print(f"{frame_id}: {len(pts)} points, image {IMG_W}x{IMG_H}, "
              f"{len([r for r in LABELS[frame_id] if not r.startswith('DontCare')])} objects")


if __name__ == "__main__":
    main()
