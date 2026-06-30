"""KITTI sensor calibration and coordinate-frame projection.

The projection chain (LiDAR point -> image pixel) is:

    x_img_homog = P2 @ R0_rect(4x4) @ Tr_velo_to_cam(4x4) @ [x, y, z, 1]^T
    (u, v) = x_img_homog[:2] / x_img_homog[2]

Shape conventions that cause the classic 3D bugs (handled here once):

  - ``P2``            is 3x4 (intrinsics + baseline of the left color camera).
  - ``R0_rect``       is 3x3, lifted to 4x4 with a 1 in the bottom-right corner.
  - ``Tr_velo_to_cam`` is 3x4, lifted to 4x4 with [0,0,0,1] as the last row.
  - Points behind the image plane (depth <= 0) must be filtered *before* the
    perspective divide or you get NaNs / wraparound pixels.
  - KITTI ``label_2`` boxes are in the rectified *camera* frame, not the LiDAR
    frame -- use :func:`boxes_camera_to_lidar` to bring them into the LiDAR frame
    used by the detector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perceptnet.geometry.heading import rotation_y_to_lidar_yaw


def cart_to_hom(points: np.ndarray) -> np.ndarray:
    """``(N, 3)`` -> ``(N, 4)`` by appending a column of ones."""
    points = np.asarray(points, dtype=np.float64)
    return np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)


class Calibration:
    """Holds KITTI calibration matrices and the projections between frames."""

    def __init__(self, P2, R0_rect, Tr_velo_to_cam):
        self.P2 = np.asarray(P2, dtype=np.float64).reshape(3, 4)
        self.R0 = np.asarray(R0_rect, dtype=np.float64).reshape(3, 3)
        self.V2C = np.asarray(Tr_velo_to_cam, dtype=np.float64).reshape(3, 4)

        self.R0_4x4 = np.eye(4)
        self.R0_4x4[:3, :3] = self.R0

        self.V2C_4x4 = np.eye(4)
        self.V2C_4x4[:3, :4] = self.V2C

        # Cached inverses for camera -> LiDAR (used for label boxes).
        self._R0_inv = np.linalg.inv(self.R0_4x4)
        self._V2C_inv = np.linalg.inv(self.V2C_4x4)

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path) -> "Calibration":
        """Parse a KITTI ``calib/xxxxxx.txt`` file."""
        values = {}
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, raw = line.split(":", 1)
            values[key.strip()] = np.array([float(v) for v in raw.split()])
        return cls(
            P2=values["P2"],
            R0_rect=values["R0_rect"],
            Tr_velo_to_cam=values["Tr_velo_to_cam"],
        )

    # ------------------------------------------------------------------ #
    # LiDAR <-> camera (rectified) <-> image
    # ------------------------------------------------------------------ #
    def velo_to_rect(self, points: np.ndarray) -> np.ndarray:
        """LiDAR ``(N, 3)`` -> rectified-camera ``(N, 3)``."""
        pts_h = cart_to_hom(points)
        rect = (self.R0_4x4 @ self.V2C_4x4 @ pts_h.T).T
        return rect[:, :3]

    def rect_to_velo(self, points_rect: np.ndarray) -> np.ndarray:
        """Rectified-camera ``(N, 3)`` -> LiDAR ``(N, 3)``."""
        pts_h = cart_to_hom(points_rect)
        velo = (self._V2C_inv @ self._R0_inv @ pts_h.T).T
        return velo[:, :3]

    def rect_to_image(self, points_rect: np.ndarray):
        """Rectified-camera ``(N, 3)`` -> (pixels ``(N, 2)``, depth ``(N,)``)."""
        pts_h = cart_to_hom(points_rect)
        img = (self.P2 @ pts_h.T).T            # (N, 3)
        depth = img[:, 2]
        # Guard the perspective divide; pixels for depth<=0 are meaningless and
        # are reported via the depth value (callers should mask on depth > 0).
        safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        uv = img[:, :2] / safe[:, None]
        return uv, depth

    def velo_to_image(self, points: np.ndarray):
        """LiDAR ``(N, 3)`` -> (pixels ``(N, 2)``, depth ``(N,)``)."""
        return self.rect_to_image(self.velo_to_rect(points))

    def project_lidar_to_image(
        self, points: np.ndarray, image_shape=None
    ):
        """Project LiDAR points to the image and return only the visible ones.

        Args:
            points: ``(N, 3)`` or ``(N, 4)`` (extra columns, e.g. intensity, ignored).
            image_shape: optional ``(height, width)`` to clip to the image bounds.

        Returns:
            ``(uv, depth, mask)`` where ``mask`` is the boolean index into the input
            of points that are in front of the camera (and inside the image, if
            ``image_shape`` is given). ``uv`` and ``depth`` are already masked.
        """
        points = np.asarray(points, dtype=np.float64)
        uv, depth = self.velo_to_image(points[:, :3])
        mask = depth > 1e-3                                  # in front of camera
        if image_shape is not None:
            h, w = image_shape[:2]
            mask &= (
                (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            )
        return uv[mask], depth[mask], mask


def boxes_camera_to_lidar(boxes_cam: np.ndarray, calib: Calibration) -> np.ndarray:
    """Convert KITTI camera-frame boxes to the canonical LiDAR box format.

    Args:
        boxes_cam: ``(N, 7)`` ``[x, y, z, l, w, h, rotation_y]`` where ``(x, y, z)`` is
            the *bottom-center* in rectified-camera coords (the KITTI label layout).
        calib: calibration for this frame.

    Returns:
        ``(N, 7)`` ``[x, y, z, dx, dy, dz, heading]`` in the LiDAR frame, with the
        center at the box centroid (z raised by h/2).
    """
    boxes_cam = np.atleast_2d(np.asarray(boxes_cam, dtype=np.float64))
    xyz_cam = boxes_cam[:, :3]
    l, w, h = boxes_cam[:, 3], boxes_cam[:, 4], boxes_cam[:, 5]
    ry = boxes_cam[:, 6]

    xyz_lidar = calib.rect_to_velo(xyz_cam)
    xyz_lidar[:, 2] += h / 2.0                       # bottom-center -> centroid
    yaw = rotation_y_to_lidar_yaw(ry)
    return np.column_stack([xyz_lidar, l, w, h, yaw])
