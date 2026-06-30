"""Unit tests for KITTI calibration / projection.

Uses a synthetic calibration whose transform is exactly the LiDAR->camera axis
permutation (velo x-fwd/y-left/z-up -> cam x-right/y-down/z-fwd), so projections
are hand-computable.
"""

import numpy as np
import pytest

from perceptnet.data.calibration import Calibration, boxes_camera_to_lidar

F, CX, CY = 700.0, 600.0, 180.0


def make_calib():
    # cam_x = -velo_y, cam_y = -velo_z, cam_z = velo_x
    V2C = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=float)
    R0 = np.eye(3)
    P2 = np.array([[F, 0, CX, 0], [0, F, CY, 0], [0, 0, 1, 0]], dtype=float)
    return Calibration(P2=P2, R0_rect=R0, Tr_velo_to_cam=V2C)


def test_point_straight_ahead_projects_to_principal_point():
    calib = make_calib()
    uv, depth = calib.velo_to_image(np.array([[10.0, 0.0, 0.0]]))
    assert np.allclose(uv, [[CX, CY]])
    assert np.allclose(depth, [10.0])


def test_point_to_the_left_projects_left_of_center():
    calib = make_calib()
    uv, depth = calib.velo_to_image(np.array([[10.0, 1.0, 0.0]]))
    # cam_x = -1 -> u = F*(-1)/10 + CX
    assert np.allclose(uv, [[F * (-1) / 10 + CX, CY]])
    assert np.allclose(depth, [10.0])


def test_point_above_projects_higher_in_image():
    calib = make_calib()
    uv, _ = calib.velo_to_image(np.array([[10.0, 0.0, 1.0]]))
    # cam_y = -1 -> v = F*(-1)/10 + CY  (smaller v = toward top)
    assert np.allclose(uv, [[CX, F * (-1) / 10 + CY]])


def test_velo_rect_roundtrip():
    calib = make_calib()
    pts = np.array([[12.3, -4.5, 0.8], [3.0, 2.0, -1.0]])
    rect = calib.velo_to_rect(pts)
    back = calib.rect_to_velo(rect)
    assert np.allclose(back, pts, atol=1e-9)


def test_project_lidar_to_image_filters_behind_camera():
    calib = make_calib()
    pts = np.array(
        [
            [10.0, 0.0, 0.0],     # in front
            [-5.0, 0.0, 0.0],     # behind camera (cam_z = -5)
            [20.0, 1.0, 0.0],     # in front
        ]
    )
    uv, depth, mask = calib.project_lidar_to_image(pts)
    assert mask.tolist() == [True, False, True]
    assert len(uv) == 2 and np.all(depth > 0)


def test_project_lidar_to_image_clips_to_bounds():
    calib = make_calib()
    # A point far to the side projects outside a small image and is dropped.
    pts = np.array([[10.0, 100.0, 0.0], [10.0, 0.0, 0.0]])
    _, _, mask = calib.project_lidar_to_image(pts, image_shape=(375, 1242))
    assert mask.tolist() == [False, True]


def test_intensity_column_ignored():
    calib = make_calib()
    pts_xyzi = np.array([[10.0, 0.0, 0.0, 0.5]])
    uv, depth, mask = calib.project_lidar_to_image(pts_xyzi)
    assert np.allclose(uv, [[CX, CY]]) and mask.tolist() == [True]


def test_boxes_camera_to_lidar_known_value():
    calib = make_calib()
    # Camera box: bottom-center at (0, 1.5, 10), l=4 w=2 h=1.5, rotation_y=0
    box_cam = np.array([[0.0, 1.5, 10.0, 4.0, 2.0, 1.5, 0.0]])
    box_lidar = boxes_camera_to_lidar(box_cam, calib)
    # rect_to_velo(0,1.5,10) = (10, 0, -1.5); centroid z += h/2 -> -0.75
    assert np.allclose(box_lidar[0, :3], [10.0, 0.0, -0.75], atol=1e-9)
    assert np.allclose(box_lidar[0, 3:6], [4.0, 2.0, 1.5])      # l, w, h preserved
    assert np.isclose(box_lidar[0, 6], -np.pi / 2)              # yaw from ry=0


def test_from_file_parses_real_kitti_calib(tmp_path):
    # A real KITTI training/calib record (000000.txt).
    calib_txt = (
        "P0: 7.215377e+02 0.0 6.095593e+02 0.0 0.0 7.215377e+02 1.728540e+02 0.0 0.0 0.0 1.0 0.0\n"
        "P1: 7.215377e+02 0.0 6.095593e+02 -3.875744e+02 0.0 7.215377e+02 1.728540e+02 0.0 0.0 0.0 1.0 0.0\n"
        "P2: 7.215377e+02 0.0 6.095593e+02 4.485728e+01 0.0 7.215377e+02 1.728540e+02 2.163791e-01 0.0 0.0 1.0 2.745884e-03\n"
        "P3: 7.215377e+02 0.0 6.095593e+02 -3.395242e+02 0.0 7.215377e+02 1.728540e+02 2.199936e+00 0.0 0.0 1.0 2.729905e-03\n"
        "R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01\n"
        "Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01\n"
        "Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01\n"
    )
    f = tmp_path / "000000.txt"
    f.write_text(calib_txt)
    calib = Calibration.from_file(f)
    assert calib.P2.shape == (3, 4)
    assert calib.R0.shape == (3, 3)
    assert calib.V2C.shape == (3, 4)
    # A LiDAR point ~10 m in front of the car should land near image center and in front.
    uv, depth, mask = calib.project_lidar_to_image(np.array([[10.0, 0.0, 0.0]]))
    assert mask[0] and depth[0] > 0
