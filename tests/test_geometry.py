"""Unit tests for the geometry primitives: box corners, heading conversions, IoU."""

import numpy as np
import pytest

from perceptnet.geometry.boxes import (
    box_bev_area,
    box_volume,
    boxes_to_bev_corners,
    boxes_to_corners_3d,
)
from perceptnet.geometry.heading import (
    decode_heading_sincos,
    encode_heading_sincos,
    lidar_yaw_to_rotation_y,
    normalize_angle,
    rotation_y_to_lidar_yaw,
)
from perceptnet.geometry.iou import iou_2d, iou_3d, iou_bev


# --------------------------------------------------------------------------- #
# Box corners
# --------------------------------------------------------------------------- #
def test_unit_box_corners_axis_aligned():
    corners = boxes_to_corners_3d([0, 0, 0, 2, 2, 2, 0])
    assert corners.shape == (8, 3)
    # A 2x2x2 box at the origin has corners at +/-1 on every axis.
    assert np.allclose(np.sort(np.unique(corners)), [-1, 1])
    assert np.allclose(corners.min(axis=0), [-1, -1, -1])
    assert np.allclose(corners.max(axis=0), [1, 1, 1])


def test_box_corner_extents_match_dims():
    corners = boxes_to_corners_3d([0, 0, 0, 2, 4, 6, 0])
    extent = corners.max(axis=0) - corners.min(axis=0)
    assert np.allclose(extent, [2, 4, 6])


def test_heading_rotation_swaps_xy_extent():
    # 90 deg yaw turns the length (dx) onto the y axis and width (dy) onto x.
    corners = boxes_to_corners_3d([0, 0, 0, 2, 1, 1, np.pi / 2])
    extent = corners.max(axis=0) - corners.min(axis=0)
    assert np.allclose(extent, [1, 2, 1], atol=1e-9)


def test_bev_corners_are_bottom_face():
    box = [1, 2, 3, 2, 2, 2, 0]
    bev = boxes_to_bev_corners(box)
    assert bev.shape == (4, 2)
    assert np.allclose(bev.mean(axis=0), [1, 2])     # centered at (x, y)


def test_box_volume_and_area():
    box = np.array([[0, 0, 0, 2, 3, 4, 0]])
    assert np.allclose(box_volume(box), [24])
    assert np.allclose(box_bev_area(box), [6])


def test_batched_corners_shape():
    boxes = np.zeros((5, 7))
    boxes[:, 3:6] = 1.0
    assert boxes_to_corners_3d(boxes).shape == (5, 8, 3)


def test_invalid_box_dim_raises():
    with pytest.raises(ValueError):
        boxes_to_corners_3d(np.zeros((3, 6)))


# --------------------------------------------------------------------------- #
# Heading
# --------------------------------------------------------------------------- #
def test_normalize_angle_range():
    angles = np.array([0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi, 7.0])
    out = normalize_angle(angles)
    assert np.all(out >= -np.pi) and np.all(out < np.pi)


def test_sincos_roundtrip():
    yaws = np.linspace(-np.pi, np.pi, 13, endpoint=False)
    decoded = decode_heading_sincos(encode_heading_sincos(yaws))
    assert np.allclose(normalize_angle(decoded - yaws), 0, atol=1e-12)


def test_rotation_y_lidar_yaw_is_involution():
    ry = np.linspace(-np.pi, np.pi, 17, endpoint=False)
    back = lidar_yaw_to_rotation_y(rotation_y_to_lidar_yaw(ry))
    assert np.allclose(normalize_angle(back - ry), 0, atol=1e-12)


def test_rotation_y_known_value():
    # rotation_y = 0 (car facing camera +x_cam) -> lidar yaw = -pi/2
    assert np.isclose(rotation_y_to_lidar_yaw(0.0), -np.pi / 2)


# --------------------------------------------------------------------------- #
# IoU — axis-aligned 2D
# --------------------------------------------------------------------------- #
def test_iou_2d_identical():
    b = np.array([[0, 0, 10, 10]])
    assert np.allclose(iou_2d(b, b), [[1.0]])


def test_iou_2d_disjoint():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[20, 20, 30, 30]])
    assert np.allclose(iou_2d(a, b), [[0.0]])


def test_iou_2d_half_overlap():
    a = np.array([[0, 0, 2, 2]])
    b = np.array([[1, 0, 3, 2]])      # overlap area 2, union 6
    assert np.allclose(iou_2d(a, b), [[2 / 6]])


# --------------------------------------------------------------------------- #
# IoU — rotated BEV / 3D
# --------------------------------------------------------------------------- #
def test_iou_bev_identical():
    box = np.array([[0, 0, 0, 2, 2, 1, 0]])
    assert np.allclose(iou_bev(box, box), [[1.0]], atol=1e-9)


def test_iou_bev_square_invariant_to_90deg():
    a = np.array([[0, 0, 0, 2, 2, 1, 0.0]])
    b = np.array([[0, 0, 0, 2, 2, 1, np.pi / 2]])     # same square, rotated 90
    assert np.allclose(iou_bev(a, b), [[1.0]], atol=1e-9)


def test_iou_bev_axis_aligned_partial():
    a = np.array([[0, 0, 0, 2, 2, 1, 0]])
    b = np.array([[1, 0, 0, 2, 2, 1, 0]])             # inter 1*2=2, union 6
    assert np.allclose(iou_bev(a, b), [[2 / 6]], atol=1e-9)


def test_iou_bev_disjoint():
    a = np.array([[0, 0, 0, 2, 2, 1, 0]])
    b = np.array([[10, 0, 0, 2, 2, 1, 0]])
    assert np.allclose(iou_bev(a, b), [[0.0]])


def test_iou_3d_identical():
    box = np.array([[1, 2, 3, 2, 3, 4, 0.3]])
    assert np.allclose(iou_3d(box, box), [[1.0]], atol=1e-9)


def test_iou_3d_no_z_overlap():
    a = np.array([[0, 0, 0, 2, 2, 2, 0]])
    b = np.array([[0, 0, 2, 2, 2, 2, 0]])             # stacked, touching, zero vol
    assert np.allclose(iou_3d(a, b), [[0.0]], atol=1e-9)


def test_iou_matrix_shape():
    a = np.zeros((3, 7)); a[:, 3:6] = 1.0
    b = np.zeros((4, 7)); b[:, 3:6] = 1.0
    assert iou_3d(a, b).shape == (3, 4)
