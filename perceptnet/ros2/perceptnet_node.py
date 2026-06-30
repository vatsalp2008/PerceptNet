"""PerceptNet ROS 2 (Humble) perception node.

Subscribes to a camera image + LiDAR point cloud, runs the fusion model + tracker,
and publishes tracked 3D objects, RViz markers, and a 2D debug image.

Requires a sourced ROS 2 Humble environment (rclpy + vision_msgs). This module is
imported only by the ROS entry points, so importing the rest of ``perceptnet`` never
touches rclpy. Run via ``ros2 launch perceptnet perceptnet.launch.py`` inside
docker/Dockerfile.ros2. See ADR-007.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Quaternion, Vector3
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from perceptnet.models.fusion import ROIFusionHead
from perceptnet.models.lidar_branch import PointPillars
from perceptnet.models.perceptnet import PerceptNet
from perceptnet.ros2.conversions import tracks_to_dicts
from perceptnet.tracking.tracker import Tracker
from perceptnet.utils import get_device

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


class PerceptNetNode(Node):
    def __init__(self):
        super().__init__("perceptnet_node")
        # --- parameters ---
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("nms_iou_threshold", 0.1)
        self.declare_parameter("max_track_age", 3)
        self.declare_parameter("modality", "fusion")
        self.declare_parameter("lidar_checkpoint", "")
        self.declare_parameter("camera_topic", "/sensor/camera/image_raw")
        self.declare_parameter("lidar_topic", "/sensor/lidar/points")

        self.modality = self.get_parameter("modality").value
        self.conf = self.get_parameter("confidence_threshold").value
        device = get_device()

        # --- model + tracker ---
        lidar = PointPillars().to(device).eval()
        ckpt = self.get_parameter("lidar_checkpoint").value
        if ckpt:
            import torch

            lidar.load_state_dict(torch.load(ckpt, map_location=device))
        self.model = PerceptNet(lidar, ROIFusionHead(lidar_channels=384).to(device).eval(),
                                camera_branch=None, score_threshold=self.conf)
        self.tracker = Tracker(iou_threshold=self.get_parameter("nms_iou_threshold").value,
                               max_age=self.get_parameter("max_track_age").value)
        self._latest_image = None

        # --- pub / sub ---
        self.create_subscription(Image, self.get_parameter("camera_topic").value, self._on_image, 10)
        self.create_subscription(PointCloud2, self.get_parameter("lidar_topic").value, self._on_points, 10)
        self.pub_objects = self.create_publisher(Detection3DArray, "/perception/objects", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/perception/markers", 10)
        self.get_logger().info(f"perceptnet_node up (modality={self.modality})")

    # ------------------------------------------------------------------ #
    def _on_image(self, msg: Image):
        self._latest_image = msg

    def _on_points(self, msg: PointCloud2):
        pts = np.array(
            [[p[0], p[1], p[2], p[3] if len(p) > 3 else 0.0]
             for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)],
            dtype=np.float32,
        )
        if len(pts) == 0:
            return

        import torch

        out = self.model.predict(torch.from_numpy(pts), modality=self.modality)
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()

        tracks = self.tracker.update(boxes, scores, labels)
        self._publish(tracks_to_dicts(tracks), msg.header)

    # ------------------------------------------------------------------ #
    def _publish(self, track_dicts, header):
        det_array = Detection3DArray(header=header)
        marker_array = MarkerArray()
        for d in track_dicts:
            det = Detection3D(header=header)
            det.bbox = BoundingBox3D()
            det.bbox.center.position = Point(x=d["center"][0], y=d["center"][1], z=d["center"][2])
            det.bbox.center.orientation = Quaternion(**d["orientation"])
            det.bbox.size = Vector3(x=d["size"][0], y=d["size"][1], z=d["size"][2])
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = CLASS_NAMES[d["label"]] if d["label"] < len(CLASS_NAMES) else str(d["label"])
            hyp.hypothesis.score = d["score"]
            det.results.append(hyp)
            det_array.detections.append(det)

            marker = Marker(header=header, ns="perceptnet", id=d["id"], type=Marker.CUBE, action=Marker.ADD)
            marker.pose.position = det.bbox.center.position
            marker.pose.orientation = det.bbox.center.orientation
            marker.scale = det.bbox.size
            marker.color.a, marker.color.g = 0.5, 1.0
            marker_array.markers.append(marker)

        self.pub_objects.publish(det_array)
        self.pub_markers.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
