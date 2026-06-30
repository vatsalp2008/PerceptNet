# ADR-007: ROS 2 (Humble) over ROS 1

**Status:** Accepted

## Context
The perception node must integrate with downstream planning/control on a real platform. ROS 1 is EOL (Noetic ends in 2025), single-master, and has weaker real-time / Python support. ROS 2 (Humble, an LTS on Ubuntu 22.04) uses DDS middleware, lifecycle nodes, and first-class `rclpy`.

## Decision
Target **ROS 2 Humble**. Publish tracked objects as `vision_msgs/Detection3DArray`, RViz cubes as `visualization_msgs/MarkerArray`, and a 2D debug image. Subscribe to `sensor_msgs/Image` and `sensor_msgs/PointCloud2`.

Implementation rules that keep the rest of the project portable:
- **All `rclpy`/`vision_msgs` imports live in `perceptnet/ros2/perceptnet_node.py`** — importing the rest of `perceptnet` never requires ROS.
- **The numpy→message-shaping logic lives in `perceptnet/ros2/conversions.py`, free of any ROS import**, so it is unit-tested on macOS (`tests/test_ros_conversions.py`).
- Run inside `docker/Dockerfile.ros2` (base `ros:humble-perception`, which bundles `vision_msgs`); `rclpy` is **not** pip-installable.

## Consequences
- **+** Industry-current; deployable on real AV/robot stacks.
- **+** The package stays importable and testable off-platform; only the node needs a ROS environment.
- **−** ROS 2 on macOS/Apple Silicon is impractical, so the node is validated in the Docker image, not on the dev laptop.
- **−** `vision_msgs` nesting (`Detection3D.results[].hypothesis`) varies across distros; we pin to the Humble layout.
