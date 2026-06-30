#!/usr/bin/env python
"""Entry point for ROS 2 inference. Delegates to the perceptnet_node.

Requires a sourced ROS 2 Humble environment (rclpy). Off-platform this prints a
clear message instead of a traceback. Prefer ``ros2 launch perceptnet
perceptnet.launch.py`` for real runs.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from perceptnet.ros2.perceptnet_node import main as node_main
    except ImportError as exc:
        print(f"ROS 2 not available ({exc}). Source /opt/ros/humble/setup.bash and run "
              "inside docker/Dockerfile.ros2.", file=sys.stderr)
        return 1
    node_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
