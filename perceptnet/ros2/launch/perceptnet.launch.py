"""ROS 2 launch file for the PerceptNet perception node (+ optional RViz2).

Usage:
    ros2 launch perceptnet perceptnet.launch.py
    ros2 launch perceptnet perceptnet.launch.py rviz:=true modality:=lidar_only
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("perceptnet"))
    default_params = str(pkg_share / "config" / "ros2.yaml")

    params_file = LaunchConfiguration("params_file")
    modality = LaunchConfiguration("modality")
    rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("modality", default_value="fusion"),
        DeclareLaunchArgument("rviz", default_value="false"),
        Node(
            package="perceptnet",
            executable="perceptnet_node",
            name="perceptnet_node",
            output="screen",
            parameters=[params_file, {"modality": modality}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            condition=IfCondition(rviz),
            arguments=["-d", str(pkg_share / "rviz" / "perceptnet.rviz")],
        ),
    ])
