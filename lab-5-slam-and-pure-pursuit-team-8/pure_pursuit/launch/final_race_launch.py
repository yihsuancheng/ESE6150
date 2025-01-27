#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    ld = LaunchDescription()
    pure_pursuit_node = Node(
        package="pure_pursuit",
        executable="final_race.py",
        name="pure_pursuit_node",
        output="screen",
        parameters=[
            {"real_test": False},
            {"lookahead": 3.2},
            {"initial_velocity_scaling": 0.9},
            {"kp1": 2.0},
            {"kd1": 1.0},
            {"kp2": 2.0},
            {"kd2": 1.0},
            {"cooldown": 1.0}
            
        ]
    )
    ld.add_action(pure_pursuit_node)

    occ_grid_node = Node(
        package="pure_pursuit",
        executable="occ_grid_node",
        name="occ_grid_node",
        output="screen",
        parameters=[
            {"grid_width": 0.5},
            {"grid_height": 2.0},
            {"inflate": 5.0},
            {"lookahead_dist": 1.0}
        ]
    )

    ld.add_action(occ_grid_node)

    return ld