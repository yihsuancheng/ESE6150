#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import atexit
import transforms3d.euler as t3d_euler
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import os
from os.path import expanduser
from time import gmtime, strftime
from numpy import linalg as LA

home = expanduser("~")
log_dir = os.path.join(home, 'logs')
os.makedirs(log_dir, exist_ok=True)

file_path = os.path.join(log_dir, strftime("wp-%Y-%m-%d-%H-%M-%S", gmtime()) + ".csv")
file = open(file_path, "w")

class WaypointsLogger(Node):
    def __init__(self):
        super().__init__("waypoints_logger")
        self.subscription = self.create_subscription(Odometry, "pf/pose/odom", self.save_waypoint, 10)

    def save_waypoint(self, data):
        # quaternion = (
        #     data.pose.pose.orientation.x,
        #     data.pose.pose.orientation.y,
        #     data.pose.pose.orientation.z,
        #     data.pose.pose.orientation.w
        # )

        quaternion = (
            data.pose.pose.orientation.w,
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z
        )

        # Convert quaternion to euler angles
        euler = t3d_euler.quat2euler(quaternion, axes="sxyz")
        
        speed = np.linalg.norm(np.array([data.twist.twist.linear.x,
                                         data.twist.twist.linear.y,
                                         data.twist.twist.linear.z]))
        
        if data.twist.twist.linear.x > 0:
            self.get_logger().info(str(data.twist.twist.linear.x))

        file.write("%f, %f, %f, %f\n" % (data.pose.pose.position.x,
                                         data.pose.pose.position.y,
                                         euler[2], # yaw angle
                                         speed))
        
    def main(args=None):
        rclpy.init(args=args)
        waypoints_logger = WaypointsLogger()
        print("Saving Waypoint...")
        try:
            rclpy.spin(waypoints_logger)
        except KeyboardInterrupt:
            pass
        finally:
            waypoints_logger.destroy_node()
            file.close()
            print("Goodbye")
            rclpy.shutdown()

    if __name__ == "__main__":
        main()


