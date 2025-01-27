#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
# TODO: include needed ROS msg type headers and libraries
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 1.0
        # TODO: create ROS subscribers and publishers.

        # Publisher for AckermannDriveStamped messages to control the car
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # Subscriber for LaserScan messages to detect obstacles
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # Subscriber for Odometry messages to get the car's current speed
        self.odom_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)

        self.get_logger().info("Safety node has been successfully launched.")


    def odom_callback(self, odom_msg):
        # TODO: update current speed
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        # TODO: calculate TTC

        # Calculate iTTC for each point in the LaserScan
        ranges = np.array(scan_msg.ranges)
        #print("ranges is", ranges)
        angle_min = scan_msg.angle_min
        #print("angle_min is ", angle_min)
        angle_increment = scan_msg.angle_increment
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        #print("angles array are ", angles)
    
        # Calculate range rate
        range_rates = -self.speed * np.cos(angles) # Negative because a decreasing range indicates a collision

        # Calculate iTTC
        threshold = 1e-3
        safe_range_rates = np.where(np.abs(range_rates) < threshold, np.sign(range_rates) * threshold, range_rates)
        ittc = np.where(range_rates < 0, ranges / -safe_range_rates, np.inf)

        # Check if any iTTC value falls below the threshold, indicating imminent collision
        ttc_threshold = 2 # Threshold in seconds
        # if np.any(ittc < ttc_threshold):
        #     self.emergency_brake()
        self.emergency_brake()
        # TODO: publish command to brake
        #pass
    def emergency_brake(self):
        # Publish an AckermannDriveStamped message to stop the car
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 1.0
        self.drive_pub.publish(drive_msg)
        self.get_logger().info('Emergency brake triggered!')
        

def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()