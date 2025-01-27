#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers
        self.lidarscan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, "/drive", 10)


        # TODO: set PID gains
        self.kp = 2.5
        self.kd = 0.1
        self.ki = 0.01

        # TODO: store history
        self.integral = 0
        self.prev_error = 0
        self.error = 0

        self.desired_distance_to_wall = 1.0

        # TODO: store any necessary values you think you'll need

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """
        # convert angle to index
        #print("length of range_data", len(range_data.ranges))
        index = int((angle - range_data.angle_min) / range_data.angle_increment)
        index = max(0, min(index, len(range_data.ranges) - 1))
        range_distance = range_data.ranges[index]

        # Handle NaNs and infs
        if np.isinf(range_distance) or np.isnan(range_distance):
            return 10 # Assign a large distance
         #TODO: implement
        return range_distance

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        #TODO:implement
        theta = np.pi / 4
        b_range =  self.get_range(range_data, np.pi / 2)
        a_range = self.get_range(range_data, theta)
        alpha = np.arctan((a_range*np.cos(theta) - b_range)/(a_range*np.sin(theta)))
        D_t = b_range * np.cos(alpha)
        L = 1
        D_t1 = D_t + L * np.sin(alpha)
        return dist - D_t1

    def pid_control(self, error, velocity):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        
        # TODO: Use kp, ki & kd to implement a PID controller
        self.integral += error
        derivative = error - self.prev_error
        angle = self.kp * error + self.ki * self.integral + self.kd * derivative
        angle = -angle
        self.prev_error = error

        # Speed control logic
        if abs(angle) < 10:
            velocity = 1.5
        elif abs(angle) < 20:
            velocity = 1.0
        else:
            velocity = 0.5

        # Create and publish AckermannDriveStamped message
    
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.drive_publisher.publish(drive_msg)
        

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        error = self.get_error(msg, self.desired_distance_to_wall) # TODO: replace with error calculated by get_error()
        velocity = 1.0 # TODO: calculate desired car velocity based on error
        self.pid_control(error, velocity) # TODO: actuate the car with PID


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()