#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
import transforms3d.euler as t3d_euler
import os
from os.path import expanduser
import csv
import math
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose
# TODO CHECK: include needed ROS msg type headers and libraries

class Waypoint(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self, waypoint_file):
        super().__init__('pure_pursuit_node')
        # TODO: create ROS subscribers and publishers
        #self.drive_publisher = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        
        self.odom_subscriber = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)
        self.waypoints_pub = self.create_publisher(PoseArray, "/waypoints_pose_array", 10)
        self.waypoints_publisher = self.create_publisher(MarkerArray, "/visualization_waypoints_array", 10)

        self.waypoint_file = waypoint_file
        self.lookahead_distance = 2.0
        self.velocity = 6.0
        self.wheelbase = 0.3302
        self.read_waypoints()

    def read_waypoints(self):
        home = expanduser("~")
        #filename = os.path.join(home, "Downloads", self.waypoint_file)
        
        #filename = "/home/yihsuan/sim_ws/src/lab-5-slam-and-pure-pursuit-team-8/pure_pursuit/waypoints/wp-2024-03-13-23-41-02.csv"
        filename = "/home/yihsuan/sim_ws/src/lab-5-slam-and-pure-pursuit-team-8/pure_pursuit/waypoints/traj_race_cl.csv"

        # filename = os.path.join("/home/sim_ws/src/lab-5-slam-and-pure-pursuit-team-8/pure_pursuit/waypoints/", self.waypoint_file)
        path_points_x_list = []
        path_points_y_list = []

        with open(filename) as f:
            csv_reader = csv.reader(f) # Create a CSV reader to parse the file
            for line in csv_reader:
                path_points_x_list.append(float(line[0]))
                path_points_y_list.append(float(line[1]))

        self.path_points_x = np.array(path_points_x_list)
        self.path_points_y = np.array(path_points_y_list)
        self.xy_points = np.hstack((self.path_points_x.reshape(-1,1), self.path_points_y.reshape(-1,1)))
        
    
    def pose_callback(self, pose_msg):
        #pass
        # TODO: find the current waypoint to track using methods mentioned in lecture

        # TODO: transform goal point to vehicle frame of reference

        # TODO: calculate curvature/steering angle

        # TODO: publish drive message, don't forget to limit the steering angle.

        quaternion = (
            pose_msg.pose.pose.orientation.w,
            pose_msg.pose.pose.orientation.x,
            pose_msg.pose.pose.orientation.y,
            pose_msg.pose.pose.orientation.z
        )
        euler = t3d_euler.quat2euler(quaternion, axes="sxyz")
        yaw = euler[2]
        #roll, pitch, yaw = self.euler_from_quaternion(quaternion)
        #print("yaw", yaw)
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y
        current_position = np.array([x,y]).reshape((1,2))
        #print("current_position", current_position)
        #print("self.xy_points", self.xy_points)
        #distance_array = np.linalg.norm(self.xy_points - current_position, axis=1)
        distance_array = self.calculate_distance(self.xy_points, current_position)
        target_index = np.where((distance_array > self.lookahead_distance - 0.3) & (distance_array < self.lookahead_distance + 0.3))[0]
        #print("target_index", target_index)

        goal_points = self.xy_points[target_index]

        goal_points_infront = []
        for i in range(len(goal_points)):
            v1 = goal_points[i] - current_position 
            v2 = [np.cos(yaw), np.sin(yaw)] # represents the direction the vehicle is facing
            angle = self.calculate_angle(v1, v2)
            if abs(angle) < np.pi/2:
                goal_points_infront.append(goal_points[i])
        
        goal_points_infront = np.array(goal_points_infront)
        #print("goal_points_infront", goal_points_infront)
        new_distance = self.calculate_distance(goal_points_infront, current_position) - self.lookahead_distance
        #new_distance = np.linalg.norm(goal_points_infront - current_position, axis=1) - self.lookahead_distance
        idx = np.argmin(new_distance)
        target_point = goal_points_infront[idx]
        #print("target_point", target_point)
        #self.publish_current_target_marker(target_point)
        #print("target_point", target_point)

        # transform into vehicle reference frame
        vector = target_point - current_position.reshape(-1)
        dx, dy = vector
        #print("vector", vector)
        x_prime = dx*np.cos(yaw) + dy*np.sin(yaw)
        y_prime = -dx*np.sin(yaw) + dy*np.cos(yaw)
        new_target = np.array([x_prime, y_prime])
        self.publish_current_target_marker(new_target)
        self.publish_waypoints_as_pose_array(new_target)
        print("target_waypoint", new_target)
        # kappa = 2 * y_prime / (self.lookahead_distance ** 2)
        # kappa = 1.0 * kappa # P control
        # steering_angle = np.arctan(kappa * self.wheelbase)
        
        # self.set_speed(steering_angle)
        #self.visualize_waypoints() # this can visualize the waypoints in RViz
        
    def publish_current_target_marker(self, target_point, r=1.0, g=0.0, b=0.0):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 9999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0  # Don't forget to set the alpha channel for visibility
        marker.pose.position.x = target_point[0]
        marker.pose.position.y = target_point[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0 # orientation of arrow, default point along x-axis

        marker.scale.x = 0.2 # length of arrow
        marker.scale.y = 0.2 # Width of arrow
        marker.scale.z = 0.2 # Height of arrow

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)

    def publish_waypoints_as_pose_array(self, target_point):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "/ego_racecar/laser"
        
        pose = Pose()
        pose.position.x = target_point[0]
        pose.position.y = target_point[1]
        pose.position.z = 0.0

        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        pose_array.poses.append(pose)

        self.waypoints_pub.publish(pose_array)

    def visualize_waypoints(self):
        marker_array = MarkerArray()
        for i, point in enumerate(self.xy_points):
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.waypoints_publisher.publish(marker_array)

    def calculate_distance(self, points, current_position):
        distance_array = []
        current_position = current_position.reshape(-1)
        for waypoint in points:
            dx = waypoint[0] - current_position[0]
            dy = waypoint[1] - current_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            distance_array.append(distance)
        distance_array = np.array(distance_array)
        return distance_array

    def calculate_angle(self, v1, v2):
        cos_angle = np.dot(v1, v2)
        #sin_angle = np.linalg.norm(np.cross(v1, v2))
        sin_angle = np.cross(v1, v2)
        angle = np.arctan2(sin_angle, cos_angle)
        #print("angle is", angle)
        return angle
    
def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized now")
    waypoint_file = "levine-waypoints.csv"
    waypoint_node = Waypoint(waypoint_file)
    rclpy.spin(waypoint_node)

    waypoint_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
