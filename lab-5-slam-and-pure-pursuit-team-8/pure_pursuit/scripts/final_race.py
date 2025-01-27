#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry, OccupancyGrid
import transforms3d.euler as t3d_euler
import os
from os.path import expanduser
import csv
import math
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
import time
from std_msgs.msg import Float64
# TODO CHECK: include needed ROS msg type headers and libraries

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self, waypoint_file):
        super().__init__('pure_pursuit_node')
        # TODO: create ROS subscribers and publishers

        self.declare_parameter("real_test")
        self.real_test = self.get_parameter("real_test").get_parameter_value().bool_value
        odom_topic = "pf/viz/inferred_pose" if self.real_test else "/ego_racecar/odom"

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, "/drive", 1)
        self.odom_subscriber = self.create_subscription(PoseStamped if self.real_test else Odometry, odom_topic, self.pose_callback, 1)
        self.lidarscan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 1)
        self.occ_grid_subscriber = self.create_subscription(OccupancyGrid, "/occ_grid", self.occ_grid_callback, 1)
        self.obs_distance_subscriber = self.create_subscription(Float64, "/obs_distance", self.obs_distance_callback, 1)

        self.marker_publisher = self.create_publisher(MarkerArray, "/visualization_marker_array", 1)
        self.waypoints_publisher = self.create_publisher(MarkerArray, "/visualization_waypoints_array", 1)
        self.waypoints_publisher2 = self.create_publisher(MarkerArray, "/visualization_waypoints_array2", 1)
        self.waypoints_pub = self.create_publisher(PoseArray, "/waypoints_pose_array", 1)

        self.waypoint_file = waypoint_file
        self.declare_parameter("lookahead")
        self.lookahead_distance = self.get_parameter("lookahead").get_parameter_value().double_value
        self.lookahead_distance = 1.0
        self.wheelbase = 0.3302

        self.declare_parameter("initial_velocity_scaling")
        self.initial_velocity_scaling = self.get_parameter("initial_velocity_scaling").get_parameter_value().double_value
        self.read_waypoints()

        # PID param
        self.prev_steer_error = 0.0
        self.steer_integral = 0.0
        self.prev_steer = 0.0
        self.prev_ditem = 0.0

        self.declare_parameter("kp1")
        self.declare_parameter("kd1")

        self.declare_parameter("kp2")
        self.declare_parameter("kd2")

        self.kp1 = self.get_parameter("kp1").get_parameter_value().double_value
        self.kd1 = self.get_parameter("kd1").get_parameter_value().double_value

        self.kp2 = self.get_parameter("kp2").get_parameter_value().double_value
        self.kd2 = self.get_parameter("kd2").get_parameter_value().double_value

        self.kp = self.kp1
        self.kd = self.kd2

        # check opponent
        self.occupied_value = 100
        self.corner_flag = False
        self.opp_detect = False
        self.prev_obs_dist = None
        self.obs_dist = None

        # Lane switching
        self.switch_lane = False
        self.last_switch_time = None
        self.declare_parameter("cooldown")
        self.switch_cooldown = self.get_parameter("cooldown").get_parameter_value().double_value

        # Initialize lane
        self.curr_lane = self.lane1
        self.prev_lane = 1
        self.current_index = None
        self.scale_velocity = 1.0

        # Do not switch langes within these ranges
        self.lane1_ranges = [(18, 30), (74, 81), (148, 178)]
        self.lane2_ranges = [(13, 26), (71, 77), (153, 183)]
        self.lane_ranges = self.lane1_ranges

    def read_waypoints(self):
        home = expanduser("~")
        
        filename = "/home/yihsuan/sim_ws/src/lab-5-slam-and-pure-pursuit-team-8/pure_pursuit/waypoints/race_line2_width_1.8.csv"
        
        path_points_x_list = []
        path_points_y_list = []
        path_points_v_list = []

        with open(filename) as f:
            csv_reader = csv.reader(f, delimiter=';') # Create a CSV reader to parse the file
            next(csv_reader)
            next(csv_reader)
            next(csv_reader)
            for line in csv_reader:
                path_points_x_list.append(float(line[1]))
                path_points_y_list.append(float(line[2]))
                path_points_v_list.append(float(line[5]))

        self.path_points_x = np.array(path_points_x_list)
        self.path_points_y = np.array(path_points_y_list) + 0.0
        self.path_points_v = np.array(path_points_v_list) * self.initial_velocity_scaling
        self.xy_points = np.hstack((self.path_points_x.reshape(-1,1), self.path_points_y.reshape(-1,1), self.path_points_v.reshape(-1,1)))
        self.lane1 = self.xy_points
        #print("xyv ", self.xy_points)

        self.path_points2_x = np.array(path_points_x_list)
        self.path_points2_y = np.array(path_points_y_list) - 0.0
        self.path_points2_v = np.array(path_points_v_list) * self.initial_velocity_scaling
        self.xy_points2 = np.hstack((self.path_points2_x.reshape(-1,1), self.path_points2_y.reshape(-1,1), self.path_points2_v.reshape(-1,1)))
        self.lane2 = self.xy_points2

    def occ_grid_callback(self, occ_grid_msg):
        max_value = max(occ_grid_msg.data)
        print("max_occ_value", max_value)
        self.opp_detect = False

        for value in occ_grid_msg.data:
            if value == self.occupied_value:
                self.opp_detect = True
                break

    def scan_callback(self, msg):
        ranges = msg.ranges
        limit_range = np.asarray(ranges).copy()

    def obs_distance_callback(self, dist_msg):
        #print("distance to obstacle", dist_msg.data)
        self.prev_obs_dist = self.obs_dist
        self.obs_dist = dist_msg.data
        print(f"Previous distance: {self.prev_obs_dist}, New distance: {self.obs_dist}")

    def pose_callback(self, pose_msg):
        #pass
        # TODO: find the current waypoint to track using methods mentioned in lecture

        # TODO: transform goal point to vehicle frame of reference

        # TODO: calculate curvature/steering angle

        # TODO: publish drive message, don't forget to limit the steering angle.
        quat = pose_msg.pose.orientation if self.real_test else pose_msg.pose.pose.orientation

        quaternion = (
            quat.w,
            quat.x,
            quat.y,
            quat.z
        )
        euler = t3d_euler.quat2euler(quaternion, axes="sxyz")
        yaw = euler[2]
        #roll, pitch, yaw = self.euler_from_quaternion(quaternion)
        #print("yaw", yaw)
        x = pose_msg.pose.position.x if self.real_test else pose_msg.pose.pose.position.x
        y = pose_msg.pose.position.y if self.real_test else pose_msg.pose.pose.position.y
        current_position = np.array([x,y]).reshape((1,2))
        print("current_position", current_position)
        
        self.check_lane(self.opp_detect, self.current_index)
        
        #distance_array = self.calculate_distance(self.xy_points[:, 0:2], current_position)
        distance_array = np.linalg.norm(self.curr_lane[:, 0:2] - current_position, axis=1)
        #end_time = time.time()
        curr_idx = np.argmin(distance_array) # estimated index of current position of car
        self.current_index = curr_idx
        #print("curr_idx", curr_idx)

        next_points_idx = np.arange(1, 50, 1)
        potential_pts_idx = (next_points_idx + curr_idx) % len(self.curr_lane)
        goal_pts = self.curr_lane[potential_pts_idx]
        goal_dist = distance_array[potential_pts_idx]
        target_point_idx = np.argmin(abs(goal_dist - self.lookahead_distance))
        target_point = goal_pts[target_point_idx, 0:2]

        #print("target_point", target_point)
        #target_velocity = goal_pts[target_point_idx, 2] * self.scale_velocity
        target_velocity = goal_pts[target_point_idx, 2]
        #print("target_velocity", target_velocity)
        #print("target_velocity with scale", target_velocity * self.scale_velocity)

        self.publish_current_target_marker(target_point)
        #print("target_point", target_point)

        # transform into vehicle reference frame
        vector = target_point - current_position.reshape(-1)
        dx, dy = vector
        #print("vector", vector)

        R = np.array([[np.cos(yaw), np.sin(yaw)],
                      [-np.sin(yaw), np.cos(yaw)]])
        x_prime, y_prime = R @ np.array([dx, dy])
    
        new_target = np.array([x_prime, y_prime])
        #self.publish_current_target_marker(new_target)
        #self.publish_waypoints_as_pose_array(new_target)
        
        kappa = 2 * y_prime / (self.lookahead_distance ** 2)
        kappa = self.kp * kappa # P control
        #steering_angle = kappa
        #print("kappa", new_kappa)
        #kappa = self.PID(kappa)
        #print("new kappa", kappa)
        steering_angle = np.arctan(kappa * self.wheelbase)
        
        self.set_speed(steering_angle, target_velocity)
        self.visualize_waypoints() # this can visualize the waypoints in RViz
        self.visualize_waypoints2()

    def check_lane(self, opp_detect, curr_idx):
        # if detect opp, switch lane
        if curr_idx is None:
            return
        current_time = time.time()

        # TODO: How to deal with detecting opponent but shouldn't switch lanes 
        if self.last_switch_time is None or (current_time - self.last_switch_time) > self.switch_cooldown:
            if opp_detect and not any(start <= curr_idx <= end for start, end in self.lane_ranges): # segments where lanes overlap, don't switch
                if self.prev_lane == 1:
                    self.curr_lane = self.lane2
                    self.prev_lane = 2
                    self.kp = self.kp2
                    self.kd = self.kd2
                    self.lane_ranges = self.lane2_ranges
                    #print("lane2")
                else:
                    self.curr_lane = self.lane1
                    self.prev_lane = 1
                    self.kp = self.kp1
                    self.kd = self.kd1
                    self.lane_ranges = self.lane1_ranges
                    #print("lane1")
                self.last_switch_time = time.time()
                self.switch_lane = True
            # self.prev_lane = self.curr_lane
            else:
                self.switch_lane = False
        
            if opp_detect and self.switch_lane is False:
                # use AEB
                self.AEB(self.prev_obs_dist, self.obs_dist)
        # just switched lane but detect opponent and has cooldown time
        elif self.last_switch_time is not None:
            can_switch = (current_time - self.last_switch_time) > self.switch_cooldown
            if (opp_detect and can_switch is False):
                self.AEB(self.prev_obs_dist, self.obs_dist)

    def AEB(self, prev_obs_dist, curr_obs_dist):
        # slow down
        if curr_obs_dist <= prev_obs_dist:
            self.scale_velocity = curr_obs_dist / self.lookahead_distance
        else:
            self.scale_velocity = 1.0

    def publish_current_target_marker(self, target_point, r=1.0, g=0.0, b=0.0):
        marker = Marker()
        #marker.header.frame_id = "/ego_racecar/laser"
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
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.waypoints_publisher.publish(marker_array)

    def visualize_waypoints2(self):
        marker_array = MarkerArray()
        for i, point in enumerate(self.xy_points2):
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
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)
        self.waypoints_publisher2.publish(marker_array)

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
    
    def euler_from_quaternion(self, quaternion):
        qx, qy, qz, qw = quaternion
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        pitch = np.arcsin(2*(qw*qy - qz*qx))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        return roll, pitch, yaw
    
    def set_speed(self, angle, speed):
       
        drive_msg = AckermannDriveStamped()
        velocity = speed
        drive_msg.drive.speed = 1.0
        #print("speed", velocity)
        drive_msg.drive.steering_angle = angle
        #print("steering angle", angle)
        self.drive_publisher.publish(drive_msg)

    def PID(self, error):
        d_error = error - self.prev_steer_error
        self.prev_ditem = d_error
        self.prev_steer_error = error

        steer = self.kp * error + self.kd * d_error
        #print(f'cur_p_item:{self.kp * error},  cur_d_item:{self.kd * d_error}')
        #print("steer", steer)
        return steer

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized now")
    waypoint_file = "levine-waypoints.csv"
    pure_pursuit_node = PurePursuit(waypoint_file)
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
