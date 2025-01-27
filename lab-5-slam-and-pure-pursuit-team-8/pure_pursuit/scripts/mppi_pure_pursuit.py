#!/usr/bin/env python3
from PIL import Image

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
from mppi_car import Config, MPPI_Numba
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
        #self.real_test = self.get_parameter("real_test").get_parameter_value().bool_value
        self.real_test = False
        #odom_topic = "pf/viz/inferred_pose" if self.real_test else "/ego_racecar/odom"
        odom_topic = "ego_racecar/odom"
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, "/drive", 1)
        self.odom_subscriber = self.create_subscription(PoseStamped if self.real_test else Odometry, odom_topic, self.pose_callback, 1)
        self.lidarscan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 1)
        self.occ_grid_subscriber = self.create_subscription(OccupancyGrid, "/occ_grid", self.occ_grid_callback, 1)
        self.obs_distance_subscriber = self.create_subscription(Float64, "/obs_distance", self.obs_distance_callback, 1)

        self.marker_publisher = self.create_publisher(MarkerArray, "/visualization_marker_array", 1)
        self.waypoints_publisher = self.create_publisher(MarkerArray, "/visualization_waypoints_array", 1)
        self.waypoints_publisher2 = self.create_publisher(MarkerArray, "/visualization_waypoints_array2", 1)
        self.waypoints_pub = self.create_publisher(PoseArray, "/waypoints_pose_array", 1)

        self.wheelbase = 0.3302
        self.lookahead_distance = 1.0

        cfg = Config(T = 5.0,
                    dt = 0.1,
                    num_control_rollouts = int(1e3), # Same as number of blocks, can be more than 1024
                    num_vis_state_rollouts = 20,
                    seed = 1)
        #self.x0 = np.array([0,0,0])
        #self.xgoal = np.array([5.88,0.559])

        self.x0 = np.array([0.0,0.0,0.0])
        #self.xgoal = np.array([-7.28,9.96])
        self.xgoal = np.array([15.95,-5.413])
        self.kp = 1.0

        def create_obstacles():# Open the image file
            filename = "/home/yihsuan/sim_ws/src/f1tenth_gym_ros/maps/levine_all.pgm"
            img = Image.open(filename)

            # Convert the image to a numpy array
            img_array = np.array(img)
            x_shape, y_shape = img_array.shape
            print("x_shape", x_shape, y_shape)
            # Define a threshold
            white_threshold = 200


            y_indices, x_indices = np.where(img_array < white_threshold)
            x_origin, y_origin = -6.13, -4.49
            x_coord, y_coord = x_origin + 0.05 * x_indices, y_origin + 0.05 * (x_shape-y_indices)
            return np.vstack([x_coord, y_coord]).T
        # obstacle_positions = create_obstacles()
        # obstacle_size = len(obstacle_positions)
        # obstacle_radius = np.full(obstacle_size, 0.05)
        # #obstacle_radius = np.array([1.5, 1])
        # assert len(obstacle_positions)==len(obstacle_radius)

        # obstacle_positions = np.array([[-1.42, 8.98], [0.19, 4.17], [-0.5, 6.18], 
        #                               [-3.06, 7.68], [-3.46, 10.39], [-5.54, 11.87],
        #                               [-8.62, 10.35], [-6.26, 8.1], [-4.34, 5], [-2.81, 2.67]])
        # obstacle_radius = np.full(10, 1.5)
        # #obstacle_radius = np.array([2.5, 1.5, 1.5, 1.5, 1.6, 1.5, 1.5, 1.6, 1.5, 1.5])
        # obstacle_radius[0] = 1.0
        # # obstacle_radius[4] = 1.0
        # # obstacle_radius[7] = 1.0
        # print("obstacle_radius", obstacle_radius)

        # assert len(obstacle_positions)==len(obstacle_radius)
        # obstacle_positions = np.array([[5,4.5], [2,1]])
        # obstacle_radius = np.array([1.5, 1])
        # assert len(obstacle_positions)==len(obstacle_radius)

        mppi_params = dict(
            # Task specification
            dt=cfg.dt,
            x0=self.x0, # Start state
            xgoal=self.xgoal, # Goal position

            # For risk-aware min time planning
            goal_tolerance=1.2,
            dist_weight=10, #  Weight for dist-to-goal cost.

            lambda_weight=1.0, # Temperature param in MPPI
            num_opt=1, # Number of steps in each solve() function call.

            # Control and sample specification
            u_std=np.array([1.0, 0.5]), # Noise std for sampling linear and angular velocities.
            vrange = np.array([0.0, 1.0]), # Linear velocity range.
            wrange=np.array([-np.pi, np.pi]), # Angular velocity range.

            # obstacles
            # obstacle_positions=obstacle_positions,
            # obstacle_radius=obstacle_radius,
            obs_penalty=1e6,
        )
        np.random.seed(1)
        self.mppi_planner = MPPI_Numba(cfg)
        self.mppi_planner.setup(mppi_params)
        self.mppi_flag = False
        self.goal_reached = False
        self.goal_tolerance = mppi_params['goal_tolerance']
        self.trajectory = self.mppi_solve(mppi_params, cfg)

    def mppi_solve(self, mppi_params, cfg):
        # Loop
        max_steps = 151
        xhist = np.zeros((max_steps+1, 3))*np.nan
        uhist = np.zeros((max_steps, 2))*np.nan
        xhist[0] = self.x0

        vis_xlim = [-1, 8]
        vis_ylim = [-1, 6]

        for t in range(max_steps):
            # Solve
            useq = self.mppi_planner.solve()
            u_curr = useq[0]
            uhist[t] = u_curr

            # Simulate state forward using the sampled map
            xhist[t+1, 0] = xhist[t, 0] + cfg.dt*np.cos(xhist[t, 2])*u_curr[0]
            xhist[t+1, 1] = xhist[t, 1] + cfg.dt*np.sin(xhist[t, 2])*u_curr[0]
            xhist[t+1, 2] = xhist[t, 2] + cfg.dt*u_curr[1]

            # Update MPPI state (x0, useq)
            self.mppi_planner.shift_and_update(xhist[t+1], useq, num_shifts=1)

            # Goal check
            if np.linalg.norm(xhist[t+1, :2] - self.xgoal) <= self.goal_tolerance:
                #print("goal reached at t={:.2f}s".format(t*cfg.dt))
                print("Goal reached at step:", t)
                self.mppi_flag = True
                xhist = xhist[:t+2] # Slice to include the last valid state
                print("xhist", xhist)
                break
        return xhist

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
        #print("current_position", current_position)
        
        #distance_array = self.calculate_distance(self.xy_points[:, 0:2], current_position)
        distance_array = np.linalg.norm(self.trajectory[:, 0:2] - current_position, axis=1)
        #end_time = time.time()
        curr_idx = np.argmin(distance_array) # estimated index of current position of car
        self.current_index = curr_idx
        #print("curr_idx", curr_idx)

        if self.mppi_flag:
            next_points_idx = np.arange(1, 50, 1)
            potential_pts_idx = (next_points_idx + curr_idx) % len(self.trajectory)
            goal_pts = self.trajectory[potential_pts_idx]
            goal_dist = distance_array[potential_pts_idx]
            target_point_idx = np.argmin(abs(goal_dist - self.lookahead_distance))
            target_point = goal_pts[target_point_idx, 0:2]

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
            
            self.set_speed(steering_angle, current_position)
            self.visualize_waypoints() # this can visualize the waypoints in RViz
            #self.visualize_waypoints2()

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
        for i, point in enumerate(self.trajectory):
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
    
    def set_speed(self, angle, current_position):
        drive_msg = AckermannDriveStamped()
        if np.linalg.norm(current_position - self.xgoal) <= self.goal_tolerance:
            drive_msg.drive.speed = 0.0
        else:
            drive_msg.drive.speed = 1.0
        drive_msg.drive.steering_angle = angle
        #print("steering angle", angle)
        self.drive_publisher.publish(drive_msg)


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
