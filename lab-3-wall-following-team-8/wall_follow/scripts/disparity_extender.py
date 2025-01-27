#!/usr/bin/env python3

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import rclpy
from rclpy.node import Node
import numpy as np
import math

class DisparityExtender(Node):
    def __init__(self):
        super().__init__("disparity_extender_node")

        lidarscan_topic = "/scan"
        drive_topic = "/drive"

        # Create publisher and subscriber
        self.lidarscan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # Configuration parameters 
        self.car_width = 0.20
        self.disparity_threshold = 0.20
        self.scan_width = 270
        self.lidar_max_range = 30
        self.front_turn_clearance = 0.15
        self.rear_turn_clearance = 0.20
        self.max_turn_angle = 34.0
        self.min_speed = 0.37
        self.max_speed = 3.5
        self.absolute_max_speed = 6.0
        self.min_distance = 0.15
        self.max_distance = 3.0
        self.absolute_max_distance = 6.0
        self.no_obstacles_distance = 6.0
        self.lidar_distance = None
        self.angle_step = 0.25 * math.pi/180

        self.friction_coefficient = 0.62
        self.wheelbase = 0.3302                       # TODO Need to find the correct wheelbase
        self.gravity = 9.81
    
    def scan_callback(self, msg):
        ranges = msg.ranges
        limit_range = np.asarray(ranges).copy()
        # Make the lidar scan range from -90 to 90
        limit_range[0:180] = 0.0
        limit_range[901:] = 0.0
        limit_range[901] = limit_range[900] # make sure the last element is not detected as disparity

        # ignore distances that are far
        #index = np.where(new_range>=10)[0]
        #new_range[index] = msg.range_max-0.1

        # calculate disparities between samples
        threshold = self.disparity_threshold
        car_width = self.car_width
        disparities = self.find_disparities(limit_range, threshold)

        # extend disparities
        new_range = self.extend_disparities(limit_range, disparities, car_width)

        # compute max value of new_range to determine where to go
        max_val = max(new_range)
        target_distances = np.where(new_range==max_val)[0]
        print("the max distance is ", target_distances)

        drive_distance = self.calculate_target_distance(target_distances)
        drive_angle = self.calculate_angle(drive_distance)

        car_front = np.asarray(ranges).copy()
        car_rear = np.asarray(ranges).copy()
        car_front_left, car_front_right = car_front[540:900], car_front[180:540]
        car_rear_left, car_rear_right = car_rear[0:180], car_rear[901:]
        #print("car_rear_left", car_rear_left)
        #print("car_rear_right", car_rear_right)
        safe_to_turn_angle = self.calculate_safe_angle(car_front_left, car_front_right, car_rear_left, car_rear_right, drive_angle)
        print("safe_to_turn_angle")
        # TODO check if turning, use max velocity
        #max_velocity = self.calculate_max_velocity_for_turning(safe_to_turn_angle)
        # velocity = self.set_velocity(safe_to_turn_angle) TODO implement this later
        #velocity = 1.5
        drive_msg = AckermannDriveStamped()
        velocity = self.set_safe_speed(safe_to_turn_angle)
        print("velocity", velocity)
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = float(safe_to_turn_angle)
        
        self.drive_publisher.publish(drive_msg)
        

    def find_disparities(self, range_data, threshold):
        disparity_array = []
        for i in range(180, 901):
            if abs(range_data[i] - range_data[i+1]) >= threshold:
                disparity_array.append(i)
        return disparity_array
   
    def extend_disparities(self, limit_range, disparity_index, car_width):
        extended_range = np.copy(limit_range)
        for i in disparity_index:
            extend_left, extend_right = False, False
            value1 = extended_range[i]
            value2 = extended_range[i+1]

            # if value1 < value2 extend left, else extend right
            if value1 < value2:
                near_value = value1
                near_index = i
                extend_left = True  # should extend left
                
            else:
                near_value = value2
                near_index = i+1
                extend_right = True # should extend right
                
            # Compute number of ranges to extend disparity using arc = angle * radius
            arc_length = self.angle_step * near_value
            numbers_extend = int(math.ceil(car_width / arc_length))

            # Now replace the lidar ranges from extending disparities
            cur_index = near_index
            for j in range(numbers_extend):
                if cur_index < 180 or cur_index >= 901:
                    break # ignore

                if extended_range[cur_index] > near_value:
                    extended_range[cur_index] = near_value # do not have to replace the first nearest value
                
                if extend_left:
                    cur_index += 1
                elif extend_right:
                    cur_index -= 1
                
        return extended_range
    
    def calculate_target_distance(self, target_distance):
        if len(target_distance) == 1:
            return target_distance[0]
        else:
            mid_index = len(target_distance) // 2 # if there are more than one index with max distance, use the middle index
            return target_distance[mid_index]
            
    def calculate_angle(self, drive_distance_index):
        angle = drive_distance_index - 540 # this is the angle relative to the car velocity where at 540 degrees the car is at the centerline
        rad = angle * self.angle_step
        max_rad = self.max_turn_angle * math.pi / 180
        
        if rad < -max_rad:
            return -max_rad
        elif rad > max_rad:
            return max_rad   # 0.59 means turning
        return rad

    def calculate_safe_angle(self, left_distance_front, right_distance_front, left_distance_behind, right_distance_behind, angle):
        # might need to check disparity
        # TODO check turn to early
        min_left_front, min_right_front, min_left_rear, min_right_rear = min(left_distance_front), min(right_distance_front), min(left_distance_behind), min(right_distance_behind)
        print("min_left_front", min_left_front)
        
        if angle > 0 and min_left_front <= self.front_turn_clearance:
            #self.get_logger().info(f"Not enough clearance to turn left {min_left}")
            angle = 0 # go straight do not turn
        elif angle > 0 and min_left_rear <= self.front_turn_clearance:
            angle = 0
        if angle < 0 and min_right_front <= self.rear_turn_clearance:
            #self.get_logger().info(f"Not enough clearance to turn right{min_right}")
            angle = 0
        elif angle < 0 and min_right_rear <= self.rear_turn_clearance:
            angle = 0
        else:
            self.get_logger().info(f"Safe to turn {angle}")
        
        return angle

    def set_safe_speed(self, angle):
        angle = abs(angle)
        if angle <= 0.05:
            speed = 6.0
        elif angle <= 0.1:
            speed = 5.5
        elif angle <= 0.2:
            speed = 5.0
        elif angle <= 0.3:
            speed = 4.5
        elif angle <= 0.4:
            speed = 4.0
        else:
            speed = 1.5
        return speed
    
def main(args=None):
    rclpy.init(args=args)
    print("Disparity Extender Initialized")
    disparity_extender_node = DisparityExtender()
    rclpy.spin(disparity_extender_node)
    disparity_extender_node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
  





