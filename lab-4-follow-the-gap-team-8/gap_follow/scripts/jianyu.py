#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: Subscribe to LIDAR
        # TODO: Publish to drive
        self.subscriber = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.subscriber  
        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.window_length=10

    def preprocess_lidar(self, ranges,start_point,end_point):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        ranges=ranges[start_point:end_point]
        proc_ranges = []
        for i in range(int(self.window_length/2),len(ranges)-int(self.window_length/2),self.window_length):
            proc_ranges.append(sum(ranges[int(i-self.window_length/2): int(i+self.window_length/2)]) / self.window_length)
        
        return np.array(proc_ranges)

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        start_idx = 0
        max_length = 0
        curr_length = 0
        threshold = 1.2
        for i in range(len(free_space_ranges)):
            if free_space_ranges[i] > threshold:
                curr_length+=1
                if curr_length>max_length:
                    max_length=curr_length
                    start_idx=i+1-max_length
            else:
                curr_length=0

        end_idx = start_idx + max_length - 1
        return start_idx, end_idx
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """

        return int((start_i+end_i)/2)
    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges
        start_point=180
        end_point=900
        proc_ranges = self.preprocess_lidar(ranges,start_point,end_point)
        # TODO:
        #Find closest point to LiDAR
        closest_point = np.argmin(proc_ranges)
        #Eliminate all points inside 'bubble' (set them to zero) 
        bubble_size=2
        free_space_ranges = np.array(proc_ranges, copy=True)
        free_space_ranges[max(closest_point-bubble_size, 0): min(closest_point+bubble_size, len(free_space_ranges)-1)+1] = 0
        #Find max length gap 
        start_idx, end_idx = self.find_max_gap(free_space_ranges)
        #Find the best point in the gap 
        best_point = self.find_best_point(start_idx, end_idx, free_space_ranges)
        #Publish Drive message
        angle=data.angle_min+start_point*data.angle_increment+ self.window_length*best_point*data.angle_increment
        if abs(angle) <= np.pi / 72:
            velocity = 5.0
        elif abs(angle) <= np.pi / 36:
            velocity = 4.0
        elif abs(angle) <= np.pi / 18:
            velocity = 4.0
        elif abs(angle) <= np.pi / 14:
            velocity = 3.0
        elif abs(angle) <= np.pi / 9:
            velocity = 2.0
        else:
            velocity = 1.0
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.publisher.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()