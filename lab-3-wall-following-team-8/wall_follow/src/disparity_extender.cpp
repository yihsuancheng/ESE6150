#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp" 
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

class DisparityExtender : public rclcpp::Node{
public:
    DisparityExtender() : Node("disparity_extender_node") {
        auto lidarscan_topic = "/scan";
        auto drive_topic = "/drive";

        lidarscan_subscription = this->create_subscription<sensor_msgs::msg::LaserScan>(
            lidarscan_topic, 10, std::bind(&DisparityExtender::scan_callback, this, std::placeholders::_1)
        );

        drive_publisher = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic, 10);

        car_width = 0.30;
        disparity_threshold = 0.30;
        scan_width = 270;
        lidar_max_range = 30;
        front_turn_clearance = 0.30;
        rear_turn_clearance = 0.20;
        max_turn_angle = 34.0;
        min_speed = 0.37;
        max_speed = 3.5;
        absolute_max_speed = 6.0;
        min_distance = 0.15;
        max_distance = 3.0;
        absolute_max_distance = 6.0;
        no_obstacles_distance = 6.0;
        //lidar_distance = None;
        angle_step = 0.25 * M_PI/180;

        friction_coefficient = 0.62;
        wheelbase = 0.3302;       // TODO Need to find the correct wheelbase
        gravity = 9.81;
    }
private:
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidarscan_subscription;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_publisher;

    double car_width, disparity_threshold, scan_width, lidar_max_range, front_turn_clearance;
    double rear_turn_clearance, max_turn_angle, min_speed, max_speed, absolute_max_speed, min_distance;
    double max_distance, absolute_max_distance, no_obstacles_distance, angle_step, friction_coefficient, wheelbase, gravity;

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg){
        auto ranges = msg->ranges;
        std::vector<float> limit_range = ranges;
        std::fill(limit_range.begin(), limit_range.begin() + 180, 0.0f);
        std::fill(limit_range.begin() + 901, limit_range.end(), 0.0f);
        limit_range[901] = limit_range[900];

        auto disparities = find_disparities(limit_range, disparity_threshold);
        auto new_range = extend_disparities(limit_range, disparities, car_width);

        auto max_val = max_element(new_range.begin(), new_range.end());

    }
};