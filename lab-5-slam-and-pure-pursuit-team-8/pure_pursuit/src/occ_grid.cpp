// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
// Make sure you have read through the header file as well

#include "occ_grid/occ_grid.h"
using namespace std;

// Destructor of the RRT class
RRT::~RRT() {
    // Do something in here, free up used memory, print message, etc.
    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "RRT shutting down");
}

// Constructor of the RRT class
RRT::RRT(): rclcpp::Node("rrt_node"), gen((std::random_device())()) {

    // ROS publishers
    // TODO: create publishers for the the drive topic, and other topics you might need
    string drive_topic = "/drive";
    string occ_grid_topic = "/occ_grid";

    occ_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "occ_grid", 1);

    distance_pub_ = this->create_publisher<std_msgs::msg::Float64>("obs_distance", 1);
    
    // ROS subscribers
    // TODO: create subscribers as you need
    string pose_topic = "ego_racecar/odom";
    string scan_topic = "/scan";
    string waypoint_topic = "/waypoints_pose_array";
    string velocity_topic = "/velocity_profile";
    local_frame = "/ego_racecar/laser";

    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, 1, std::bind(&RRT::scan_callback, this, std::placeholders::_1));

    velocity_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        velocity_topic, 1, std::bind(&RRT::velocity_callback, this, std::placeholders::_1));

    // TODO: tune the right parameters for grid


    grid_width = this->declare_parameter("grid_width", 0.5);
    grid_height = this->declare_parameter("grid_height", 2.0);
    inflate = this->declare_parameter<double>("inflate", 5);
    lookahead_distance = this->declare_parameter("lookahead_dist", 1.0);
    MAX_ITER = 1000;

    // vehicle parameters
    //lookahead_distance = 1.0;

    this->get_parameter("grid_width", grid_width);
    this->get_parameter("grid_height", grid_height);
    this->get_parameter("inflate", inflate);
    this->get_parameter("lookahead_dist", lookahead_distance);
    
    grid_resolution = 0.05;
    occupied_value = 100;

    // vehicle parameters
    //lookahead_distance = 1.0;
    wheelbase = 0.3302;
    velocity = 0.0;

    // random generator
    std::uniform_real_distribution<> x_temp(
            0.0,
            grid_height);
    std::uniform_real_distribution<> y_temp(
            -grid_width/2.0,
            grid_width/2.0);

    x_dist.param(x_temp.param());
    y_dist.param(y_temp.param());

    max_dist = sqrt(pow(grid_width/2, 2) + pow(grid_height, 2));
    

    occ_grid.header.frame_id = "/ego_racecar/laser";
    occ_grid.info.width = static_cast<int>(grid_width / grid_resolution);
    occ_grid.info.height = static_cast<int>(grid_height / grid_resolution);
    occ_grid.info.resolution = grid_resolution;

    occ_grid.info.origin.position.x = 0;
    occ_grid.info.origin.position.y = grid_width / 2.0;
    occ_grid.info.origin.position.z = 0;

    occ_grid.info.origin.orientation.x = 0;
    occ_grid.info.origin.orientation.y = 0;
    occ_grid.info.origin.orientation.z = -1/sqrt(2);
    occ_grid.info.origin.orientation.w = 1/sqrt(2);

    grid_theta = atan2(occ_grid.info.height , occ_grid.info.width / 2.0);

    //occ_grid.data.assign(occ_grid.info.width * occ_grid.info.height, 0);
    for (int i=0; i < occ_grid.info.width*occ_grid.info.height; i++){
        occ_grid.data.push_back(0);
    }

    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "Created new RRT Object.");
}

void RRT::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    // The scan callback, update your occupancy grid here
    // Args:
    //    scan_msg (*LaserScan): pointer to the incoming scan message
    // Returns:
    //

    // clear the grid data
    occ_grid.data.assign(occ_grid.info.width * occ_grid.info.height, 0);
    float min_distance = std::numeric_limits<float>::max();
    float angle_90 = M_PI / 2;

    double angle_inc = scan_msg->angle_increment;
    double angle_min = scan_msg->angle_min;

    // get the range data
    //int lower_range = (int) ((-angle_90 - angle_min) / angle_inc);
    //int upper_range = (int) ((angle_90 - angle_min) / angle_inc);
    int lower_range = 500;
    int upper_range = 580;
    //RCLCPP_INFO(this->get_logger(), "lower range: %d", lower_range);
    //RCLCPP_INFO(this->get_logger(), "Upper range: %d", upper_range);
    for (int i = lower_range; i < upper_range; i++) {

        float curr_value = scan_msg->ranges[i];

        // TODO: add max dist: max_occ_dist
        // if the value is NaN or Inf  or greater than max dist, skip it
        if (isnan(curr_value) || isinf(curr_value) || curr_value > max_dist) {
            continue;
        }

        float theta = angle_min + i * angle_inc;
        float dist_front = curr_value * cos(theta);
        float dist_side = curr_value * sin(theta);

        // valid distance to update the grid
        if (dist_front < grid_height && abs(dist_side) < grid_width/ 2.0) {
            // Convert x, y coordinates to 2D coordinates in the occupancy grid.
            std::vector<int>indices = get_grid_coordinates(dist_front, dist_side);

            // Mark the obstacle area as occupied.
            mark_obstacles_area(indices[0], indices[1]);
            if (curr_value < min_distance) {
                min_distance = curr_value; // Update minimum distance
            }
        }
    }
    occ_grid_pub_->publish(occ_grid);
    std_msgs::msg::Float64 dist_msg;
    dist_msg.data = min_distance;
    distance_pub_ ->publish(dist_msg);

}

std::vector<int> RRT::get_grid_coordinates(double x, double y) {
    int x_ind = static_cast<int>(floor(x / occ_grid.info.resolution));
    int y_ind = static_cast<int>(-(floor(y / occ_grid.info.resolution) - occ_grid.info.width / 2)) - 1;

    return std::vector<int>{x_ind, y_ind};
}

void RRT::mark_obstacles_area(int x, int y) {
    for (int i = -inflate; i <= inflate; i++) {
        int current_x = x + i;
        if (current_x < 0 || current_x >= occ_grid.info.height) {
            continue; // Skip if outside the grid bounds.
        }

        for (int j = -inflate; j <= inflate; j++) {
            int current_y = y + j;
            if (current_y < 0 || current_y >= occ_grid.info.width) {
                continue; // Skip if outside the grid bounds.
            }

            occ_grid.data[current_x * occ_grid.info.width + current_y] = occupied_value;
        }
    }
}

void RRT::velocity_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    velocity = msg->linear.x; // Assuming linear.x contains the desired velocity
    RCLCPP_INFO(this->get_logger(), "Updated velocity to: %f", velocity);
}


bool RRT::xy_occupied(double x, double y){
    std::vector<int>xy_pose = get_grid_coordinates(x,y);
    int index = xy_pose[0] * occ_grid.info.width + xy_pose[1];

    if (occ_grid.data[index] == occupied_value){
        return true; // cell is occupied
    } else {
        return false;
    }

}

bool RRT::grid_has_obstacles() {
    for (int i = 0; i < occ_grid.data.size(); i++) {
        if (occ_grid.data[i] == occupied_value) {  // Assuming 100 denotes an obstacle
            return true;
        }
    }
    return false;  // No obstacles found
}


double sign(double num){
    if (num > 0){
        return 1;
    } else if (num < 0) {
        return -1;
    } else {
        return 0;
    }
}