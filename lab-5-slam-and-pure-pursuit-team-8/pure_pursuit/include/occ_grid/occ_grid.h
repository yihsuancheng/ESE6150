// RRT assignment

// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf

#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <vector>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <std_msgs/msg/float64.hpp>

/// CHECK: include needed ROS msg type headers and libraries

using namespace std;

// Struct defining the RRT_Node object in the RRT tree.
// More fields could be added to thiis struct if more info needed.
// You can choose to use this or not
typedef struct RRT_Node {
    double x, y;
    double cost; // only used for RRT*
    int parent; // index of parent node in the tree vector
    bool is_root = false;
} RRT_Node;


class RRT : public rclcpp::Node {
public:
    RRT();
    virtual ~RRT();
private:

    // TODO: add the publishers and subscribers you need

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr velocity_sub_;
    
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid >::SharedPtr occ_grid_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64 >::SharedPtr distance_pub_;

    // random generator, use this
    std::mt19937 gen;
    std::uniform_real_distribution<> x_dist;
    std::uniform_real_distribution<> y_dist;
    
    nav_msgs::msg::OccupancyGrid occ_grid;
    std::string local_frame;
    double max_dist;
    double grid_width;
    double grid_height;
    double grid_theta;
    double grid_resolution;
    double occupied_value;
    double inflate;

    int MAX_ITER;

    double goal_x;
    double goal_y;

  
    double lookahead_distance;
    double wheelbase;
    double KP;
    double velocity;

    // callbacks
    // where rrt actually happens

    // updates occupancy grid
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);

    void velocity_callback(const geometry_msgs::msg::Twist::SharedPtr msg);

    // RRT methods

    std::vector<int>get_grid_coordinates(double x, double y);
    void mark_obstacles_area(int x, int y);
    bool xy_occupied(double x, double y);
    bool grid_has_obstacles();

};

double sign(double num);

