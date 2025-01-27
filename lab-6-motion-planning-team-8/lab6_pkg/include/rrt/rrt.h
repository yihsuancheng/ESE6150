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

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr waypoint_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr velocity_sub_;
    
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid >::SharedPtr occ_grid_pub_;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_vis_pub;          // goal 
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tree_vis_pub;     // every node
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_vis_pub;          // final path
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr branch_vis_pub;        // all branches
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr cur_waypoint_vis_pub_;
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

    // tree variables
    std::vector<RRT_Node> rrt_path;
    std::vector<RRT_Node> tree;
    RRT_Node root;

    double lookahead_distance;
    double wheelbase;
    double KP;
    double velocity;
    void visualize_tree(std::vector<RRT_Node> &tree);

    // visualization markers
    visualization_msgs::msg::Marker goal_marker;
    visualization_msgs::msg::Marker tree_marker;
    visualization_msgs::msg::Marker path_marker;
    visualization_msgs::msg::Marker branch_marker;
    visualization_msgs::msg::MarkerArray branch_marker_arr;

    // visualization_msgs::msg::Marker branch_marker;
    visualization_msgs::msg::Marker cur_waypoint_marker;

    // callbacks
    // where rrt actually happens

    void pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg);
    // updates occupancy grid
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);

    void waypoint_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg);

    void velocity_callback(const geometry_msgs::msg::Twist::SharedPtr msg);

    // RRT methods
    std::vector<double> sample();
    int nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point);
    RRT_Node steer(RRT_Node &nearest_node, std::vector<double> &sampled_point);
    bool check_collision(RRT_Node &nearest_node, RRT_Node &new_node);
    bool is_goal(RRT_Node &latest_added_node, double goal_x, double goal_y);
    std::vector<RRT_Node> find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node);
    // RRT* methods
    double cost(std::vector<RRT_Node> &tree, RRT_Node &node);
    double line_cost(RRT_Node &n1, RRT_Node &n2);
    std::vector<int> near(std::vector<RRT_Node> &tree, RRT_Node &node);

    std::vector<int>get_grid_coordinates(double x, double y);
    void mark_obstacles_area(int x, int y);
    bool xy_occupied(double x, double y);
    bool grid_has_obstacles();
    void publish_drive();

};

double sign(double num);

