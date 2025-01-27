// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
// Make sure you have read through the header file as well

#include "rrt/rrt.h"
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

    drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      drive_topic, 1);
    occ_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "occ_grid", 1);
    
    // ROS subscribers
    // TODO: create subscribers as you need
    string pose_topic = "ego_racecar/odom";
    string scan_topic = "/scan";
    string waypoint_topic = "/waypoints_pose_array";
    string velocity_topic = "/velocity_profile";
    local_frame = "/ego_racecar/laser";

    pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      pose_topic, 1, std::bind(&RRT::pose_callback, this, std::placeholders::_1));
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, 1, std::bind(&RRT::scan_callback, this, std::placeholders::_1));

    waypoint_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
        waypoint_topic, 1, std::bind(&RRT::waypoint_callback, this, std::placeholders::_1));

    velocity_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        velocity_topic, 1, std::bind(&RRT::velocity_callback, this, std::placeholders::_1));

    // TODO: tune the right parameters for grid
    grid_width = 1.0;
    grid_height = 2.0;
    grid_resolution = 0.05;
    occupied_value = 100;
    inflate = 5;

    MAX_ITER = 1000;

    // vehicle parameters
    lookahead_distance = 1.0;
    wheelbase = 0.3302;
    KP = 3.0;
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

    // Visualization
    goal_vis_pub = this->create_publisher<visualization_msgs::msg::Marker>("goal", 1);
    path_vis_pub = this->create_publisher<visualization_msgs::msg::Marker>("path", 1);
    // waypoints_vis_pub=this->create_publisher<visualization_msgs::msg::Marker>("waypoints", 1);
    tree_vis_pub = this->create_publisher<visualization_msgs::msg::Marker>("tree", 1);
    branch_vis_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("branch", 1);
    cur_waypoint_vis_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("rrt_waypoint", 1);


    goal_marker.type = visualization_msgs::msg::Marker::SPHERE;
    goal_marker.id = 1;
    goal_marker.scale.x = 0.2;
    goal_marker.scale.y = 0.2;
    goal_marker.scale.z = 0.2;
    goal_marker.color.a = 0.5;
    goal_marker.color.r = 1.0;
    goal_marker.color.g = 0.0;
    goal_marker.color.b = 1.0;
    goal_marker.header.frame_id = local_frame;
    // TODO: when the goal is set, publish it
    // goal_marker.pose.position.x = goal_x;
    // goal_marker.pose.position.y = goal_y;
    // rrt_goal_vis_pub_->publish(goal_marker);

    // waypoints_marker.type = visualization_msgs::msg::Marker::SPHERE;
    // waypoints_marker.id = 2;
    // waypoints_marker.scale.x = 0.1;
    // waypoints_marker.scale.y = 0.1;
    // waypoints_marker.scale.z = 0.1;
    // waypoints_marker.color.a = 1.0;
    // waypoints_marker.color.r = 0.1;
    // waypoints_marker.color.g = 0.1;
    // waypoints_marker.color.b = 0.1;
    // waypoints_marker.header.frame_id = local_frame;

    path_marker.header.frame_id = local_frame;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    
    path_marker.id = 3;
    path_marker.scale.x = 0.05;
    path_marker.color.a = 1.0;
    path_marker.color.r = 0.0;
    path_marker.color.g = 0.0;
    path_marker.color.b = 1.0;
  

    branch_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    branch_marker.id = 2000;
    branch_marker.scale.x = 0.1;
    branch_marker.color.a = 0.5;
    branch_marker.color.r = 0.0;
    branch_marker.color.g = 0.0;
    branch_marker.color.b = 1.0;
    branch_marker.header.frame_id = local_frame;


    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "Created new RRT Object.");
}

// This function is used to visualize the RRT Tree
void RRT::visualize_tree(std::vector<RRT_Node> &tree){

    tree_marker.action = visualization_msgs::msg::Marker::ADD;
    tree_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    tree_marker.id = 4;
    tree_marker.scale.x = 0.04;
    tree_marker.scale.y = 0.04;
    tree_marker.scale.z = 0.04;
    tree_marker.color.a = 1.0;
    tree_marker.color.r = 1.0;
    tree_marker.color.g = 0.0;
    tree_marker.color.b = 0.0;
    tree_marker.header.frame_id = local_frame;
    tree_marker.points.clear();

    for (const auto& node : tree)
    {
        geometry_msgs::msg::Point p;
        p.x = node.x;
        p.y = node.y;
        tree_marker.points.push_back(p);
    }

    tree_vis_pub->publish(tree_marker);
}


void RRT::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    // The scan callback, update your occupancy grid here
    // Args:
    //    scan_msg (*LaserScan): pointer to the incoming scan message
    // Returns:
    //

    // clear the grid data
    occ_grid.data.assign(occ_grid.info.width * occ_grid.info.height, 0);

    float angle_90 = M_PI / 2;

    double angle_inc = scan_msg->angle_increment;
    double angle_min = scan_msg->angle_min;

    // get the range data
    int lower_range = (int) ((-angle_90 - angle_min) / angle_inc);
    int upper_range = (int) ((angle_90 - angle_min) / angle_inc);


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
        }
    }
    occ_grid_pub_->publish(occ_grid);
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


void RRT::pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
    // The pose callback when subscribed to particle filter's inferred pose
    // The RRT main loop happens here
    // Args:
    //    pose_msg (*PoseStamped): pointer to the incoming pose message
    // Returns:
    //

    // tree as std::vector

    tree.clear();
    root.is_root = true;
    tree.push_back(root);

    int iter = 0;

    //double goal_x = grid_height;
    //double goal_y = 0.0;
    RRT_Node new_node;

    // TODO: fill in the RRT main loop
    while (!is_goal(tree.back(), goal_x, goal_y) && iter < MAX_ITER) {
        iter++;

        // TODO: fill sample
        std::vector<double> sampled_point = sample();
        int nearest_node = nearest(tree, sampled_point);
        //RCLCPP_INFO(this->get_logger(), "nearest_node: %d", nearest_node);
        new_node = steer(tree[nearest_node], sampled_point);

        // TODO: check the right condition - is xy occupied
        if (xy_occupied(new_node.x, new_node.y)){
            continue;
        }
        
        if (!check_collision(tree[nearest_node], new_node)) {
            tree.push_back(new_node);
            tree.back().parent = nearest_node;
            if (is_goal(tree.back(), goal_x, goal_y)) {
                break;
            }
        }
    }
    //RCLCPP_INFO(this->get_logger(), "Found path");
    rrt_path = find_path(tree, new_node);

    // RCLCPP_INFO(this->get_logger(), "Size of RRT path: %zu", rrt_path.size());

    // RCLCPP_INFO(this->get_logger(), "rrt_path: ", rrt_path);
    // path found as Path message
    std::stringstream path_stream;
    path_stream << "RRT Path: ";
    for (const auto& node : rrt_path) {
        path_stream << "(" << node.x << ", " << node.y << ") ";
    }

    RCLCPP_INFO(this->get_logger(), "%s", path_stream.str().c_str());
    visualize_tree(tree);
    publish_drive();

}

void RRT::waypoint_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg){
    if (!msg->poses.empty()){
        auto pose = msg->poses[0];
        double x = pose.position.x;
        double y = pose.position.y;
        // RCLCPP_INFO(this->get_logger(), "Waypoints: x = %f, y = %f", x, y);
        // TODO: Need to check if the waypoints are occupied and how to deal with it
        if (x < grid_height && abs(y) < grid_width / 2.0) {
            // waypoint is inside grid
            goal_x = x;
            goal_y = y;     
        } 
        else {
            // waypoint is outside grid, so project waypoints onto the edge of grid
            double waypoint_theta = atan2(x, abs(y));
            if (abs(waypoint_theta) > grid_theta) {
                // project onto the top edge of occupancy grid
                goal_x = grid_height;
                goal_y = grid_height * (y / x);
            }
            else {
                // waypoint is projected onto side edge of occupancy grid
                goal_x = abs((grid_width / 2) * (x / y));
                goal_y = sign(y) * (grid_width) / 2;
            }
        }

        // publish goal marker
        goal_marker.pose.position.x = goal_x;
        goal_marker.pose.position.y = goal_y;
        goal_vis_pub->publish(goal_marker);


    } else{
        RCLCPP_INFO(this->get_logger(), "Received empty PoseArray");
    }

}

void RRT::velocity_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    velocity = msg->linear.x; // Assuming linear.x contains the desired velocity
    RCLCPP_INFO(this->get_logger(), "Updated velocity to: %f", velocity);
}


void RRT::publish_drive() {
    // Find the target_node on the RRT path that is closest to the lookahead_distance from the vehicle
    RRT_Node target_node;
    double dist;
    // double distance;
    // const auto& last_node = rrt_path.back();
    // double last_node_distance = sqrt(pow(last_node.x, 2) + pow(last_node.y, 2));
    // dist = last_node_distance; // if distance < lookahead_distance, get the last (largest) distance
    dist = 1.0;
    for (const auto& node : rrt_path){
        double distance = sqrt(pow(node.x, 2) + pow(node.y, 2));
        RCLCPP_INFO(this->get_logger(), "Distance: %f", distance);
        if (distance > lookahead_distance) {
            target_node = node;
            dist = distance;
            break;
        }
    }
    
    if (!xy_occupied(goal_x, goal_y) && !grid_has_obstacles()){
        cur_waypoint_marker.pose.position.x = goal_x;
        cur_waypoint_marker.pose.position.y = goal_y;
        dist = lookahead_distance;
        RCLCPP_INFO(this->get_logger(), "FOLLOWING WAYPOINTS");
    }
    else if (target_node.x != 0.0 && target_node.y != 0.0) {
        cur_waypoint_marker.pose.position.x = target_node.x;
        cur_waypoint_marker.pose.position.y = target_node.y;
        //velocity = 2.0;

        // double kappa = 2 * target_node.y / pow(dist, 2);
        // RCLCPP_INFO(this->get_logger(), "target_node.x: %f", target_node.x);
        // RCLCPP_INFO(this->get_logger(), "target_node.y: %f", target_node.y);
        // RCLCPP_INFO(this->get_logger(), "dist: %f", dist);
        
        // double theta = KP * kappa;
        // RCLCPP_INFO(this->get_logger(), "theta: %f", theta);
        // double steering_angle = atan(theta * wheelbase);

        // ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        // drive_msg.drive.steering_angle = steering_angle;
        // drive_msg.drive.speed = velocity;
        // drive_pub_->publish(drive_msg);
        // cur_waypoint_vis_pub_->publish(cur_waypoint_marker);
        // RCLCPP_INFO(this->get_logger(), "steering angle: %f", steering_angle);

    }
    double kappa = 2 * cur_waypoint_marker.pose.position.y / pow(dist, 2);
    RCLCPP_INFO(this->get_logger(), "target_node.x: %f", cur_waypoint_marker.pose.position.x);
    RCLCPP_INFO(this->get_logger(), "target_node.y: %f", cur_waypoint_marker.pose.position.y);
    RCLCPP_INFO(this->get_logger(), "dist: %f", dist);
    
    double theta = KP * kappa;
    RCLCPP_INFO(this->get_logger(), "theta: %f", theta);
    double steering_angle = atan(theta * wheelbase);

    ackermann_msgs::msg::AckermannDriveStamped drive_msg;
    drive_msg.drive.steering_angle = steering_angle;
    drive_msg.drive.speed = 1.0;
    drive_pub_->publish(drive_msg);
    cur_waypoint_vis_pub_->publish(cur_waypoint_marker);
    RCLCPP_INFO(this->get_logger(), "steering angle: %f", steering_angle);

}
std::vector<double> RRT::sample() {
    // This method returns a sampled point from the free space
    // You should restrict so that it only samples a small region
    // of interest around the car's current position
    // Args:
    // Returns:
    //     sampled_point (std::vector<double>): the sampled point in free space

    std::vector<double> sampled_point;
    // TODO: fill in this method
    // look up the documentation on how to use std::mt19937 devices with a distribution
    // the generator and the distribution is created for you (check the header file)

    double x_new, y_new;

    while (true) {
        x_new = x_dist(gen);
        y_new = y_dist(gen);

        std::vector<int>indices = get_grid_coordinates(x_new, y_new);

        if (occ_grid.data[indices[0] * occ_grid.info.width + indices[1]] != occupied_value){
            break;
        }
    }

    sampled_point.push_back(x_new);
    sampled_point.push_back(y_new);

    return sampled_point;
}


int RRT::nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point) {
    // This method returns the nearest node on the tree to the sampled point
    // Args:
    //     tree (std::vector<RRT_Node>): the current RRT tree
    //     sampled_point (std::vector<double>): the sampled point in free space
    // Returns:
    //     nearest_node (int): index of nearest node on the tree

    int nearest_node = 0;
    double nearest_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < tree.size(); i++) {
        double x_dist = tree[i].x - sampled_point[0];
        double y_dist = tree[i].y - sampled_point[1];
        double dist = std::sqrt(x_dist * x_dist + y_dist * y_dist);

        if (dist < nearest_dist) {
            nearest_dist = dist;
            nearest_node = i;
        }
    }

    return nearest_node;
}

RRT_Node RRT::steer(RRT_Node &nearest_node, std::vector<double> &sampled_point) {
    // The function steer:(x,y)->z returns a point such that z is “closer” 
    // to y than x is. The point z returned by the function steer will be 
    // such that z minimizes ||z−y|| while at the same time maintaining 
    //||z−x|| <= max_expansion_dist, for a prespecified max_expansion_dist > 0

    // basically, expand the tree towards the sample point (within a max dist)

    // Args:
    //    nearest_node (RRT_Node): nearest node on the tree to the sampled point
    //    sampled_point (std::vector<double>): the sampled point in free space
    // Returns:
    //    new_node (RRT_Node): new node created from steering

    RRT_Node new_node;
    // TODO: fill in this method
    double distance = std::sqrt(std::pow(nearest_node.x-sampled_point[0],2) + std::pow(nearest_node.y - sampled_point[1],2));
    double STEER_RANGE = 0.3;

    new_node.x = nearest_node.x + min(STEER_RANGE, distance)*(sampled_point[0]-nearest_node.x)/distance;
    new_node.y = nearest_node.y + min(STEER_RANGE, distance)*(sampled_point[1]-nearest_node.y)/distance;

    return new_node;
}

bool RRT::check_collision(RRT_Node &nearest_node, RRT_Node &new_node) {
    // This method returns a boolean indicating if the path between the 
    // nearest node and the new node created from steering is collision free
    // Args:
    //    nearest_node (RRT_Node): nearest node on the tree to the sampled point
    //    new_node (RRT_Node): new node created from steering
    // Returns:
    //    collision (bool): true if in collision, false otherwise

    bool collision = false;
    // TODO: change params based on testing
    int num_steps = 10;

    double dist = std::sqrt(std::pow(nearest_node.x - new_node.x, 2) + std::pow(nearest_node.y - new_node.y, 2));

    double unit_x = (new_node.x - nearest_node.x) / dist;
    double unit_y = (new_node.y - nearest_node.y) / dist;
    double dx = dist/ num_steps;

    for (int i = 0; i < num_steps; i++) {
        int x = nearest_node.x + (i * dx * unit_x);
        int y = nearest_node.y + (i * dx * unit_y);

        std::vector<int> indices = get_grid_coordinates(x, y);

        if (occ_grid.data[indices[0] * occ_grid.info.width + indices[1]] == occupied_value) {
            collision = true;
            break;
        }
    }

    return collision;
}

bool RRT::is_goal(RRT_Node &latest_added_node, double goal_x, double goal_y) {
    // This method checks if the latest node added to the tree is close
    // enough (defined by goal_threshold) to the goal so we can terminate
    // the search and find a path
    // Args:
    //   latest_added_node (RRT_Node): latest addition to the tree
    //   goal_x (double): x coordinate of the current goal
    //   goal_y (double): y coordinate of the current goal
    // Returns:
    //   close_enough (bool): true if node close enough to the goal

    double distance = std::sqrt(
      std::pow(latest_added_node.x - goal_x, 2) + std::pow(latest_added_node.y - goal_y, 2));

    // TODO: move this to yaml file
    double goal_threshold = 0.5;

    return distance < goal_threshold;
}

std::vector<RRT_Node> RRT::find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node) {
    // This method traverses the tree from the node that has been determined
    // as goal
    // Args:
    //   latest_added_node (RRT_Node): latest addition to the tree that has been
    //      determined to be close enough to the goal
    // Returns:
    //   path (std::vector<RRT_Node>): the vector that represents the order of
    //      of the nodes traversed as the found path
    
    std::vector<RRT_Node> found_path;
    found_path.clear();
    path_marker.points.clear();
    RRT_Node current_node = tree.back();
    geometry_msgs::msg::Point p;

    while (!current_node.is_root) {
        found_path.push_back(current_node);
        p.x = current_node.x;
        p.y = current_node.y;
        path_marker.points.push_back(p);
        current_node = tree.at(current_node.parent);
    }

    p.x = current_node.x;
    p.y = current_node.y;
    path_marker.points.push_back(p);
    path_vis_pub->publish(path_marker);
    found_path.push_back(current_node); // add the root node
    reverse(found_path.begin(), found_path.end());

    if (found_path.empty()){
        RCLCPP_INFO(this->get_logger(), "RRT path is empty");
    }

    return found_path;
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


// RRT* methods
double RRT::cost(std::vector<RRT_Node> &tree, RRT_Node &node) {
    // This method returns the cost associated with a node
    // Args:
    //    tree (std::vector<RRT_Node>): the current tree
    //    node (RRT_Node): the node the cost is calculated for
    // Returns:
    //    cost (double): the cost value associated with the node

    return tree.at(node.parent).cost + line_cost(tree.at(node.parent), node);;
}

double RRT::line_cost(RRT_Node &n1, RRT_Node &n2) {
    // This method returns the cost of the straight line path between two nodes
    // Args:
    //    n1 (RRT_Node): the RRT_Node at one end of the path
    //    n2 (RRT_Node): the RRT_Node at the other end of the path
    // Returns:
    //    cost (double): the cost value associated with the path

    double cost = 0;

    cost = std::sqrt(std::pow(n1.x - n2.x, 2) + std::pow(n1.y - n2.y, 2));

    return cost;
}

std::vector<int> RRT::near(std::vector<RRT_Node> &tree, RRT_Node &node) {
    // This method returns the set of Nodes in the neighborhood of a 
    // node.
    // Args:
    //   tree (std::vector<RRT_Node>): the current tree
    //   node (RRT_Node): the node to find the neighborhood for
    // Returns:
    //   neighborhood (std::vector<int>): the index of the nodes in the neighborhood

    std::vector<int> neighborhood;
    // TODO:: fill in this method
    double NEAR_RANGE = 1.0;
    for (int i=0; i<tree.size(); i++){
        if (line_cost(tree.at(i), node) < NEAR_RANGE){
            neighborhood.push_back(i);
        }
    }
    return neighborhood;
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