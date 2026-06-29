#pragma once

#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <nav2_costmap_2d/layer.hpp>
#include <nav2_costmap_2d/layered_costmap.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>

#include "path_segmentor/msg/corridor_shrink_status.hpp"

namespace path_segmentor {

class PathCorridorLayer : public nav2_costmap_2d::Layer
{
public:
  PathCorridorLayer();
  ~PathCorridorLayer() override;

  void onInitialize() override;

  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y,
    double * max_x, double * max_y) override;

  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;

  bool isClearable() override { return true; }
  void reset() override;

private:
  void planCb(const nav_msgs::msg::Path::SharedPtr msg);
  void widthCmdCb(const std_msgs::msg::Float32::SharedPtr msg);
  void checkDeviationAndStatus();
  double distToPath(double rx, double ry) const;  // caller must hold plan_mutex_

  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr    plan_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr width_cmd_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr       deviation_pub_;
  rclcpp::Publisher<path_segmentor::msg::CorridorShrinkStatus>::SharedPtr shrink_pub_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  nav_msgs::msg::Path::SharedPtr current_plan_;
  mutable std::mutex plan_mutex_;
  mutable std::mutex width_mutex_;

  double max_half_width_{1.0};
  double min_half_width_{0.35};
  double current_half_width_{1.0};
  std::string base_frame_{"base_footprint"};
};

}  // namespace path_segmentor
