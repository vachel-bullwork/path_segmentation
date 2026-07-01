#include "path_segmentor/path_corridor_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <pluginlib/class_list_macros.hpp>
#include <nav2_costmap_2d/costmap_math.hpp>

PLUGINLIB_EXPORT_CLASS(path_segmentor::PathCorridorLayer, nav2_costmap_2d::Layer)

namespace path_segmentor {

PathCorridorLayer::PathCorridorLayer() = default;
PathCorridorLayer::~PathCorridorLayer() = default;

// ── onInitialize ─────────────────────────────────────────────────────────────
void PathCorridorLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("PathCorridorLayer: failed to lock node pointer");
  }

  declareParameter("corridor_half_width", rclcpp::ParameterValue(1.0));
  declareParameter("min_half_width",      rclcpp::ParameterValue(0.35));
  declareParameter("base_frame",          rclcpp::ParameterValue(std::string("base_footprint")));

  node->get_parameter(name_ + ".corridor_half_width", max_half_width_);
  node->get_parameter(name_ + ".min_half_width",      min_half_width_);
  node->get_parameter(name_ + ".base_frame",          base_frame_);

  // Clamp min so it never exceeds max
  min_half_width_     = std::min(min_half_width_, max_half_width_);
  current_half_width_ = max_half_width_;

  // Subscriptions
  plan_sub_ = node->create_subscription<nav_msgs::msg::Path>(
    "/plan", rclcpp::QoS(10).reliable(),
    std::bind(&PathCorridorLayer::planCb, this, std::placeholders::_1));

  width_cmd_sub_ = node->create_subscription<std_msgs::msg::Float32>(
    "/corridor/width_cmd", rclcpp::SystemDefaultsQoS(),
    std::bind(&PathCorridorLayer::widthCmdCb, this, std::placeholders::_1));

  // Publishers
  deviation_pub_ = node->create_publisher<std_msgs::msg::Bool>(
    "/corridor/plan_deviation", rclcpp::SystemDefaultsQoS());

  shrink_pub_ = node->create_publisher<path_segmentor::msg::CorridorShrinkStatus>(
    "/corridor/shrink_status", rclcpp::SystemDefaultsQoS());

  // 10 Hz status timer
  status_timer_ = node->create_wall_timer(
    std::chrono::milliseconds(100),
    std::bind(&PathCorridorLayer::checkDeviationAndStatus, this));

  enabled_ = true;
  current_ = true;
}

// ── Callbacks ─────────────────────────────────────────────────────────────────
void PathCorridorLayer::planCb(const nav_msgs::msg::Path::SharedPtr msg)
{
  std::lock_guard<std::mutex> lk(plan_mutex_);
  current_plan_ = msg;
}

void PathCorridorLayer::widthCmdCb(const std_msgs::msg::Float32::SharedPtr msg)
{
  std::lock_guard<std::mutex> lk(width_mutex_);
  current_half_width_ = std::clamp(
    static_cast<double>(msg->data), min_half_width_, max_half_width_);
}

// ── updateBounds — claim entire map so we mark everything outside corridor ────
void PathCorridorLayer::updateBounds(
  double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  auto * cm = layered_costmap_->getCostmap();
  *min_x = std::min(*min_x, cm->getOriginX());
  *min_y = std::min(*min_y, cm->getOriginY());
  *max_x = std::max(*max_x, cm->getOriginX() + cm->getSizeInMetersX());
  *max_y = std::max(*max_y, cm->getOriginY() + cm->getSizeInMetersY());
}

// ── updateCosts — mark cells outside corridor as LETHAL ───────────────────────
void PathCorridorLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) return;

  nav_msgs::msg::Path::SharedPtr plan;
  double hw;
  {
    std::lock_guard<std::mutex> lk(plan_mutex_);
    plan = current_plan_;
  }
  {
    std::lock_guard<std::mutex> lk(width_mutex_);
    hw = current_half_width_;
  }

  if (!plan || plan->poses.size() < 2) return;

  const double hw2 = hw * hw;
  const auto & raw_poses = plan->poses;

  // Build effective point list. When SKIP_GLOBAL_PLANNER=true the FSM publishes
  // a path whose first waypoint is meters ahead of the robot — so the robot
  // starts OUTSIDE the corridor and is immediately in LETHAL territory.
  // Fix: if the robot is farther than hw from the nearest plan segment, prepend
  // its current position so the corridor extends back to cover the robot.
  // When SKIP=false NavFn's path already starts at the robot, so min_dist_sq ≤ hw²
  // and no prepend happens — this branch is a no-op in that case.
  std::vector<std::pair<double, double>> pts;
  pts.reserve(raw_poses.size() + 1);

  double rx = 0.0, ry = 0.0;
  bool got_robot = false;
  try {
    auto tf_msg = tf_->lookupTransform(
      "map", base_frame_, rclcpp::Time(0),
      rclcpp::Duration::from_seconds(0.05));
    rx = tf_msg.transform.translation.x;
    ry = tf_msg.transform.translation.y;
    got_robot = true;
  } catch (const tf2::TransformException &) {}

  if (got_robot) {
    double min_d2 = std::numeric_limits<double>::max();
    for (size_t k = 1; k < raw_poses.size(); ++k) {
      const double ax = raw_poses[k-1].pose.position.x;
      const double ay = raw_poses[k-1].pose.position.y;
      const double bx = raw_poses[k].pose.position.x;
      const double by = raw_poses[k].pose.position.y;
      const double dx = bx - ax, dy = by - ay;
      const double lsq = dx*dx + dy*dy;
      double t = 0.0;
      if (lsq > 1e-9) {
        t = std::clamp(((rx-ax)*dx + (ry-ay)*dy) / lsq, 0.0, 1.0);
      }
      const double ex = ax + t*dx - rx, ey = ay + t*dy - ry;
      min_d2 = std::min(min_d2, ex*ex + ey*ey);
    }
    if (min_d2 > hw2) {
      pts.emplace_back(rx, ry);  // robot is outside corridor — extend it back
    }
  }

  for (const auto & ps : raw_poses) {
    pts.emplace_back(ps.pose.position.x, ps.pose.position.y);
  }

  for (int j = min_j; j < max_j; ++j) {
    for (int i = min_i; i < max_i; ++i) {
      double wx, wy;
      master_grid.mapToWorld(i, j, wx, wy);

      bool inside = false;
      for (size_t k = 1; k < pts.size() && !inside; ++k) {
        const double ax = pts[k-1].first,  ay = pts[k-1].second;
        const double bx = pts[k].first,    by = pts[k].second;
        const double dx = bx - ax, dy = by - ay;
        const double lsq = dx*dx + dy*dy;
        double t = 0.0;
        if (lsq > 1e-9) {
          t = std::clamp(((wx-ax)*dx + (wy-ay)*dy) / lsq, 0.0, 1.0);
        }
        const double ex = ax + t*dx - wx;
        const double ey = ay + t*dy - wy;
        if (ex*ex + ey*ey <= hw2) inside = true;
      }

      if (!inside) {
        master_grid.setCost(i, j, nav2_costmap_2d::LETHAL_OBSTACLE);
      } else {
        master_grid.setCost(i, j, nav2_costmap_2d::FREE_SPACE);
      }
    }
  }
}

// ── reset — called by ClearEntireCostmap BT node ─────────────────────────────
void PathCorridorLayer::reset()
{
  std::lock_guard<std::mutex> lk(plan_mutex_);
  current_plan_.reset();
}

// ── Helpers ───────────────────────────────────────────────────────────────────
double PathCorridorLayer::distToPath(double rx, double ry) const
{
  if (!current_plan_ || current_plan_->poses.size() < 2) {
    return std::numeric_limits<double>::max();
  }
  double best = std::numeric_limits<double>::max();
  const auto & poses = current_plan_->poses;
  for (size_t k = 1; k < poses.size(); ++k) {
    const double ax = poses[k-1].pose.position.x;
    const double ay = poses[k-1].pose.position.y;
    const double bx = poses[k].pose.position.x;
    const double by = poses[k].pose.position.y;
    const double dx = bx-ax, dy = by-ay;
    const double lsq = dx*dx + dy*dy;
    double t = 0.0;
    if (lsq > 1e-9) {
      t = std::clamp(((rx-ax)*dx + (ry-ay)*dy) / lsq, 0.0, 1.0);
    }
    const double ex = ax + t*dx - rx, ey = ay + t*dy - ry;
    const double d = std::sqrt(ex*ex + ey*ey);
    if (d < best) best = d;
  }
  return best;
}

// ── 10 Hz timer: deviation + shrink status ────────────────────────────────────
void PathCorridorLayer::checkDeviationAndStatus()
{
  double hw;
  {
    std::lock_guard<std::mutex> lk(width_mutex_);
    hw = current_half_width_;
  }

  // TF lookup: map → base_frame_
  double rx = 0.0, ry = 0.0;
  bool tf_ok = false;
  try {
    auto tf = tf_->lookupTransform(
      "map", base_frame_, rclcpp::Time(0),
      rclcpp::Duration::from_seconds(0.05));
    rx = tf.transform.translation.x;
    ry = tf.transform.translation.y;
    tf_ok = true;
  } catch (const tf2::TransformException &) {}

  // plan_deviation
  std_msgs::msg::Bool dev;
  if (tf_ok) {
    double d;
    {
      std::lock_guard<std::mutex> lk(plan_mutex_);
      d = distToPath(rx, ry);
    }
    dev.data = (d > hw);
  } else {
    dev.data = false;
  }
  deviation_pub_->publish(dev);

  // shrink_status
  path_segmentor::msg::CorridorShrinkStatus status;
  status.current_half_width = static_cast<float>(hw);
  status.min_half_width     = static_cast<float>(min_half_width_);
  status.max_half_width     = static_cast<float>(max_half_width_);
  status.at_minimum         = (hw <= min_half_width_ + 1e-4);
  status.can_shrink         = (hw >  min_half_width_ + 1e-4);
  status.can_expand         = (hw <  max_half_width_ - 1e-4);
  shrink_pub_->publish(status);
}

}  // namespace path_segmentor
