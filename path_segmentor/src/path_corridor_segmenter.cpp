// ═══════════════════════════════════════════════════════════════════════════════
// path_corridor_segmenter.cpp
//
// Standalone ROS 2 visualiser: projects the Nav2 path corridor onto the ZED X
// camera feed, classifies terrain as ground/wall using CUDA surface normals,
// and draws a colour-coded overlay showing the driveable corridor.
//
// Phase 1: see it working.  Phase 2: feed into costmap plugin.
// ═══════════════════════════════════════════════════════════════════════════════

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <opencv2/opencv.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "path_segmentor/terrain_backend.hpp"

#include <std_msgs/msg/bool.hpp>

#include <mutex>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>

using namespace std::placeholders;

class PathCorridorSegmenter : public rclcpp::Node
{
public:
  PathCorridorSegmenter()
  : Node("path_corridor_segmenter"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // ── Parameters ────────────────────────────────────────────────────────
    this->declare_parameter("max_corridor_half_m",   1.0);
    this->declare_parameter("robot_half_width_m",    0.3);
    this->declare_parameter("ground_angle_thresh_deg", 30.0);
    this->declare_parameter("bilateral_sigma_s",     2.0);
    this->declare_parameter("bilateral_sigma_d",     0.10);
    this->declare_parameter("max_depth_jump_m",      0.5);
    this->declare_parameter("min_depth_m",           0.3);
    this->declare_parameter("max_depth_m",           12.0);
    this->declare_parameter("overlay_alpha",         0.35);
    this->declare_parameter("shrink_safety_margin_m", 0.10);
    this->declare_parameter("shrink_samples",        6);
    this->declare_parameter("base_frame",  std::string("base_link"));
    this->declare_parameter("publish_terrain_debug",  true);
    this->declare_parameter("cloud_stride", 2);
    this->declare_parameter("boundary_lookahead_m", 15.0);

    max_half_         = this->get_parameter("max_corridor_half_m").as_double();
    robot_half_       = this->get_parameter("robot_half_width_m").as_double();
    bilateral_sigma_s_= this->get_parameter("bilateral_sigma_s").as_double();
    bilateral_sigma_d_= this->get_parameter("bilateral_sigma_d").as_double();
    max_depth_jump_   = this->get_parameter("max_depth_jump_m").as_double();
    min_depth_        = this->get_parameter("min_depth_m").as_double();
    max_depth_        = this->get_parameter("max_depth_m").as_double();
    overlay_alpha_    = this->get_parameter("overlay_alpha").as_double();
    shrink_margin_    = this->get_parameter("shrink_safety_margin_m").as_double();
    shrink_samples_   = this->get_parameter("shrink_samples").as_int();
    base_frame_       = this->get_parameter("base_frame").as_string();
    publish_debug_    = this->get_parameter("publish_terrain_debug").as_bool();
    cloud_stride_          = std::max(1, (int)this->get_parameter("cloud_stride").as_int());
    boundary_lookahead_m_  = this->get_parameter("boundary_lookahead_m").as_double();

    double angle_deg  = this->get_parameter("ground_angle_thresh_deg").as_double();
    cos_ground_thresh_= std::cos(angle_deg * M_PI / 180.0);

    // ── Terrain backend (CUDA or CPU) ─────────────────────────────────────
    // Env var PATH_SEGMENTOR_USE_CUDA = 0/false/cpu forces the CPU backend.
    // Anything else (or unset) prefers CUDA, but the factory falls back to CPU
    // automatically if the package was built without a CUDA toolchain.
    bool prefer_cuda = true;
    if (const char * e = std::getenv("PATH_SEGMENTOR_USE_CUDA")) {
      std::string val(e);
      if (val == "0" || val == "false" || val == "FALSE" || val == "cpu" || val == "CPU")
        prefer_cuda = false;
    }
    backend_ = pcv::createTerrainBackend(prefer_cuda);

    // ── Subscriptions ────────────────────────────────────────────────────
    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        "/plan", rclcpp::SystemDefaultsQoS(),
        std::bind(&PathCorridorSegmenter::pathCb, this, _1));

    info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/zed/zed_node/left/camera_info", rclcpp::SystemDefaultsQoS(),
        std::bind(&PathCorridorSegmenter::infoCb, this, _1));

    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/zed/zed_node/depth/depth_registered", rclcpp::SensorDataQoS(),
        std::bind(&PathCorridorSegmenter::depthCb, this, _1));

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/zed/zed_node/left/image_rect_color", rclcpp::SensorDataQoS(),
        std::bind(&PathCorridorSegmenter::imageCb, this, _1));

    // ── Publishers ───────────────────────────────────────────────────────
    overlay_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/corridor/overlay", rclcpp::SensorDataQoS());

    if (publish_debug_) {
      terrain_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
          "/corridor/terrain_debug", rclcpp::SensorDataQoS());
    }

    safety_pub_ = this->create_publisher<std_msgs::msg::Bool>(
        "/corridor/safety_violation", rclcpp::SystemDefaultsQoS());

    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/corridor/obstacle_cloud", rclcpp::SensorDataQoS());

    boundary_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/corridor/boundary_cloud", rclcpp::SensorDataQoS());

    // Boundary crossing check runs independently of the camera at 10 Hz.
    // The old approach gated it inside imageCb behind left_pts.size() >= 2,
    // which meant it only fired when path poses were visible in the camera FOV.
    // This timer uses only the latest path + TF — no image or terrain needed.
    boundary_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&PathCorridorSegmenter::boundaryCb, this));

    RCLCPP_INFO(this->get_logger(),
        "PathCorridorSegmenter ready. backend=%s, corridor=%.1fm, ground_thresh=%.0f°, "
        "bilateral σ_s=%.1f σ_d=%.2fm",
        backend_->name(), max_half_, angle_deg, bilateral_sigma_s_, bilateral_sigma_d_);
  }

private:

  // ═══════════════════════════════════════════════════════════════════════════
  //  CALLBACKS
  // ═══════════════════════════════════════════════════════════════════════════

  void pathCb(const nav_msgs::msg::Path::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(path_mutex_);
    latest_path_ = msg;
  }

  // Boundary crossing check — runs at 10 Hz, fully camera-independent.
  // Builds the corridor polygon directly from the latest path poses in world XY
  // (no projection, no terrain data, no image required). This means the vehicle
  // is monitored even when the camera is facing away from the path.
  void boundaryCb() {
    nav_msgs::msg::Path::SharedPtr path;
    {
      std::lock_guard<std::mutex> lk(path_mutex_);
      path = latest_path_;
    }

    std_msgs::msg::Bool out;
    out.data = false;

    if (!path || (int)path->poses.size() < 2) {
      safety_pub_->publish(out);
      return;
    }

    auto & poses = path->poses;
    int N = (int)poses.size();

    std::vector<cv::Point2f> left_world, right_world;
    left_world.reserve(N);
    right_world.reserve(N);

    for (int i = 0; i < N; i++) {
      double dx, dy;
      if (i < N - 1) {
        dx = poses[i+1].pose.position.x - poses[i].pose.position.x;
        dy = poses[i+1].pose.position.y - poses[i].pose.position.y;
      } else {
        dx = poses[i].pose.position.x - poses[i-1].pose.position.x;
        dy = poses[i].pose.position.y - poses[i-1].pose.position.y;
      }
      double yaw = std::atan2(dy, dx);
      // Use the nominal max_half_ (not shrunk) — this is a safety boundary,
      // not a visual one. Shrinking is for obstacle avoidance; the safety zone
      // is the full declared corridor width.
      left_world.emplace_back(
          (float)(poses[i].pose.position.x - max_half_ * std::sin(yaw)),
          (float)(poses[i].pose.position.y + max_half_ * std::cos(yaw)));
      right_world.emplace_back(
          (float)(poses[i].pose.position.x + max_half_ * std::sin(yaw)),
          (float)(poses[i].pose.position.y - max_half_ * std::cos(yaw)));
    }

    checkBoundaryCrossing(left_world, right_world,
                          path->header.frame_id);

    // ── Corridor boundary fence for Nav2 planner ──────────────────────────
    // Published HERE instead of imageCb so that:
    //   1. Width = fixed max_half_ — imageCb uses shrunk left_hw which changes
    //      every frame; with clearing:false the costmap accumulates all widths → scatter.
    //   2. Timestamp = now() — path->header.stamp is stale after the initial plan
    //      is published (e.g. 718 s old). Nav2 / RViz look up TF at that stamp.
    //   3. Camera-independent — fence fires at 10 Hz regardless of camera state.
    if (boundary_pub_->get_subscription_count() > 0) {
      size_t fence_n = trimToLookahead(left_world, boundary_lookahead_m_);
      std::vector<cv::Point2f> lw(left_world.begin(), left_world.begin() + fence_n);
      std::vector<cv::Point2f> rw(right_world.begin(), right_world.begin() + fence_n);
      std_msgs::msg::Header fence_hdr;
      fence_hdr.stamp    = this->get_clock()->now();
      fence_hdr.frame_id = path->header.frame_id;  // map
      publishBoundaryCloud(fence_hdr, lw, rw);
    }
  }

  void infoCb(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if (cam_ready_) return;  // latch
    cam_model_.fromCameraInfo(*msg);
    cam_frame_ = msg->header.frame_id;
    width_  = (int)msg->width;
    height_ = (int)msg->height;

    // Extract raw intrinsics for CUDA (K matrix)
    fx_ = (float)msg->k[0];
    fy_ = (float)msg->k[4];
    cx_ = (float)msg->k[2];
    cy_ = (float)msg->k[5];

    cam_ready_ = true;
    RCLCPP_INFO(this->get_logger(),
        "Camera: %dx%d  fx=%.1f fy=%.1f cx=%.1f cy=%.1f  frame=%s",
        width_, height_, fx_, fy_, cx_, cy_, cam_frame_.c_str());
  }

  // ── Depth callback: upload + run terrain pipeline ──────────────────────

  void depthCb(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (!cam_ready_) return;

    // Accept 32FC1 (metres, ZED default) or 16UC1 (mm, some configs)
    cv::Mat depth_cpu;
    if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      depth_cpu = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
    } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      cv::Mat raw = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
      raw.convertTo(depth_cpu, CV_32FC1, 0.001);  // mm → m
    } else {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "Unexpected depth encoding: %s", msg->encoding.c_str());
      return;
    }

    // Save raw depth for pointcloud unprojection in imageCb.
    // Single-threaded executor → no mutex needed.
    latest_depth_cpu_ = depth_cpu;

    // Compute expected ground normal in camera frame from vehicle orientation
    float gnd_nx, gnd_ny, gnd_nz;
    if (!computeExpectedGroundNormal(gnd_nx, gnd_ny, gnd_nz)) {
      // Fallback: assume camera is upright → ground normal ≈ (0, -1, 0)
      // in optical frame (+Y = down, so ground pointing "up" = -Y)
      gnd_nx = 0.0f; gnd_ny = -1.0f; gnd_nz = 0.0f;
    }

    // Terrain pipeline: bilateral → normals → classify (CUDA or CPU backend).
    // Result labels land in cpu_labels_ on the host for boundary sampling.
    pcv::TerrainParams tp;
    tp.bilateral_sigma_s = (float)bilateral_sigma_s_;
    tp.bilateral_sigma_d = (float)bilateral_sigma_d_;
    tp.max_depth_jump    = (float)max_depth_jump_;
    tp.cos_ground_thresh = (float)cos_ground_thresh_;
    tp.min_depth         = (float)min_depth_;
    tp.max_depth         = (float)max_depth_;

    backend_->computeTerrain(
        depth_cpu, fx_, fy_, cx_, cy_,
        gnd_nx, gnd_ny, gnd_nz, tp, cpu_labels_);

    terrain_stamp_ = msg->header.stamp;
    terrain_ready_ = true;
  }

  // ── Image callback: project corridor, build overlay ────────────────────

  void imageCb(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (!cam_ready_ || !terrain_ready_) return;

    nav_msgs::msg::Path::SharedPtr path;
    {
      std::lock_guard<std::mutex> lk(path_mutex_);
      path = latest_path_;
    }
    if (!path || path->poses.size() < 2) return;

    cv_bridge::CvImagePtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat rgb_cpu = cv_ptr->image;

    // ── TF: camera ← path frame ──────────────────────────────────────────
    geometry_msgs::msg::TransformStamped tf_cam_map;
    try {
      tf_cam_map = tf_buffer_.lookupTransform(
          cam_frame_, path->header.frame_id,
          msg->header.stamp,
          rclcpp::Duration::from_seconds(0.1));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
          "TF lookup [%s → %s]: %s",
          path->header.frame_id.c_str(), cam_frame_.c_str(), ex.what());
      return;
    }

    // ── Project path + build corridor with continuous shrink ──────────────
    auto & poses = path->poses;
    int N = (int)poses.size();

    std::vector<cv::Point> left_pts, right_pts, centre_pts;
    left_pts.reserve(N);
    right_pts.reserve(N);
    centre_pts.reserve(N);

    for (int i = 0; i < N; i++) {
      // Direction vector from this pose to next (or previous for last)
      double dx, dy;
      if (i < N - 1) {
        dx = poses[i+1].pose.position.x - poses[i].pose.position.x;
        dy = poses[i+1].pose.position.y - poses[i].pose.position.y;
      } else {
        dx = poses[i].pose.position.x - poses[i-1].pose.position.x;
        dy = poses[i].pose.position.y - poses[i-1].pose.position.y;
      }
      double yaw = std::atan2(dy, dx);

      // ── Continuous vector compression ────────────────────────────────
      // Sample terrain labels at decreasing lateral distances.
      // If a wall is found, shrink the corridor at this segment.
      double left_hw  = max_half_;
      double right_hw = max_half_;

      double step_size = max_half_ / shrink_samples_;

      for (int s = shrink_samples_; s >= 1; s--) {
        double d = s * step_size;

        // Left sample
        if (left_hw >= d) {
          auto px_l = projectOffsetPoint(
              poses[i], yaw, d, +1.0, tf_cam_map);
          if (px_l && sampleLabel(*px_l) == pcv::LABEL_WALL) {
            left_hw = std::max(d - shrink_margin_, robot_half_);
          }
        }

        // Right sample
        if (right_hw >= d) {
          auto px_r = projectOffsetPoint(
              poses[i], yaw, d, -1.0, tf_cam_map);
          if (px_r && sampleLabel(*px_r) == pcv::LABEL_WALL) {
            right_hw = std::max(d - shrink_margin_, robot_half_);
          }
        }
      }

      // ── Project final boundary points ────────────────────────────────
      auto left_px  = projectOffsetPoint(poses[i], yaw, left_hw,  +1.0, tf_cam_map);
      auto right_px = projectOffsetPoint(poses[i], yaw, right_hw, -1.0, tf_cam_map);
      auto ctr_px   = projectOffsetPoint(poses[i], yaw, 0.0,       0.0, tf_cam_map);

      if (left_px && right_px) {
        left_pts.push_back(*left_px);
        right_pts.push_back(*right_px);
      }
      if (ctr_px) {
        centre_pts.push_back(*ctr_px);
      }
    }

    // ── Extend corridor polygon to image bottom (near-field fix) ──────────
    // Nav2's nearest path pose is typically 0.5–2 m ahead of the robot.
    // That pose projects to somewhere in the lower portion of the image,
    // leaving the bottom strip (ground immediately in front of the camera)
    // outside the polygon → classified black even though it is the nearest
    // driveable surface. Fix: find the bottommost (highest v) point in each
    // boundary and extend straight down to the image edge.
    if (!left_pts.empty() && !right_pts.empty()) {
      cv::Point near_l = left_pts[0], near_r = right_pts[0];
      for (const auto & p : left_pts)  if (p.y > near_l.y) near_l = p;
      for (const auto & p : right_pts) if (p.y > near_r.y) near_r = p;
      // Only extend if the polygon doesn't already reach the bottom 5 rows.
      if (near_l.y < height_ - 5 || near_r.y < height_ - 5) {
        left_pts.push_back(cv::Point(near_l.x, height_ - 1));
        right_pts.push_back(cv::Point(near_r.x, height_ - 1));
      }
    }

    if (left_pts.size() < 2) return;

    // ── Build corridor polygon mask ──────────────────────────────────────
    cv::Mat corridor_mask = cv::Mat::zeros(height_, width_, CV_8UC1);
    {
      std::vector<cv::Point> polygon;
      polygon.reserve(left_pts.size() + right_pts.size());
      polygon.insert(polygon.end(), left_pts.begin(), left_pts.end());
      polygon.insert(polygon.end(), right_pts.rbegin(), right_pts.rend());
      std::vector<std::vector<cv::Point>> contours = {polygon};
      cv::fillPoly(corridor_mask, contours, cv::Scalar(255));
    }

    // ── Debug publish (corridor-masked) ──────────────────────────────────
    if (publish_debug_ && terrain_pub_->get_subscription_count() > 0) {
      publishTerrainDebug(msg->header, corridor_mask);
    }

    // ── Obstacle pointcloud for Nav2 costmap ──────────────────────────────
    if (cloud_pub_->get_subscription_count() > 0) {
      publishObstacleCloud(msg->header, corridor_mask);
    }

    // ── Overlay compositing (CUDA or CPU backend) ────────────────────────
    cv::Mat result;
    backend_->composite(rgb_cpu, corridor_mask, cpu_labels_,
                         (float)overlay_alpha_, result);

    // ── Draw path centreline + corridor edges (CPU, few hundred points) ──
    if (centre_pts.size() >= 2) {
      cv::polylines(result, centre_pts, false,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);  // yellow
    }
    if (left_pts.size() >= 2) {
      cv::polylines(result, left_pts, false,
                    cv::Scalar(255, 180, 0), 1, cv::LINE_AA);  // cyan-ish
    }
    if (right_pts.size() >= 2) {
      cv::polylines(result, right_pts, false,
                    cv::Scalar(255, 180, 0), 1, cv::LINE_AA);
    }

    // ── Publish ──────────────────────────────────────────────────────────
    auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", result).toImageMsg();
    overlay_pub_->publish(*out_msg);
  }


  // ═══════════════════════════════════════════════════════════════════════════
  //  HELPERS
  // ═══════════════════════════════════════════════════════════════════════════

  // Return the index (exclusive end) that limits world-frame boundary points
  // to at most `max_m` metres of arc-length from the first pose.
  // Used to prevent publishing the full global plan as a fence.
  static size_t trimToLookahead(const std::vector<cv::Point2f> & pts, double max_m) {
    if (pts.size() <= 1) return pts.size();
    double accum = 0.0;
    for (size_t i = 1; i < pts.size(); ++i) {
      float dx = pts[i].x - pts[i-1].x;
      float dy = pts[i].y - pts[i-1].y;
      accum += std::sqrt(dx*dx + dy*dy);
      if (accum >= max_m) return i + 1;
    }
    return pts.size();
  }

  // Project a point that is `dist` metres laterally from a path pose.
  // sign: +1 = left of heading, -1 = right (0 = on centreline).
  // Returns pixel coords if in front of camera and inside image, else nullopt.
  std::optional<cv::Point> projectOffsetPoint(
      const geometry_msgs::msg::PoseStamped & pose,
      double yaw, double dist, double sign,
      const geometry_msgs::msg::TransformStamped & tf_cam_map)
  {
    geometry_msgs::msg::PoseStamped p_map, p_cam;
    p_map.pose.position.x = pose.pose.position.x
                           + dist * sign * std::cos(yaw + M_PI_2);
    p_map.pose.position.y = pose.pose.position.y
                           + dist * sign * std::sin(yaw + M_PI_2);
    p_map.pose.position.z = pose.pose.position.z;
    p_map.pose.orientation.w = 1.0;

    tf2::doTransform(p_map, p_cam, tf_cam_map);

    // Behind camera check
    if (p_cam.pose.position.z < 0.15) return std::nullopt;

    // Project through full camera model (handles distortion)
    cv::Point2d px = cam_model_.project3dToPixel(cv::Point3d(
        p_cam.pose.position.x,
        p_cam.pose.position.y,
        p_cam.pose.position.z));

    int u = (int)std::round(px.x);
    int v = (int)std::round(px.y);

    if (u < 0 || u >= width_ || v < 0 || v >= height_)
      return std::nullopt;

    return cv::Point(u, v);
  }

  // Sample the terrain label at a pixel location (from CPU-side copy)
  uint8_t sampleLabel(const cv::Point & px) {
    if (cpu_labels_.empty()) return pcv::LABEL_INVALID;
    if (px.x < 0 || px.x >= cpu_labels_.cols ||
        px.y < 0 || px.y >= cpu_labels_.rows)
      return pcv::LABEL_INVALID;
    return cpu_labels_.at<uint8_t>(px.y, px.x);
  }

  // Compute expected ground normal in camera optical frame.
  // = rotation part of TF(camera_optical ← base_link) applied to [0, 0, 1].
  // On a slope, base_link z-axis tilts with the vehicle, so this automatically
  // adapts the "what is ground" reference to the current terrain pitch/roll.
  bool computeExpectedGroundNormal(
      float & gnx, float & gny, float & gnz)
  {
    try {
      auto tf = tf_buffer_.lookupTransform(
          cam_frame_, base_frame_, rclcpp::Time(0));

      // Extract rotation (quaternion → rotate unit Z vector)
      double qx = tf.transform.rotation.x;
      double qy = tf.transform.rotation.y;
      double qz = tf.transform.rotation.z;
      double qw = tf.transform.rotation.w;

      // Rotate [0, 0, 1] by quaternion q:
      // v' = q * [0,0,1] * q_conj
      // Expanded for just the z-basis vector:
      gnx = (float)(2.0 * (qx*qz + qy*qw));
      gny = (float)(2.0 * (qy*qz - qx*qw));
      gnz = (float)(1.0 - 2.0*(qx*qx + qy*qy));

      // Normalise (should already be unit but safety)
      float len = std::sqrt(gnx*gnx + gny*gny + gnz*gnz);
      if (len < 1e-6f) return false;
      gnx /= len; gny /= len; gnz /= len;

      return true;
    } catch (const tf2::TransformException &) {
      return false;
    }
  }

  // Check if base_link has drifted outside the corridor polygon (world XY).
  // Publishes True on /corridor/safety_violation if vehicle is outside.
  void checkBoundaryCrossing(
      const std::vector<cv::Point2f> & left_world,
      const std::vector<cv::Point2f> & right_world,
      const std::string & world_frame)
  {
    std_msgs::msg::Bool out;
    out.data = false;

    try {
      // Use Time(0) = latest available transform.
      // boundaryCb fires from a wall-timer; using get_clock()->now() breaks in
      // simulation because wall-clock time >> sim time, causing "extrapolation
      // into the future". Time(0) always returns the most recent transform
      // regardless of clock source.
      auto tf = tf_buffer_.lookupTransform(
          world_frame, base_frame_, rclcpp::Time(0));

      cv::Point2f base_xy(
          (float)tf.transform.translation.x,
          (float)tf.transform.translation.y);

      // Build corridor polygon: left boundary forward + right boundary reversed
      std::vector<cv::Point2f> poly;
      poly.reserve(left_world.size() + right_world.size());
      poly.insert(poly.end(), left_world.begin(),  left_world.end());
      poly.insert(poly.end(), right_world.rbegin(), right_world.rend());

      // pointPolygonTest: positive = inside, negative = outside
      double dist = cv::pointPolygonTest(poly, base_xy, false);
      out.data = (dist < 0.0);

    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "BoundaryCrossing TF [%s→%s]: %s",
          base_frame_.c_str(), world_frame.c_str(), ex.what());
    }

    safety_pub_->publish(out);
  }

  // Publish terrain classification masked to the corridor (black outside).
  void publishTerrainDebug(const std_msgs::msg::Header & header,
                           const cv::Mat & mask) {
    cv::Mat debug(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));  // black bg
    for (int v = 0; v < height_; v++) {
      for (int u = 0; u < width_; u++) {
        if (!mask.at<uint8_t>(v, u)) continue;  // outside corridor — stay black
        uint8_t l = cpu_labels_.at<uint8_t>(v, u);
        auto & px = debug.at<cv::Vec3b>(v, u);
        if      (l == pcv::LABEL_GROUND) px = {50, 200, 50};   // green
        else if (l == pcv::LABEL_WALL)   px = {50, 50, 220};   // red
        // INVALID inside corridor stays black
      }
    }
    auto msg = cv_bridge::CvImage(header, "bgr8", debug).toImageMsg();
    terrain_pub_->publish(*msg);
  }


  // Publish a virtual fence along both corridor edges so Nav2 planners cannot
  // route outside the ±1 m safety zone.
  //
  // Takes left_world and right_world (world XY of the corridor boundaries in
  // the path frame), densifies to fence_step_m spacing, and stacks points at
  // multiple heights to form a solid "wall" the costmap marks as LETHAL.
  //
  // Published in path->header.frame_id (= map frame) so the costmap needs no
  // TF transform — it sees the fence directly in its own reference frame.
  void publishBoundaryCloud(
      const std_msgs::msg::Header & path_header,
      const std::vector<cv::Point2f> & left_world,
      const std::vector<cv::Point2f> & right_world)
  {
    // Fence heights: create a solid wall from ~ground to 2 m so all costmap
    // height filters will capture it (min_height 0.0 → max_height 2.0).
    static constexpr float FENCE_HEIGHTS[] = {0.1f, 0.5f, 1.0f, 1.5f};
    static constexpr int   N_HEIGHTS = 4;
    static constexpr float STEP = 0.15f;  // densification step (m) — ≤ costmap resolution

    std::vector<std::array<float, 3>> pts;
    pts.reserve((left_world.size() + right_world.size()) * 20);

    // Helper: densify a boundary segment and add fence points
    auto addSegment = [&](const cv::Point2f & a, const cv::Point2f & b) {
      float dx = b.x - a.x, dy = b.y - a.y;
      float len = std::sqrt(dx*dx + dy*dy);
      if (len < 1e-4f) return;
      int steps = std::max(1, (int)std::ceil(len / STEP));
      for (int s = 0; s <= steps; ++s) {
        float t = (float)s / steps;
        float x = a.x + t * dx;
        float y = a.y + t * dy;
        for (int h = 0; h < N_HEIGHTS; ++h) {
          pts.push_back({x, y, FENCE_HEIGHTS[h]});
        }
      }
    };

    for (size_t i = 0; i + 1 < left_world.size(); ++i)
      addSegment(left_world[i], left_world[i+1]);
    for (size_t i = 0; i + 1 < right_world.size(); ++i)
      addSegment(right_world[i], right_world[i+1]);

    auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header          = path_header;   // map frame — costmap uses this directly
    cloud->height          = 1;
    cloud->is_dense        = false;

    sensor_msgs::PointCloud2Modifier mod(*cloud);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(pts.size());

    sensor_msgs::PointCloud2Iterator<float> ix(*cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iy(*cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iz(*cloud, "z");
    for (const auto & p : pts) {
      *ix = p[0]; *iy = p[1]; *iz = p[2];
      ++ix; ++iy; ++iz;
    }

    boundary_pub_->publish(*cloud);
  }

  // Unproject WALL pixels inside the corridor to 3D points and publish as
  // PointCloud2 in the camera optical frame. Nav2 costmap layers (voxel_layer,
  // obstacle_layer, STVL) subscribe to this topic to mark obstacles.
  void publishObstacleCloud(const std_msgs::msg::Header & header,
                            const cv::Mat & mask)
  {
    if (latest_depth_cpu_.empty()) return;

    // Collect 3D points: WALL pixels inside corridor only.
    // Unproject (u, v, d) → camera optical frame: X=(u-cx)*d/fx, Y=(v-cy)*d/fy, Z=d
    std::vector<std::array<float, 3>> pts;
    pts.reserve((height_ / cloud_stride_) * (width_ / cloud_stride_));

    for (int v = 0; v < height_; v += cloud_stride_) {
      const uint8_t * mrow = mask.ptr<uint8_t>(v);
      const uint8_t * lrow = cpu_labels_.ptr<uint8_t>(v);
      const float   * drow = latest_depth_cpu_.ptr<float>(v);

      for (int u = 0; u < width_; u += cloud_stride_) {
        if (!mrow[u]) continue;
        if (lrow[u] != pcv::LABEL_WALL) continue;

        const float d = drow[u];
        if (d <= 0.0f || std::isnan(d) || std::isinf(d)) continue;
        if (d < (float)min_depth_ || d > (float)max_depth_) continue;

        pts.push_back({
            ((float)u - cx_) * d / fx_,
            ((float)v - cy_) * d / fy_,
            d
        });
      }
    }

    // Build PointCloud2 in the camera optical frame.
    // The Nav2 costmap performs the TF lookup to its own frame internally.
    auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header          = header;
    cloud->header.frame_id = cam_frame_;
    cloud->height          = 1;          // unorganised
    cloud->is_dense        = false;

    sensor_msgs::PointCloud2Modifier mod(*cloud);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(pts.size());

    sensor_msgs::PointCloud2Iterator<float> ix(*cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iy(*cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iz(*cloud, "z");
    for (const auto & p : pts) {
      *ix = p[0]; *iy = p[1]; *iz = p[2];
      ++ix; ++iy; ++iz;
    }

    cloud_pub_->publish(*cloud);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  MEMBER VARIABLES
  // ═══════════════════════════════════════════════════════════════════════════

  // ROS
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr          path_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr      depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr      image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr         overlay_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr         terrain_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr             safety_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr   cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr   boundary_pub_;
  rclcpp::TimerBase::SharedPtr                                  boundary_timer_;

  tf2_ros::Buffer            tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  image_geometry::PinholeCameraModel cam_model_;

  // Parameters
  double max_half_, robot_half_;
  double bilateral_sigma_s_, bilateral_sigma_d_;
  double max_depth_jump_;
  double min_depth_, max_depth_;
  double cos_ground_thresh_;
  double overlay_alpha_;
  double shrink_margin_;
  int    shrink_samples_;
  std::string base_frame_;
  bool   publish_debug_;

  // Camera
  std::string cam_frame_;
  float fx_, fy_, cx_, cy_;
  int width_{0}, height_{0};
  bool cam_ready_{false};

  // Path (guarded by mutex)
  std::mutex path_mutex_;
  nav_msgs::msg::Path::SharedPtr latest_path_;

  // Terrain state
  cv::Mat       cpu_labels_;
  cv::Mat       latest_depth_cpu_;   // raw depth, saved each depthCb for pointcloud unprojection
  rclcpp::Time  terrain_stamp_;
  bool          terrain_ready_{false};
  int           cloud_stride_{2};    // downsample stride for obstacle pointcloud

  double boundary_lookahead_m_{15.0};

  // Terrain processing backend (CUDA or CPU, chosen at construction)
  std::unique_ptr<pcv::TerrainBackend> backend_;
};


// ═════════════════════════════════════════════════════════════════════════════
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathCorridorSegmenter>());
  rclcpp::shutdown();
  return 0;
}
