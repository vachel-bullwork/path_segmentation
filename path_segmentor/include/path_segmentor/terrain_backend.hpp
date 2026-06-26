#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include "path_segmentor/terrain_types.hpp"

// ═══════════════════════════════════════════════════════════════════════════════
// TerrainBackend — abstraction over the terrain-classification pipeline.
//
// Two implementations:
//   • CpuTerrainBackend  — pure cv::Mat math, always available.
//   • CudaTerrainBackend — GpuMat + custom CUDA kernels, compiled only when
//                          the build found a CUDA toolchain (HAVE_CUDA).
//
// The node is CUDA-agnostic: it holds a std::unique_ptr<TerrainBackend> and
// calls computeTerrain() / composite().  Inputs and outputs are ordinary
// cv::Mat on the host, so the GPU/CPU choice never leaks into the node.
//
// Backend selection (createTerrainBackend) honours the environment variable
//   PATH_SEGMENTOR_USE_CUDA = 1/true/cuda (default)  → CUDA if compiled in
//                           = 0/false/cpu            → force CPU
// but always falls back to CPU when CUDA was not compiled in.
// ═══════════════════════════════════════════════════════════════════════════════

namespace pcv
{

struct TerrainParams {
  float bilateral_sigma_s = 2.0f;    // spatial sigma (pixels)
  float bilateral_sigma_d = 0.10f;   // depth sigma (metres)
  float max_depth_jump    = 0.5f;    // normal-rejection discontinuity (metres)
  float cos_ground_thresh = 0.866f;  // cos(30°); dot above this = GROUND
  float min_depth         = 0.3f;    // valid depth range (metres)
  float max_depth         = 12.0f;
};

class TerrainBackend
{
public:
  virtual ~TerrainBackend() = default;

  // depth_in : CV_32FC1, metres (0 / NaN / Inf = invalid)
  // intrinsics: fx,fy,cx,cy of the (rectified) camera
  // gnd_n*   : expected ground normal in camera optical frame
  // labels_out : CV_8UC1, TerrainLabel per pixel (allocated/resized as needed)
  virtual void computeTerrain(
      const cv::Mat & depth_in,
      float fx, float fy, float cx, float cy,
      float gnd_nx, float gnd_ny, float gnd_nz,
      const TerrainParams & params,
      cv::Mat & labels_out) = 0;

  // rgb_in   : CV_8UC3 BGR
  // mask     : CV_8UC1 corridor mask (0 = outside, 255 = inside)
  // labels   : CV_8UC1 TerrainLabel per pixel
  // overlay_out : CV_8UC3 BGR blended result (allocated/resized as needed)
  virtual void composite(
      const cv::Mat & rgb_in,
      const cv::Mat & mask,
      const cv::Mat & labels,
      float alpha,
      cv::Mat & overlay_out) = 0;

  // Human-readable backend name for logging ("CUDA" / "CPU").
  virtual const char * name() const = 0;
};

// Factory.  prefer_cuda is the resolved env-var preference; the returned
// backend is CUDA only if prefer_cuda AND the build has HAVE_CUDA.
std::unique_ptr<TerrainBackend> createTerrainBackend(bool prefer_cuda);

}  // namespace pcv
