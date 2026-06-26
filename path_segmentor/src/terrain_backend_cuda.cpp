// ═══════════════════════════════════════════════════════════════════════════════
// terrain_backend_cuda.cpp
//
// GPU implementation of the terrain pipeline. Holds the GpuMat scratch buffers
// and the CUDA stream, uploads host depth, runs the custom kernels, and
// downloads the resulting label map back to a host cv::Mat.
//
// Compiled ONLY when the build found a CUDA toolchain (guarded by HAVE_CUDA in
// CMake — this file is not added to the target otherwise). All cv::cuda and
// raw-CUDA usage is therefore isolated here; the node never sees it.
// ═══════════════════════════════════════════════════════════════════════════════

#include "path_segmentor/terrain_backend.hpp"
#include "path_segmentor/terrain_kernels.cuh"

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

namespace pcv
{

class CudaTerrainBackend : public TerrainBackend
{
public:
  CudaTerrainBackend() {
    cudaStreamCreate(&stream_);
  }

  ~CudaTerrainBackend() override {
    if (stream_) cudaStreamDestroy(stream_);
  }

  void computeTerrain(
      const cv::Mat & depth_in,
      float fx, float fy, float cx, float cy,
      float gnd_nx, float gnd_ny, float gnd_nz,
      const TerrainParams & p,
      cv::Mat & labels_out) override
  {
    const int w = depth_in.cols, h = depth_in.rows;
    ensureSize(w, h);

    gpu_depth_raw_.upload(depth_in);

    launchBilateralDepthFilter(
        gpu_depth_raw_.ptr<float>(),    (int)gpu_depth_raw_.step,
        gpu_depth_smooth_.ptr<float>(), (int)gpu_depth_smooth_.step,
        w, h, p.bilateral_sigma_s, p.bilateral_sigma_d, stream_);

    launchComputeNormals(
        gpu_depth_smooth_.ptr<float>(), (int)gpu_depth_smooth_.step,
        gpu_normals_.ptr<float>(),      (int)gpu_normals_.step,
        w, h, fx, fy, cx, cy, p.max_depth_jump, stream_);

    launchClassifyTerrain(
        gpu_normals_.ptr<float>(),      (int)gpu_normals_.step,
        gpu_depth_smooth_.ptr<float>(), (int)gpu_depth_smooth_.step,
        gpu_labels_.ptr<uint8_t>(),     (int)gpu_labels_.step,
        w, h, gnd_nx, gnd_ny, gnd_nz,
        p.cos_ground_thresh, p.min_depth, p.max_depth, stream_);

    cudaStreamSynchronize(stream_);
    gpu_labels_.download(labels_out);
  }

  void composite(
      const cv::Mat & rgb_in, const cv::Mat & mask,
      const cv::Mat & labels, float alpha,
      cv::Mat & overlay_out) override
  {
    const int w = rgb_in.cols, h = rgb_in.rows;
    ensureSize(w, h);

    gpu_rgb_.upload(rgb_in);
    gpu_mask_.upload(mask);
    gpu_labels_.upload(labels);

    launchOverlayComposite(
        gpu_rgb_.ptr<uint8_t>(),     (int)gpu_rgb_.step,
        gpu_overlay_.ptr<uint8_t>(), (int)gpu_overlay_.step,
        gpu_mask_.ptr<uint8_t>(),    (int)gpu_mask_.step,
        gpu_labels_.ptr<uint8_t>(),  (int)gpu_labels_.step,
        w, h, alpha, stream_);

    cudaStreamSynchronize(stream_);
    gpu_overlay_.download(overlay_out);
  }

  const char * name() const override { return "CUDA"; }

private:
  void ensureSize(int w, int h) {
    if (w == buf_w_ && h == buf_h_) return;
    gpu_depth_raw_    = cv::cuda::GpuMat(h, w, CV_32FC1);
    gpu_depth_smooth_ = cv::cuda::GpuMat(h, w, CV_32FC1);
    gpu_normals_      = cv::cuda::GpuMat(h, w, CV_32FC3);
    gpu_labels_       = cv::cuda::GpuMat(h, w, CV_8UC1);
    gpu_rgb_          = cv::cuda::GpuMat(h, w, CV_8UC3);
    gpu_overlay_      = cv::cuda::GpuMat(h, w, CV_8UC3);
    gpu_mask_         = cv::cuda::GpuMat(h, w, CV_8UC1);
    buf_w_ = w; buf_h_ = h;
  }

  cv::cuda::GpuMat gpu_depth_raw_;
  cv::cuda::GpuMat gpu_depth_smooth_;
  cv::cuda::GpuMat gpu_normals_;
  cv::cuda::GpuMat gpu_labels_;
  cv::cuda::GpuMat gpu_rgb_;
  cv::cuda::GpuMat gpu_overlay_;
  cv::cuda::GpuMat gpu_mask_;
  int buf_w_{0}, buf_h_{0};

  cudaStream_t stream_{nullptr};
};

std::unique_ptr<TerrainBackend> createCudaBackend() {
  return std::make_unique<CudaTerrainBackend>();
}

}  // namespace pcv
