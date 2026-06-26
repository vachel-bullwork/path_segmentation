// ═══════════════════════════════════════════════════════════════════════════════
// terrain_backend_cpu.cpp
//
// Pure-CPU implementation of the terrain pipeline. The per-pixel math is a
// direct port of the CUDA kernels in terrain_kernels.cu, so CPU and GPU
// classification are identical (the CPU path is just slower). This lets the
// package build and run on a simulation host with no CUDA toolchain.
//
// Also hosts the createTerrainBackend() factory, which picks the CUDA backend
// when available (HAVE_CUDA) and the env var allows, else falls back here.
// ═══════════════════════════════════════════════════════════════════════════════

#include "path_segmentor/terrain_backend.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdint>

namespace pcv
{

namespace {

inline bool isValidDepth(float d) {
  return d > 0.0f && !std::isnan(d) && !std::isinf(d);
}

// ── KERNEL 1 (CPU) — Bilateral depth filter, 5×5 ─────────────────────────────
void cpuBilateralDepthFilter(
    const cv::Mat & din, cv::Mat & dout,
    float sigma_s, float sigma_d)
{
  const int w = din.cols, h = din.rows;
  dout.create(h, w, CV_32FC1);

  const float inv_2ss = 1.0f / (2.0f * sigma_s * sigma_s);
  const float inv_2sd = 1.0f / (2.0f * sigma_d * sigma_d);

  for (int v = 0; v < h; ++v) {
    const float * row_c = din.ptr<float>(v);
    float *       row_o = dout.ptr<float>(v);

    for (int u = 0; u < w; ++u) {
      float center = row_c[u];
      if (!isValidDepth(center)) { row_o[u] = 0.0f; continue; }

      float sum_w = 0.0f, sum_d = 0.0f;
      for (int dv = -2; dv <= 2; ++dv) {
        int vv = v + dv;
        if (vv < 0 || vv >= h) continue;
        const float * row_n = din.ptr<float>(vv);
        for (int du = -2; du <= 2; ++du) {
          int uu = u + du;
          if (uu < 0 || uu >= w) continue;
          float nd = row_n[uu];
          if (!isValidDepth(nd)) continue;

          float spatial_dist2 = (float)(du*du + dv*dv);
          float depth_diff    = nd - center;
          float depth_dist2   = depth_diff * depth_diff;
          float weight = std::exp(-spatial_dist2 * inv_2ss
                                  - depth_dist2  * inv_2sd);
          sum_w += weight;
          sum_d += weight * nd;
        }
      }
      row_o[u] = (sum_w > 1e-6f) ? (sum_d / sum_w) : 0.0f;
    }
  }
}

// ── KERNEL 2 (CPU) — Surface normals, 5×5 cross (±2) ─────────────────────────
void cpuComputeNormals(
    const cv::Mat & depth, cv::Mat & normals,
    float fx, float fy, float cx, float cy,
    float max_depth_jump)
{
  const int w = depth.cols, h = depth.rows;
  normals.create(h, w, CV_32FC3);

  for (int v = 0; v < h; ++v) {
    float * out_row = normals.ptr<float>(v);

    for (int u = 0; u < w; ++u) {
      float * out = out_row + u * 3;

      if (u < 2 || v < 2 || u >= w-2 || v >= h-2) {
        out[0] = out[1] = out[2] = 0.0f;
        continue;
      }

      const float * rc = depth.ptr<float>(v);
      const float * ru = depth.ptr<float>(v-2);
      const float * rd = depth.ptr<float>(v+2);

      float dc = rc[u];
      float dl = rc[u-2];
      float dr = rc[u+2];
      float dt = ru[u];
      float db = rd[u];

      if (!isValidDepth(dc) || !isValidDepth(dl) || !isValidDepth(dr) ||
          !isValidDepth(dt) || !isValidDepth(db)) {
        out[0] = out[1] = out[2] = 0.0f;
        continue;
      }

      float maxj = max_depth_jump;
      if (std::fabs(dl - dc) > maxj || std::fabs(dr - dc) > maxj ||
          std::fabs(dt - dc) > maxj || std::fabs(db - dc) > maxj) {
        out[0] = out[1] = out[2] = 0.0f;
        continue;
      }

      // Unproject the 4 cross samples to 3D camera frame
      float Lx = dl * ((float)(u-2) - cx) / fx;
      float Ly = dl * ((float)v     - cy) / fy;
      float Lz = dl;

      float Rx = dr * ((float)(u+2) - cx) / fx;
      float Ry = dr * ((float)v     - cy) / fy;
      float Rz = dr;

      float Tx = dt * ((float)u     - cx) / fx;
      float Ty = dt * ((float)(v-2) - cy) / fy;
      float Tz = dt;

      float Bx = db * ((float)u     - cx) / fx;
      float By = db * ((float)(v+2) - cy) / fy;
      float Bz = db;

      float dxU = Rx - Lx, dxV = Ry - Ly, dxW = Rz - Lz;
      float dyU = Bx - Tx, dyV = By - Ty, dyW = Bz - Tz;

      float nx = dxV * dyW - dxW * dyV;
      float ny = dxW * dyU - dxU * dyW;
      float nz = dxU * dyV - dxV * dyU;

      float len = std::sqrt(nx*nx + ny*ny + nz*nz);
      if (len < 1e-8f) { out[0] = out[1] = out[2] = 0.0f; continue; }

      float inv_len = 1.0f / len;
      nx *= inv_len; ny *= inv_len; nz *= inv_len;
      if (nz > 0.0f) { nx = -nx; ny = -ny; nz = -nz; }

      out[0] = nx; out[1] = ny; out[2] = nz;
    }
  }
}

// ── KERNEL 3 (CPU) — Terrain classification ──────────────────────────────────
void cpuClassifyTerrain(
    const cv::Mat & normals, const cv::Mat & depth, cv::Mat & labels,
    float gnd_nx, float gnd_ny, float gnd_nz,
    float cos_thresh, float min_depth, float max_depth)
{
  const int w = depth.cols, h = depth.rows;
  labels.create(h, w, CV_8UC1);

  for (int v = 0; v < h; ++v) {
    const float *   drow = depth.ptr<float>(v);
    const float *   nrow = normals.ptr<float>(v);
    uint8_t *       lrow = labels.ptr<uint8_t>(v);

    for (int u = 0; u < w; ++u) {
      float d = drow[u];
      if (!isValidDepth(d) || d < min_depth || d > max_depth) {
        lrow[u] = LABEL_INVALID;
        continue;
      }

      const float * n = nrow + u * 3;
      float nx = n[0], ny = n[1], nz = n[2];
      if (nx == 0.0f && ny == 0.0f && nz == 0.0f) {
        lrow[u] = LABEL_INVALID;
        continue;
      }

      float dot = std::fabs(nx*gnd_nx + ny*gnd_ny + nz*gnd_nz);
      lrow[u] = (dot > cos_thresh) ? LABEL_GROUND : LABEL_WALL;
    }
  }
}

// ── KERNEL 4 (CPU) — Overlay compositing ─────────────────────────────────────
void cpuOverlayComposite(
    const cv::Mat & rgb_in, cv::Mat & rgb_out,
    const cv::Mat & mask, const cv::Mat & labels,
    float alpha)
{
  const int w = rgb_in.cols, h = rgb_in.rows;
  rgb_out.create(h, w, CV_8UC3);
  const float inv_alpha = 1.0f - alpha;

  for (int v = 0; v < h; ++v) {
    const uint8_t * src  = rgb_in.ptr<uint8_t>(v);
    uint8_t *       dst  = rgb_out.ptr<uint8_t>(v);
    const uint8_t * mrow = mask.ptr<uint8_t>(v);
    const uint8_t * lrow = labels.ptr<uint8_t>(v);

    for (int u = 0; u < w; ++u) {
      const uint8_t * s = src + u * 3;
      uint8_t *       d = dst + u * 3;

      if (mrow[u] == 0) {
        d[0] = 0; d[1] = 0; d[2] = 0;  // black outside corridor
        continue;
      }

      float tint_b, tint_g, tint_r;
      switch (lrow[u]) {
        case LABEL_GROUND: tint_b = 30.0f;  tint_g = 210.0f; tint_r = 50.0f;  break;
        case LABEL_WALL:   tint_b = 40.0f;  tint_g = 40.0f;  tint_r = 230.0f; break;
        default:           tint_b = 20.0f;  tint_g = 190.0f; tint_r = 210.0f; break;
      }

      d[0] = (uint8_t)(inv_alpha * s[0] + alpha * tint_b);
      d[1] = (uint8_t)(inv_alpha * s[1] + alpha * tint_g);
      d[2] = (uint8_t)(inv_alpha * s[2] + alpha * tint_r);
    }
  }
}

}  // anonymous namespace


// ═════════════════════════════════════════════════════════════════════════════
// CpuTerrainBackend
// ═════════════════════════════════════════════════════════════════════════════
class CpuTerrainBackend : public TerrainBackend
{
public:
  void computeTerrain(
      const cv::Mat & depth_in,
      float fx, float fy, float cx, float cy,
      float gnd_nx, float gnd_ny, float gnd_nz,
      const TerrainParams & p,
      cv::Mat & labels_out) override
  {
    cpuBilateralDepthFilter(depth_in, smooth_, p.bilateral_sigma_s, p.bilateral_sigma_d);
    cpuComputeNormals(smooth_, normals_, fx, fy, cx, cy, p.max_depth_jump);
    cpuClassifyTerrain(normals_, smooth_, labels_out,
                       gnd_nx, gnd_ny, gnd_nz,
                       p.cos_ground_thresh, p.min_depth, p.max_depth);
  }

  void composite(
      const cv::Mat & rgb_in, const cv::Mat & mask,
      const cv::Mat & labels, float alpha,
      cv::Mat & overlay_out) override
  {
    cpuOverlayComposite(rgb_in, overlay_out, mask, labels, alpha);
  }

  const char * name() const override { return "CPU"; }

private:
  cv::Mat smooth_;    // CV_32FC1 reused scratch
  cv::Mat normals_;   // CV_32FC3 reused scratch
};

std::unique_ptr<TerrainBackend> createCpuBackend() {
  return std::make_unique<CpuTerrainBackend>();
}


// ═════════════════════════════════════════════════════════════════════════════
// Factory
// ═════════════════════════════════════════════════════════════════════════════
#ifdef HAVE_CUDA
std::unique_ptr<TerrainBackend> createCudaBackend();  // defined in terrain_backend_cuda.cpp
#endif

std::unique_ptr<TerrainBackend> createTerrainBackend(bool prefer_cuda)
{
#ifdef HAVE_CUDA
  if (prefer_cuda) {
    return createCudaBackend();
  }
#else
  (void)prefer_cuda;  // no CUDA compiled in — always CPU
#endif
  return createCpuBackend();
}

}  // namespace pcv
