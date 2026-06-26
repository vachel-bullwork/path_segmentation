#include "path_segmentor/terrain_kernels.cuh"
#include <math.h>
#include <float.h>

namespace pcv
{

// ── Helpers ──────────────────────────────────────────────────────────────────

// Row pointer with byte-pitch awareness (GpuMat stores step in bytes)
template<typename T>
__device__ __forceinline__ const T* row_ptr(const T* base, int row, int step_bytes)
{
  return reinterpret_cast<const T*>(
      reinterpret_cast<const char*>(base) + row * step_bytes);
}

template<typename T>
__device__ __forceinline__ T* row_ptr_mut(T* base, int row, int step_bytes)
{
  return reinterpret_cast<T*>(
      reinterpret_cast<char*>(base) + row * step_bytes);
}

__device__ __forceinline__ bool is_valid_depth(float d)
{
  return d > 0.0f && !isnan(d) && !isinf(d);
}


// ═════════════════════════════════════════════════════════════════════════════
// KERNEL 1 — Bilateral depth filter
//
// 5×5 window.  Each neighbour's weight = G_spatial * G_depth.
// Gravel: stones cause 1-3 cm ripples.  sigma_d=0.1m smooths these.
// Walls:  depth jumps >10 cm at edges.  sigma_d=0.1m preserves them.
// ═════════════════════════════════════════════════════════════════════════════

__global__ void bilateralDepthFilterKernel(
    const float* __restrict__ depth_in,  int step_in,
    float*       __restrict__ depth_out, int step_out,
    int w, int h,
    float inv_2sigma_s2,   // precomputed: 1 / (2 * sigma_s^2)
    float inv_2sigma_d2)   // precomputed: 1 / (2 * sigma_d^2)
{
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= w || v >= h) return;

  const float* row_c = row_ptr(depth_in, v, step_in);
  float center = row_c[u];

  if (!is_valid_depth(center)) {
    row_ptr_mut(depth_out, v, step_out)[u] = 0.0f;
    return;
  }

  float sum_w = 0.0f;
  float sum_d = 0.0f;

  // 5×5 window = radius 2
  for (int dv = -2; dv <= 2; dv++) {
    int vv = v + dv;
    if (vv < 0 || vv >= h) continue;
    const float* row_n = row_ptr(depth_in, vv, step_in);

    for (int du = -2; du <= 2; du++) {
      int uu = u + du;
      if (uu < 0 || uu >= w) continue;

      float nd = row_n[uu];
      if (!is_valid_depth(nd)) continue;

      float spatial_dist2 = (float)(du*du + dv*dv);
      float depth_diff    = nd - center;
      float depth_dist2   = depth_diff * depth_diff;

      float weight = expf(-spatial_dist2 * inv_2sigma_s2
                          -depth_dist2   * inv_2sigma_d2);
      sum_w += weight;
      sum_d += weight * nd;
    }
  }

  float result = (sum_w > 1e-6f) ? (sum_d / sum_w) : 0.0f;
  row_ptr_mut(depth_out, v, step_out)[u] = result;
}


// ═════════════════════════════════════════════════════════════════════════════
// KERNEL 2 — Surface normal estimation (5×5 cross, ±2 pixel stride)
//
// Uses wider stride than 3×3 for robustness on textured surfaces (gravel).
// Cross pattern: only samples the 4 cardinal neighbours at offset ±2,
// which averages out per-stone depth jitter while staying cheap.
//
// Normals are stored as 3-channel interleaved float (nx, ny, nz).
// In camera optical frame: +X=right, +Y=down, +Z=forward.
// Ground normal ≈ (0, -1, 0) on flat ground (pointing up = -Y in optical).
// ═════════════════════════════════════════════════════════════════════════════

__global__ void computeNormalsKernel(
    const float* __restrict__ depth,   int step_d,
    float*       __restrict__ normals, int step_n,   // 3-channel float
    int w, int h,
    float fx, float fy, float cx, float cy,
    float max_depth_jump)
{
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= w || v >= h) return;

  // Output pointer for this pixel (3 floats per pixel)
  float* out = row_ptr_mut(normals, v, step_n) + u * 3;

  // Need ±2 margin
  if (u < 2 || v < 2 || u >= w-2 || v >= h-2) {
    out[0] = out[1] = out[2] = 0.0f;
    return;
  }

  // Sample cross pattern: center, left(-2), right(+2), up(-2), down(+2)
  const float* rc = row_ptr(depth, v,   step_d);
  const float* ru = row_ptr(depth, v-2, step_d);
  const float* rd = row_ptr(depth, v+2, step_d);

  float dc = rc[u];
  float dl = rc[u-2];
  float dr = rc[u+2];
  float dt = ru[u];     // top (v-2)
  float db = rd[u];     // bottom (v+2)

  if (!is_valid_depth(dc) || !is_valid_depth(dl) ||
      !is_valid_depth(dr) || !is_valid_depth(dt) ||
      !is_valid_depth(db)) {
    out[0] = out[1] = out[2] = 0.0f;
    return;
  }

  // Depth discontinuity check — reject if any cross sample jumps too much
  float maxj = max_depth_jump;
  if (fabsf(dl - dc) > maxj || fabsf(dr - dc) > maxj ||
      fabsf(dt - dc) > maxj || fabsf(db - dc) > maxj) {
    out[0] = out[1] = out[2] = 0.0f;
    return;
  }

  // Unproject the 5 sample points to 3D camera frame
  // P(u,v) = ((u-cx)*d/fx, (v-cy)*d/fy, d)
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

  // Tangent vectors from cross differences
  float dxU = Rx - Lx, dxV = Ry - Ly, dxW = Rz - Lz;  // du direction
  float dyU = Bx - Tx, dyV = By - Ty, dyW = Bz - Tz;  // dv direction

  // Cross product → normal
  float nx = dxV * dyW - dxW * dyV;
  float ny = dxW * dyU - dxU * dyW;
  float nz = dxU * dyV - dxV * dyU;

  float len = sqrtf(nx*nx + ny*ny + nz*nz);
  if (len < 1e-8f) {
    out[0] = out[1] = out[2] = 0.0f;
    return;
  }

  // Normalise. Convention: normal should point toward camera (nz < 0 for
  // surfaces facing us). Flip if needed.
  float inv_len = 1.0f / len;
  nx *= inv_len;
  ny *= inv_len;
  nz *= inv_len;
  if (nz > 0.0f) { nx = -nx; ny = -ny; nz = -nz; }

  out[0] = nx;
  out[1] = ny;
  out[2] = nz;
}


// ═════════════════════════════════════════════════════════════════════════════
// KERNEL 3 — Terrain classification (vehicle-relative)
//
// Dot product between surface normal and expected ground normal.
// If dot > cos_thresh → GROUND.  Else → WALL.
// Also rejects depth outside [min_depth, max_depth].
//
// Expected ground normal is computed by the host from:
//   R_cam_base * [0, 0, 1]^T
// where R_cam_base is the rotation part of TF(camera_optical ← base_link).
//
// On a slope, base_link tilts with the vehicle, so the expected normal
// tilts too → ground still matches. A wall perpendicular to the slope
// still reads as wall because its normal is ~90° from expected.
// ═════════════════════════════════════════════════════════════════════════════

__global__ void classifyTerrainKernel(
    const float*   __restrict__ normals, int step_n,
    const float*   __restrict__ depth,   int step_d,
    uint8_t*       __restrict__ labels,  int step_l,
    int w, int h,
    float gnd_nx, float gnd_ny, float gnd_nz,
    float cos_thresh,
    float min_depth, float max_depth)
{
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= w || v >= h) return;

  uint8_t* lbl = row_ptr_mut(labels, v, step_l) + u;

  // Depth range check
  float d = row_ptr(depth, v, step_d)[u];
  if (!is_valid_depth(d) || d < min_depth || d > max_depth) {
    *lbl = LABEL_INVALID;
    return;
  }

  // Normal check
  const float* n = row_ptr(normals, v, step_n) + u * 3;
  float nx = n[0], ny = n[1], nz = n[2];

  // Zero normal = invalid from kernel 2
  if (nx == 0.0f && ny == 0.0f && nz == 0.0f) {
    *lbl = LABEL_INVALID;
    return;
  }

  // Dot product with expected ground normal
  // Both normals point toward camera (-Z direction), so dot > 0 = similar
  float dot = fabsf(nx*gnd_nx + ny*gnd_ny + nz*gnd_nz);

  *lbl = (dot > cos_thresh) ? LABEL_GROUND : LABEL_WALL;
}


// ═════════════════════════════════════════════════════════════════════════════
// KERNEL 4 — Overlay compositing
// ═════════════════════════════════════════════════════════════════════════════

__global__ void overlayCompositeKernel(
    const uint8_t* __restrict__ rgb_in,  int step_rgb_in,
    uint8_t*       __restrict__ rgb_out, int step_rgb_out,
    const uint8_t* __restrict__ mask,    int step_mask,
    const uint8_t* __restrict__ labels,  int step_labels,
    int w, int h,
    float alpha)
{
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= w || v >= h) return;

  const uint8_t* src = row_ptr(rgb_in, v, step_rgb_in) + u * 3;
  uint8_t*       dst = row_ptr_mut(rgb_out, v, step_rgb_out) + u * 3;

  uint8_t m = row_ptr(mask,   v, step_mask)[u];
  uint8_t l = row_ptr(labels, v, step_labels)[u];

  if (m == 0) {
    // Outside corridor — black
    dst[0] = 0; dst[1] = 0; dst[2] = 0;
    return;
  }

  // Inside corridor — blend with terrain-dependent colour
  float tint_b, tint_g, tint_r;

  switch (l) {
    case LABEL_GROUND:
      // Green — driveable ground
      tint_b = 30.0f;  tint_g = 210.0f; tint_r = 50.0f;
      break;
    case LABEL_WALL:
      // Red — wall / steep obstacle
      tint_b = 40.0f;  tint_g = 40.0f;  tint_r = 230.0f;
      break;
    default:  // LABEL_INVALID
      // Yellow — unknown / no data
      tint_b = 20.0f;  tint_g = 190.0f; tint_r = 210.0f;
      break;
  }

  float inv_alpha = 1.0f - alpha;
  dst[0] = (uint8_t)(inv_alpha * src[0] + alpha * tint_b);
  dst[1] = (uint8_t)(inv_alpha * src[1] + alpha * tint_g);
  dst[2] = (uint8_t)(inv_alpha * src[2] + alpha * tint_r);
}


// ═════════════════════════════════════════════════════════════════════════════
// HOST LAUNCHERS
// ═════════════════════════════════════════════════════════════════════════════

static dim3 grid2d(int w, int h, int bx = 16, int by = 16) {
  return dim3((w + bx - 1) / bx, (h + by - 1) / by);
}

void launchBilateralDepthFilter(
    const float* depth_in,  int step_in,
    float*       depth_out, int step_out,
    int w, int h,
    float sigma_s, float sigma_d,
    cudaStream_t stream)
{
  dim3 block(16, 16);
  float inv_2ss = 1.0f / (2.0f * sigma_s * sigma_s);
  float inv_2sd = 1.0f / (2.0f * sigma_d * sigma_d);
  bilateralDepthFilterKernel<<<grid2d(w,h), block, 0, stream>>>(
      depth_in, step_in, depth_out, step_out,
      w, h, inv_2ss, inv_2sd);
}

void launchComputeNormals(
    const float* depth,   int step_d,
    float*       normals, int step_n,
    int w, int h,
    float fx, float fy, float cx, float cy,
    float max_depth_jump,
    cudaStream_t stream)
{
  dim3 block(16, 16);
  computeNormalsKernel<<<grid2d(w,h), block, 0, stream>>>(
      depth, step_d, normals, step_n,
      w, h, fx, fy, cx, cy, max_depth_jump);
}

void launchClassifyTerrain(
    const float*   normals, int step_n,
    const float*   depth,   int step_d,
    uint8_t*       labels,  int step_l,
    int w, int h,
    float gnd_nx, float gnd_ny, float gnd_nz,
    float cos_thresh,
    float min_depth, float max_depth,
    cudaStream_t stream)
{
  dim3 block(16, 16);
  classifyTerrainKernel<<<grid2d(w,h), block, 0, stream>>>(
      normals, step_n, depth, step_d, labels, step_l,
      w, h, gnd_nx, gnd_ny, gnd_nz,
      cos_thresh, min_depth, max_depth);
}

void launchOverlayComposite(
    const uint8_t* rgb_in,  int step_rgb_in,
    uint8_t*       rgb_out, int step_rgb_out,
    const uint8_t* mask,    int step_mask,
    const uint8_t* labels,  int step_labels,
    int w, int h,
    float alpha,
    cudaStream_t stream)
{
  dim3 block(16, 16);
  overlayCompositeKernel<<<grid2d(w,h), block, 0, stream>>>(
      rgb_in, step_rgb_in, rgb_out, step_rgb_out,
      mask, step_mask, labels, step_labels,
      w, h, alpha);
}

}  // namespace pcv
