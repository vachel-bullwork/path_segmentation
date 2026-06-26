#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "path_segmentor/terrain_types.hpp"   // TerrainLabel (CUDA-free)

namespace pcv  // path_segmentor
{

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 1 — Depth-aware bilateral filter
//
// Smooths gravel micro-noise while preserving wall edges.
// For each pixel, computes a Gaussian-weighted average of the 5×5 neighbourhood,
// but weights drop to near-zero across large depth discontinuities (wall edges).
//
// depth_in   : float*, row-major, metres (NaN or 0 = invalid)
// depth_out  : float*, same layout, smoothed output
// step_in/out: pitch in BYTES (GpuMat::step)
// sigma_s    : spatial sigma in pixels     (default 2.0)
// sigma_d    : depth sigma in metres       (default 0.1 — 10cm)
// ═══════════════════════════════════════════════════════════════════════════════
void launchBilateralDepthFilter(
    const float* depth_in,  int step_in,
    float*       depth_out, int step_out,
    int w, int h,
    float sigma_s, float sigma_d,
    cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 2 — Surface normal estimation (5×5 cross pattern)
//
// Computes per-pixel surface normals in camera optical frame.
// Uses ±2 pixel cross-pattern finite differences for robustness on gravel.
// Outputs float3 normals (nx, ny, nz) normalised; (0,0,0) if invalid.
//
// depth      : smoothed float depth, metres
// normals    : float output, 3-channel interleaved (nx,ny,nz per pixel)
// step_d/step_n : pitch in BYTES
// fx,fy,cx,cy: camera intrinsics
// max_depth_jump: if depth delta in the cross exceeds this, mark invalid (0.5m)
// ═══════════════════════════════════════════════════════════════════════════════
void launchComputeNormals(
    const float* depth,   int step_d,
    float*       normals, int step_n,
    int w, int h,
    float fx, float fy, float cx, float cy,
    float max_depth_jump,
    cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 3 — Terrain classification (vehicle-relative)
//
// Compares each pixel's surface normal against the expected ground normal
// (derived from the vehicle's base_link orientation in camera frame).
//
// normals       : float 3-channel, from kernel 2
// depth         : smoothed depth (for min/max range check)
// labels        : uint8_t output, TerrainLabel per pixel
// gnd_nx/ny/nz  : expected ground normal in camera optical frame
// cos_thresh    : dot product threshold; cos(30°)≈0.866 for offroad
// min_depth     : minimum valid depth in metres (0.3 for ZED X)
// max_depth     : maximum valid depth in metres
// ═══════════════════════════════════════════════════════════════════════════════
void launchClassifyTerrain(
    const float*   normals, int step_n,
    const float*   depth,   int step_d,
    uint8_t*       labels,  int step_l,
    int w, int h,
    float gnd_nx, float gnd_ny, float gnd_nz,
    float cos_thresh,
    float min_depth, float max_depth,
    cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 4 — Overlay compositing
//
// Blends terrain-classified corridor overlay onto the RGB image.
//   - Outside corridor mask (mask==0) → original pixel
//   - Inside corridor + GROUND  → green tint   (0,200,80) at alpha
//   - Inside corridor + WALL    → red tint     (60,60,220) at alpha
//   - Inside corridor + INVALID → yellow tint  (0,180,200) at alpha
//
// rgb_in   : uint8_t 3-channel BGR input
// rgb_out  : uint8_t 3-channel BGR output (can alias rgb_in for in-place)
// mask     : uint8_t 1-channel corridor mask (0 or 255)
// labels   : uint8_t 1-channel terrain labels
// alpha    : blend strength [0.0–1.0], 0.35 is good
// ═══════════════════════════════════════════════════════════════════════════════
void launchOverlayComposite(
    const uint8_t* rgb_in,  int step_rgb_in,
    uint8_t*       rgb_out, int step_rgb_out,
    const uint8_t* mask,    int step_mask,
    const uint8_t* labels,  int step_labels,
    int w, int h,
    float alpha,
    cudaStream_t stream);

}  // namespace pcv
