#pragma once
#include <cstdint>

// CUDA-free shared types. Safe to include from pure-C++ translation units
// (CPU backend, node) as well as from .cu / CUDA backend code.

namespace pcv  // path_segmentor
{

// ── Terrain labels ──────────────────────────────────────────────────────────
enum TerrainLabel : uint8_t {
  LABEL_INVALID = 0,   // no depth, out of range, or edge discontinuity
  LABEL_GROUND  = 1,   // surface normal close to expected ground normal
  LABEL_WALL    = 2,   // surface normal far from expected ground (vertical/steep)
};

}  // namespace pcv
