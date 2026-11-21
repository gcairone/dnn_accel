#pragma once
// Per-target tuning constants that the backend-independent code (Linear,
// the buffer sizing in layer.cpp) has to agree with the active kernels on.
// Everything here is a measured number for one fixed CPU, selected at compile
// time by the same FASTMLP_BACKEND_* macro as the kernels themselves.

#include <cstddef>

namespace fastmlp::simd::tuning {

// Widest K-slice (in floats per row) any backend's dX/dW kernels copy into
// the caller's pack scratch: avx2 uses 16 (2 ymm), avx512 up to 64 (a 4-zmm
// panel). Sized for the widest so the same buffer is valid for every
// backend -- tests run all of them against one buffer.
inline constexpr int kPackSliceFloats = 64;

#if defined(FASTMLP_BACKEND_AVX512)
// Linear::update_weights: rows per optimizer/repack block. A multiple of 16
// so each block transposes into whole 64-byte lines of Wp, and at least
// kUpdateMinElems elements so a narrow layer is not cut into many tiny
// optimizer calls. The block (W rows + gradient + optimizer state) must stay
// L1-resident between the update and the transpose: 48 KB L1D here.
inline constexpr int kUpdateRowAlign = 16;
inline constexpr int kUpdateMinElems = 2048;
#else
// Zen 2: measured best of 8/16/32/64/whole-matrix rows (32 KB L1D).
inline constexpr int kUpdateRowAlign = 16;
inline constexpr int kUpdateMinElems = 2048;
#endif

}  // namespace fastmlp::simd::tuning
