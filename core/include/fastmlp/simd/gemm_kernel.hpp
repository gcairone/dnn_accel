#pragma once
// GEMM shape here is NOT the generic C = A@B: it's the shape every Linear
// forward/backward pass actually needs, matching the project's weight
// layout decision (row-major (out_features, in_features), like nn.Linear):
//
//   Y(B,N)  = X(B,K) @ W(N,K)^T [+ bias(N)]      -- forward
//   dX(B,K) = dY(B,N) @ W(N,K)                    -- input gradient
//   dW(N,K) += X(B,K)^T @ dY(B,N)                 -- weight gradient (accumulates)
//   db(N)   += sum_b dY(b,:)                      -- bias gradient (accumulates)
//
// All buffers row-major, no allocation inside any of these functions --
// every buffer is caller-owned and preallocated (idea.md: zero allocation
// in the hot path).
//
// PACKED FORWARD (the fast path Linear::forward actually uses)
// ------------------------------------------------------------
// The forward pass is the one place the (N,K) weight layout fights the
// hardware: Y = X @ W^T needs W's N dimension contiguous, but W stores K
// contiguously. A dot-product kernel can work directly on (N,K), and the
// original one did, but it pays twice -- one horizontal reduction per
// output (which dominates at small K: measured 2.97 GFLOP/s on a 20->64
// layer, slower than the scalar reference), and each vector accumulator
// only ever yields a single scalar output, wasting 7 of 8 lanes' worth of
// register capacity.
//
// So the forward path instead consumes W pre-transposed into (K, ldWp)
// "packed" form, where a row is one k and holds all N values contiguously.
// That turns the kernel into a standard outer-product accumulation: every
// vector accumulator holds 8 *distinct* outputs, there is no horizontal
// reduction at all, and the register budget covers a much larger output
// tile. Packing costs O(N*K) once per weight change, against O(B*N*K)
// per forward -- and for an inference_only model the weights never change,
// so it is paid exactly once, at construction.
//
// gemm_xwt_f32 (unpacked, (N,K)) is kept as the correctness reference and
// is what tests compare the packed path against.

namespace fastmlp::simd {

// Ground truth: portable, ISA-independent, no intrinsics. Always built and
// always available -- this is what every other backend is checked against.
namespace scalar {

void gemm_xwt_f32(const float* X, const float* W, const float* bias,
                   float* Y, int B, int K, int N);

void gemm_dX_f32(const float* dY, const float* W, float* dX, int B, int K,
                  int N);

void gemm_dW_accum_f32(const float* X, const float* dY, float* dW, int B,
                        int K, int N);

void bias_grad_accum_f32(const float* dY, float* db, int B, int N);

// dst[c*ld_dst + r] = src[r*src_cols + c], i.e. a plain transpose of a
// (src_rows, src_cols) row-major matrix into (src_cols, ld_dst) row-major.
// Columns [src_rows, ld_dst) are zero-filled so a kernel reading whole
// vectors out of dst never needs masking. Two callers, both hot-path
// prerequisites:
//   pack W(N,K)  -> Wp(K, ldWp)  for gemm_xwtp_f32
//   pack dY(B,N) -> dYt(N, B)    for gemm_dW_accum_t_f32
void pack_transpose_f32(const float* src, float* dst, int src_rows, int src_cols, int ld_dst);

// Y(B,N) = X(B,K) @ Wp + bias, Wp being W packed by pack_transpose_f32.
void gemm_xwtp_f32(const float* X, const float* Wp, const float* bias,
                    float* Y, int B, int K, int N, int ldWp);

// dW(N,K) += dYt(N,B) @ X(B,K), dYt being dY packed by pack_transpose_f32.
// Same result as gemm_dW_accum_f32, which stays as the correctness oracle;
// this variant exists because dW's reduction walks dY down a column, and at
// a power-of-two row stride (N=256 -> 1024 B) that hits only 4 of the L1's
// 64 sets, thrashing an 8-way cache with 128 live lines. Transposing dY
// first turns those strided broadcasts into sequential streams.
void gemm_dW_accum_t_f32(const float* X, const float* dYt, float* dW, int B, int K, int N);

}  // namespace scalar

// Row stride the packed forward expects. Two things are being arranged here:
//
// 1. N is rounded up to a full register tile, so the kernel never has to
//    mask an N-tail load.
// 2. The result is then forced to an ODD number of tiles, i.e. an odd number
//    of 64-byte cache lines per row. The forward kernel walks Wp down one
//    row per k, so the row stride *is* its access stride; at a power-of-two
//    stride (N=256 -> 1024 B -> 16 lines) every row maps onto just 4 of the
//    L1's 64 sets, and an 8-way cache thrashes long before K rows are read.
//    Making the line count odd makes it coprime with 64, so consecutive rows
//    walk all 64 sets. Costs at most 16 floats per row.
inline int packed_ldw(int N) {
  constexpr int kNTile = 16;  // kNR(2) * 8 floats, see gemm_kernel_avx2.cpp
  int ld = ((N + kNTile - 1) / kNTile) * kNTile;
  if ((ld / kNTile) % 2 == 0) ld += kNTile;
  return ld;
}

// Hand-written AVX2+FMA3 microkernel (M8). Same 4 signatures as scalar::,
// so both can be linked and compared directly regardless of which one is
// "active" by default.
namespace avx2 {

void gemm_xwt_f32(const float* X, const float* W, const float* bias,
                   float* Y, int B, int K, int N);

void gemm_dX_f32(const float* dY, const float* W, float* dX, int B, int K,
                  int N);

void gemm_dW_accum_f32(const float* X, const float* dY, float* dW, int B,
                        int K, int N);

void bias_grad_accum_f32(const float* dY, float* db, int B, int N);

void pack_transpose_f32(const float* src, float* dst, int src_rows, int src_cols, int ld_dst);

void gemm_xwtp_f32(const float* X, const float* Wp, const float* bias,
                    float* Y, int B, int K, int N, int ldWp);

void gemm_dW_accum_t_f32(const float* X, const float* dYt, float* dW, int B, int K, int N);

}  // namespace avx2

// Dev/testing selector ONLY -- the shipped/production build always selects
// avx2 (FASTMLP_BACKEND_AVX2 defined by CMake once M8 lands and
// FASTMLP_SIMD_BACKEND=AVX2 is configured). scalar exists purely as the
// numerical ground truth for tests and is never chosen at runtime; this is
// a compile-time CMake option, not a portability fallback -- idea.md rules
// that out explicitly, since the target CPU is fixed and known in advance.
#if defined(FASTMLP_BACKEND_AVX2)
using namespace avx2;
#else
using namespace scalar;
#endif

}  // namespace fastmlp::simd
