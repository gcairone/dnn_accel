// Hand-written AVX2+FMA3 microkernels for the target CPU (AMD Ryzen 7
// 4800H / Zen 2 -- see benchmarks/hardware_analysis/hw.md): no AVX-512
// hardware exists on this part, so AVX2+FMA is not a choice, it's the only
// available SIMD path. 16 architectural YMM (256-bit / 8x fp32) registers,
// not 32 -- every tile below is sized to that budget.
//
// THREE THINGS THE FIRST VERSION GOT WRONG, all found by measurement
// (benchmarks/cpp/bench_breakdown), not by reading the code:
//
// 1. ACCESS PATTERN. Every kernel ran its innermost loop over the full
//    reduction dimension while the outer loop walked the batch, so the large
//    matrix was re-streamed from memory once per outer step. On MNIST's
//    784->256 layer at B=128 that was ~102 MB of traffic for dX and for dW,
//    and ~26 MB for the forward, against 51 MFLOP of actual work. Fix: block
//    the reduction dimension outermost so the slice being reduced over stays
//    L1-resident, and register-block the other two.
//
// 2. TOO FEW ACCUMULATORS. The first fix used 4-row tiles = 8 accumulators.
//    Zen 2 issues 2 FMAs/cycle at 5-cycle latency, so it needs ~10
//    independent chains to avoid bubbles; 8 left it stalling at roughly half
//    throughput (dX 57 GFLOP/s, dW 29). Widening to 6-row tiles = 12
//    accumulators took dX to 110 and dW to 106 GFLOP/s. This was the single
//    largest win, and it is invisible to a traffic analysis -- it only shows
//    up as "the kernel is nowhere near its roofline".
//
// 3. POWER-OF-TWO ROW STRIDE. The packed forward walks Wp one row per k, so
//    Wp's row stride is its access stride. At N=256 that stride was 1024 B =
//    16 cache lines, mapping every row onto just 4 of the L1's 64 sets; an
//    8-way cache thrashes long before K rows are read. Padding the stride to
//    an odd number of lines (see packed_ldw()) took the forward from 41 to
//    107 GFLOP/s. Note the same trap does NOT apply to dX/dW, whose stride is
//    K and therefore not something this code controls.
//
// Tiles (all within the 16-register budget, leaving room for operands):
//   gemm_xwtp_f32       6 batch rows  x 2 vector-columns = 12 accumulators
//   gemm_dX_f32         6 batch rows  x 2 k-vectors      = 12 accumulators
//   gemm_dW_accum_*_f32 6 output rows x 2 k-vectors      = 12 accumulators
//
// K-remainders use masked loads at the same tile width rather than falling
// back to a scalar loop: shapes like usages.md ex1's K=20 leave a 4-wide
// remainder that a scalar path would make the dominant cost of the kernel.
//
// gemm_xwt_f32 (the original unpacked dot-product forward) is deliberately
// left as-is: it is the correctness reference the packed path is tested
// against, and nothing performance-critical calls it any more.
#include "fastmlp/simd/gemm_kernel.hpp"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace fastmlp::simd::avx2 {

namespace {

constexpr int kMR = 4;
constexpr int kNR = 2;
constexpr int kVW = 8;  // AVX2 vector width for fp32

constexpr int kFwdMR = 6;
constexpr int kNTile = 16;  // must match packed_ldw()'s tile in the header

constexpr int kBwdMR = 6;
constexpr int kKTile = 16;  // kBwdKV(2) * kVW

// Below this K the operand's natural row stride is under 1 KB, so a whole
// tile's worth of rows fits in L1 without conflicting and copying the
// K-slice into scratch is pure overhead (measured -6% on a 20->64 layer).
// Above it, packing is worth up to +70%. See pack_scratch_floats().
constexpr int kPackMinK = 256;

constexpr int kGemvNR = 6;  // 6 output rows x 2 k-vectors = 12 FMA chains

inline int min_int(int a, int b) { return a < b ? a : b; }

// Sign-bit mask selecting the first `rem` lanes (rem in [1,8]).
inline __m256i lane_mask(int rem) {
  alignas(32) static const int32_t tbl[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                               0,  0,  0,  0,  0,  0,  0,  0};
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tbl + (8 - rem)));
}

inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

// --- unpacked reference forward (kept as the correctness oracle) ----------

void microkernel_xwt(const float* X, int ldX, const float* W, int ldW,
                      const float* bias_tile, float* Y, int ldY, int mr,
                      int nr, int K) {
  __m256 vacc[kMR][kNR];
  for (int m = 0; m < mr; m++)
    for (int n = 0; n < nr; n++) vacc[m][n] = _mm256_setzero_ps();

  int k = 0;
  const int K8 = K - (K % kVW);
  for (; k < K8; k += kVW) {
    __m256 xv[kMR];
    for (int m = 0; m < mr; m++) xv[m] = _mm256_loadu_ps(X + (size_t)m * ldX + k);
    __m256 wv[kNR];
    for (int n = 0; n < nr; n++) wv[n] = _mm256_loadu_ps(W + (size_t)n * ldW + k);
    for (int m = 0; m < mr; m++) {
      for (int n = 0; n < nr; n++) {
        vacc[m][n] = _mm256_fmadd_ps(xv[m], wv[n], vacc[m][n]);
      }
    }
  }

  float acc[kMR][kNR];
  for (int m = 0; m < mr; m++)
    for (int n = 0; n < nr; n++) acc[m][n] = hsum256(vacc[m][n]);

  for (; k < K; k++) {  // scalar K-tail
    for (int m = 0; m < mr; m++) {
      float xk = X[(size_t)m * ldX + k];
      for (int n = 0; n < nr; n++) acc[m][n] += xk * W[(size_t)n * ldW + k];
    }
  }

  for (int m = 0; m < mr; m++) {
    for (int n = 0; n < nr; n++) {
      float v = acc[m][n];
      if (bias_tile) v += bias_tile[n];
      Y[(size_t)m * ldY + n] = v;
    }
  }
}

// --- packed forward ------------------------------------------------------

// MR is a template parameter, not a runtime one, so the accumulator array is
// fully unrolled into registers -- with a runtime bound GCC spills it to the
// stack and the whole point of register blocking is lost. Same reason every
// tile below is templated.
template <int MR>
inline void xwtp_tile(const float* X, const float* Wp, __m256 bv0, __m256 bv1, float* Y,
                       int b0, int K, int N, int ldWp, int n0, int ncols) {
  __m256 a0[MR], a1[MR];
  for (int m = 0; m < MR; m++) {
    a0[m] = bv0;
    a1[m] = bv1;
  }

  const float* wp = Wp + n0;
  for (int k = 0; k < K; k++) {
    __m256 w0 = _mm256_loadu_ps(wp);
    __m256 w1 = _mm256_loadu_ps(wp + kVW);
    wp += ldWp;
    for (int m = 0; m < MR; m++) {
      __m256 xv = _mm256_set1_ps(X[(size_t)(b0 + m) * K + k]);
      a0[m] = _mm256_fmadd_ps(xv, w0, a0[m]);
      a1[m] = _mm256_fmadd_ps(xv, w1, a1[m]);
    }
  }

  if (ncols == kNTile) {
    for (int m = 0; m < MR; m++) {
      float* y = Y + (size_t)(b0 + m) * N + n0;
      _mm256_storeu_ps(y, a0[m]);
      _mm256_storeu_ps(y + kVW, a1[m]);
    }
  } else {
    // N-tail: Wp is zero-padded so the compute above is valid, but Y has no
    // padding, so only the real columns may be written.
    float tmp[kNTile];
    for (int m = 0; m < MR; m++) {
      _mm256_storeu_ps(tmp, a0[m]);
      _mm256_storeu_ps(tmp + kVW, a1[m]);
      float* y = Y + (size_t)(b0 + m) * N + n0;
      for (int j = 0; j < ncols; j++) y[j] = tmp[j];
    }
  }
}

// --- batch-1 forward: y(N) = x(K) @ W(N,K)^T -----------------------------

// NR output rows reduced along K at once. Unlike the packed tile this reads
// W exactly as stored, so the whole matrix streams sequentially in one pass;
// at batch 1 that matters more than the horizontal reductions cost, because
// there is no batch reuse left to amortise anything against.
template <int NR>
inline void gemv_tile(const float* x, const float* W, const float* bias, float* y, int K, int n0) {
  __m256 a0[NR], a1[NR];
  for (int j = 0; j < NR; j++) {
    a0[j] = _mm256_setzero_ps();
    a1[j] = _mm256_setzero_ps();
  }

  int k = 0;
  for (; k + 2 * kVW <= K; k += 2 * kVW) {
    __m256 x0 = _mm256_loadu_ps(x + k);
    __m256 x1 = _mm256_loadu_ps(x + k + kVW);
    for (int j = 0; j < NR; j++) {
      const float* w = W + (size_t)(n0 + j) * K + k;
      a0[j] = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w), a0[j]);
      a1[j] = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w + kVW), a1[j]);
    }
  }
  if (k + kVW <= K) {
    __m256 x0 = _mm256_loadu_ps(x + k);
    for (int j = 0; j < NR; j++)
      a0[j] = _mm256_fmadd_ps(x0, _mm256_loadu_ps(W + (size_t)(n0 + j) * K + k), a0[j]);
    k += kVW;
  }
  if (k < K) {  // masked K-tail, same as every other kernel here
    __m256i mask = lane_mask(K - k);
    __m256 x0 = _mm256_maskload_ps(x + k, mask);
    for (int j = 0; j < NR; j++)
      a0[j] = _mm256_fmadd_ps(x0, _mm256_maskload_ps(W + (size_t)(n0 + j) * K + k, mask), a0[j]);
  }

  for (int j = 0; j < NR; j++) {
    float v = hsum256(_mm256_add_ps(a0[j], a1[j]));
    y[n0 + j] = bias ? v + bias[n0 + j] : v;
  }
}

// --- backward dX: dX(B,K) = dY(B,N) @ W(N,K) -----------------------------

// Two k-vectors per tile -- the hot path. The accumulators are two separate
// flat arrays rather than one __m256 a[MR][KV]: with the 2D form GCC stops
// promoting them to registers and spills the tile to the stack, which
// measured as a 2x regression (dX 112 -> 56 GFLOP/s on MNIST's 784->256).
// Hence the deliberate duplication between this and dx_tile1 below.
// `Wsrc`/`ldW` are the K-slice's location and row stride: either W itself at
// stride K, or the packed 16-wide scratch at stride 16 (see kPackMinK).
template <int MR>
inline void dx_tile2(const float* dY, const float* Wsrc, int ldW, float* dX, int b0, int K, int N,
                      int k0) {
  __m256 a0[MR], a1[MR];
  for (int m = 0; m < MR; m++) {
    a0[m] = _mm256_setzero_ps();
    a1[m] = _mm256_setzero_ps();
  }

  const float* w = Wsrc;
  for (int n = 0; n < N; n++) {
    __m256 w0 = _mm256_loadu_ps(w);
    __m256 w1 = _mm256_loadu_ps(w + kVW);
    w += ldW;
    for (int m = 0; m < MR; m++) {
      __m256 dyv = _mm256_set1_ps(dY[(size_t)(b0 + m) * N + n]);
      a0[m] = _mm256_fmadd_ps(dyv, w0, a0[m]);
      a1[m] = _mm256_fmadd_ps(dyv, w1, a1[m]);
    }
  }

  for (int m = 0; m < MR; m++) {
    float* d = dX + (size_t)(b0 + m) * K + k0;
    _mm256_storeu_ps(d, a0[m]);
    _mm256_storeu_ps(d + kVW, a1[m]);
  }
}

// One full k-vector: only reaches 8 <= K-remainder < 16, so not hot.
template <int MR>
inline void dx_tile1(const float* dY, const float* W, float* dX, int b0, int K, int N, int k0) {
  __m256 a0[MR];
  for (int m = 0; m < MR; m++) a0[m] = _mm256_setzero_ps();

  const float* w = W + k0;
  for (int n = 0; n < N; n++) {
    __m256 w0 = _mm256_loadu_ps(w);
    w += K;
    for (int m = 0; m < MR; m++) {
      a0[m] = _mm256_fmadd_ps(_mm256_set1_ps(dY[(size_t)(b0 + m) * N + n]), w0, a0[m]);
    }
  }

  for (int m = 0; m < MR; m++) _mm256_storeu_ps(dX + (size_t)(b0 + m) * K + k0, a0[m]);
}

// One partial k-vector, same tile width. Keeps K-remainders (e.g. K=20's
// leftover 4, or K=2 outright) on the register-blocked path.
template <int MR>
inline void dx_tile_masked(const float* dY, const float* W, float* dX, int b0, int K, int N,
                            int k0, __m256i mask) {
  __m256 a[MR];
  for (int m = 0; m < MR; m++) a[m] = _mm256_setzero_ps();

  const float* w = W + k0;
  for (int n = 0; n < N; n++) {
    __m256 wv = _mm256_maskload_ps(w, mask);
    w += K;
    for (int m = 0; m < MR; m++) {
      __m256 dyv = _mm256_set1_ps(dY[(size_t)(b0 + m) * N + n]);
      a[m] = _mm256_fmadd_ps(dyv, wv, a[m]);
    }
  }

  for (int m = 0; m < MR; m++)
    _mm256_maskstore_ps(dX + (size_t)(b0 + m) * K + k0, mask, a[m]);
}

inline void dx_k_block2(const float* dY, const float* Wsrc, int ldW, float* dX, int B, int K,
                         int N, int k0) {
  int b0 = 0;
  for (; b0 + kBwdMR <= B; b0 += kBwdMR) dx_tile2<kBwdMR>(dY, Wsrc, ldW, dX, b0, K, N, k0);
  switch (B - b0) {  // batch tail, still fully unrolled per case
    case 5: dx_tile2<5>(dY, Wsrc, ldW, dX, b0, K, N, k0); break;
    case 4: dx_tile2<4>(dY, Wsrc, ldW, dX, b0, K, N, k0); break;
    case 3: dx_tile2<3>(dY, Wsrc, ldW, dX, b0, K, N, k0); break;
    case 2: dx_tile2<2>(dY, Wsrc, ldW, dX, b0, K, N, k0); break;
    case 1: dx_tile2<1>(dY, Wsrc, ldW, dX, b0, K, N, k0); break;
    default: break;
  }
}

inline void dx_k_block1(const float* dY, const float* W, float* dX, int B, int K, int N, int k0) {
  int b0 = 0;
  for (; b0 + kBwdMR <= B; b0 += kBwdMR) dx_tile1<kBwdMR>(dY, W, dX, b0, K, N, k0);
  switch (B - b0) {
    case 5: dx_tile1<5>(dY, W, dX, b0, K, N, k0); break;
    case 4: dx_tile1<4>(dY, W, dX, b0, K, N, k0); break;
    case 3: dx_tile1<3>(dY, W, dX, b0, K, N, k0); break;
    case 2: dx_tile1<2>(dY, W, dX, b0, K, N, k0); break;
    case 1: dx_tile1<1>(dY, W, dX, b0, K, N, k0); break;
    default: break;
  }
}

inline void dx_k_block_masked(const float* dY, const float* W, float* dX, int B, int K, int N,
                               int k0, __m256i mask) {
  int b0 = 0;
  for (; b0 + kBwdMR <= B; b0 += kBwdMR) dx_tile_masked<kBwdMR>(dY, W, dX, b0, K, N, k0, mask);
  switch (B - b0) {
    case 5: dx_tile_masked<5>(dY, W, dX, b0, K, N, k0, mask); break;
    case 4: dx_tile_masked<4>(dY, W, dX, b0, K, N, k0, mask); break;
    case 3: dx_tile_masked<3>(dY, W, dX, b0, K, N, k0, mask); break;
    case 2: dx_tile_masked<2>(dY, W, dX, b0, K, N, k0, mask); break;
    case 1: dx_tile_masked<1>(dY, W, dX, b0, K, N, k0, mask); break;
    default: break;
  }
}

// --- backward dW: dW(N,K) += dY^T @ X(B,K) -------------------------------

// dYt variant: broadcasts come from the transposed dY, so the reduction over
// b reads NR sequential streams. Worth ~9% on large shapes (measured), which
// is why Linear::backward pays for the transpose.
// Two k-vectors, the hot path. Separate a0/a1 arrays for the same reason as
// dx_tile2 -- a 2D accumulator array gets spilled to the stack.
// `Xsrc`/`ldX`: the X-cache's K-slice in place at stride K, or the packed
// 16-wide scratch at stride 16 -- same choice dx_tile2 makes over W.
template <int NR>
inline void dw_tile_t2(const float* Xsrc, int ldX, const float* dYt, float* dW, int n0, int B,
                        int K, int N, int k0, bool acc) {
  __m256 a0[NR], a1[NR];
  for (int j = 0; j < NR; j++) {
    const float* d = dW + (size_t)(n0 + j) * K + k0;
    a0[j] = acc ? _mm256_loadu_ps(d) : _mm256_setzero_ps();
    a1[j] = acc ? _mm256_loadu_ps(d + kVW) : _mm256_setzero_ps();
  }

  const float* dyt[NR];
  for (int j = 0; j < NR; j++) dyt[j] = dYt + (size_t)(n0 + j) * B;

  for (int b = 0; b < B; b++) {
    const float* x = Xsrc + (size_t)b * ldX;
    __m256 x0 = _mm256_loadu_ps(x);
    __m256 x1 = _mm256_loadu_ps(x + kVW);
    for (int j = 0; j < NR; j++) {
      __m256 dyv = _mm256_set1_ps(dyt[j][b]);
      a0[j] = _mm256_fmadd_ps(dyv, x0, a0[j]);
      a1[j] = _mm256_fmadd_ps(dyv, x1, a1[j]);
    }
  }

  for (int j = 0; j < NR; j++) {
    float* d = dW + (size_t)(n0 + j) * K + k0;
    _mm256_storeu_ps(d, a0[j]);
    _mm256_storeu_ps(d + kVW, a1[j]);
  }
}

template <int NR>
inline void dw_tile_t1(const float* X, const float* dYt, float* dW, int n0, int B, int K, int N,
                        int k0, bool acc) {
  __m256 a0[NR];
  for (int j = 0; j < NR; j++)
    a0[j] = acc ? _mm256_loadu_ps(dW + (size_t)(n0 + j) * K + k0) : _mm256_setzero_ps();

  const float* dyt[NR];
  for (int j = 0; j < NR; j++) dyt[j] = dYt + (size_t)(n0 + j) * B;

  for (int b = 0; b < B; b++) {
    __m256 x0 = _mm256_loadu_ps(X + (size_t)b * K + k0);
    for (int j = 0; j < NR; j++) {
      a0[j] = _mm256_fmadd_ps(_mm256_set1_ps(dyt[j][b]), x0, a0[j]);
    }
  }

  for (int j = 0; j < NR; j++) _mm256_storeu_ps(dW + (size_t)(n0 + j) * K + k0, a0[j]);
}

template <int NR>
inline void dw_tile_t_masked(const float* X, const float* dYt, float* dW, int n0, int B, int K,
                              int N, int k0, __m256i mask, bool acc) {
  __m256 a[NR];
  for (int j = 0; j < NR; j++)
    a[j] = acc ? _mm256_maskload_ps(dW + (size_t)(n0 + j) * K + k0, mask) : _mm256_setzero_ps();

  const float* dyt[NR];
  for (int j = 0; j < NR; j++) dyt[j] = dYt + (size_t)(n0 + j) * B;

  for (int b = 0; b < B; b++) {
    __m256 xv = _mm256_maskload_ps(X + (size_t)b * K + k0, mask);
    for (int j = 0; j < NR; j++) {
      __m256 dyv = _mm256_set1_ps(dyt[j][b]);
      a[j] = _mm256_fmadd_ps(dyv, xv, a[j]);
    }
  }

  for (int j = 0; j < NR; j++)
    _mm256_maskstore_ps(dW + (size_t)(n0 + j) * K + k0, mask, a[j]);
}

inline void dw_t_k_block2(const float* Xsrc, int ldX, const float* dYt, float* dW, int B, int K,
                           int N, int k0, bool acc) {
  int n0 = 0;
  for (; n0 + kBwdMR <= N; n0 += kBwdMR)
    dw_tile_t2<kBwdMR>(Xsrc, ldX, dYt, dW, n0, B, K, N, k0, acc);
  switch (N - n0) {
    case 5: dw_tile_t2<5>(Xsrc, ldX, dYt, dW, n0, B, K, N, k0, acc); break;
    case 4: dw_tile_t2<4>(Xsrc, ldX, dYt, dW, n0, B, K, N, k0, acc); break;
    case 3: dw_tile_t2<3>(Xsrc, ldX, dYt, dW, n0, B, K, N, k0, acc); break;
    case 2: dw_tile_t2<2>(Xsrc, ldX, dYt, dW, n0, B, K, N, k0, acc); break;
    case 1: dw_tile_t2<1>(Xsrc, ldX, dYt, dW, n0, B, K, N, k0, acc); break;
    default: break;
  }
}

inline void dw_t_k_block1(const float* X, const float* dYt, float* dW, int B, int K, int N,
                           int k0, bool acc) {
  int n0 = 0;
  for (; n0 + kBwdMR <= N; n0 += kBwdMR) dw_tile_t1<kBwdMR>(X, dYt, dW, n0, B, K, N, k0, acc);
  switch (N - n0) {
    case 5: dw_tile_t1<5>(X, dYt, dW, n0, B, K, N, k0, acc); break;
    case 4: dw_tile_t1<4>(X, dYt, dW, n0, B, K, N, k0, acc); break;
    case 3: dw_tile_t1<3>(X, dYt, dW, n0, B, K, N, k0, acc); break;
    case 2: dw_tile_t1<2>(X, dYt, dW, n0, B, K, N, k0, acc); break;
    case 1: dw_tile_t1<1>(X, dYt, dW, n0, B, K, N, k0, acc); break;
    default: break;
  }
}

inline void dw_t_k_block_masked(const float* X, const float* dYt, float* dW, int B, int K, int N,
                                 int k0, __m256i mask, bool acc) {
  int n0 = 0;
  for (; n0 + kBwdMR <= N; n0 += kBwdMR)
    dw_tile_t_masked<kBwdMR>(X, dYt, dW, n0, B, K, N, k0, mask, acc);
  switch (N - n0) {
    case 5: dw_tile_t_masked<5>(X, dYt, dW, n0, B, K, N, k0, mask, acc); break;
    case 4: dw_tile_t_masked<4>(X, dYt, dW, n0, B, K, N, k0, mask, acc); break;
    case 3: dw_tile_t_masked<3>(X, dYt, dW, n0, B, K, N, k0, mask, acc); break;
    case 2: dw_tile_t_masked<2>(X, dYt, dW, n0, B, K, N, k0, mask, acc); break;
    case 1: dw_tile_t_masked<1>(X, dYt, dW, n0, B, K, N, k0, mask, acc); break;
    default: break;
  }
}

}  // namespace

void gemm_xwt_f32(const float* X, const float* W, const float* bias, float* Y,
                   int B, int K, int N) {
  for (int b0 = 0; b0 < B; b0 += kMR) {
    int mr = min_int(kMR, B - b0);
    for (int n0 = 0; n0 < N; n0 += kNR) {
      int nr = min_int(kNR, N - n0);
      microkernel_xwt(X + (size_t)b0 * K, K, W + (size_t)n0 * K, K,
                       bias ? bias + n0 : nullptr, Y + (size_t)b0 * N + n0, N,
                       mr, nr, K);
    }
  }
}

// 8x8 in-register transpose: s is 8 rows at stride ls, d receives them as 8
// columns at stride ld.
inline void transpose8x8(const float* s, size_t ls, float* d, size_t ld) {
  __m256 r0 = _mm256_loadu_ps(s), r1 = _mm256_loadu_ps(s + ls);
  __m256 r2 = _mm256_loadu_ps(s + 2 * ls), r3 = _mm256_loadu_ps(s + 3 * ls);
  __m256 r4 = _mm256_loadu_ps(s + 4 * ls), r5 = _mm256_loadu_ps(s + 5 * ls);
  __m256 r6 = _mm256_loadu_ps(s + 6 * ls), r7 = _mm256_loadu_ps(s + 7 * ls);
  __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1);
  __m256 t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3);
  __m256 t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5);
  __m256 t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
  __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
  __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
  __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
  _mm256_storeu_ps(d, _mm256_permute2f128_ps(u0, u4, 0x20));
  _mm256_storeu_ps(d + ld, _mm256_permute2f128_ps(u1, u5, 0x20));
  _mm256_storeu_ps(d + 2 * ld, _mm256_permute2f128_ps(u2, u6, 0x20));
  _mm256_storeu_ps(d + 3 * ld, _mm256_permute2f128_ps(u3, u7, 0x20));
  _mm256_storeu_ps(d + 4 * ld, _mm256_permute2f128_ps(u0, u4, 0x31));
  _mm256_storeu_ps(d + 5 * ld, _mm256_permute2f128_ps(u1, u5, 0x31));
  _mm256_storeu_ps(d + 6 * ld, _mm256_permute2f128_ps(u2, u6, 0x31));
  _mm256_storeu_ps(d + 7 * ld, _mm256_permute2f128_ps(u3, u7, 0x31));
}

void transpose_rows_f32(const float* src, float* dst, int r0, int r1, int src_cols, int ld_dst) {
  // Row blocks outermost, so the source -- W in the repack, usually the one
  // coming from DRAM -- is read as 8 sequential streams the prefetcher
  // follows. The previous column-block-outer order read it at a stride of a
  // whole row, a new 4 KB page per access at K>=1024, and was latency-bound
  // at 3.2 GB/s cold; this order measures 8.4 (and ~2x hot, from the
  // register transpose replacing 64 scalar moves per 8x8 block).
  const size_t C = (size_t)src_cols, ld = (size_t)ld_dst;
  const int C8 = src_cols - (src_cols % 8);
  int r = r0;
  for (; r + 8 <= r1; r += 8) {
    const float* s = src + (size_t)r * C;
    for (int c = 0; c < C8; c += 8) transpose8x8(s + c, C, dst + (size_t)c * ld + r, ld);
    for (int c = C8; c < src_cols; c++)
      for (int j = 0; j < 8; j++) dst[(size_t)c * ld + r + j] = s[(size_t)j * C + c];
  }
  for (; r < r1; r++)
    for (int c = 0; c < src_cols; c++) dst[(size_t)c * ld + r] = src[(size_t)r * C + c];
}

void pack_transpose_f32(const float* src, float* dst, int src_rows, int src_cols, int ld_dst) {
  transpose_rows_f32(src, dst, 0, src_rows, src_cols, ld_dst);
  if (ld_dst > src_rows) {
    for (int c = 0; c < src_cols; c++) {
      float* d = dst + (size_t)c * ld_dst;
      for (int r = src_rows; r < ld_dst; r++) d[r] = 0.0f;
    }
  }
}

void gemm_xwtp_f32(const float* X, const float* Wp, const float* bias, float* Y, int B, int K,
                    int N, int ldWp) {
  for (int n0 = 0; n0 < N; n0 += kNTile) {
    const int ncols = min_int(kNTile, N - n0);

    // bias[] is only N long while the tile is always kNTile wide, so the
    // tail block must read a padded copy rather than run off the end.
    float btile[kNTile];
    for (int j = 0; j < kNTile; j++) btile[j] = (bias && j < ncols) ? bias[n0 + j] : 0.0f;
    const __m256 bv0 = _mm256_loadu_ps(btile);
    const __m256 bv1 = _mm256_loadu_ps(btile + kVW);

    int b0 = 0;
    for (; b0 + kFwdMR <= B; b0 += kFwdMR) {
      xwtp_tile<kFwdMR>(X, Wp, bv0, bv1, Y, b0, K, N, ldWp, n0, ncols);
    }
    switch (B - b0) {  // batch tail, still fully unrolled per case
      case 5: xwtp_tile<5>(X, Wp, bv0, bv1, Y, b0, K, N, ldWp, n0, ncols); break;
      case 4: xwtp_tile<4>(X, Wp, bv0, bv1, Y, b0, K, N, ldWp, n0, ncols); break;
      case 3: xwtp_tile<3>(X, Wp, bv0, bv1, Y, b0, K, N, ldWp, n0, ncols); break;
      case 2: xwtp_tile<2>(X, Wp, bv0, bv1, Y, b0, K, N, ldWp, n0, ncols); break;
      case 1: xwtp_tile<1>(X, Wp, bv0, bv1, Y, b0, K, N, ldWp, n0, ncols); break;
      default: break;
    }
  }
}

void gemv_xwt_f32(const float* x, const float* W, const float* bias, float* y, int K, int N) {
  int n0 = 0;
  for (; n0 + kGemvNR <= N; n0 += kGemvNR) gemv_tile<kGemvNR>(x, W, bias, y, K, n0);
  switch (N - n0) {
    case 5: gemv_tile<5>(x, W, bias, y, K, n0); break;
    case 4: gemv_tile<4>(x, W, bias, y, K, n0); break;
    case 3: gemv_tile<3>(x, W, bias, y, K, n0); break;
    case 2: gemv_tile<2>(x, W, bias, y, K, n0); break;
    case 1: gemv_tile<1>(x, W, bias, y, K, n0); break;
    default: break;
  }
}

void gemm_dX_f32(const float* dY, const float* W, float* dX, int B, int K, int N, float* pack) {
  // Packing only pays if the slice is then reused by more than a tile or two,
  // hence the batch check alongside the stride one.
  const bool do_pack = pack != nullptr && K >= kPackMinK && B >= kBwdMR;
  int k0 = 0;
  for (; k0 + kKTile <= K; k0 += kKTile) {
    if (do_pack) {
      for (int n = 0; n < N; n++) {
        std::memcpy(pack + (size_t)n * kKTile, W + (size_t)n * K + k0, kKTile * sizeof(float));
      }
      dx_k_block2(dY, pack, kKTile, dX, B, K, N, k0);
    } else {
      dx_k_block2(dY, W + k0, K, dX, B, K, N, k0);
    }
  }
  for (; k0 + kVW <= K; k0 += kVW) dx_k_block1(dY, W, dX, B, K, N, k0);
  if (k0 < K) dx_k_block_masked(dY, W, dX, B, K, N, k0, lane_mask(K - k0));
}

void gemm_dW_accum_t_f32(const float* X, const float* dYt, float* dW, int B, int K, int N,
                          float* pack, bool accumulate) {
  const bool do_pack = pack != nullptr && K >= kPackMinK && N >= kBwdMR;
  int k0 = 0;
  for (; k0 + kKTile <= K; k0 += kKTile) {
    if (do_pack) {
      for (int b = 0; b < B; b++) {
        std::memcpy(pack + (size_t)b * kKTile, X + (size_t)b * K + k0, kKTile * sizeof(float));
      }
      dw_t_k_block2(pack, kKTile, dYt, dW, B, K, N, k0, accumulate);
    } else {
      dw_t_k_block2(X + k0, K, dYt, dW, B, K, N, k0, accumulate);
    }
  }
  for (; k0 + kVW <= K; k0 += kVW) dw_t_k_block1(X, dYt, dW, B, K, N, k0, accumulate);
  if (k0 < K) dw_t_k_block_masked(X, dYt, dW, B, K, N, k0, lane_mask(K - k0), accumulate);
}

// Untransposed dW: kept as the correctness oracle for gemm_dW_accum_t_f32
// (which is what the hot path uses) and as the A/B baseline showing what the
// transpose buys. Same tiling, broadcasts strided down dY's columns instead.
void gemm_dW_accum_f32(const float* X, const float* dY, float* dW, int B, int K, int N) {
  for (int n = 0; n < N; n++) {
    float* dw_row = dW + (size_t)n * K;
    int k = 0;
    for (; k + kVW <= K; k += kVW) {
      __m256 acc = _mm256_loadu_ps(dw_row + k);
      for (int b = 0; b < B; b++) {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dY[(size_t)b * N + n]),
                               _mm256_loadu_ps(X + (size_t)b * K + k), acc);
      }
      _mm256_storeu_ps(dw_row + k, acc);
    }
    for (; k < K; k++) {
      float s = dw_row[k];
      for (int b = 0; b < B; b++) s += dY[(size_t)b * N + n] * X[(size_t)b * K + k];
      dw_row[k] = s;
    }
  }
}

void bias_grad_accum_f32(const float* dY, float* db, int B, int N, bool accumulate) {
  int n = 0;
  const int N8 = N - (N % kVW);
  for (; n < N8; n += kVW) {
    __m256 acc = accumulate ? _mm256_loadu_ps(db + n) : _mm256_setzero_ps();
    for (int b = 0; b < B; b++) {
      __m256 dyv = _mm256_loadu_ps(dY + (size_t)b * N + n);
      acc = _mm256_add_ps(acc, dyv);
    }
    _mm256_storeu_ps(db + n, acc);
  }
  for (; n < N; n++) {
    float s = accumulate ? db[n] : 0.0f;
    for (int b = 0; b < B; b++) s += dY[(size_t)b * N + n];
    db[n] = s;
  }
}

}  // namespace fastmlp::simd::avx2
