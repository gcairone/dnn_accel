// Hand-written AVX-512 kernels for the Emerald Rapids target (Intel Xeon,
// family 6 model 207, Raptor Cove cores). Every number quoted here is from
// benchmarks/hardware_analysis/results/emr/ unless it says otherwise.
//
// The machine, as far as these kernels care:
//   - 2 x 512-bit FMA units, latency 4: 64 flop/cycle, >= 8 independent
//     accumulator chains to saturate; 32 zmm registers.
//   - 2 loads/cycle at any width, 1 zmm store/cycle. But a load FOLDED into
//     an arithmetic op costs throughput: zmm FMA with an m512 operand runs at
//     ~1.5/cycle, with an m32{1to16} broadcast ~1.9, against 2.0 reg-reg. So
//     the register tile must keep loads per FMA low.
//   - L1D 48 KB / 12-way, L2 2 MB / 16-way, L3 slow for one core (~190 cycle
//     latency, ~7 B/cycle). L2 read 46 B/cycle sequential, but ~19 B/cycle
//     when walked at a large row stride (the prefetcher does not follow).
//   - Shuffles/permutes: one port (1/cycle).
//
// One microkernel serves all three GEMMs of a Linear layer. Each is
//     C(M x P) (+)= A(M x D) @ Bm(D x P)
// with A read by broadcast (row-major, stride lda) and Bm's rows contiguous
// along P (stride ldb):
//     forward  Y  = X   @ Wp  (+ bias)   M=B, D=K, P=N   A=X,   Bm=Wp
//     dX       dX = dY  @ W              M=B, D=N, P=K   A=dY,  Bm=W
//     dW       dW = dYt @ X  (+= dW)     M=N, D=B, P=K   A=dYt, Bm=X
// The register tile is MR rows of A x NR zmm of Bm. Per depth step it issues
// NR loads + MR broadcasts for MR*NR FMAs.
//
// Deliberately no STL or other shared inline code in this file: it is
// compiled with AVX-512 flags even in a Zen 2 build (DetectISA.cmake), and an
// AVX-512 copy of an inline function must not win an ODR merge there.
#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "fastmlp/simd/gemm_kernel.hpp"

namespace fastmlp::simd::avx512 {

namespace {

constexpr int kVW = 16;  // fp32 lanes per zmm

inline int min_i(int a, int b) { return a < b ? a : b; }
inline long min_l(long a, long b) { return a < b ? a : b; }

// First `n` lanes, n in [0, 16].
inline __mmask16 head_mask(int n) { return (__mmask16)((n >= 16) ? 0xFFFF : ((1u << n) - 1u)); }

enum Init { kZero, kAccum, kBias };

// --- the microkernel ---------------------------------------------------------
//
// MR / NR are template parameters so the accumulator array is fully
// register-allocated (a runtime bound makes GCC spill it). MASKED: the last of
// the NR vectors is partial -- its Bm loads and C loads/stores use `m`.
//
// pf > 0: software-prefetch the Bm row pf rows ahead. Wp's rows are a page
// apart or more on wide layers, so the hardware prefetchers (which do not
// cross 4 KB pages) leave every row of a panel's first pass a demand miss --
// from L3 at ~190 cycles once the weights outgrow the L2.
template <int MR, int NR, bool MASKED>
inline void tile(const float* __restrict A, long lda, const float* __restrict Bm, long ldb,
                 float* __restrict C, long ldc, int D, Init init, const float* __restrict bias,
                 __mmask16 m, int pf) {
  __m512 acc[MR][NR];
  if (init == kAccum) {
#pragma GCC unroll 16
    for (int i = 0; i < MR; i++) {
#pragma GCC unroll 4
      for (int j = 0; j < NR; j++) {
        const float* c = C + i * ldc + j * kVW;
        acc[i][j] = (MASKED && j == NR - 1) ? _mm512_maskz_loadu_ps(m, c) : _mm512_loadu_ps(c);
      }
    }
  } else if (init == kBias && bias) {
    __m512 bv[NR];
#pragma GCC unroll 4
    for (int j = 0; j < NR; j++)
      bv[j] = (MASKED && j == NR - 1) ? _mm512_maskz_loadu_ps(m, bias + j * kVW) : _mm512_loadu_ps(bias + j * kVW);
#pragma GCC unroll 16
    for (int i = 0; i < MR; i++)
#pragma GCC unroll 4
      for (int j = 0; j < NR; j++) acc[i][j] = bv[j];
  } else {
#pragma GCC unroll 16
    for (int i = 0; i < MR; i++)
#pragma GCC unroll 4
      for (int j = 0; j < NR; j++) acc[i][j] = _mm512_setzero_ps();
  }

  const float* b = Bm;
  const long pfo = (long)pf * ldb;
  for (int d = 0; d < D; d++, b += ldb) {
    if (pf) {
#pragma GCC unroll 4
      for (int j = 0; j < NR; j++) _mm_prefetch((const char*)(b + pfo + j * kVW), _MM_HINT_T0);
    }
    __m512 bv[NR];
#pragma GCC unroll 4
    for (int j = 0; j < NR; j++)
      bv[j] = (MASKED && j == NR - 1) ? _mm512_maskz_loadu_ps(m, b + j * kVW) : _mm512_loadu_ps(b + j * kVW);
#pragma GCC unroll 16
    for (int i = 0; i < MR; i++) {
      const __m512 a = _mm512_set1_ps(A[i * lda + d]);
#pragma GCC unroll 4
      for (int j = 0; j < NR; j++) acc[i][j] = _mm512_fmadd_ps(a, bv[j], acc[i][j]);
    }
  }

#pragma GCC unroll 16
  for (int i = 0; i < MR; i++) {
#pragma GCC unroll 4
    for (int j = 0; j < NR; j++) {
      float* c = C + i * ldc + j * kVW;
      if (MASKED && j == NR - 1)
        _mm512_mask_storeu_ps(c, m, acc[i][j]);
      else
        _mm512_storeu_ps(c, acc[i][j]);
    }
  }
}

// Rows of A per tile for a given NR: MR*NR = 24 accumulators (of 32 zmm),
// leaving room for the NR Bm vectors and the broadcast. Loads per FMA:
// (MR+NR)/(MR*NR) = 0.58 / 0.46 / 0.42 for NR = 2 / 3 / 4.
template <int NR>
constexpr int mr_for() {
  return NR == 1 ? 12 : NR == 2 ? 12 : NR == 3 ? 8 : 6;
}

// All M rows for one (panel, depth block): full tiles, then the remainder.
// Only the first tile of a panel pass prefetches its own slice (pf): it is the
// one that pulls the slice in, the rest find it in L1/L2.
template <int NR, bool MASKED>
void run_m(const float* A, long lda, const float* Bm, long ldb, float* C, long ldc, int M, int D, Init init,
           const float* bias, __mmask16 m, int pf) {
  constexpr int MR = mr_for<NR>();
  int i = 0;
  for (; i + MR <= M; i += MR) {
    tile<MR, NR, MASKED>(A + i * lda, lda, Bm, ldb, C + i * ldc, ldc, D, init, bias, m, pf);
    pf = 0;
  }
  const float* a = A + i * lda;
  float* c = C + i * ldc;
  switch (M - i) {
#define FASTMLP_TAIL(R) \
  case R: if constexpr (R < MR) tile<R, NR, MASKED>(a, lda, Bm, ldb, c, ldc, D, init, bias, m, pf); break;
    FASTMLP_TAIL(1) FASTMLP_TAIL(2) FASTMLP_TAIL(3) FASTMLP_TAIL(4) FASTMLP_TAIL(5) FASTMLP_TAIL(6)
    FASTMLP_TAIL(7) FASTMLP_TAIL(8) FASTMLP_TAIL(9) FASTMLP_TAIL(10) FASTMLP_TAIL(11)
#undef FASTMLP_TAIL
    default: break;
  }
}

template <int NR>
void run_m_any(bool masked, const float* A, long lda, const float* Bm, long ldb, float* C, long ldc, int M, int D,
               Init init, const float* bias, __mmask16 m, int pf) {
  if (masked)
    run_m<NR, true>(A, lda, Bm, ldb, C, ldc, M, D, init, bias, m, pf);
  else
    run_m<NR, false>(A, lda, Bm, ldb, C, ldc, M, D, init, bias, m, pf);
}

// --- the driver ------------------------------------------------------------------

// Widest panel, in zmm vectors. 2 = 32 columns: NR 2-4 measured within noise
// of each other (bench_kernels), and 2 wastes the least on the many 16-32
// wide layers.
#ifndef FASTMLP_AVX512_NRMAX
#define FASTMLP_AVX512_NRMAX 2
#endif
constexpr int kNRMax = FASTMLP_AVX512_NRMAX;

// Blocking. The loop nest is m-block > depth-block > panel > tile, so the A
// block (mc x dc) is reused by every panel of P and should stay in L2, and
// the Bm panel slice (dc x 32) is reused by every tile of the m-block.
// Values from interleaved A/B runs (each config alternated sample by sample
// with the others in one process, so all see the same host load) on the
// suite's large layers:
//   a_budget: bytes of A block kept L2-resident (the L2 is 2 MB; the rest
//             holds the Bm slices streaming through and the C block). 512 KB
//             and 1 MB measured within 1-2%; 1.5 MB loses 7% on dW 2048x512.
//   dc_max:   cap on the depth block (none: deeper = C touched fewer times;
//             a 96-384 cap measured worse on every large shape).
//   pf:       Bm prefetch distance in rows for a panel's first tile. Off
//             costs 7-33% on forwards whose Wp outgrows the L2; 8 vs 16
//             within 1-4%, 16 ahead on batch 64.
//   pf_pack:  the same for the packing copy's source rows: 3-4% on batch-64
//             dX of the 1024-wide layers, neutral elsewhere.
// Tried and measured neutral or worse, so not done: prefetching the NEXT
// panel's slice into L2 across the current pass's tiles (0-4% slower),
// prefetching A rows ahead within a tile (up to 1.7x slower on dW, whose A
// rows are short), prefetchw of a tile's C lines at its start (+-2%), capping
// the depth block so the Bm slice stays L1-resident (5-12% slower at
// 128-192), and cldemote of finished C lines (up to 1.5x slower).
}  // namespace
struct Tune {
  long a_budget = 768 * 1024;
  int dc_max = 1 << 30;
  int pf = 16;
  int pf_pack = 8;
  int fwd_pack = 1;
};
#ifdef FASTMLP_AVX512_TUNING
Tune g_tune;  // development builds only: benchmarks sweep it
#else
namespace {
constexpr Tune g_tune{};
}  // namespace
#endif
namespace {

constexpr int kFwdPackFloats = 64 * 1024;

struct Gemm {
  const float* A;
  long lda;
  const float* Bm;
  long ldb;
  float* C;
  long ldc;
  int M, D, P;
  Init init;
  const float* bias;  // kBias only; P long
  float* pack;        // nullptr = never pack
  bool pack_bm;       // copy each Bm panel slice into `pack` first
};

// Traffic that leaves the L2 for a blocking choice (mc rows of A by dc of
// depth per block): Bm is re-read once per m-block, C re-read and re-written
// once per extra depth block -- each only counted if it does not fit in L2
// anyway. The L3 behind it delivers ~7 B/cycle to one core, so this is what
// decides large layers.
double blocking_cost(const Gemm& g, int mc, int dc) {
  constexpr double kL2Resident = 768.0 * 1024;
  const double b_bytes = 4.0 * g.D * g.P, c_bytes = 4.0 * g.M * g.P;
  const int m_blocks = (g.M + mc - 1) / mc, d_blocks = (g.D + dc - 1) / dc;
  double cost = 0;
  if (b_bytes > kL2Resident) cost += b_bytes * (m_blocks - 1);
  if (c_bytes > kL2Resident) cost += c_bytes * 2.0 * (d_blocks - 1);
  return cost;
}

void run(const Gemm& g) {
  // If the whole A (M x D) fits the budget, no blocking (C is then written
  // exactly once and Bm read once). Otherwise either cut the depth (full-M
  // slices) or cut M (full-depth blocks), whichever moves less data.
  int mc = g.M, dc = g.D;
  if (4.0 * g.M * g.D > (double)g_tune.a_budget) {
    int dc_a = (int)(g_tune.a_budget / (4L * g.M));
    dc_a = dc_a < 64 ? min_i(g.D, 64) : dc_a;
    int mc_b = (int)(g_tune.a_budget / (4L * g.D));
    mc_b = mc_b < 12 ? 12 : mc_b / 12 * 12;
    if (blocking_cost(g, g.M, dc_a) <= blocking_cost(g, mc_b, g.D)) {
      dc = dc_a;
    } else {
      mc = mc_b;
    }
  }
  dc = min_i(dc, g_tune.dc_max);
  if (dc < 1) dc = 1;

  for (int m0 = 0; m0 < g.M; m0 += mc) {
    const int mrows = min_i(mc, g.M - m0);
    for (int d0 = 0; d0 < g.D; d0 += dc) {
      const int dcur = min_i(dc, g.D - d0);
      const Init init = d0 == 0 ? g.init : kAccum;
      for (int p0 = 0; p0 < g.P; p0 += kNRMax * kVW) {
        const int cols = min_i(kNRMax * kVW, g.P - p0);
        const int nr = (cols + kVW - 1) / kVW;
        const int tail = cols - (nr - 1) * kVW;
        const bool masked = tail != kVW;
        const __mmask16 m = head_mask(tail);
        const float* bias = g.bias ? g.bias + p0 : nullptr;

        const float* panel = g.Bm + (long)d0 * g.ldb + p0;
        long ldp = g.ldb;
        // The forward gets no pack scratch (Wp is already packed, at a row
        // stride of one or more pages); a thread-local one serves when the
        // contiguous copy pays: M >= 128 rows reusing it and a Wp beyond the
        // L2 -- +4-5% on the 2048-wide layers at batch 256, while on smaller
        // M or an L2-resident Wp the copy costs 3-15% (interleaved A/B).
        static thread_local float tl_pack[kFwdPackFloats];
        float* packbuf = g.pack;
        bool do_pack = g.pack_bm;
        if (!do_pack && g_tune.fwd_pack && g.M >= 128 && 4.0 * g.D * g.ldb > 2.0 * 1024 * 1024 &&
            dcur * nr * kVW <= kFwdPackFloats) {
          do_pack = true;
          packbuf = tl_pack;
        }
        if (do_pack) {
          // Contiguous copy of the slice: walks all L1 sets (a power-of-two
          // ldb maps every row onto a few of them) and turns the tiles'
          // strided walk into a sequential one. Zero-padded past a masked
          // tail.
          float* dst = packbuf;
          const int w = nr * kVW;
          const long pfo = (long)g_tune.pf_pack * g.ldb;
          for (int d = 0; d < dcur; d++) {
            const float* src = panel + (long)d * g.ldb;
            if (g_tune.pf_pack)
              for (int j = 0; j < nr; j++) _mm_prefetch((const char*)(src + pfo + j * kVW), _MM_HINT_T0);
            for (int j = 0; j < nr; j++) {
              __mmask16 mj = (j == nr - 1) ? m : (__mmask16)0xFFFF;
              _mm512_storeu_ps(dst + d * w + j * kVW, _mm512_maskz_loadu_ps(mj, src + j * kVW));
            }
          }
          panel = dst;
          ldp = w;
        }
        const float* a = g.A + (long)m0 * g.lda + d0;
        float* c = g.C + (long)m0 * g.ldc + p0;
        const int pf = do_pack ? 0 : g_tune.pf;
        switch (nr) {
          case 1: run_m_any<1>(masked, a, g.lda, panel, ldp, c, g.ldc, mrows, dcur, init, bias, m, pf); break;
          case 2: run_m_any<2>(masked, a, g.lda, panel, ldp, c, g.ldc, mrows, dcur, init, bias, m, pf); break;
          case 3: run_m_any<3>(masked, a, g.lda, panel, ldp, c, g.ldc, mrows, dcur, init, bias, m, pf); break;
          case 4: run_m_any<4>(masked, a, g.lda, panel, ldp, c, g.ldc, mrows, dcur, init, bias, m, pf); break;
          default: break;
        }
      }
    }
  }
}

// Packing a panel slice costs dc*NR*16 copies against M*dc*NR*16 FMAs that
// reuse it: worth it once M covers a couple of tiles.
inline bool pack_worth(int M) { return M >= 24; }

// --- horizontal reduction of 16 accumulators ----------------------------------
//
// out[r] = sum of the 16 lanes of v[r], r = 0..15, in 30 shuffles + 15 adds
// (16 separate _mm512_reduce_add_ps cost ~64 shuffles, and shuffles have one
// port). Stage by stage, within each 128-bit lane:
//   1. pairs (a,b):  unpacklo + unpackhi   -> [a0+a2, b0+b2, a1+a3, b1+b3]
//   2. pairs of those: shuffle_ps 0x44/0xEE -> [a, b, c, d] lane-partial sums
//   3./4. shuffle_f32x4 0x88/0xDD twice: add the four 128-bit lanes together
//      and put rows 0-3, 4-7, 8-11, 12-15 in lanes 0, 1, 2, 3.
inline __m512 reduce16(const __m512* v) {
  __m512 s[8];
  for (int i = 0; i < 8; i++)
    s[i] = _mm512_add_ps(_mm512_unpacklo_ps(v[2 * i], v[2 * i + 1]), _mm512_unpackhi_ps(v[2 * i], v[2 * i + 1]));
  __m512 t[4];
  for (int i = 0; i < 4; i++)
    t[i] = _mm512_add_ps(_mm512_shuffle_ps(s[2 * i], s[2 * i + 1], 0x44),
                         _mm512_shuffle_ps(s[2 * i], s[2 * i + 1], 0xEE));
  __m512 u01 = _mm512_add_ps(_mm512_shuffle_f32x4(t[0], t[1], 0x88), _mm512_shuffle_f32x4(t[0], t[1], 0xDD));
  __m512 u23 = _mm512_add_ps(_mm512_shuffle_f32x4(t[2], t[3], 0x88), _mm512_shuffle_f32x4(t[2], t[3], 0xDD));
  return _mm512_add_ps(_mm512_shuffle_f32x4(u01, u23, 0x88), _mm512_shuffle_f32x4(u01, u23, 0xDD));
}

// --- batch-1 forward: y(N) = W(N,K) @ x(K) + bias -------------------------
//
// R output rows at once, reduced along K over the unpacked W, so the whole
// matrix streams sequentially once; x's vector is loaded once per k-step and
// reused by all R rows. The R horizontal sums go through reduce16.
template <int R>
inline void gemv_rows(const float* __restrict x, const float* __restrict W, const float* __restrict bias,
                      float* __restrict y, int K, int n0) {
  __m512 acc[16];
  for (int r = 0; r < 16; r++) acc[r] = _mm512_setzero_ps();
  const float* w = W + (long)n0 * K;
  int k = 0;
  for (; k + kVW <= K; k += kVW) {
    const __m512 xv = _mm512_loadu_ps(x + k);
#pragma GCC unroll 16
    for (int r = 0; r < R; r++) acc[r] = _mm512_fmadd_ps(_mm512_loadu_ps(w + (long)r * K + k), xv, acc[r]);
  }
  if (k < K) {
    const __mmask16 m = head_mask(K - k);
    const __m512 xv = _mm512_maskz_loadu_ps(m, x + k);
#pragma GCC unroll 16
    for (int r = 0; r < R; r++)
      acc[r] = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w + (long)r * K + k), xv, acc[r]);
  }
  __m512 sums = reduce16(acc);
  const __mmask16 om = head_mask(R);
  if (bias) sums = _mm512_add_ps(sums, _mm512_maskz_loadu_ps(om, bias + n0));
  _mm512_mask_storeu_ps(y + n0, om, sums);
}

// --- narrow forward (N <= kNarrowN): dot products over the batch ------------
//
// A 16-column tile computes N of its 16 lanes: at N=1 that is 1/16 of the
// work done usefully. Instead, extract W's N rows from Wp (a K x N transpose,
// N*K elements -- cheap next to B*N*K), then reduce along K for 16 batch rows
// at a time and reduce16() the results, exactly like the batch-1 path.
constexpr int kNarrowN = 8;
constexpr int kNarrowMaxK = 4096;  // stack buffer for the extracted rows

template <int NN>
inline void narrow_rows(const float* __restrict X, const float* __restrict Wt, const float* __restrict bias,
                        float* __restrict Y, int b0, int rows, int K, int N) {
  // acc[n][r]: output (b0+r, n). 16 rows x NN outputs > 24 registers for
  // NN >= 2, so rows are processed in groups of RG with NN*RG <= 16.
  constexpr int RG = 16 / NN;
  for (int g0 = 0; g0 < rows; g0 += RG) {
    const int gr = min_i(RG, rows - g0);
    __m512 acc[NN][RG];
    for (int n = 0; n < NN; n++)
      for (int r = 0; r < RG; r++) acc[n][r] = _mm512_setzero_ps();
    const float* xb = X + (long)(b0 + g0) * K;
    int k = 0;
    for (; k + kVW <= K; k += kVW) {
      __m512 wv[NN];
#pragma GCC unroll 8
      for (int n = 0; n < NN; n++) wv[n] = _mm512_loadu_ps(Wt + (long)n * K + k);
#pragma GCC unroll 16
      for (int r = 0; r < RG; r++) {
        if (r < gr) {
          const __m512 xv = _mm512_loadu_ps(xb + (long)r * K + k);
#pragma GCC unroll 8
          for (int n = 0; n < NN; n++) acc[n][r] = _mm512_fmadd_ps(xv, wv[n], acc[n][r]);
        }
      }
    }
    if (k < K) {
      const __mmask16 m = head_mask(K - k);
      __m512 wv[NN];
      for (int n = 0; n < NN; n++) wv[n] = _mm512_maskz_loadu_ps(m, Wt + (long)n * K + k);
      for (int r = 0; r < gr; r++) {
        const __m512 xv = _mm512_maskz_loadu_ps(m, xb + (long)r * K + k);
        for (int n = 0; n < NN; n++) acc[n][r] = _mm512_fmadd_ps(xv, wv[n], acc[n][r]);
      }
    }
    // Reduce all NN*RG (<= 16) accumulators in one reduce16; lane n*RG + r.
    __m512 v[16];
    for (int i = 0; i < 16; i++) v[i] = _mm512_setzero_ps();
    for (int n = 0; n < NN; n++)
      for (int r = 0; r < RG; r++) v[n * RG + r] = acc[n][r];
    alignas(64) float s[16];
    _mm512_store_ps(s, reduce16(v));
    for (int r = 0; r < gr; r++) {
      float* y = Y + (long)(b0 + g0 + r) * N;
      for (int n = 0; n < NN; n++) y[n] = s[n * RG + r] + (bias ? bias[n] : 0.0f);
    }
  }
}

template <int NN>
void narrow_forward(const float* X, const float* Wp, const float* bias, float* Y, int B, int K, int N, int ldWp) {
  alignas(64) float Wt[NN * kNarrowMaxK];
  for (int k = 0; k < K; k++)
    for (int n = 0; n < NN; n++) Wt[n * K + k] = Wp[(long)k * ldWp + n];
  for (int b0 = 0; b0 < B; b0 += 16) narrow_rows<NN>(X, Wt, bias, Y, b0, min_i(16, B - b0), K, N);
}

// --- narrow dW (K <= kNarrowDwK): dot products over the batch -----------------
//
// dW(N,K) is C with P = K: at K = 2 a 16-wide panel computes 2 useful lanes.
// Transposed, each element is a dot product of two contiguous rows,
// dW[n][k] = <dYt[n][:], Xt[k][:]>, Xt being X's K columns transposed once
// (K*B elements, cheap next to the N*K*B multiply-adds) into a stack buffer.
// RN rows of dYt x KK columns per group, RN*KK <= 16, reduce16'd together.
constexpr int kNarrowDwK = 8;
constexpr int kNarrowDwMaxB = 4096;

template <int KK>
void narrow_dw(const float* X, const float* dYt, float* dW, int B, int N, bool accumulate) {
  alignas(64) float Xt[KK * kNarrowDwMaxB];
  for (int b = 0; b < B; b++)
    for (int k = 0; k < KK; k++) Xt[k * B + b] = X[(long)b * KK + k];
  constexpr int RN = 16 / KK;
  for (int n0 = 0; n0 < N; n0 += RN) {
    const int rn = min_i(RN, N - n0);
    __m512 acc[RN][KK];
    for (int r = 0; r < RN; r++)
      for (int k = 0; k < KK; k++) acc[r][k] = _mm512_setzero_ps();
    for (int b = 0; b < B; b += kVW) {
      const __mmask16 m = head_mask(B - b);
      __m512 xv[KK];
#pragma GCC unroll 8
      for (int k = 0; k < KK; k++) xv[k] = _mm512_maskz_loadu_ps(m, Xt + k * B + b);
#pragma GCC unroll 16
      for (int r = 0; r < RN; r++) {
        if (r < rn) {
          const __m512 dv = _mm512_maskz_loadu_ps(m, dYt + (long)(n0 + r) * B + b);
#pragma GCC unroll 8
          for (int k = 0; k < KK; k++) acc[r][k] = _mm512_fmadd_ps(dv, xv[k], acc[r][k]);
        }
      }
    }
    __m512 v[16];
    for (int i = 0; i < 16; i++) v[i] = _mm512_setzero_ps();
    for (int r = 0; r < RN; r++)
      for (int k = 0; k < KK; k++) v[r * KK + k] = acc[r][k];
    alignas(64) float sum[16];
    _mm512_store_ps(sum, reduce16(v));
    // rows n0..n0+rn of dW are rn*KK contiguous floats, in the same r*KK+k order
    float* d = dW + (long)n0 * KK;
    for (int i = 0; i < rn * KK; i++) d[i] = accumulate ? d[i] + sum[i] : sum[i];
  }
}

// --- 16 x 16 transpose ----------------------------------------------------------
//
// dst[c * ldd + r] = src[r * lds + c] for r < rows, c < cols (both <= 16).
// 64 shuffles per 256 elements (the shuffle port is the bound).
inline void transpose16(const float* src, long lds, float* dst, long ldd, int rows, int cols) {
  const __mmask16 cm = head_mask(cols);
  __m512 r[16];
  for (int i = 0; i < 16; i++) r[i] = i < rows ? _mm512_maskz_loadu_ps(cm, src + i * lds) : _mm512_setzero_ps();
  __m512 t[16];
  for (int i = 0; i < 8; i++) {
    t[2 * i] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
    t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
  }
  // t[0]=(r0,r1 lo) t[1]=(r0,r1 hi) t[2]=(r2,r3 lo) ...
  for (int i = 0; i < 4; i++) {
    const __m512 a = t[4 * i], b = t[4 * i + 1], c = t[4 * i + 2], d = t[4 * i + 3];
    r[4 * i + 0] = _mm512_shuffle_ps(a, c, 0x44);
    r[4 * i + 1] = _mm512_shuffle_ps(a, c, 0xEE);
    r[4 * i + 2] = _mm512_shuffle_ps(b, d, 0x44);
    r[4 * i + 3] = _mm512_shuffle_ps(b, d, 0xEE);
  }
  // Now r[4i+j] holds, in each 128-bit lane L, column (4L + j) of rows 4i..4i+3.
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      const __m512 a = r[j * 8 + i], b = r[j * 8 + 4 + i];
      t[j * 8 + i] = _mm512_shuffle_f32x4(a, b, 0x88);      // lanes L0,L2 of a | L0,L2 of b
      t[j * 8 + 4 + i] = _mm512_shuffle_f32x4(a, b, 0xDD);  // lanes L1,L3 of a | L1,L3 of b
    }
  }
  // t[i]     (i<4): col i      for rows 0-3,8-11? -> fixed up in the last stage
  const __mmask16 rm = head_mask(rows);
  for (int i = 0; i < 4; i++) {
    // t[i]   = [c_i r0-3 | c_{8+i} r0-3 | c_i r4-7 | c_{8+i} r4-7]
    // t[8+i] = [c_i r8-11| c_{8+i} r8-11| c_i r12-15| c_{8+i} r12-15]
    // t[4+i] / t[12+i]: the same for columns 4+i and 12+i.
    const __m512 c_i = _mm512_shuffle_f32x4(t[i], t[8 + i], 0x88);
    const __m512 c_8i = _mm512_shuffle_f32x4(t[i], t[8 + i], 0xDD);
    const __m512 c_4i = _mm512_shuffle_f32x4(t[4 + i], t[12 + i], 0x88);
    const __m512 c_12i = _mm512_shuffle_f32x4(t[4 + i], t[12 + i], 0xDD);
    if (i < cols) _mm512_mask_storeu_ps(dst + i * ldd, rm, c_i);
    if (4 + i < cols) _mm512_mask_storeu_ps(dst + (4 + i) * ldd, rm, c_4i);
    if (8 + i < cols) _mm512_mask_storeu_ps(dst + (8 + i) * ldd, rm, c_8i);
    if (12 + i < cols) _mm512_mask_storeu_ps(dst + (12 + i) * ldd, rm, c_12i);
  }
}

}  // namespace

// --- public entry points -------------------------------------------------------

void gemm_xwtp_f32(const float* X, const float* Wp, const float* bias, float* Y, int B, int K, int N, int ldWp) {
  if (N <= kNarrowN && K <= kNarrowMaxK) {
    switch (N) {
      case 1: narrow_forward<1>(X, Wp, bias, Y, B, K, N, ldWp); return;
      case 2: narrow_forward<2>(X, Wp, bias, Y, B, K, N, ldWp); return;
      case 3: narrow_forward<3>(X, Wp, bias, Y, B, K, N, ldWp); return;
      case 4: narrow_forward<4>(X, Wp, bias, Y, B, K, N, ldWp); return;
      case 5: narrow_forward<5>(X, Wp, bias, Y, B, K, N, ldWp); return;
      case 6: narrow_forward<6>(X, Wp, bias, Y, B, K, N, ldWp); return;
      case 7: narrow_forward<7>(X, Wp, bias, Y, B, K, N, ldWp); return;
      case 8: narrow_forward<8>(X, Wp, bias, Y, B, K, N, ldWp); return;
      default: break;
    }
  }
  run({X, K, Wp, ldWp, Y, N, B, K, N, kBias, bias, nullptr, false});
}

void gemv_xwt_f32(const float* x, const float* W, const float* bias, float* y, int K, int N) {
  int n0 = 0;
  for (; n0 + 16 <= N; n0 += 16) gemv_rows<16>(x, W, bias, y, K, n0);
  switch (N - n0) {
#define FASTMLP_TAIL(R) \
  case R: gemv_rows<R>(x, W, bias, y, K, n0); break;
    FASTMLP_TAIL(1) FASTMLP_TAIL(2) FASTMLP_TAIL(3) FASTMLP_TAIL(4) FASTMLP_TAIL(5) FASTMLP_TAIL(6)
    FASTMLP_TAIL(7) FASTMLP_TAIL(8) FASTMLP_TAIL(9) FASTMLP_TAIL(10) FASTMLP_TAIL(11) FASTMLP_TAIL(12)
    FASTMLP_TAIL(13) FASTMLP_TAIL(14) FASTMLP_TAIL(15)
#undef FASTMLP_TAIL
    default: break;
  }
}

void gemm_dX_f32(const float* dY, const float* W, float* dX, int B, int K, int N, float* pack) {
  run({dY, N, W, K, dX, K, B, N, K, kZero, nullptr, pack, pack != nullptr && pack_worth(B)});
}

void gemm_dW_accum_t_f32(const float* X, const float* dYt, float* dW, int B, int K, int N, float* pack,
                          bool accumulate) {
  if (K <= kNarrowDwK && B <= kNarrowDwMaxB) {
    switch (K) {
      case 1: narrow_dw<1>(X, dYt, dW, B, N, accumulate); return;
      case 2: narrow_dw<2>(X, dYt, dW, B, N, accumulate); return;
      case 3: narrow_dw<3>(X, dYt, dW, B, N, accumulate); return;
      case 4: narrow_dw<4>(X, dYt, dW, B, N, accumulate); return;
      case 5: narrow_dw<5>(X, dYt, dW, B, N, accumulate); return;
      case 6: narrow_dw<6>(X, dYt, dW, B, N, accumulate); return;
      case 7: narrow_dw<7>(X, dYt, dW, B, N, accumulate); return;
      case 8: narrow_dw<8>(X, dYt, dW, B, N, accumulate); return;
      default: break;
    }
  }
  run({dYt, B, X, K, dW, K, N, B, K, accumulate ? kAccum : kZero, nullptr, pack, pack != nullptr && pack_worth(N)});
}

// Correctness oracle for the transposed variant above, not a hot path:
// broadcasts walk dY's column n at stride N.
void gemm_dW_accum_f32(const float* X, const float* dY, float* dW, int B, int K, int N) {
  for (int n = 0; n < N; n++) {
    float* row = dW + (long)n * K;
    for (int k = 0; k < K; k += kVW) {
      const __mmask16 m = head_mask(K - k);
      __m512 acc = _mm512_maskz_loadu_ps(m, row + k);
      for (int b = 0; b < B; b++)
        acc = _mm512_fmadd_ps(_mm512_set1_ps(dY[(long)b * N + n]), _mm512_maskz_loadu_ps(m, X + (long)b * K + k), acc);
      _mm512_mask_storeu_ps(row + k, m, acc);
    }
  }
}

// Unpacked reference forward, kept as the oracle the packed path is tested
// against: one dot product per output.
void gemm_xwt_f32(const float* X, const float* W, const float* bias, float* Y, int B, int K, int N) {
  for (int b = 0; b < B; b++) {
    for (int n = 0; n < N; n++) {
      __m512 acc = _mm512_setzero_ps();
      for (int k = 0; k < K; k += kVW) {
        const __mmask16 m = head_mask(K - k);
        acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, X + (long)b * K + k),
                              _mm512_maskz_loadu_ps(m, W + (long)n * K + k), acc);
      }
      Y[(long)b * N + n] = _mm512_reduce_add_ps(acc) + (bias ? bias[n] : 0.0f);
    }
  }
}

void bias_grad_accum_f32(const float* dY, float* db, int B, int N, bool accumulate) {
  if (N == 1) {  // a column of B contiguous floats: a plain sum
    __m512 s = _mm512_setzero_ps();
    for (int b = 0; b < B; b += kVW) s = _mm512_add_ps(s, _mm512_maskz_loadu_ps(head_mask(B - b), dY + b));
    const float t = _mm512_reduce_add_ps(s);
    db[0] = accumulate ? db[0] + t : t;
    return;
  }
  for (int n0 = 0; n0 < N; n0 += 4 * kVW) {
    __m512 acc[4];
    __mmask16 m[4];
    const int nv = min_i(4, (N - n0 + kVW - 1) / kVW);
    for (int j = 0; j < 4; j++) {
      m[j] = j < nv ? head_mask(N - n0 - j * kVW) : (__mmask16)0;
      acc[j] = accumulate ? _mm512_maskz_loadu_ps(m[j], db + n0 + j * kVW) : _mm512_setzero_ps();
    }
    for (int b = 0; b < B; b++) {
      const float* row = dY + (long)b * N + n0;
      for (int j = 0; j < nv; j++) acc[j] = _mm512_add_ps(acc[j], _mm512_maskz_loadu_ps(m[j], row + j * kVW));
    }
    for (int j = 0; j < nv; j++) _mm512_mask_storeu_ps(db + n0 + j * kVW, m[j], acc[j]);
  }
}

void transpose_rows_f32(const float* src, float* dst, int r0, int r1, int src_cols, int ld_dst) {
  // One column (a layer with N = 1 transposing its dY): a plain copy, not a
  // 64-shuffle transpose per 16 elements.
  if (src_cols == 1) {
    for (int r = r0; r < r1; r += kVW) {
      const __mmask16 m = head_mask(r1 - r);
      _mm512_mask_storeu_ps(dst + r, m, _mm512_maskz_loadu_ps(m, src + r));
    }
    return;
  }
  // One or two source rows (W -> Wp of a layer with N <= 2, every optimizer
  // step): a 16x16 block would spend 64 shuffles and 16 masked stores on
  // 16-32 elements (~6.5 cycles each, measured); a scalar scatter is ~1.
  if (r1 - r0 <= 2) {
    for (int r = r0; r < r1; r++)
      for (int c = 0; c < src_cols; c++) dst[(long)c * ld_dst + r] = src[(long)r * src_cols + c];
    return;
  }
  // Row blocks outermost: the source (W in the repack, often cold) is read as
  // 16 sequential streams.
  for (int r = r0; r < r1; r += 16) {
    const int rows = min_i(16, r1 - r);
    const float* s = src + (long)r * src_cols;
    for (int c = 0; c < src_cols; c += 16)
      transpose16(s + c, src_cols, dst + (long)c * ld_dst + r, ld_dst, rows, min_i(16, src_cols - c));
  }
}

void pack_transpose_f32(const float* src, float* dst, int src_rows, int src_cols, int ld_dst) {
  transpose_rows_f32(src, dst, 0, src_rows, src_cols, ld_dst);
  if (ld_dst > src_rows) {
    for (int c = 0; c < src_cols; c++) {
      float* d = dst + (long)c * ld_dst;
      for (int r = src_rows; r < ld_dst; r += kVW) _mm512_mask_storeu_ps(d + r, head_mask(ld_dst - r), _mm512_setzero_ps());
    }
  }
}

}  // namespace fastmlp::simd::avx512
