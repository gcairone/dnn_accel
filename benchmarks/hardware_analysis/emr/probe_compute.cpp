// Compute-side probe for the Emerald Rapids target:
//   1. core clock idle and under ymm / zmm FMA load (is there a license drop?)
//   2. FMA latency and throughput vs number of independent chains, ymm and zmm
//      (-> number of FMA units, chains a microkernel needs). The ymm column
//      uses the clock read right after (valid outside the AVX-512 license)
//      and plateaus at exactly 2.00 FMA/cycle; the zmm column is relative to
//      the 12-chain zmm reference, taken as 2/cycle, and the 1-chain latency
//      it yields (~4 cycles, the documented figure) is the consistency check
//      on that assumption -- with one zmm FMA unit it would read ~2.
//   3. register-tile microkernels MR x NR (zmm), the shape of the forward GEMM
//      inner loop, on L1-resident operands (-> tile shape)
//   4. latency / throughput of the other instructions the kernels and
//      activations would use
//
// All cycle figures come from cycles_it() (common.hpp): each sample is
// converted with the clock measured right after it, because this VM's clock
// drifts by +-10%.
//
//   g++ -O3 -march=emeraldrapids -o probe_compute probe_compute.cpp
#include <utility>

#include "common.hpp"

using namespace probe;

// --- 1. clock under load -----------------------------------------------------
// Loop body: CH independent FMAs plus a chain of 16 dependent reg-reg adds.
// The add chain is the critical path (16 cycles vs CH/2 for the FMAs), so
// adds/s is the core clock *while* the FMA units are busy at CH/32 of peak.
template <int CH, class V, class FMA>
static double clock_under(FMA fma, V a, V b, double secs) {
  V acc[CH];
  for (int i = 0; i < CH; ++i) acc[i] = a;
  uint64_t x = 0, one = 1, iters = 0;
  asm volatile("" : "+r"(one));
  double t0 = now(), t;
  do {
    for (int r = 0; r < 20000; ++r) {
#pragma GCC unroll 32
      for (int i = 0; i < CH; ++i) acc[i] = fma(acc[i], a, b);
      asm volatile(
          "add %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\t"
          "add %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\t"
          "add %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\t"
          "add %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\tadd %1, %0"
          : "+r"(x)
          : "r"(one));
    }
    iters += 20000;
    t = now();
  } while (t - t0 < secs);
  for (int i = 0; i < CH; ++i) do_not_optimize(acc[i]);
  return iters * 16.0 / (t - t0);
}

static void section_clock() {
  std::printf("== 1. core clock, MHz (5 repetitions each: min / median / max) ==\n");
  auto stats = [](const char* name, auto f) {
    std::vector<double> v;
    for (int i = 0; i < 5; ++i) v.push_back(f() / 1e6);
    std::sort(v.begin(), v.end());
    std::printf("%-40s %6.0f %6.0f %6.0f\n", name, v[0], v[2], v[4]);
  };
  stats("idle, add r,r chain", [] { return add_chain_hz(0.3); });
  stats("idle, imul chain (cross-check)", [] { return imul_chain_hz(0.3); });
  stats("under ymm FMA (1.5 FMA/cyc)", [] {
    return clock_under<24, __m256>([](__m256 c, __m256 a, __m256 b) { return _mm256_fmadd_ps(c, a, b); },
                                   _mm256_set1_ps(0.999999f), _mm256_set1_ps(1e-7f), 0.3);
  });
  stats("under zmm FMA (1.5 FMA/cyc)", [] {
    return clock_under<24, __m512>([](__m512 c, __m512 a, __m512 b) { return _mm512_fmadd_ps(c, a, b); },
                                   _mm512_set1_ps(0.999999f), _mm512_set1_ps(1e-7f), 0.3);
  });
  stats("under zmm FMA, 2 s sustained", [] {
    return clock_under<24, __m512>([](__m512 c, __m512 a, __m512 b) { return _mm512_fmadd_ps(c, a, b); },
                                   _mm512_set1_ps(0.999999f), _mm512_set1_ps(1e-7f), 2.0);
  });
  std::printf("\n");
}

// --- 2. FMA chains -------------------------------------------------------------
constexpr int kChainIters = 2000;
template <int CH>
__attribute__((noinline)) static void fma_zmm(void) {
  __m512 acc[CH];
  for (int i = 0; i < CH; ++i) acc[i] = _mm512_set1_ps(0.001f * i);
  const __m512 a = _mm512_set1_ps(0.999999f), b = _mm512_set1_ps(1e-7f);
  for (int r = 0; r < kChainIters; ++r) {
#pragma GCC unroll 32
    for (int i = 0; i < CH; ++i) acc[i] = _mm512_fmadd_ps(acc[i], a, b);
  }
  for (int i = 0; i < CH; ++i) do_not_optimize(acc[i]);
}
template <int CH>
__attribute__((noinline)) static void fma_ymm(void) {
  __m256 acc[CH];
  for (int i = 0; i < CH; ++i) acc[i] = _mm256_set1_ps(0.001f * i);
  const __m256 a = _mm256_set1_ps(0.999999f), b = _mm256_set1_ps(1e-7f);
  for (int r = 0; r < kChainIters; ++r) {
#pragma GCC unroll 32
    for (int i = 0; i < CH; ++i) acc[i] = _mm256_fmadd_ps(acc[i], a, b);
  }
  for (int i = 0; i < CH; ++i) do_not_optimize(acc[i]);
}

template <int... CH>
static void fma_table(std::integer_sequence<int, CH...>) {
  std::printf("%6s %12s %12s %12s %12s\n", "chains", "ymm FMA/cyc", "ymm flop/cyc", "zmm FMA/cyc", "zmm flop/cyc");
  // ymm code runs outside the AVX-512 license, so its clock-after reading is
  // valid (cycles_it); zmm is measured against the zmm FMA reference.
  auto row = [](int ch, Cycles y, double zc) {
    double fy = ch * (double)kChainIters / y.med, fz = ch * (double)kChainIters / zc;
    std::printf("%6d %12.2f %12.1f %12.2f %12.1f\n", ch, fy, fy * 16, fz, fz * 32);
  };
  (row(CH, cycles_it(fma_ymm<CH>), zmm_cycles(fma_zmm<CH>)), ...);
}

static void section_fma() {
  std::printf("== 2. FMA throughput vs independent chains (core cycles, median) ==\n");
  std::printf("1 chain = 1/latency; plateau = number of FMA units\n");
  fma_table(std::integer_sequence<int, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 30>{});
  std::printf("\n");
}

// --- 3. register tiles -------------------------------------------------------
// acc[MR][NR] += X[m][k] (broadcast) * W[k][16*NR] (NR vector loads), k over
// an L1-resident block: exactly the forward GEMM's inner loop on packed W.
constexpr int kTileK = 128;
alignas(64) static float g_X[32 * kTileK];
alignas(64) static float g_W[kTileK * 16 * 8];
alignas(64) static float g_out[32 * 8 * 16];
constexpr int kTileReps = 20;

template <int MR, int NR>
__attribute__((noinline)) static void tile_zmm() {
  __m512 acc[MR][NR];
  for (int m = 0; m < MR; ++m)
    for (int j = 0; j < NR; ++j) acc[m][j] = _mm512_setzero_ps();
  for (int r = 0; r < kTileReps; ++r) {
    const float* w = g_W;
    for (int k = 0; k < kTileK; ++k, w += 16 * NR) {
      __m512 wv[NR];
#pragma GCC unroll 8
      for (int j = 0; j < NR; ++j) wv[j] = _mm512_load_ps(w + 16 * j);
#pragma GCC unroll 32
      for (int m = 0; m < MR; ++m) {
        __m512 xb = _mm512_set1_ps(g_X[m * kTileK + k]);
#pragma GCC unroll 8
        for (int j = 0; j < NR; ++j) acc[m][j] = _mm512_fmadd_ps(xb, wv[j], acc[m][j]);
      }
    }
  }
  for (int m = 0; m < MR; ++m)
    for (int j = 0; j < NR; ++j) _mm512_storeu_ps(g_out + (m * NR + j) * 16, acc[m][j]);
}

// The Zen 2 kernel's shape (6 x 2 ymm) for reference.
template <int MR, int NR>
__attribute__((noinline)) static void tile_ymm() {
  __m256 acc[MR][NR];
  for (int m = 0; m < MR; ++m)
    for (int j = 0; j < NR; ++j) acc[m][j] = _mm256_setzero_ps();
  for (int r = 0; r < kTileReps; ++r) {
    const float* w = g_W;
    for (int k = 0; k < kTileK; ++k, w += 8 * NR) {
      __m256 wv[NR];
#pragma GCC unroll 8
      for (int j = 0; j < NR; ++j) wv[j] = _mm256_load_ps(w + 8 * j);
#pragma GCC unroll 32
      for (int m = 0; m < MR; ++m) {
        __m256 xb = _mm256_set1_ps(g_X[m * kTileK + k]);
#pragma GCC unroll 8
        for (int j = 0; j < NR; ++j) acc[m][j] = _mm256_fmadd_ps(xb, wv[j], acc[m][j]);
      }
    }
  }
  for (int m = 0; m < MR; ++m)
    for (int j = 0; j < NR; ++j) _mm256_storeu_ps(g_out + (m * NR + j) * 8, acc[m][j]);
}

template <int MR, int NR, bool Z>
static void tile_row() {
  const double flops = 2.0 * MR * NR * (Z ? 16 : 8) * kTileK * kTileReps;
  double cyc;
  if (Z) {
    cyc = zmm_cycles(tile_zmm<MR, NR>);
  } else {
    cyc = cycles_it(tile_ymm<MR, NR>).med;
  }
  // absolute rate too, from a plain wall-clock timing (noisy, indicative)
  auto t = time_it(Z ? tile_zmm<MR, NR> : tile_ymm<MR, NR>, 0.05, 7);
  const double fpc = flops / cyc;
  std::printf("%3s %2dx%d  acc=%2d  %5.1f flop/cyc  %5.1f%% of 64  (%6.1f GFLOP/s wall)\n", Z ? "zmm" : "ymm",
              MR, NR, MR * NR, fpc, 100 * fpc / 64, flops / t.med / 1e9);
}

static void section_tiles() {
  std::printf("== 3. register tiles on L1-resident operands (K=%d), median ==\n", kTileK);
  for (auto& v : g_X) v = 1e-3f;
  for (auto& v : g_W) v = 1e-3f;
  tile_row<6, 2, false>();
  tile_row<8, 1, true>();
  tile_row<12, 1, true>();
  tile_row<16, 1, true>();
  tile_row<24, 1, true>();
  tile_row<28, 1, true>();
  tile_row<4, 2, true>();
  tile_row<6, 2, true>();
  tile_row<8, 2, true>();
  tile_row<10, 2, true>();
  tile_row<12, 2, true>();
  tile_row<14, 2, true>();
  tile_row<4, 3, true>();
  tile_row<6, 3, true>();
  tile_row<8, 3, true>();
  tile_row<9, 3, true>();
  tile_row<4, 4, true>();
  tile_row<5, 4, true>();
  tile_row<6, 4, true>();
  tile_row<7, 4, true>();
  tile_row<3, 8, true>();
  std::printf("\n");
}

// --- 4. instruction latency / throughput -------------------------------------
// latency: v = op(v) in one chain; throughput: 12 independent chains.
constexpr int kOpIters = 4000;
template <class OP>
static void lat_tp(const char* name, OP op, __m512 init) {
  // Every chain's start value goes through an empty asm, so the compiler can
  // neither constant-fold the chain nor CSE identical chains into one (it did
  // both for vdivps and vshufps before this).
  double lat = zmm_cycles([&] {
    __m512 v = init;
    asm volatile("" : "+v"(v));
    for (int i = 0; i < kOpIters; ++i) v = op(v);
    do_not_optimize(v);
  }, 0.003, 9);
  double tp = zmm_cycles([&] {
    __m512 v[12];
    for (int j = 0; j < 12; ++j) {
      v[j] = init;
      asm volatile("" : "+v"(v[j]));
    }
    for (int i = 0; i < kOpIters; ++i) {
#pragma GCC unroll 12
      for (int j = 0; j < 12; ++j) v[j] = op(v[j]);
    }
    for (int j = 0; j < 12; ++j) do_not_optimize(v[j]);
  }, 0.003, 9);
  double lat_c = lat / kOpIters;
  double tp_c = tp / (kOpIters * 12.0);
  std::printf("%-28s lat %5.2f cyc   recip.tp %5.2f cyc  (%4.2f/cyc)\n", name, lat_c, tp_c, 1.0 / tp_c);
}

static void section_instr() {
  std::printf("== 4. zmm instruction latency / reciprocal throughput (core cycles, median) ==\n");
  const __m512 one = _mm512_set1_ps(1.0f), half = _mm512_set1_ps(0.5f), c = _mm512_set1_ps(1.0000001f);
  const __m512 zero = _mm512_setzero_ps();
  const __m512i idx = _mm512_set_epi32(0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31);
  const __m512i pidx = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  lat_tp("vfmadd231ps", [&](__m512 v) { return _mm512_fmadd_ps(v, c, zero); }, one);
  lat_tp("vaddps", [&](__m512 v) { return _mm512_add_ps(v, zero); }, one);
  lat_tp("vmulps", [&](__m512 v) { return _mm512_mul_ps(v, c); }, one);
  lat_tp("vmaxps", [&](__m512 v) { return _mm512_max_ps(v, half); }, one);
  lat_tp("vdivps", [&](__m512 v) { return _mm512_div_ps(c, v); }, one);
  lat_tp("vsqrtps", [&](__m512 v) { return _mm512_sqrt_ps(v); }, c);
  lat_tp("vrcp14ps", [&](__m512 v) { return _mm512_rcp14_ps(v); }, c);
  lat_tp("vrsqrt14ps", [&](__m512 v) { return _mm512_rsqrt14_ps(v); }, c);
  lat_tp("vscalefps", [&](__m512 v) { return _mm512_scalef_ps(v, zero); }, c);
  lat_tp("vgetexpps", [&](__m512 v) { return _mm512_getexp_ps(v); }, _mm512_set1_ps(1e30f));
  lat_tp("vgetmantps", [&](__m512 v) {
    return _mm512_getmant_ps(v, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
  }, c);
  lat_tp("vrndscaleps", [&](__m512 v) {
    return _mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }, c);
  lat_tp("vpermps", [&](__m512 v) { return _mm512_permutexvar_ps(pidx, v); }, c);
  lat_tp("vpermt2ps", [&](__m512 v) { return _mm512_permutex2var_ps(v, idx, v); }, c);
  lat_tp("vshufps", [&](__m512 v) { return _mm512_shuffle_ps(v, c, 0x4e); }, c);
  lat_tp("vunpcklps", [&](__m512 v) { return _mm512_unpacklo_ps(v, c); }, c);
  lat_tp("vshuff32x4", [&](__m512 v) { return _mm512_shuffle_f32x4(v, v, 0x4e); }, c);
  lat_tp("vcvtps2dq+vcvtdq2ps", [&](__m512 v) { return _mm512_cvtepi32_ps(_mm512_cvtps_epi32(v)); }, c);
  lat_tp("vpaddd", [&](__m512 v) {
    return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(v), _mm512_castps_si512(c)));
  }, c);
  lat_tp("vcmpps->k + vblendmps", [&](__m512 v) {
    __mmask16 k = _mm512_cmp_ps_mask(v, half, _CMP_GT_OQ);
    return _mm512_mask_blend_ps(k, half, v);
  }, c);
  lat_tp("vfixupimmps", [&](__m512 v) { return _mm512_fixupimm_ps(v, v, _mm512_setzero_si512(), 0); }, c);
  std::printf("\n");
}

// --- 5. memory operands ----------------------------------------------------------
// The same 12 independent chains, the second operand either a register or an
// L1-resident memory operand (12 distinct lines). A load folded into the op
// costs throughput on this core even though the load ports alone do 2/cycle:
// this decides how many loads per FMA a microkernel can afford.
alignas(4096) static float g_fusebuf[4096];
#define CH12(OP)                                                                                  \
  asm volatile(OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7) OP(8) OP(9) OP(10) OP(11)          \
               :                                                                                  \
               : "r"(p)                                                                           \
               : "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", \
                 "xmm10", "xmm11", "xmm12")
#define ORR(i) "vorps %%zmm12, %%zmm" #i ", %%zmm" #i "\n\t"
#define ORM(i) "vorps " #i "*64(%0), %%zmm" #i ", %%zmm" #i "\n\t"
#define ADDR(i) "vaddps %%zmm12, %%zmm" #i ", %%zmm" #i "\n\t"
#define ADDM(i) "vaddps " #i "*64(%0), %%zmm" #i ", %%zmm" #i "\n\t"
#define FMAR(i) "vfmadd231ps %%zmm12, %%zmm12, %%zmm" #i "\n\t"
#define FMAM(i) "vfmadd231ps " #i "*64(%0), %%zmm12, %%zmm" #i "\n\t"
#define FMAB(i) "vfmadd231ps " #i "*4(%0)%{1to16%}, %%zmm12, %%zmm" #i "\n\t"
#define FUSE_FN(name, OP)                                                 \
  __attribute__((noinline)) static void name() {                          \
    const float* p = g_fusebuf;                                           \
    asm volatile("vxorps %%xmm12,%%xmm12,%%xmm12" ::: "xmm12");           \
    for (int r = 0; r < 4000; ++r) CH12(OP);                              \
  }
FUSE_FN(fu_orr, ORR)
FUSE_FN(fu_orm, ORM)
FUSE_FN(fu_addr, ADDR)
FUSE_FN(fu_addm, ADDM)
FUSE_FN(fu_fmar, FMAR)
FUSE_FN(fu_fmam, FMAM)
FUSE_FN(fu_fmab, FMAB)

static void section_fuse() {
  std::printf("== 5. op throughput, register vs folded L1 memory operand (/cyc) ==\n");
  std::printf("first column via the zmm FMA reference (exact for FMA/add), second via clock-after\n");
  const double n = 12.0 * 4000;
  auto row = [&](const char* s, auto f) {
    std::printf("%-28s %5.2f  %5.2f\n", s, n / zmm_cycles(f), n / cycles_it(f).med);
  };
  row("vorps reg", fu_orr);
  row("vorps m512", fu_orm);
  row("vaddps reg", fu_addr);
  row("vaddps m512", fu_addm);
  row("vfmadd231ps reg", fu_fmar);
  row("vfmadd231ps m512", fu_fmam);
  row("vfmadd231ps m32{1to16}", fu_fmab);
  std::printf("\n");
}

int main(int argc, char** argv) {
  pin(core_from_env());
  std::printf("# probe_compute, core %d\n\n", core_from_env());
  const char* only = argc > 1 ? argv[1] : "";
  auto want = [&](const char* s) { return !*only || std::strstr(only, s); };
  if (want("clock")) section_clock();
  if (want("fma")) section_fma();
  if (want("tile")) section_tiles();
  if (want("instr")) section_instr();
  if (want("fuse")) section_fuse();
  return 0;
}
