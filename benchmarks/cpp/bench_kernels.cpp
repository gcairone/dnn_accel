// Per-kernel benchmark for the Emerald Rapids target: avx2 vs avx512 vs MKL
// (the vendor sgemm for this CPU, single thread) on the GEMM shapes the
// benchmark suite's architectures actually run, at the training batch sizes.
//
// Timing: this VM's clock drifts by +-10% (3.0-4.1 GHz phases, measured in
// benchmarks/hardware_analysis/results/emr/noise.txt) and it has no PMU, so
// wall-clock GFLOP/s scatter by up to 20% run to run. Each kernel is instead
// timed in alternation with a fixed zmm FMA loop of known cycle count
// (2 FMA/cycle), and reported as time in units of that loop's cycles. For an
// AVX-512 kernel that is its true core-cycle count, so "% of peak" is
// flops / (64 * cycles); for ymm or scalar code it is a time-equivalent (they
// may run at a ~10% higher clock outside the AVX-512 license). Run-to-run
// spread of this measure: ~1-2%.
//
//   bench_kernels [filter]   e.g. "fwd", "dX", "dW", "gemv", "transpose"
//   BENCH_UNALIGNED=1        offset every buffer by 16 B (glibc malloc's
//                            alignment) instead of 64 B-aligning them
#include <immintrin.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "fastmlp/simd/gemm_kernel.hpp"

#ifdef FASTMLP_HAVE_MKL
#include <mkl.h>
#endif

using namespace fastmlp;

namespace {

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// --- the paired reference ----------------------------------------------------------
constexpr int kRefIters = 2000;
constexpr double kRefCycles = 12.0 * kRefIters / 2.0;
__attribute__((noinline)) void ref_fma_zmm() {
  __m512 acc[12];
  for (int i = 0; i < 12; ++i) acc[i] = _mm512_set1_ps(0.001f * i);
  const __m512 a = _mm512_set1_ps(0.999999f), b = _mm512_set1_ps(1e-7f);
  for (int r = 0; r < kRefIters; ++r) {
#pragma GCC unroll 12
    for (int i = 0; i < 12; ++i) acc[i] = _mm512_fmadd_ps(acc[i], a, b);
  }
  for (int i = 0; i < 12; ++i) asm volatile("" : : "v"(acc[i]));
}

template <class F>
double ref_cycles(F&& f, double batch_secs = 0.004, int samples = 15) {
  auto calib = [&](auto& g) {
    g();
    double t0 = now();
    g();
    return std::max(1L, (long)(batch_secs / std::max(now() - t0, 1e-9)));
  };
  long nf = calib(f), nr = calib(ref_fma_zmm);
  std::vector<double> r;
  for (int s = 0; s < samples; ++s) {
    double a = now();
    for (long i = 0; i < nf; ++i) f();
    double tf = (now() - a) / nf;
    a = now();
    for (long i = 0; i < nr; ++i) ref_fma_zmm();
    double tr = (now() - a) / nr;
    r.push_back(tf / tr);
  }
  std::sort(r.begin(), r.end());
  return r[r.size() / 2] * kRefCycles;
}

// --- buffers --------------------------------------------------------------------------
bool g_unaligned = false;

struct Buf {
  float* p = nullptr;
  void* base = nullptr;
  explicit Buf(size_t n) {
    base = std::aligned_alloc(64, ((n * 4 + 64) + 63) / 64 * 64);
    p = (float*)((char*)base + (g_unaligned ? 16 : 0));
  }
  ~Buf() { std::free(base); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

void fill(float* p, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  for (size_t i = 0; i < n; i++) p[i] = d(rng);
}

struct Shape {
  int K, N;
  const char* model;
};

// Every distinct Linear shape (in -> out) of benchmarks/suite/cases.py.
const Shape kShapes[] = {
    {16, 32, "tabular_s"},     {32, 16, "tabular_s"},   {16, 1, "tabular_s"},     {64, 256, "tabular_m"},
    {256, 256, "tabular_m"},   {256, 128, "tabular_m"}, {128, 1, "tabular_m"},    {48, 64, "rl_policy"},
    {64, 64, "rl_policy"},     {64, 8, "rl_policy"},    {376, 256, "sac"},        {256, 17, "sac_actor"},
    {17, 64, "sac_critic"},    {320, 256, "sac_critic"}, {256, 1, "sac_critic"},  {784, 256, "mnist_mlp"},
    {128, 10, "mnist_mlp"},    {784, 1024, "mnist_wide"}, {1024, 1024, "mnist_wide"}, {1024, 10, "mnist_wide"},
    {784, 512, "autoencoder"}, {512, 128, "autoencoder"}, {128, 32, "autoencoder"}, {32, 128, "autoencoder"},
    {128, 512, "autoencoder"}, {512, 784, "autoencoder"}, {512, 2048, "ffn_block"}, {2048, 512, "ffn_block"},
    {63, 256, "nerf"},         {256, 4, "nerf"},        {3, 64, "pinn"},          {64, 1, "pinn"},
};

const char* g_filter = "";
bool want(const char* op) { return !*g_filter || std::strstr(op, g_filter) || std::strstr(g_filter, op); }

struct Row {
  double c_avx2, c_512, c_mkl;
};

void print_header(const char* op) {
  std::printf("\n=== %s  (%% of 64 flop/cycle; ratio = avx2 time / avx512 time, mkl time / avx512 time) ===\n", op);
  std::printf("%-12s %5s %5s %5s | %7s %7s %7s | %6s %6s\n", "model", "B", "K", "N", "avx2", "avx512", "mkl",
              "x avx2", "x mkl");
}
void print_row(const Shape& s, int B, double flops, const Row& r) {
  auto pct = [&](double c) { return c > 0 ? 100.0 * flops / (64.0 * c) : 0.0; };
  std::printf("%-12s %5d %5d %5d | %6.1f%% %6.1f%% ", s.model, B, s.K, s.N, pct(r.c_avx2), pct(r.c_512));
  if (r.c_mkl > 0)
    std::printf("%6.1f%% | %6.2f %6.2f\n", pct(r.c_mkl), r.c_avx2 / r.c_512, r.c_mkl / r.c_512);
  else
    std::printf("%7s | %6.2f %6s\n", "-", r.c_avx2 / r.c_512, "-");
}

void bench_fwd(int B) {
  print_header(("fwd  Y = X W^T + b, B=" + std::to_string(B)).c_str());
  for (const auto& s : kShapes) {
    const int K = s.K, N = s.N, ld = simd::packed_ldw(N);
    Buf X((size_t)B * K), W((size_t)N * K), Wp((size_t)K * ld), bias(N), Y((size_t)B * N);
    fill(X.p, (size_t)B * K, 1);
    fill(W.p, (size_t)N * K, 2);
    fill(bias.p, N, 3);
    simd::scalar::pack_transpose_f32(W.p, Wp.p, N, K, ld);
    Row r{};
    r.c_avx2 = ref_cycles([&] { simd::avx2::gemm_xwtp_f32(X.p, Wp.p, bias.p, Y.p, B, K, N, ld); });
    r.c_512 = ref_cycles([&] { simd::avx512::gemm_xwtp_f32(X.p, Wp.p, bias.p, Y.p, B, K, N, ld); });
#ifdef FASTMLP_HAVE_MKL
    r.c_mkl = ref_cycles(
        [&] { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, N, K, 1.0f, X.p, K, W.p, K, 0.0f, Y.p, N); });
#endif
    print_row(s, B, 2.0 * B * K * N, r);
  }
}

void bench_dX(int B) {
  print_header(("dX   dX = dY W, B=" + std::to_string(B)).c_str());
  for (const auto& s : kShapes) {
    const int K = s.K, N = s.N;
    Buf dY((size_t)B * N), W((size_t)N * K), dX((size_t)B * K), pack(simd::pack_scratch_floats(std::max(B, N)));
    fill(dY.p, (size_t)B * N, 4);
    fill(W.p, (size_t)N * K, 5);
    Row r{};
    r.c_avx2 = ref_cycles([&] { simd::avx2::gemm_dX_f32(dY.p, W.p, dX.p, B, K, N, pack.p); });
    r.c_512 = ref_cycles([&] { simd::avx512::gemm_dX_f32(dY.p, W.p, dX.p, B, K, N, pack.p); });
#ifdef FASTMLP_HAVE_MKL
    r.c_mkl = ref_cycles(
        [&] { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B, K, N, 1.0f, dY.p, N, W.p, K, 0.0f, dX.p, K); });
#endif
    print_row(s, B, 2.0 * B * K * N, r);
  }
}

void bench_dW(int B) {
  print_header(("dW   dW = dY^T X (overwrite, transposed dY), B=" + std::to_string(B)).c_str());
  for (const auto& s : kShapes) {
    const int K = s.K, N = s.N;
    Buf X((size_t)B * K), dY((size_t)B * N), dYt((size_t)N * B), dW((size_t)N * K),
        pack(simd::pack_scratch_floats(std::max(B, N)));
    fill(X.p, (size_t)B * K, 6);
    fill(dY.p, (size_t)B * N, 7);
    simd::scalar::pack_transpose_f32(dY.p, dYt.p, B, N, B);
    Row r{};
    r.c_avx2 = ref_cycles([&] { simd::avx2::gemm_dW_accum_t_f32(X.p, dYt.p, dW.p, B, K, N, pack.p, false); });
    r.c_512 = ref_cycles([&] { simd::avx512::gemm_dW_accum_t_f32(X.p, dYt.p, dW.p, B, K, N, pack.p, false); });
#ifdef FASTMLP_HAVE_MKL
    r.c_mkl = ref_cycles(
        [&] { cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, K, B, 1.0f, dY.p, N, X.p, K, 0.0f, dW.p, K); });
#endif
    print_row(s, B, 2.0 * B * K * N, r);
  }
}

void bench_gemv() {
  print_header("gemv  y = W x + b (batch 1, unpacked W)");
  for (const auto& s : kShapes) {
    const int K = s.K, N = s.N;
    Buf x(K), W((size_t)N * K), bias(N), y(N);
    fill(x.p, K, 8);
    fill(W.p, (size_t)N * K, 9);
    fill(bias.p, N, 10);
    Row r{};
    r.c_avx2 = ref_cycles([&] { simd::avx2::gemv_xwt_f32(x.p, W.p, bias.p, y.p, K, N); });
    r.c_512 = ref_cycles([&] { simd::avx512::gemv_xwt_f32(x.p, W.p, bias.p, y.p, K, N); });
#ifdef FASTMLP_HAVE_MKL
    r.c_mkl = ref_cycles([&] { cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0f, W.p, K, x.p, 1, 0.0f, y.p, 1); });
#endif
    print_row(s, 1, 2.0 * K * N, r);
  }
}

void bench_transpose(int B) {
  std::printf("\n=== transposes, B=%d  (cycles per element; avx2 / avx512) ===\n", B);
  std::printf("%-12s %5s %5s | %-22s %-22s\n", "model", "K", "N", "W -> Wp (N x K)", "dY -> dYt (B x N)");
  for (const auto& s : kShapes) {
    const int K = s.K, N = s.N, ld = simd::packed_ldw(N);
    Buf W((size_t)N * K), Wp((size_t)K * ld), dY((size_t)B * N), dYt((size_t)N * B);
    fill(W.p, (size_t)N * K, 11);
    fill(dY.p, (size_t)B * N, 12);
    double w2 = ref_cycles([&] { simd::avx2::pack_transpose_f32(W.p, Wp.p, N, K, ld); });
    double w5 = ref_cycles([&] { simd::avx512::pack_transpose_f32(W.p, Wp.p, N, K, ld); });
    double d2 = ref_cycles([&] { simd::avx2::pack_transpose_f32(dY.p, dYt.p, B, N, B); });
    double d5 = ref_cycles([&] { simd::avx512::pack_transpose_f32(dY.p, dYt.p, B, N, B); });
    const double nw = (double)N * K, nd = (double)B * N;
    std::printf("%-12s %5d %5d | %6.3f / %6.3f        %6.3f / %6.3f\n", s.model, K, N, w2 / nw, w5 / nw, d2 / nd,
                d5 / nd);
  }
}

}  // namespace

int main(int argc, char** argv) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  sched_setaffinity(0, sizeof(set), &set);
#ifdef FASTMLP_HAVE_MKL
  mkl_set_num_threads(1);
#endif
  g_unaligned = std::getenv("BENCH_UNALIGNED") != nullptr;
  g_filter = argc > 1 ? argv[1] : "";
  std::printf("bench_kernels: buffers %s\n", g_unaligned ? "16 B-aligned (malloc-like)" : "64 B-aligned");
  if (want("fwd")) {
    bench_fwd(64);
    bench_fwd(256);
  }
  if (want("dX")) {
    bench_dX(64);
    bench_dX(256);
  }
  if (want("dW")) {
    bench_dW(64);
    bench_dW(256);
  }
  if (want("gemv")) bench_gemv();
  if (want("transpose")) bench_transpose(256);
  return 0;
}
