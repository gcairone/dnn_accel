// Measurement-noise probe: how stable is a fixed AVX-512 kernel on this VM?
//   1. per core 0-3: 40 wall-clock samples of a fixed L1-resident FMA tile,
//      and the same kernel's cycle count via the paired FMA reference
//   2. drift over ~20 s on one core
// Decides the benchmark protocol (pinning, sample count, statistic).
//
//   g++ -O3 -march=emeraldrapids -o probe_noise probe_noise.cpp
#include <cmath>

#include "common.hpp"

using namespace probe;

alignas(64) static float g_X[12 * 128];
alignas(64) static float g_W[128 * 32];
alignas(64) static float g_out[24 * 16];

// 12x2 zmm tile over K=128, 20 reps: ~0.25 M flop, ~4 us
__attribute__((noinline)) static void kernel() {
  __m512 acc[12][2];
  for (auto& r : acc)
    for (auto& v : r) v = _mm512_setzero_ps();
  for (int rep = 0; rep < 20; ++rep) {
    const float* w = g_W;
    for (int k = 0; k < 128; ++k, w += 32) {
      __m512 w0 = _mm512_load_ps(w), w1 = _mm512_load_ps(w + 16);
#pragma GCC unroll 12
      for (int m = 0; m < 12; ++m) {
        __m512 xb = _mm512_set1_ps(g_X[m * 128 + k]);
        acc[m][0] = _mm512_fmadd_ps(xb, w0, acc[m][0]);
        acc[m][1] = _mm512_fmadd_ps(xb, w1, acc[m][1]);
      }
    }
  }
  for (int m = 0; m < 12; ++m) {
    _mm512_storeu_ps(g_out + m * 32, acc[m][0]);
    _mm512_storeu_ps(g_out + m * 32 + 16, acc[m][1]);
  }
}
constexpr double kFlops = 2.0 * 12 * 32 * 128 * 20;

struct Stats {
  double min, p10, med, p90, max, cv;
};
static Stats stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  double mean = 0, var = 0;
  for (double x : v) mean += x;
  mean /= v.size();
  for (double x : v) var += (x - mean) * (x - mean);
  auto q = [&](double f) { return v[(size_t)(f * (v.size() - 1))]; };
  return {v.front(), q(0.1), q(0.5), q(0.9), v.back(), std::sqrt(var / v.size()) / mean};
}

int main() {
  for (auto& v : g_X) v = 1e-3f;
  for (auto& v : g_W) v = 1e-3f;
  std::printf("# probe_noise\n\n== 1. per core: 40 samples x 5 ms of a 12x2 zmm tile ==\n");
  std::printf("%4s | %-44s | %-28s\n", "core", "wall GFLOP/s  min / p10 / med / p90 / max (cv)",
              "cycles via FMA ref, med (cv)");
  for (int core = 0; core < 4; ++core) {
    pin(core);
    std::vector<double> g, c;
    for (int s = 0; s < 40; ++s) {
      auto t = time_it(kernel, 0.005, 1);
      g.push_back(kFlops / t.med / 1e9);
    }
    for (int s = 0; s < 10; ++s) c.push_back(zmm_cycles(kernel, 0.002, 5));
    Stats a = stats(g), b = stats(c);
    std::printf("%4d | %6.1f %6.1f %6.1f %6.1f %6.1f (%4.1f%%) | %8.0f (%4.1f%%) = %4.1f%% of peak\n", core, a.min,
                a.p10, a.med, a.p90, a.max, 100 * a.cv, b.med, 100 * b.cv, 100 * kFlops / 64 / b.med);
  }
  std::printf("\n== 2. drift on core 0: 1 s windows over 20 s ==\n");
  pin(0);
  for (int w = 0; w < 20; ++w) {
    std::vector<double> g;
    double t0 = now();
    while (now() - t0 < 1.0) {
      auto t = time_it(kernel, 0.01, 1);
      g.push_back(kFlops / t.med / 1e9);
    }
    Stats a = stats(g);
    std::printf("  t=%2ds  med %6.1f  min %6.1f  max %6.1f GFLOP/s  clock %5.0f MHz\n", w, a.med, a.min, a.max,
                add_chain_hz(0.02) / 1e6);
  }
  return 0;
}
