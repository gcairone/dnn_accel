#pragma once
// Tiny shared timing utility, not a benchmark framework: core-pinning,
// warmup, median-of-N wall-clock timing -- the same manual methodology as
// benchmarks/hardware_analysis/src/bench_gemm.c, factored out only because
// three files now need it (avoiding 3x copy-paste, not introducing a
// dependency -- the user explicitly chose this manual style over Google
// Benchmark).
#include <sched.h>
#include <time.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace fastmlp::bench {

inline void pin_to_core0() {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  sched_setaffinity(0, sizeof(set), &set);
}

inline double now_sec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Runs `fn` `inner_iters` times per sample (back-to-back, cache-hot --
// matches real training-loop steady state, not cold-cache latency),
// `outer_reps` samples, returns the median per-call seconds.
inline double median_time_per_call(const std::function<void()>& fn, int warmup_blocks,
                                    int inner_iters, int outer_reps) {
  for (int b = 0; b < warmup_blocks; b++) {
    for (int i = 0; i < inner_iters; i++) fn();
  }
  std::vector<double> samples((size_t)outer_reps);
  for (int r = 0; r < outer_reps; r++) {
    double t0 = now_sec();
    for (int i = 0; i < inner_iters; i++) fn();
    double t1 = now_sec();
    samples[r] = (t1 - t0) / inner_iters;
  }
  std::sort(samples.begin(), samples.end());
  int n = outer_reps;
  return (n % 2) ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

}  // namespace fastmlp::bench
