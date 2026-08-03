#pragma once
// Minimal per-iteration timing recorder for example programs. Distinct from
// benchmarks/cpp/bench_common.hpp: that harness times a fn called
// back-to-back in isolation (no real training dynamics); examples run one
// real training step at a time and record its wall-clock cost as training
// actually proceeds, then report a summary at the end -- same
// warmup-then-median-of-N spirit as the rest of the project's benchmarks,
// applied to a live loop instead of a synthetic one.
#include <sched.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <vector>

namespace fastmlp::example {

// Same rationale as benchmarks/cpp/bench_common.hpp's pin_to_core0(): without
// pinning, OS scheduling noise (core migration, frequency scaling) swamps
// microsecond-scale per-iteration measurements. Duplicated here rather than
// included from benchmarks/cpp/ to keep examples/ independent of it.
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

class StepTimer {
 public:
  explicit StepTimer(int warmup_steps) : warmup_steps_(warmup_steps) {}

  void begin() { t0_ = now_sec(); }
  void end() {
    double dt = now_sec() - t0_;
    step_count_++;
    if (step_count_ > warmup_steps_) samples_.push_back(dt);
  }

  void print_summary(const char* label) const {
    std::vector<double> s = samples_;
    std::sort(s.begin(), s.end());
    size_t n = s.size();
    double median = n ? (n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2])) : 0.0;
    double mean = 0.0;
    for (double v : s) mean += v;
    mean = n ? mean / n : 0.0;
    std::printf(
        "%s %zu timed iterations (after %d warmup): median %.1f us/iter, "
        "mean %.1f us/iter, %.0f iter/s\n",
        label, n, warmup_steps_, median * 1e6, mean * 1e6, mean > 0 ? 1.0 / mean : 0.0);
  }

 private:
  int warmup_steps_;
  int step_count_ = 0;
  double t0_ = 0.0;
  std::vector<double> samples_;
};

// For benchmarks that call the SAME thing repeatedly with nothing else
// happening in between (inference throughput, not a live training loop) --
// runs `fn` `inner_iters` times back-to-back per sample, `outer_reps`
// samples, returns the median per-call seconds. Identical methodology to
// benchmarks/cpp/bench_common.hpp's function of the same name, duplicated
// here for the same reason pin_to_core0() is (keeping examples/ independent
// of benchmarks/cpp/).
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

}  // namespace fastmlp::example
