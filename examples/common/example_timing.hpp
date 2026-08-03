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

}  // namespace fastmlp::example
