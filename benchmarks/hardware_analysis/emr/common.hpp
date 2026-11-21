#pragma once
// Shared helpers for the Emerald Rapids probes: pinning, TSC timing, and the
// core clock measured from a dependent integer-add chain.
//
// Two clocks matter on this machine and they are not the same:
//   - the TSC ticks at a fixed rate (constant_tsc, tsc_known_freq; 2.1 GHz),
//   - the core clock is whatever the (virtualized) turbo logic gives us.
// Every "cycles" figure printed by these probes is CORE cycles, obtained as
//   seconds * core_hz, with core_hz measured by an add chain (latency 1 on
//   every Intel core since P6), not TSC ticks.
#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace probe {

using clk = std::chrono::steady_clock;
inline double now() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

inline void pin(int core) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  sched_setaffinity(0, sizeof(set), &set);
}

inline int core_from_env() {
  const char* e = std::getenv("PROBE_CORE");
  return e ? std::atoi(e) : 0;
}

// Core clock from a dependent chain of `add reg, reg` (1 cycle latency).
//
// NOT `add reg, imm`: Golden Cove-derived cores (this one is Raptor Cove)
// fold chains of immediate adds in the renamer, so a `add $1, %0` chain runs
// at ~3.7 adds/cycle and reports a ~15 GHz "clock" (measured). A
// register-register add cannot be folded. imul_chain_hz() is an independent
// cross-check (imul r64 latency 3).
// Returns adds/second == core Hz.
inline double add_chain_hz(double secs) {
  uint64_t x = 0, one = 1, iters = 0;
  asm volatile("" : "+r"(one));  // opaque to the compiler: not a constant
  double t0 = now(), t;
  do {
    for (int i = 0; i < 200000; ++i) {
      asm volatile(
          "add %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\t"
          "add %1, %0\n\tadd %1, %0\n\tadd %1, %0\n\tadd %1, %0"
          : "+r"(x)
          : "r"(one));
    }
    iters += 1600000;
    t = now();
  } while (t - t0 < secs);
  return iters / (t - t0);
}

inline double imul_chain_hz(double secs) {
  uint64_t x = 3, m = 1, iters = 0;
  asm volatile("" : "+r"(m));
  double t0 = now(), t;
  do {
    for (int i = 0; i < 100000; ++i) {
      asm volatile(
          "imul %1, %0\n\timul %1, %0\n\timul %1, %0\n\timul %1, %0\n\t"
          "imul %1, %0\n\timul %1, %0\n\timul %1, %0\n\timul %1, %0"
          : "+r"(x)
          : "r"(m));
    }
    iters += 800000;
    t = now();
  } while (t - t0 < secs);
  return iters * 3.0 / (t - t0);
}

// Times f() (which must take >= ~1 us) in `samples` batches, each about
// secs/samples long, and returns per-call seconds: the minimum and the median.
struct Timing {
  double min, med;
};
template <class F>
inline Timing time_it(F&& f, double secs = 0.2, int samples = 11) {
  f();  // warm
  double t0 = now();
  f();
  double one = std::max(now() - t0, 1e-9);
  long inner = std::max(1L, (long)(secs / samples / one));
  std::vector<double> s;
  s.reserve(samples);
  for (int r = 0; r < samples; ++r) {
    double a = now();
    for (long i = 0; i < inner; ++i) f();
    s.push_back((now() - a) / inner);
  }
  std::sort(s.begin(), s.end());
  return {s.front(), s[s.size() / 2]};
}

// Core cycles per call of f(), robust to the clock drifting under us.
//
// The VM's core clock wanders by +-10% between runs and within one
// (3.0-3.7 GHz measured), so one clock sample normalizing a whole table gives
// nonsense (tiles at 114% of peak). Instead every sample times a ~10 ms batch
// of f() and measures the clock for ~1 ms right after, while the core is still
// in the same frequency state (an AVX-512 license, if any, outlives the code
// that triggered it by far more than that), and converts that batch with its
// own clock. Returns min and median over samples.
struct Cycles {
  double min, med, hz;  // hz: median clock seen across samples
};
template <class F>
inline Cycles cycles_it(F&& f, double batch_secs = 0.01, int samples = 15) {
  f();
  double t0 = now();
  f();
  double one = std::max(now() - t0, 1e-9);
  long inner = std::max(1L, (long)(batch_secs / one));
  std::vector<double> c, h;
  c.reserve(samples);
  h.reserve(samples);
  for (int r = 0; r < samples; ++r) {
    double a = now();
    for (long i = 0; i < inner; ++i) f();
    double t = (now() - a) / inner;
    double hz = add_chain_hz(0.001);
    c.push_back(t * hz);
    h.push_back(hz);
  }
  std::sort(c.begin(), c.end());
  std::sort(h.begin(), h.end());
  return {c.front(), c[c.size() / 2], h[h.size() / 2]};
}

// Median over samples of time(f) / time(ref), the two alternated batch by
// batch, so both see the same clock and the same AVX-512 frequency state.
// This VM exposes no APERF/MPERF and no PMU, and the clock measured after a
// zmm loop already runs ~10% above the clock during it, so this ratio against
// a reference of known cycle count is the only drift-free cycle measure.
template <class F, class R>
inline double paired_ratio(F&& f, R&& ref, double batch_secs = 0.004, int samples = 15) {
  auto calib = [&](auto& g) {
    g();
    double t0 = now();
    g();
    return std::max(1L, (long)(batch_secs / std::max(now() - t0, 1e-9)));
  };
  long nf = calib(f), nr = calib(ref);
  std::vector<double> r;
  r.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    double a = now();
    for (long i = 0; i < nf; ++i) f();
    double tf = (now() - a) / nf;
    a = now();
    for (long i = 0; i < nr; ++i) ref();
    double tr = (now() - a) / nr;
    r.push_back(tf / tr);
  }
  std::sort(r.begin(), r.end());
  return r[r.size() / 2];
}

// The reference for paired_ratio on AVX-512 code: 12 independent zmm FMA
// chains x kRefIters. At 2 FMA/cycle (verified in probe_compute section 1)
// it takes exactly kRefCycles core cycles.
constexpr int kRefIters = 2000;
constexpr double kRefCycles = 12.0 * kRefIters / 2.0;
__attribute__((noinline)) inline void ref_fma_zmm() {
  __m512 acc[12];
  for (int i = 0; i < 12; ++i) acc[i] = _mm512_set1_ps(0.001f * i);
  const __m512 a = _mm512_set1_ps(0.999999f), b = _mm512_set1_ps(1e-7f);
  for (int r = 0; r < kRefIters; ++r) {
#pragma GCC unroll 12
    for (int i = 0; i < 12; ++i) acc[i] = _mm512_fmadd_ps(acc[i], a, b);
  }
  for (int i = 0; i < 12; ++i) asm volatile("" : : "v"(acc[i]));
}
// Core cycles of one f() call, via the zmm FMA reference.
template <class F>
inline double zmm_cycles(F&& f, double batch_secs = 0.004, int samples = 15) {
  return paired_ratio(f, ref_fma_zmm, batch_secs, samples) * kRefCycles;
}

// 2 MB-aligned allocation; huge=true asks for THP (madvise mode on this
// host), huge=false forbids it, so the two variants can be compared.
inline void* alloc_pages(size_t bytes, bool huge) {
  const size_t align = 2u << 20;
  bytes = (bytes + align - 1) / align * align;
  void* p = std::aligned_alloc(align, bytes);
  madvise(p, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  std::memset(p, 0, bytes);  // fault in now, not inside the timed region
  return p;
}

template <class T>
inline void do_not_optimize(T const& v) {
  asm volatile("" : : "r,m"(v) : "memory");
}

}  // namespace probe
