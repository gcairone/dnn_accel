// Roofline ceilings for one core of the target CPU: sustained clock, FMA peak,
// and read / write / read+write bandwidth at every level of the hierarchy.
// These are the denominators every "% of peak" in the repo should use; before
// this, hw.md had only a live clock sample (4292 MHz) and no bandwidth figure.
//
//   g++ -O3 -march=znver2 -o roofline roofline.cpp && ./roofline
//
// Pins itself to core 0 (FASTMLP_SUITE_CORE overrides), single-threaded.

#include <immintrin.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using clk = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

// --- clock: a chain of dependent 1-cycle integer adds -----------------------
static double measure_clock_hz(double secs) {
    uint64_t x = 0, iters = 0;
    double t0 = now(), t;
    do {
        for (int i = 0; i < 1000000; ++i) {
            // 8 dependent adds per iteration; the asm keeps the compiler from
            // folding the chain into one multiply.
            asm volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                         "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
        }
        iters += 8000000;
        t = now();
    } while (t - t0 < secs);
    return iters / (t - t0);
}

// --- FMA peak: CH independent accumulator chains -----------------------------
template <int CH>
static double fma_gflops(double secs, double* hz_during) {
    __m256 acc[CH];
    for (int i = 0; i < CH; ++i) acc[i] = _mm256_set1_ps(0.001f * i);
    const __m256 a = _mm256_set1_ps(0.999999f), b = _mm256_set1_ps(1e-7f);
    uint64_t n = 0;
    double t0 = now(), t;
    do {
        for (int r = 0; r < 100000; ++r)
            for (int i = 0; i < CH; ++i) acc[i] = _mm256_fmadd_ps(acc[i], a, b);
        n += 100000ull * CH;
        t = now();
    } while (t - t0 < secs);
    float sink = 0;
    for (int i = 0; i < CH; ++i) sink += _mm256_cvtss_f32(acc[i]);
    if (sink == 12345.f) std::puts("");
    double gf = n * 16.0 / (t - t0) / 1e9;   // 8 lanes x 2 flops
    // clock right after, still hot: tells us whether AVX load drops it
    *hz_during = measure_clock_hz(0.2);
    return gf;
}

// --- bandwidth ---------------------------------------------------------------
// read: 8 accumulators of vaddps over loadu (2 loads/cycle is the Zen 2 limit)
static float read_kernel(const float* p, size_t n) {
    __m256 s[8];
    for (auto& v : s) v = _mm256_setzero_ps();
    for (size_t i = 0; i < n; i += 64)
        for (int j = 0; j < 8; ++j) s[j] = _mm256_add_ps(s[j], _mm256_loadu_ps(p + i + 8 * j));
    for (int j = 1; j < 8; ++j) s[0] = _mm256_add_ps(s[0], s[j]);
    return _mm256_cvtss_f32(s[0]);
}
static void write_kernel(float* p, size_t n, float v) {
    const __m256 x = _mm256_set1_ps(v);
    for (size_t i = 0; i < n; i += 8) _mm256_storeu_ps(p + i, x);
}
// a[i] += s * b[i]: one read stream + one read-modify-write stream, the
// shape of an axpy / an optimizer update
static void rmw_kernel(float* a, const float* b, size_t n, float s) {
    const __m256 x = _mm256_set1_ps(s);
    for (size_t i = 0; i < n; i += 8)
        _mm256_storeu_ps(a + i, _mm256_fmadd_ps(x, _mm256_loadu_ps(b + i), _mm256_loadu_ps(a + i)));
}

template <class F>
static double best_rate(F f, double bytes_per_call, double min_secs = 0.15) {
    // median of 9 samples, each long enough to be resolved
    f();
    double t0 = now();
    f();
    double one = std::max(now() - t0, 1e-8);
    int inner = std::max(1, int(min_secs / 9 / one));
    std::vector<double> s;
    for (int r = 0; r < 9; ++r) {
        double a = now();
        for (int i = 0; i < inner; ++i) f();
        s.push_back((now() - a) / inner);
    }
    std::sort(s.begin(), s.end());
    return bytes_per_call / s[4] / 1e9;
}

int main() {
    int core = getenv("FASTMLP_SUITE_CORE") ? atoi(getenv("FASTMLP_SUITE_CORE")) : 0;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    sched_setaffinity(0, sizeof(set), &set);

    std::printf("== clock / FMA peak (core %d) ==\n", core);
    double hz_idle = measure_clock_hz(0.5);
    std::printf("scalar clock (int add chain)      %.0f MHz\n", hz_idle / 1e6);
    double hz;
    double g8 = fma_gflops<8>(1.0, &hz);
    std::printf("FMA  8 chains  %6.1f GFLOP/s  (%.1f flop/cycle), clock after %.0f MHz\n", g8, g8 * 1e9 / hz, hz / 1e6);
    double g10 = fma_gflops<10>(1.0, &hz);
    std::printf("FMA 10 chains  %6.1f GFLOP/s  (%.1f flop/cycle), clock after %.0f MHz\n", g10, g10 * 1e9 / hz, hz / 1e6);
    double g12 = fma_gflops<12>(3.0, &hz);
    std::printf("FMA 12 chains  %6.1f GFLOP/s  (%.1f flop/cycle), clock after %.0f MHz  [3 s sustained]\n",
                g12, g12 * 1e9 / hz, hz / 1e6);

    std::printf("\n== bandwidth, GB/s (bytes moved / time; rmw counts 3 x 4 B per element) ==\n");
    std::printf("%10s %8s %8s %8s\n", "size", "read", "write", "rmw");
    const size_t sizes_kb[] = {16, 24, 128, 256, 384, 1024, 2048, 3072, 8192, 32768, 262144};
    for (size_t kb : sizes_kb) {
        size_t n = kb * 1024 / 4;
        // rmw uses two arrays of n/2 so its total footprint equals `kb`
        float* a = static_cast<float*>(std::aligned_alloc(64, n * 4));
        float* b = static_cast<float*>(std::aligned_alloc(64, n * 4));
        for (size_t i = 0; i < n; ++i) a[i] = b[i] = 1e-3f;
        volatile float sink = 0;
        double r = best_rate([&] { sink = sink + read_kernel(a, n); }, n * 4.0);
        double w = best_rate([&] { write_kernel(a, n, 1e-3f); }, n * 4.0);
        double m = best_rate([&] { rmw_kernel(a, b, n / 2, 1e-9f); }, (n / 2) * 12.0);
        std::printf("%8zu K %8.1f %8.1f %8.1f\n", kb, r, w, m);
        std::free(a);
        std::free(b);
    }
    return 0;
}
