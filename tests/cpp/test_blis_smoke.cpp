// M0 smoke test: de-risks the toolchain BEFORE any fastmlp code exists.
// Three independent checks:
//   1. Compile-time ISA guard -- if the aggressive -march=znver2/-mavx2/-mfma
//      flags aren't actually reaching this translation unit, the BUILD
//      fails here, which is a stronger guarantee than any runtime check.
//   2. Exact-value correctness of BLIS's cblas_sgemm against hand-written
//      arithmetic, using the project's actual weight-layout convention:
//      Y(B,N) = X(B,K) @ W(N,K)^T + bias(N), W stored row-major (out,in).
//   3. A conservative GFLOP/s floor canary -- catches "linked the wrong
//      BLAS build" (e.g. an unoptimized reference BLAS) before it poisons
//      later benchmark numbers. Real benchmarking is M9, not here.
//
// BLIS is used here purely as a correctness oracle / performance reference
// (idea.md: never a shipped runtime dependency of fastmlp_core itself).

#include "test_harness.hpp"
#include <cblas.h>
#include <sched.h>
#include <time.h>
#include <vector>

#if !defined(__AVX2__) || !defined(__FMA__)
#error "fastmlp: aggressive compile flags (-mavx2 -mfma) did not reach test_blis_smoke.cpp"
#endif

TEST(blis_exact_small_case) {
  // X(2,3), W(2,3) i.e. (out=2, in=3), bias(2) -- small integer values,
  // exactly representable in float32, so exact equality is a valid check.
  const int B = 2, K = 3, N = 2;
  float X[B * K] = {1, 2, 3,
                     4, 5, 6};
  float W[N * K] = {1, 0, 1,
                     0, 1, 1};
  float bias[N] = {10, 20};
  float Y[B * N] = {0};

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, N, K, 1.0f, X, K,
              W, K, 0.0f, Y, N);
  for (int i = 0; i < B; i++)
    for (int n = 0; n < N; n++) Y[i * N + n] += bias[n];

  float expected[B * N] = {14, 25, 20, 31};
  for (int i = 0; i < B * N; i++) {
    ASSERT_TRUE(Y[i] == expected[i]);
  }
}

TEST(blis_perf_floor_canary) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  sched_setaffinity(0, sizeof(set), &set);

  const int B = 128, K = 64, N = 32;
  std::vector<float> X((size_t)B * K), W((size_t)N * K), Y((size_t)B * N);
  for (size_t i = 0; i < X.size(); i++) X[i] = (float)((int)(i % 13) - 6) * 0.1f;
  for (size_t i = 0; i < W.size(); i++) W[i] = (float)((int)(i % 7) - 3) * 0.1f;

  const int warmup = 50, reps = 500;
  for (int r = 0; r < warmup; r++) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, N, K, 1.0f,
                X.data(), K, W.data(), K, 0.0f, Y.data(), N);
  }

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int r = 0; r < reps; r++) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, N, K, 1.0f,
                X.data(), K, W.data(), K, 0.0f, Y.data(), N);
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);

  double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
  double gflops = 2.0 * B * K * N * reps / sec / 1e9;
  // Conservative floor: hw.md measured BLIS at ~94-116 GFLOP/s on shapes
  // this size; 20 GFLOP/s only needs to catch "wrong/unoptimized BLAS
  // build linked", not validate real performance.
  ASSERT_TRUE(gflops > 20.0);
}

FASTMLP_TEST_MAIN()
