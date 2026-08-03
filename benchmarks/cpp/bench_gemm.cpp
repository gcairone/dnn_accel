// The real M9 benchmark: scalar vs. hand-written AVX2 vs. BLIS, single-core,
// on hw.md's exact (B,K,N) shape table -- so these numbers sit directly
// next to the OpenBLAS/BLIS baseline already recorded there.
#include "bench_common.hpp"
#include "fastmlp/simd/gemm_kernel.hpp"

#include <cstdio>
#include <vector>

#ifdef FASTMLP_HAVE_BLIS
#include <cblas.h>
#endif

using namespace fastmlp;
using namespace fastmlp::bench;

namespace {
struct Shape {
  int B, K, N;
  const char* label;
};
}  // namespace

int main() {
  pin_to_core0();

  Shape shapes[] = {
      {128, 20, 64, "ex1 L1 (20->64)"},
      {128, 64, 32, "ex1 L2 (64->32)"},
      {128, 32, 1, "ex1 L3 (32->1)"},
      {64, 512, 128, "ex2 enc_img L1 (512->128)"},
      {64, 128, 64, "ex2 enc_img L2 (128->64)"},
      {256, 16, 32, "ex3 enc_a L1 (16->32)"},
      {256, 32, 16, "ex3 enc_a L2 (32->16)"},
      {1, 64, 128, "ex5 inference B=1 (64->128)"},
  };

  printf("=== fastmlp GEMM forward: scalar vs avx2%s (GFLOP/s) ===\n",
#ifdef FASTMLP_HAVE_BLIS
         " vs blis"
#else
         ""
#endif
  );
  printf("%-28s %-10s %10s %10s", "shape", "BxKxN", "scalar", "avx2");
#ifdef FASTMLP_HAVE_BLIS
  printf(" %10s", "blis");
#endif
  printf("\n");

  for (auto& s : shapes) {
    std::vector<float> X((size_t)s.B * s.K), W((size_t)s.N * s.K), bias((size_t)s.N),
        Y((size_t)s.B * s.N);
    for (size_t i = 0; i < X.size(); i++) X[i] = 0.1f * (float)((int)(i % 13) - 6);
    for (size_t i = 0; i < W.size(); i++) W[i] = 0.1f * (float)((int)(i % 7) - 3);
    for (size_t i = 0; i < bias.size(); i++) bias[i] = 0.1f * (float)((int)(i % 5) - 2);

    double flops = 2.0 * s.B * s.K * s.N;

    double t_scalar = median_time_per_call(
        [&] { simd::scalar::gemm_xwt_f32(X.data(), W.data(), bias.data(), Y.data(), s.B, s.K, s.N); },
        3, 200, 15);
    double t_avx2 = median_time_per_call(
        [&] { simd::avx2::gemm_xwt_f32(X.data(), W.data(), bias.data(), Y.data(), s.B, s.K, s.N); },
        3, 200, 15);

    char shape_str[32];
    std::snprintf(shape_str, sizeof(shape_str), "%dx%dx%d", s.B, s.K, s.N);
    printf("%-28s %-10s %10.3f %10.3f", s.label, shape_str, flops / t_scalar / 1e9,
           flops / t_avx2 / 1e9);

#ifdef FASTMLP_HAVE_BLIS
    double t_blis = median_time_per_call(
        [&] {
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, s.B, s.N, s.K, 1.0f, X.data(),
                      s.K, W.data(), s.K, 0.0f, Y.data(), s.N);
        },
        3, 200, 15);
    printf(" %10.3f", flops / t_blis / 1e9);
#endif
    printf("\n");
  }
  return 0;
}
