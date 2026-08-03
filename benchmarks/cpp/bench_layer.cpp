// Full Linear::forward/backward (bias + activation overhead included, not
// just the raw GEMM) on hw.md's shape table, using whichever backend this
// binary was configured with (FASTMLP_SIMD_BACKEND at cmake-configure time
// -- see the printed backend name in the output).
#include "bench_common.hpp"
#include "fastmlp/layer.hpp"

#include <cstdio>
#include <vector>

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

  printf("=== fastmlp Linear::forward/backward -- active backend: %s ===\n",
#if defined(FASTMLP_BACKEND_AVX2)
         "AVX2"
#else
         "SCALAR"
#endif
  );
  printf("%-28s %10s %10s\n", "shape", "fwd GF/s", "bwd GF/s");

  for (auto& s : shapes) {
    LinearConfig cfg;
    cfg.in_features = s.K;
    cfg.out_features = s.N;
    cfg.max_batch = s.B;
    cfg.activation = ActivationKind::ReLU;
    cfg.seed = 1;
    Linear layer(cfg);

    std::vector<float> x((size_t)s.B * s.K), dOut((size_t)s.B * s.N);
    for (size_t i = 0; i < x.size(); i++) x[i] = 0.1f * (float)((int)(i % 13) - 6);
    for (size_t i = 0; i < dOut.size(); i++) dOut[i] = 0.1f * (float)((int)(i % 7) - 3);

    double flops_fwd = 2.0 * s.B * s.K * s.N;  // dominant term (bias/activation are O(B*N))
    double t_fwd = median_time_per_call([&] { layer.forward(x.data(), s.B); }, 3, 200, 15);

    layer.forward(x.data(), s.B);  // repopulate caches right before timing backward
    double flops_bwd = 4.0 * s.B * s.K * s.N;  // dX (2*B*K*N) + dW (2*B*K*N)
    double t_bwd = median_time_per_call([&] { layer.backward(dOut.data(), s.B); }, 3, 200, 15);

    printf("%-28s %10.3f %10.3f\n", s.label, flops_fwd / t_fwd / 1e9, flops_bwd / t_bwd / 1e9);
  }
  return 0;
}
