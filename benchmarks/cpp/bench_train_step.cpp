// End-to-end training-step throughput (forward + backward + SGD step +
// zero_grad) on a small [20,64,32,1] MLP, entirely inside core/ (no Python):
// the C++ counterpart of benchmarks/python/bench_train_vs_torch.py.
#include "bench_common.hpp"
#include "fastmlp/module.hpp"
#include "fastmlp/optimizer.hpp"

#include <cstdio>
#include <vector>

using namespace fastmlp;
using namespace fastmlp::bench;

int main() {
  pin_to_core0();

  ModuleConfig cfg;
  cfg.layer_sizes = {20, 64, 32, 1};
  cfg.activation = ActivationKind::ReLU;
  cfg.max_batch = 128;
  cfg.seed = 1;
  Module model(cfg);
  SGD opt(model, 0.01f);

  const int B = 128;
  std::vector<float> x((size_t)B * 20), dOut((size_t)B * 1);
  for (size_t i = 0; i < x.size(); i++) x[i] = 0.1f * (float)((int)(i % 13) - 6);
  for (size_t i = 0; i < dOut.size(); i++) dOut[i] = 0.1f * (float)((int)(i % 7) - 3);

  double t = median_time_per_call(
      [&] {
        model.forward({x.data()}, B);
        model.backward(dOut.data(), B);
        opt.step();
        opt.zero_grad();
      },
      3, 100, 15);

  printf("=== fastmlp full train step -- active backend: %s ===\n",
#if defined(FASTMLP_BACKEND_AVX2)
         "AVX2"
#else
         "SCALAR"
#endif
  );
  printf("[20,64,32,1], B=%d: %.3f us/step, %.1f steps/sec\n", B, t * 1e6, 1.0 / t);
  return 0;
}
