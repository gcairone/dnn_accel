// Inference-only throughput across a handful of common MLP architectures,
// via the raw fastmlp_core C++ API -- no training data needed, since only
// the forward pass's cost is being measured (unlike the other examples/
// tasks, where the actual predictions have to agree across implementations).
//
// inference_only=true (skips allocating gradient/backward-cache buffers) is
// the normal way this library is used once a model is trained and deployed,
// not an inference-specific speed trick -- and this
// binary is compiled with the project's standard -O3 arch flags, the same
// ones every other target here uses, nothing extra. What IS deliberately
// NOT done: no manual buffer/kernel-level tricks beyond what the library
// already does internally -- this is plain repeated model.forward() calls,
// same as any other caller's code.
//
// Batch sizes tested: 1 (the common "one request at a time" serving case)
// and the model's max_batch, 128 (batched/offline throughput).
//
// Run from the repo root:
//   ./build-examples/examples/inference_cpp
#include "fastmlp/module.hpp"

#include "../common/example_timing.hpp"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

using fastmlp::ActivationKind;
using fastmlp::Module;
using fastmlp::ModuleConfig;

namespace {

struct Arch {
  std::string name;
  std::vector<int> layers;
};

long count_params(const std::vector<int>& layers) {
  long n = 0;
  for (size_t i = 0; i + 1 < layers.size(); i++) {
    n += (long)layers[i] * layers[i + 1] + layers[i + 1];
  }
  return n;
}

}  // namespace

int main() {
  fastmlp::example::pin_to_core0();
  const int MAX_BATCH = 128;
  const int WARMUP_BLOCKS = 3, INNER_ITERS = 200, OUTER_REPS = 7;

  // Mostly architectures one would actually encounter (a tiny tabular
  // classifier, the classic single-hidden-layer MNIST baseline, a deeper
  // MNIST-style classifier, a mid-size tabular/feature classifier, a wide
  // shallow net), plus one deliberately unusual shape (deep_narrow) for
  // variety, plus square_WxD (width W in {64,128,256,512}, depth 1 or 2
  // hidden->hidden layers, in_features==out_features==W throughout) --
  // a clean width/depth sweep at equal width, isolating how each scales
  // independently of the mixed-width shapes above.
  std::vector<Arch> archs = {
      {"tiny_tabular", {8, 16, 8, 1}},
      {"small_classifier", {20, 64, 32, 1}},
      {"mnist_shallow", {784, 128, 10}},
      {"mnist_deep", {784, 256, 128, 10}},
      {"medium_classifier", {128, 256, 128, 64, 10}},
      {"wide_shallow", {100, 1024, 10}},
      {"deep_narrow", {64, 64, 64, 64, 64, 64, 64, 64, 64, 1}},
      {"square_64x2", {64, 64}},
      {"square_64x3", {64, 64, 64}},
      {"square_128x2", {128, 128}},
      {"square_128x3", {128, 128, 128}},
      {"square_256x2", {256, 256}},
      {"square_256x3", {256, 256, 256}},
      {"square_512x2", {512, 512}},
      {"square_512x3", {512, 512, 512}},
  };

  std::printf("%-18s %9s %6s %10s %12s\n", "arch", "params", "batch", "us/call", "infer/s");
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto& a : archs) {
    ModuleConfig cfg;
    cfg.layer_sizes = a.layers;
    cfg.activation = ActivationKind::ReLU;
    cfg.max_batch = MAX_BATCH;
    cfg.inference_only = true;
    cfg.seed = 1;
    Module model(cfg);

    int in_features = a.layers.front();
    std::vector<float> x((size_t)MAX_BATCH * in_features);
    for (auto& v : x) v = dist(rng);

    for (int batch : {1, MAX_BATCH}) {
      double t = fastmlp::example::median_time_per_call(
          [&] { model.forward({x.data()}, batch); }, WARMUP_BLOCKS, INNER_ITERS, OUTER_REPS);
      std::printf("%-18s %9ld %6d %10.2f %12.0f\n", a.name.c_str(), count_params(a.layers), batch,
                   t * 1e6, (double)batch / t);
    }
  }
  return 0;
}
