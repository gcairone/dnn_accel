// two-moons training via the raw fastmlp_core C++ API (no Python/bindings
// involved) -- reads the same fixed dataset generate_data.py writes, so
// results/timing are directly comparable to train_fastmlp.py and
// train_torch.py. Same architecture/hyperparameters/activation choice; see
// train_fastmlp.py's header comment for why activation=Sigmoid (not ReLU).
//
// The loss gradient (BCE) is not part of fastmlp_core, whose boundary is
// forward/backward/optimizer (python/fastmlp/losses.py lives outside the
// native extension too). Computed inline here, as a C++ caller would;
// matches losses.py::bce_grad exactly (same eps clipping, same 1/batch
// averaging).
//
// Run from the repo root (paths below are relative to it):
//   python3 examples/two_moons/generate_data.py   # once
//   ./build-examples/two_moons_cpp
#include "fastmlp/module.hpp"
#include "fastmlp/optimizer.hpp"

#include "../common/example_timing.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using fastmlp::ActivationKind;
using fastmlp::Module;
using fastmlp::ModuleConfig;
using fastmlp::SGD;

namespace {

std::vector<float> read_bin(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s (run from repo root, after generate_data.py)\n",
                 path.c_str());
    std::exit(1);
  }
  std::vector<float> data(count);
  f.read(reinterpret_cast<char*>(data.data()), (std::streamsize)(count * sizeof(float)));
  if (!f) {
    std::fprintf(stderr, "short read on %s\n", path.c_str());
    std::exit(1);
  }
  return data;
}

// Matches python/fastmlp/losses.py::bce_grad exactly.
void bce_grad(const float* pred, const float* target, float* grad, int batch) {
  const float eps = 1e-7f;
  for (int i = 0; i < batch; i++) {
    float p = std::min(std::max(pred[i], eps), 1.0f - eps);
    grad[i] = (p - target[i]) / (p * (1.0f - p)) / (float)batch;
  }
}

float accuracy(Module& model, const std::vector<float>& X, const std::vector<float>& y, int n,
               int batch_size) {
  int correct = 0;
  for (int i = 0; i < n; i += batch_size) {
    int b = std::min(batch_size, n - i);
    const float* pred = model.forward({X.data() + (size_t)i * 2}, b);
    for (int j = 0; j < b; j++) {
      bool p = pred[j] > 0.5f;
      bool t = y[i + j] > 0.5f;
      if (p == t) correct++;
    }
  }
  return (float)correct / (float)n;
}

}  // namespace

int main() {
  fastmlp::example::pin_to_core0();
  const std::string data_dir = "examples/two_moons/data/";
  const int N_TRAIN = 1600, N_TEST = 400;
  const int BATCH = 64, N_EPOCHS = 200, WARMUP_STEPS = 25;
  const float LR = 0.5f;

  std::vector<float> X_train = read_bin(data_dir + "X_train.bin", (size_t)N_TRAIN * 2);
  std::vector<float> y_train = read_bin(data_dir + "y_train.bin", N_TRAIN);
  std::vector<float> X_test = read_bin(data_dir + "X_test.bin", (size_t)N_TEST * 2);
  std::vector<float> y_test = read_bin(data_dir + "y_test.bin", N_TEST);

  ModuleConfig cfg;
  cfg.layer_sizes = {2, 32, 16, 1};
  cfg.activation = ActivationKind::Sigmoid;
  cfg.max_batch = BATCH;
  cfg.seed = 1;
  Module model(cfg);
  SGD opt(model, LR);

  // Permuted once per epoch into a scratch buffer, mirroring
  // train_fastmlp.py's `Xs, ys = X_train[perm], y_train[perm]`; each batch
  // is then a contiguous slice (see examples/mnist/train_cpp.cpp for why).
  std::vector<float> X_shuf((size_t)N_TRAIN * 2), y_shuf(N_TRAIN), grad(BATCH);
  std::vector<int> perm(N_TRAIN);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 rng(42);

  fastmlp::example::StepTimer timer(WARMUP_STEPS);
  for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int k = 0; k < N_TRAIN; k++) {
      int idx = perm[k];
      X_shuf[(size_t)k * 2 + 0] = X_train[(size_t)idx * 2 + 0];
      X_shuf[(size_t)k * 2 + 1] = X_train[(size_t)idx * 2 + 1];
      y_shuf[k] = y_train[idx];
    }
    for (int i = 0; i < N_TRAIN; i += BATCH) {
      int b = std::min(BATCH, N_TRAIN - i);

      timer.begin();
      const float* pred = model.forward({X_shuf.data() + (size_t)i * 2}, b);
      bce_grad(pred, y_shuf.data() + i, grad.data(), b);
      model.backward(grad.data(), b);
      opt.step();
      opt.zero_grad();
      timer.end();
    }
  }

  float train_acc = accuracy(model, X_train, y_train, N_TRAIN, BATCH);
  float test_acc = accuracy(model, X_test, y_test, N_TEST, BATCH);
  std::printf("[fastmlp/cpp] train_acc=%.3f test_acc=%.3f\n", train_acc, test_acc);
  timer.print_summary("[fastmlp/cpp]");
  return 0;
}
