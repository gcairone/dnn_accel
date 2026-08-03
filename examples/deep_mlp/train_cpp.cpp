// Deep single-branch regression via the raw fastmlp_core C++ API -- 6
// layers ([20,128,128,128,64,32,1]), a different point in the shape space
// than the other examples' 2-3 layer stacks. Reads the same fixed dataset
// generate_data.py writes; directly comparable to train_fastmlp.py (same
// data, architecture, hyperparameters) and train_torch.py.
//
// mse_grad is computed inline -- loss gradients are not part of
// fastmlp_core; matches python/fastmlp/losses.py::mse_grad exactly.
//
// Run from the repo root (paths below are relative to it):
//   python3 examples/deep_mlp/generate_data.py   # once
//   ./build-examples/examples/deep_mlp_cpp
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
using fastmlp::Adam;
using fastmlp::Module;
using fastmlp::ModuleConfig;

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

// Matches python/fastmlp/losses.py::mse_grad exactly: (2/B) * (pred - target).
void mse_grad(const float* pred, const float* target, float* grad, int batch) {
  float scale = 2.0f / (float)batch;
  for (int i = 0; i < batch; i++) grad[i] = scale * (pred[i] - target[i]);
}

double mse(Module& model, const std::vector<float>& X, const std::vector<float>& y, int n,
           int in_features, int batch) {
  double total_se = 0.0;
  for (int i = 0; i < n; i += batch) {
    int b = std::min(batch, n - i);
    const float* pred = model.forward({X.data() + (size_t)i * in_features}, b);
    for (int j = 0; j < b; j++) {
      double d = pred[j] - y[i + j];
      total_se += d * d;
    }
  }
  return total_se / n;
}

}  // namespace

int main() {
  fastmlp::example::pin_to_core0();
  const std::string data_dir = "examples/deep_mlp/data/";
  const int N_TRAIN = 8000, N_TEST = 2000, IN = 20;
  const int BATCH = 128, N_EPOCHS = 60, WARMUP_STEPS = N_TRAIN / BATCH;
  const float LR = 1e-3f, WEIGHT_DECAY = 1e-4f;

  std::vector<float> X_train = read_bin(data_dir + "X_train.bin", (size_t)N_TRAIN * IN);
  std::vector<float> y_train = read_bin(data_dir + "y_train.bin", N_TRAIN);
  std::vector<float> X_test = read_bin(data_dir + "X_test.bin", (size_t)N_TEST * IN);
  std::vector<float> y_test = read_bin(data_dir + "y_test.bin", N_TEST);

  ModuleConfig cfg;
  // Narrow relative to N_TRAIN on purpose -- see train_fastmlp.py's header
  // comment: a wider stack here memorized the training set outright.
  cfg.layer_sizes = {IN, 48, 48, 48, 24, 12, 1};
  cfg.activation = ActivationKind::Tanh;
  cfg.max_batch = BATCH;
  cfg.seed = 1;
  Module model(cfg);
  Adam opt(model, LR, /*beta1=*/0.9f, /*beta2=*/0.999f, /*eps=*/1e-8f, WEIGHT_DECAY);

  std::vector<float> X_shuf((size_t)N_TRAIN * IN), y_shuf(N_TRAIN), grad(BATCH);
  std::vector<int> perm(N_TRAIN);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 rng(7);

  fastmlp::example::StepTimer timer(WARMUP_STEPS);
  for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int k = 0; k < N_TRAIN; k++) {
      int idx = perm[k];
      std::copy(X_train.begin() + (size_t)idx * IN, X_train.begin() + (size_t)(idx + 1) * IN,
                 X_shuf.begin() + (size_t)k * IN);
      y_shuf[k] = y_train[idx];
    }
    for (int i = 0; i < N_TRAIN; i += BATCH) {
      int b = std::min(BATCH, N_TRAIN - i);

      timer.begin();
      const float* pred = model.forward({X_shuf.data() + (size_t)i * IN}, b);
      mse_grad(pred, y_shuf.data() + i, grad.data(), b);
      model.backward(grad.data(), b);
      opt.step();
      opt.zero_grad();
      timer.end();
    }
  }

  double train_mse = mse(model, X_train, y_train, N_TRAIN, IN, BATCH);
  double test_mse = mse(model, X_test, y_test, N_TEST, IN, BATCH);
  std::printf("[fastmlp/cpp] train_mse=%.5f test_mse=%.5f\n", train_mse, test_mse);
  timer.print_summary("[fastmlp/cpp]");
  return 0;
}
