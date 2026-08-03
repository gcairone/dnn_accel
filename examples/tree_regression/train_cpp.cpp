// Tree-regression training via the raw fastmlp_core C++ API -- the
// Module(inputs={&enc_a, &enc_b}) case, matching Module::forward's
// documented depth-first leaf order (enc_a's input first, then enc_b's).
// Reads the same fixed dataset generate_data.py writes; directly comparable
// to train_fastmlp.py (same data, architecture, hyperparameters) and
// train_torch.py.
//
// mse_grad is computed inline -- loss gradients are not part of
// fastmlp_core (idea.md draws the core's boundary at forward/backward/
// optimizer only); matches python/fastmlp/losses.py::mse_grad exactly.
//
// Run from the repo root (paths below are relative to it):
//   python3 examples/tree_regression/generate_data.py   # once
//   ./build-examples/examples/tree_regression_cpp
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
void mse_grad(const float* pred, const float* target, float* grad, int n, int batch) {
  float scale = 2.0f / (float)batch;
  for (int i = 0; i < n; i++) grad[i] = scale * (pred[i] - target[i]);
}

double mse(Module& model, const std::vector<float>& A, const std::vector<float>& B,
           const std::vector<float>& y, int n, int na, int nb, int batch) {
  double total_se = 0.0;
  for (int i = 0; i < n; i += batch) {
    int b = std::min(batch, n - i);
    const float* pred =
        model.forward({A.data() + (size_t)i * na, B.data() + (size_t)i * nb}, b);
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
  const std::string data_dir = "examples/tree_regression/data/";
  const int N_TRAIN = 4000, N_TEST = 1000, NA = 32, NB = 16;
  const int BATCH = 64, N_EPOCHS = 100, WARMUP_STEPS = N_TRAIN / BATCH;
  const float LR = 1e-3f;

  std::vector<float> A_train = read_bin(data_dir + "A_train.bin", (size_t)N_TRAIN * NA);
  std::vector<float> B_train = read_bin(data_dir + "B_train.bin", (size_t)N_TRAIN * NB);
  std::vector<float> y_train = read_bin(data_dir + "y_train.bin", N_TRAIN);
  std::vector<float> A_test = read_bin(data_dir + "A_test.bin", (size_t)N_TEST * NA);
  std::vector<float> B_test = read_bin(data_dir + "B_test.bin", (size_t)N_TEST * NB);
  std::vector<float> y_test = read_bin(data_dir + "y_test.bin", N_TEST);

  ModuleConfig enc_a_cfg;
  enc_a_cfg.layer_sizes = {NA, 64, 32};
  enc_a_cfg.activation = ActivationKind::Sigmoid;
  enc_a_cfg.max_batch = BATCH;
  enc_a_cfg.seed = 1;
  Module enc_a(enc_a_cfg);

  ModuleConfig enc_b_cfg;
  enc_b_cfg.layer_sizes = {NB, 32, 32};
  enc_b_cfg.activation = ActivationKind::Sigmoid;
  enc_b_cfg.max_batch = BATCH;
  enc_b_cfg.seed = 2;
  Module enc_b(enc_b_cfg);

  ModuleConfig root_cfg;
  root_cfg.layer_sizes = {64, 32, 1};  // 64 = enc_a.out_features(32) + enc_b.out_features(32)
  root_cfg.activation = ActivationKind::Sigmoid;
  root_cfg.max_batch = BATCH;
  root_cfg.seed = 3;
  Module model(root_cfg, {&enc_a, &enc_b});
  Adam opt(model, LR);

  // Whole training set permuted into a scratch buffer once per epoch (see
  // examples/mnist/train_cpp.cpp's header comment for why this matters for
  // measurement, not just style: cache state right before the timed region).
  std::vector<float> A_shuf((size_t)N_TRAIN * NA), B_shuf((size_t)N_TRAIN * NB),
      y_shuf(N_TRAIN), grad(BATCH);
  std::vector<int> perm(N_TRAIN);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 rng(42);

  fastmlp::example::StepTimer timer(WARMUP_STEPS);
  for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int k = 0; k < N_TRAIN; k++) {
      int idx = perm[k];
      std::copy(A_train.begin() + (size_t)idx * NA, A_train.begin() + (size_t)(idx + 1) * NA,
                 A_shuf.begin() + (size_t)k * NA);
      std::copy(B_train.begin() + (size_t)idx * NB, B_train.begin() + (size_t)(idx + 1) * NB,
                 B_shuf.begin() + (size_t)k * NB);
      y_shuf[k] = y_train[idx];
    }
    for (int i = 0; i < N_TRAIN; i += BATCH) {
      int b = std::min(BATCH, N_TRAIN - i);

      timer.begin();
      const float* pred = model.forward(
          {A_shuf.data() + (size_t)i * NA, B_shuf.data() + (size_t)i * NB}, b);
      mse_grad(pred, y_shuf.data() + i, grad.data(), b, b);
      model.backward(grad.data(), b);
      opt.step();
      opt.zero_grad();
      timer.end();
    }
  }

  double train_mse = mse(model, A_train, B_train, y_train, N_TRAIN, NA, NB, BATCH);
  double test_mse = mse(model, A_test, B_test, y_test, N_TEST, NA, NB, BATCH);
  std::printf("[fastmlp/cpp] train_mse=%.5f test_mse=%.5f\n", train_mse, test_mse);
  timer.print_summary("[fastmlp/cpp]");
  return 0;
}
