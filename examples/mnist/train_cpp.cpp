// MNIST training via the raw fastmlp_core C++ API (no Python/bindings).
// Reads the raw uint8 files prepare_data.py writes -- byte-identical to
// train_fastmlp.py/train_torch.py. mse_grad is computed inline (loss
// gradients are not part of fastmlp_core, the same boundary the Python API
// draws by keeping losses.py outside the native extension); matches
// python/fastmlp/losses.py::mse_grad exactly.
//
// Run from the repo root (paths below are relative to it):
//   python3 examples/mnist/prepare_data.py   # once
//   ./build-examples/mnist_cpp
#include "fastmlp/module.hpp"
#include "fastmlp/optimizer.hpp"

#include "../common/example_timing.hpp"

#include <algorithm>
#include <cstdint>
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

std::vector<uint8_t> read_bin_u8(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s (run from repo root, after prepare_data.py)\n",
                 path.c_str());
    std::exit(1);
  }
  std::vector<uint8_t> data(count);
  f.read(reinterpret_cast<char*>(data.data()), (std::streamsize)count);
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

}  // namespace

int main() {
  fastmlp::example::pin_to_core0();
  const std::string data_dir = "examples/mnist/data/";
  const int N_TRAIN = 60000, N_TEST = 10000, IN = 784, OUT = 10;
  const int BATCH = 128, N_EPOCHS = 15, WARMUP_STEPS = 469;
  const float LR = 1e-3f;

  std::vector<uint8_t> X_train_u8 = read_bin_u8(data_dir + "train_images.bin", (size_t)N_TRAIN * IN);
  std::vector<uint8_t> y_train_u8 = read_bin_u8(data_dir + "train_labels.bin", N_TRAIN);
  std::vector<uint8_t> X_test_u8 = read_bin_u8(data_dir + "test_images.bin", (size_t)N_TEST * IN);
  std::vector<uint8_t> y_test_u8 = read_bin_u8(data_dir + "test_labels.bin", N_TEST);

  std::vector<float> X_train((size_t)N_TRAIN * IN), X_test((size_t)N_TEST * IN);
  for (size_t i = 0; i < X_train.size(); i++) X_train[i] = X_train_u8[i] / 255.0f;
  for (size_t i = 0; i < X_test.size(); i++) X_test[i] = X_test_u8[i] / 255.0f;

  // One-hot targets, precomputed once (matches train_fastmlp.py's
  // `Y_train = np.eye(10)[y_train_int]`, built once before the epoch loop).
  std::vector<float> Y_train((size_t)N_TRAIN * OUT, 0.0f);
  for (int i = 0; i < N_TRAIN; i++) Y_train[(size_t)i * OUT + y_train_u8[i]] = 1.0f;

  ModuleConfig cfg;
  cfg.layer_sizes = {IN, 256, 128, OUT};
  cfg.activation = ActivationKind::Sigmoid;
  cfg.max_batch = BATCH;
  cfg.seed = 1;
  Module model(cfg);
  Adam opt(model, LR);

  // The whole training set is permuted into a scratch buffer ONCE per
  // epoch, mirroring train_fastmlp.py's `Xs, Ys = X_train[perm],
  // Y_train[perm]` exactly, then each batch is a contiguous slice of it
  // (no per-batch gather). Gathering scattered rows into a fresh xb right
  // before every timed step (as an earlier version of this file did)
  // thrashes the cache immediately before forward()/backward() run,
  // inflating the measured per-iteration cost with a shuffle artifact
  // Python's once-per-epoch permutation doesn't pay per batch.
  std::vector<float> X_shuf((size_t)N_TRAIN * IN), Y_shuf((size_t)N_TRAIN * OUT),
      grad((size_t)BATCH * OUT);
  std::vector<int> perm(N_TRAIN);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 rng(0);

  fastmlp::example::StepTimer timer(WARMUP_STEPS);
  for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int k = 0; k < N_TRAIN; k++) {
      int idx = perm[k];
      std::copy(X_train.begin() + (size_t)idx * IN, X_train.begin() + (size_t)(idx + 1) * IN,
                 X_shuf.begin() + (size_t)k * IN);
      std::copy(Y_train.begin() + (size_t)idx * OUT, Y_train.begin() + (size_t)(idx + 1) * OUT,
                 Y_shuf.begin() + (size_t)k * OUT);
    }
    for (int i = 0; i < N_TRAIN; i += BATCH) {
      int b = std::min(BATCH, N_TRAIN - i);

      timer.begin();
      const float* pred = model.forward({X_shuf.data() + (size_t)i * IN}, b);
      mse_grad(pred, Y_shuf.data() + (size_t)i * OUT, grad.data(), b * OUT, b);
      model.backward(grad.data(), b);
      opt.step();
      opt.zero_grad();
      timer.end();
    }
    std::printf("  epoch %d/%d done\n", epoch + 1, N_EPOCHS);
  }

  auto evaluate = [&](const std::vector<float>& X, const std::vector<uint8_t>& y_int, int n) {
    int correct = 0;
    for (int i = 0; i < n; i += BATCH) {
      int b = std::min(BATCH, n - i);
      const float* pred = model.forward({X.data() + (size_t)i * IN}, b);
      for (int j = 0; j < b; j++) {
        int argmax = 0;
        float best = pred[(size_t)j * OUT];
        for (int k = 1; k < OUT; k++) {
          if (pred[(size_t)j * OUT + k] > best) {
            best = pred[(size_t)j * OUT + k];
            argmax = k;
          }
        }
        if (argmax == y_int[i + j]) correct++;
      }
    }
    return (float)correct / (float)n;
  };

  float train_acc = evaluate(X_train, y_train_u8, N_TRAIN);
  float test_acc = evaluate(X_test, y_test_u8, N_TEST);
  std::printf("[fastmlp/cpp] train_acc=%.4f test_acc=%.4f\n", train_acc, test_acc);
  timer.print_summary("[fastmlp/cpp]");
  return 0;
}
