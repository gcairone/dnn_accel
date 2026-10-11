// The "cpp" implementation of the benchmark suite: the raw fastmlp_core API,
// no Python anywhere in the process. Driven by run.py, which prints and
// records what this writes; mirrors worker.py case for case.
//
//   suite_cpp infer <arch> <model.mlpf> <batch>...
//   suite_cpp train <arch> <model.mlpf> <ce|bce|mse> <batch>...
//   suite_cpp loop  <task> <spec.txt>
//
// Harness constants, synthetic-input kinds, loss gradients and metrics all
// match benchmarks/suite/common.py / worker.py / models.py -- keep in sync.
#include "fastmlp/mlpf.hpp"
#include "fastmlp/optimizer.hpp"

#include <sched.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using fastmlp::Module;

namespace {

constexpr int kWarmupCalls = 3;
constexpr double kTargetSampleS = 0.05;
constexpr int kReps = 7;
constexpr double kBceEps = 1e-7;

void pin() {
  const char* c = std::getenv("FASTMLP_SUITE_CORE");
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(c ? std::atoi(c) : 0, &set);
  sched_setaffinity(0, sizeof(set), &set);
}

double now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

struct Measured {
  double sec;
  int inner;
};

// Same shape as common.measure(): warmup calls, calibrate inner count to one
// ~50 ms sample, one discarded warmup sample, median of kReps.
template <class F>
Measured measure(F&& fn) {
  for (int i = 0; i < kWarmupCalls; i++) fn();
  double t0 = now();
  fn();
  double one = std::max(now() - t0, 1e-7);
  int inner = std::max(1, (int)(kTargetSampleS / one));
  std::vector<double> samples;
  for (int r = 0; r <= kReps; r++) {
    t0 = now();
    for (int i = 0; i < inner; i++) fn();
    double dt = (now() - t0) / inner;
    if (r) samples.push_back(dt);
  }
  std::sort(samples.begin(), samples.end());
  return {samples[samples.size() / 2], inner};
}

void collect_leaves(Module& m, std::vector<Module*>& out) {
  if (m.is_leaf()) {
    out.push_back(&m);
    return;
  }
  for (Module* c : m.inputs()) collect_leaves(*c, out);
}

enum class Loss { CE, BCE, MSE };

Loss parse_loss(const std::string& s) {
  if (s == "ce") return Loss::CE;
  if (s == "bce") return Loss::BCE;
  if (s == "mse") return Loss::MSE;
  std::fprintf(stderr, "unknown loss %s\n", s.c_str());
  std::exit(2);
}

// The gradients fastmlp's losses.py computes (mse_grad, bce_grad,
// cross_entropy_grad). For CE, y holds one class index (as a float) per row.
void loss_grad(Loss loss, const float* pred, const float* y, float* g, int batch, int c) {
  const float inv_b = 1.0f / (float)batch;
  if (loss == Loss::MSE) {
    const float s = 2.0f * inv_b;
    for (int i = 0; i < batch * c; i++) g[i] = s * (pred[i] - y[i]);
  } else if (loss == Loss::BCE) {
    for (int i = 0; i < batch * c; i++) {
      float p = std::min(std::max(pred[i], (float)kBceEps), 1.0f - (float)kBceEps);
      g[i] = (p - y[i]) / (p * (1.0f - p)) * inv_b;
    }
  } else {
    for (int b = 0; b < batch; b++) {
      const float* z = pred + (size_t)b * c;
      float* gr = g + (size_t)b * c;
      float mx = z[0];
      for (int j = 1; j < c; j++) mx = std::max(mx, z[j]);
      float sum = 0.0f;
      for (int j = 0; j < c; j++) {
        gr[j] = std::exp(z[j] - mx);
        sum += gr[j];
      }
      const float s = inv_b / sum;
      for (int j = 0; j < c; j++) gr[j] *= s;
      gr[(int)y[b]] -= inv_b;
    }
  }
}

// common.metrics(), same definitions, float64 accumulation.
void print_metrics(Loss loss, const std::vector<float>& pred, const std::vector<float>& y, int n,
                   int c) {
  if (loss == Loss::CE) {
    double nll = 0.0;
    int correct = 0;
    for (int i = 0; i < n; i++) {
      const float* z = pred.data() + (size_t)i * c;
      int label = (int)y[i];
      int arg = 0;
      double mx = z[0];
      for (int j = 1; j < c; j++) {
        if (z[j] > mx) {
          mx = z[j];
          arg = j;
        }
      }
      double sum = 0.0;
      for (int j = 0; j < c; j++) sum += std::exp((double)z[j] - mx);
      nll -= ((double)z[label] - mx) - std::log(sum);
      correct += (arg == label);
    }
    std::printf("\"test_loss\":%.9g,\"test_acc\":%.9g", nll / n, (double)correct / n);
  } else if (loss == Loss::BCE) {
    double ll = 0.0;
    int correct = 0;
    for (int i = 0; i < n; i++) {
      double p = std::min(std::max((double)pred[(size_t)i * c], kBceEps), 1.0 - kBceEps);
      double t = y[i];
      ll -= t * std::log(p) + (1.0 - t) * std::log(1.0 - p);
      correct += ((p > 0.5) == (t > 0.5));
    }
    std::printf("\"test_loss\":%.9g,\"test_acc\":%.9g", ll / n, (double)correct / n);
  } else {
    double se = 0.0;
    for (size_t i = 0; i < (size_t)n * c; i++) {
      double d = (double)pred[i] - y[i];
      se += d * d;
    }
    std::printf("\"test_loss\":%.9g", se / ((double)n * c));
  }
}

// Random leaf inputs + target, same kinds as worker.synthetic() (values differ:
// different generators, and timing does not depend on them).
struct Synthetic {
  std::vector<std::vector<float>> xs;
  std::vector<const float*> ptrs;
  std::vector<float> y;
};

Synthetic synthetic(const std::vector<Module*>& leaves, Loss loss, int out, int batch,
                    std::mt19937_64& rng) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  Synthetic s;
  for (Module* l : leaves) {
    s.xs.emplace_back((size_t)batch * l->in_features());
    for (float& v : s.xs.back()) v = normal(rng);
  }
  for (auto& x : s.xs) s.ptrs.push_back(x.data());
  if (loss == Loss::CE) {
    std::uniform_int_distribution<int> cls(0, out - 1);
    s.y.resize(batch);
    for (float& v : s.y) v = (float)cls(rng);
  } else {
    s.y.resize((size_t)batch * out);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (float& v : s.y) v = (loss == Loss::BCE) ? (u(rng) > 0.5f ? 1.0f : 0.0f) : normal(rng);
  }
  return s;
}

int run_infer(const std::string& arch, const std::string& path, const std::vector<int>& batches) {
  std::mt19937_64 rng(0);
  for (int batch : batches) {
    fastmlp::mlpf::LoadOptions opts;
    opts.max_batch_override = batch;
    opts.inference_only = true;
    auto model = fastmlp::mlpf::load(path, opts);
    Module& root = model.root();
    std::vector<Module*> leaves;
    collect_leaves(root, leaves);
    Synthetic s = synthetic(leaves, Loss::MSE, root.out_features(), batch, rng);
    Measured m = measure([&] { root.forward(s.ptrs, batch); });
    std::printf(
        "RESULT {\"mode\":\"infer\",\"impl\":\"cpp\",\"arch\":\"%s\",\"batch\":%d,\"sec\":%.9g,"
        "\"inner\":%d,\"compile_s\":null}\n",
        arch.c_str(), batch, m.sec, m.inner);
    std::fflush(stdout);
  }
  return 0;
}

int run_train(const std::string& arch, const std::string& path, Loss loss,
              const std::vector<int>& batches) {
  std::mt19937_64 rng(0);
  for (int batch : batches) {
    fastmlp::mlpf::LoadOptions opts;
    opts.max_batch_override = batch;
    auto model = fastmlp::mlpf::load(path, opts);
    Module& root = model.root();
    std::vector<Module*> leaves;
    collect_leaves(root, leaves);
    const int out = root.out_features();
    Synthetic s = synthetic(leaves, loss, out, batch, rng);
    fastmlp::Adam opt(root, 1e-3f);
    std::vector<float> grad((size_t)batch * out);
    Measured m = measure([&] {
      const float* pred = root.forward(s.ptrs, batch);
      loss_grad(loss, pred, s.y.data(), grad.data(), batch, out);
      root.backward(grad.data(), batch);
      opt.step();
      opt.zero_grad();
    });
    std::printf(
        "RESULT {\"mode\":\"train\",\"impl\":\"cpp\",\"arch\":\"%s\",\"batch\":%d,\"sec\":%.9g,"
        "\"inner\":%d,\"compile_s\":null}\n",
        arch.c_str(), batch, m.sec, m.inner);
    std::fflush(stdout);
  }
  return 0;
}

template <class T>
std::vector<T> read_bin(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary);
  std::vector<T> v(count);
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(count * sizeof(T)));
  if (!f) {
    std::fprintf(stderr, "cannot read %zu x %zu bytes from %s (run prepare_data.py)\n", count,
                 sizeof(T), path.c_str());
    std::exit(1);
  }
  return v;
}

struct Leaf {
  int width;
  std::vector<float> train, test;
};

int run_loop(const std::string& task, const std::string& spec_path) {
  std::ifstream spec(spec_path);
  if (!spec) {
    std::fprintf(stderr, "cannot open %s (run prepare_data.py)\n", spec_path.c_str());
    return 1;
  }
  std::map<std::string, std::string> kv;
  std::vector<std::array<std::string, 3>> leaf_specs;
  std::string key;
  while (spec >> key) {
    if (key == "leaf") {
      std::array<std::string, 3> l;
      spec >> l[0] >> l[1] >> l[2];
      leaf_specs.push_back(l);
    } else {
      spec >> kv[key];
    }
  }
  const Loss loss = parse_loss(kv["loss"]);
  const int batch = std::stoi(kv["batch"]), epochs = std::stoi(kv["epochs"]);
  const int n_train = std::stoi(kv["n_train"]), n_test = std::stoi(kv["n_test"]);
  const int yw = std::stoi(kv["y_width"]);

  std::vector<Leaf> data;
  for (auto& l : leaf_specs) {
    int w = std::stoi(l[0]);
    data.push_back({w, read_bin<float>(l[1], (size_t)n_train * w),
                    read_bin<float>(l[2], (size_t)n_test * w)});
  }
  auto y_train = read_bin<float>(kv["y_train"], (size_t)n_train * yw);
  auto y_test = read_bin<float>(kv["y_test"], (size_t)n_test * yw);
  auto perm = read_bin<int32_t>(kv["perm"], (size_t)epochs * n_train);

  fastmlp::mlpf::LoadOptions opts;
  opts.max_batch_override = batch;
  auto model = fastmlp::mlpf::load(kv["model"], opts);
  Module& root = model.root();
  const int out = root.out_features();
  std::unique_ptr<fastmlp::Optimizer> opt;
  if (kv["optimizer"] == "adam") {
    opt = std::make_unique<fastmlp::Adam>(root, std::stof(kv["lr"]));
  } else {
    opt = std::make_unique<fastmlp::SGD>(root, std::stof(kv["lr"]), std::stof(kv["momentum"]));
  }

  // Per-step row gather into contiguous batch buffers -- what X[idx] does in
  // the Python implementations.
  std::vector<std::vector<float>> xb;
  std::vector<const float*> xptr;
  for (auto& l : data) xb.emplace_back((size_t)batch * l.width);
  for (auto& b : xb) xptr.push_back(b.data());
  std::vector<float> yb((size_t)batch * yw), grad((size_t)batch * out);

  const int steps = n_train / batch;  // drop_last, as in worker.py
  double total = 0.0, first_step_s = -1.0;
  std::vector<float> pred_test((size_t)n_test * out);
  std::vector<const float*> tptr(data.size());
  for (int e = 0; e < epochs; e++) {
    const int32_t* p = perm.data() + (size_t)e * n_train;
    double t0 = now();
    for (int s = 0; s < steps; s++) {
      double ts = now();
      const int32_t* idx = p + (size_t)s * batch;
      for (size_t li = 0; li < data.size(); li++) {
        const int w = data[li].width;
        for (int r = 0; r < batch; r++) {
          std::memcpy(xb[li].data() + (size_t)r * w, data[li].train.data() + (size_t)idx[r] * w,
                      sizeof(float) * w);
        }
      }
      for (int r = 0; r < batch; r++) {
        std::memcpy(yb.data() + (size_t)r * yw, y_train.data() + (size_t)idx[r] * yw,
                    sizeof(float) * yw);
      }
      const float* pred = root.forward(xptr, batch);
      loss_grad(loss, pred, yb.data(), grad.data(), batch, out);
      root.backward(grad.data(), batch);
      opt->step();
      opt->zero_grad();
      if (first_step_s < 0) first_step_s = now() - ts;
    }
    double dt = now() - t0;
    total += dt;

    // Untimed evaluation on the test set.
    for (int a = 0; a < n_test; a += batch) {
      int nb = std::min(batch, n_test - a);
      for (size_t li = 0; li < data.size(); li++) {
        tptr[li] = data[li].test.data() + (size_t)a * data[li].width;
      }
      const float* o = root.forward(tptr, nb);
      std::memcpy(pred_test.data() + (size_t)a * out, o, sizeof(float) * nb * out);
    }
    std::printf("RESULT {\"mode\":\"loop_epoch\",\"impl\":\"cpp\",\"task\":\"%s\",\"epoch\":%d,"
                "\"train_s\":%.9g,",
                task.c_str(), e + 1, dt);
    print_metrics(loss, pred_test, y_test, n_test, out);
    std::printf("}\n");
    std::fflush(stdout);
  }
  std::printf("RESULT {\"mode\":\"loop\",\"impl\":\"cpp\",\"task\":\"%s\",\"total_s\":%.9g,"
              "\"first_step_s\":%.9g,\"steps\":%d,",
              task.c_str(), total, first_step_s, steps * epochs);
  print_metrics(loss, pred_test, y_test, n_test, out);
  std::printf("}\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  pin();
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: suite_cpp infer <arch> <mlpf> <batch>...\n"
                 "       suite_cpp train <arch> <mlpf> <loss> <batch>...\n"
                 "       suite_cpp loop  <task> <spec.txt>\n");
    return 2;
  }
  const std::string mode = argv[1];
  try {
    if (mode == "infer") {
      std::vector<int> b;
      for (int i = 4; i < argc; i++) b.push_back(std::atoi(argv[i]));
      return run_infer(argv[2], argv[3], b);
    }
    if (mode == "train") {
      std::vector<int> b;
      for (int i = 5; i < argc; i++) b.push_back(std::atoi(argv[i]));
      return run_train(argv[2], argv[3], parse_loss(argv[4]), b);
    }
    if (mode == "loop") return run_loop(argv[2], argv[3]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "suite_cpp: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
