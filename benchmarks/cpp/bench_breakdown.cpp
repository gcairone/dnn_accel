// Phase-by-phase breakdown of a real training step -- the analysis tool that
// says WHERE the time actually goes, rather than inferring it from
// bench_gemm's isolated-kernel numbers.
//
// bench_gemm measures each kernel on its own, cache-hot, in a tight loop.
// That is the right way to tune a kernel in isolation but it systematically
// FLATTERS the real thing: in a tight loop over one shape, the weight matrix
// stays resident in L2/L3, while in a real training step every layer's W, dW,
// X-cache and optimizer state compete for the same cache. This file times the
// phases as they actually run, back to back, on realistic model shapes, and
// reports each phase's share of the step plus its achieved GFLOP/s against the
// hw.md peak (134.4 GFLOP/s at the observed 4292 MHz).
//
// Build:  cmake -DFASTMLP_BUILD_BENCHMARKS=ON ...  &&  ./bench_breakdown
#include "fastmlp/activation.hpp"
#include "fastmlp/layer.hpp"
#include "fastmlp/module.hpp"
#include "fastmlp/optimizer.hpp"
#include "fastmlp/simd/gemm_kernel.hpp"

#include "bench_common.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace fastmlp;

namespace {

constexpr double kPeakGflops = 134.4;  // hw.md: 32 flops/cycle @ 4292 MHz

struct Phase {
  std::string name;
  double seconds;
  double flops;      // 0 = not a FLOP-bound phase (memory/elementwise)
  double bytes;      // rough traffic estimate for memory-bound phases
};

// Repeats fn enough times to get a stable measurement, returns per-call seconds.
double time_phase(const std::function<void()>& fn, int inner, int reps) {
  return bench::median_time_per_call(fn, 2, inner, reps);
}

void report(const std::string& title, std::vector<Phase>& phases) {
  double total = 0.0;
  for (const auto& p : phases) total += p.seconds;

  std::printf("\n=== %s ===\n", title.c_str());
  std::printf("%-28s %10s %7s %10s %8s\n", "phase", "us", "%step", "GFLOP/s", "%peak");
  for (const auto& p : phases) {
    double pct = 100.0 * p.seconds / total;
    if (p.flops > 0) {
      double gf = p.flops / p.seconds * 1e-9;
      std::printf("%-28s %10.1f %6.1f%% %10.2f %7.1f%%\n", p.name.c_str(), p.seconds * 1e6, pct,
                   gf, 100.0 * gf / kPeakGflops);
    } else if (p.bytes > 0) {
      double gbs = p.bytes / p.seconds * 1e-9;
      std::printf("%-28s %10.1f %6.1f%% %10s %8.1f GB/s\n", p.name.c_str(), p.seconds * 1e6, pct,
                   "-", gbs);
    } else {
      std::printf("%-28s %10.1f %6.1f%% %10s %8s\n", p.name.c_str(), p.seconds * 1e6, pct, "-", "-");
    }
  }
  std::printf("%-28s %10.1f %6.1f%%\n", "TOTAL", total * 1e6, 100.0);
}

// Decomposes one training step of a plain [in, h1, h2, out] MLP into the
// individual kernel calls Linear::forward/backward make, timing each in
// isolation but on buffers sized exactly as the real thing uses them.
void breakdown_mlp(const std::string& title, const std::vector<int>& sizes, int B,
                    ActivationKind act, int inner, int reps) {
  const int L = (int)sizes.size() - 1;

  // Per-layer buffers, laid out exactly like Linear's members.
  std::vector<std::vector<float>> W(L), dW(L), Xc(L), Z(L), A(L), dZ(L), dX(L), bias(L), db(L);
  for (int l = 0; l < L; l++) {
    int in = sizes[l], out = sizes[l + 1];
    W[l].assign((size_t)out * in, 0.01f);
    dW[l].assign((size_t)out * in, 0.0f);
    bias[l].assign(out, 0.0f);
    db[l].assign(out, 0.0f);
    Xc[l].assign((size_t)B * in, 0.5f);
    Z[l].assign((size_t)B * out, 0.0f);
    A[l].assign((size_t)B * out, 0.0f);
    dZ[l].assign((size_t)B * out, 0.1f);
    dX[l].assign((size_t)B * in, 0.0f);
  }

  std::vector<Phase> phases;

  // --- forward -----------------------------------------------------------
  // Measures the packed path Linear::forward actually takes, including the
  // repack itself -- which a training step really does pay once per
  // optimizer step, so hiding it would flatter the result.
  std::vector<std::vector<float>> Wp(L);
  std::vector<int> ldWp(L);
  for (int l = 0; l < L; l++) {
    int in = sizes[l], out = sizes[l + 1];
    ldWp[l] = simd::packed_ldw(out);
    Wp[l].assign((size_t)in * ldWp[l], 0.0f);
    simd::pack_transpose_f32(W[l].data(), Wp[l].data(), out, in, ldWp[l]);
  }
  for (int l = 0; l < L; l++) {
    int in = sizes[l], out = sizes[l + 1];
    double flops = 2.0 * B * in * out;
    double t = time_phase(
        [&] {
          simd::gemm_xwtp_f32(Xc[l].data(), Wp[l].data(), bias[l].data(), Z[l].data(), B, in, out,
                               ldWp[l]);
        },
        inner, reps);
    phases.push_back({"fwd gemm L" + std::to_string(l) + " (" + std::to_string(in) + "->" +
                        std::to_string(out) + ")",
                       t, flops, 0});
  }
  for (int l = 0; l < L; l++) {
    int in = sizes[l], out = sizes[l + 1];
    double t = time_phase(
        [&] { simd::pack_transpose_f32(W[l].data(), Wp[l].data(), out, in, ldWp[l]); }, inner, reps);
    phases.push_back({"fwd repack W L" + std::to_string(l), t, 0, 2.0 * in * out * 4});
  }
  for (int l = 0; l < L; l++) {
    int out = sizes[l + 1];
    double t = time_phase([&] { activation_forward(act, Z[l].data(), A[l].data(), B * out); }, inner,
                           reps);
    phases.push_back({"fwd act L" + std::to_string(l), t, 0, 2.0 * B * out * 4});
  }
  // Linear::forward's x_cache_ memcpy (backward needs the input for dW).
  for (int l = 0; l < L; l++) {
    int in = sizes[l];
    std::vector<float> src((size_t)B * in, 0.5f);
    double t = time_phase(
        [&] { std::memcpy(Xc[l].data(), src.data(), sizeof(float) * (size_t)B * in); }, inner, reps);
    phases.push_back({"fwd x_cache memcpy L" + std::to_string(l), t, 0, 2.0 * B * in * 4});
  }

  // --- backward ----------------------------------------------------------
  for (int l = L - 1; l >= 0; l--) {
    int out = sizes[l + 1];
    double t = time_phase(
        [&] { activation_backward(act, Z[l].data(), A[l].data(), dZ[l].data(), dZ[l].data(), B * out); },
        inner, reps);
    phases.push_back({"bwd act L" + std::to_string(l), t, 0, 4.0 * B * out * 4});
  }
  for (int l = L - 1; l >= 0; l--) {
    int in = sizes[l], out = sizes[l + 1];
    double flops = 2.0 * B * in * out;
    double t = time_phase(
        [&] { simd::gemm_dX_f32(dZ[l].data(), W[l].data(), dX[l].data(), B, in, out); }, inner, reps);
    phases.push_back({"bwd dX L" + std::to_string(l) + " (" + std::to_string(in) + "->" +
                        std::to_string(out) + ")",
                       t, flops, 0});
  }
  // The dW path Linear::backward actually takes: transpose dZ, then
  // accumulate. Both phases reported, so the transpose is never hidden.
  std::vector<std::vector<float>> dZt(L);
  for (int l = 0; l < L; l++) dZt[l].assign((size_t)sizes[l + 1] * B, 0.0f);
  for (int l = L - 1; l >= 0; l--) {
    int out = sizes[l + 1];
    double t = time_phase(
        [&] { simd::pack_transpose_f32(dZ[l].data(), dZt[l].data(), B, out, B); }, inner, reps);
    phases.push_back({"bwd dZ transpose L" + std::to_string(l), t, 0, 2.0 * B * out * 4});
  }
  for (int l = L - 1; l >= 0; l--) {
    int in = sizes[l], out = sizes[l + 1];
    double flops = 2.0 * B * in * out;
    double t = time_phase(
        [&] { simd::gemm_dW_accum_t_f32(Xc[l].data(), dZt[l].data(), dW[l].data(), B, in, out); },
        inner, reps);
    phases.push_back({"bwd dW L" + std::to_string(l) + " (" + std::to_string(in) + "->" +
                        std::to_string(out) + ")",
                       t, flops, 0});
  }
  for (int l = L - 1; l >= 0; l--) {
    int out = sizes[l + 1];
    double t = time_phase([&] { simd::bias_grad_accum_f32(dZ[l].data(), db[l].data(), B, out); },
                           inner, reps);
    phases.push_back({"bwd dbias L" + std::to_string(l), t, 0, (double)B * out * 4});
  }

  // --- optimizer ---------------------------------------------------------
  {
    ModuleConfig cfg;
    cfg.layer_sizes = sizes;
    cfg.activation = act;
    cfg.max_batch = B;
    cfg.seed = 1;
    Module model(cfg);
    Adam opt(model, 1e-3f);
    size_t nparams = 0;
    for (int l = 0; l < L; l++) nparams += (size_t)sizes[l] * sizes[l + 1];

    double t_step = time_phase([&] { opt.step(); }, inner, reps);
    // Adam touches W, dW, m, v: read dW/m/v/W + write m/v/W = 7 streams.
    phases.push_back({"opt Adam step", t_step, 0, 7.0 * nparams * 4});

    double t_zero = time_phase([&] { opt.zero_grad(); }, inner, reps);
    phases.push_back({"opt zero_grad", t_zero, 0, (double)nparams * 4});
  }

  report(title, phases);
}

}  // namespace

int main() {
  bench::pin_to_core0();
  std::printf("fastmlp training-step breakdown (peak reference %.1f GFLOP/s, hw.md)\n",
               kPeakGflops);
#if defined(FASTMLP_BACKEND_AVX2)
  std::printf("backend: AVX2\n");
#else
  std::printf("backend: SCALAR\n");
#endif

  // examples/mnist: the shape that currently loses ~3.7x to PyTorch.
  breakdown_mlp("MNIST [784,256,128,10] B=128", {784, 256, 128, 10}, 128, ActivationKind::Sigmoid,
                 20, 5);
  // usages.md ex1: the small shape fastmlp already wins on.
  breakdown_mlp("ex1 [20,64,32,1] B=128", {20, 64, 32, 1}, 128, ActivationKind::ReLU, 200, 5);
  // examples/two_moons.
  breakdown_mlp("two_moons [2,32,16,1] B=64", {2, 32, 16, 1}, 64, ActivationKind::Sigmoid, 500, 5);
  // The two remaining M12 acceptance shapes (benchmarks/python/results.md).
  breakdown_mlp("ex2-scale [512,128,64,1] B=64", {512, 128, 64, 1}, 64, ActivationKind::ReLU, 100,
                 5);
  breakdown_mlp("wide [512,256,128,1] B=256", {512, 256, 128, 1}, 256, ActivationKind::ReLU, 20, 5);
  return 0;
}
