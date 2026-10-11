// Per-activation cost, scalar (std:: libm) vs avx2 (hand-vectorized), forward
// and backward, on one buffer the size of MNIST's first hidden layer at B=128
// (32768 floats, L2-resident) -- i.e. what one layer's activation phase costs
// inside bench_breakdown, isolated per kind.
#include <cstdio>
#include <vector>

#include "bench_common.hpp"
#include "fastmlp/activation.hpp"
#include "fastmlp/simd/activation_kernel.hpp"

using namespace fastmlp;

int main() {
  bench::pin_to_core0();
  const int n = 128 * 256;
  std::vector<float> z(n), a(n), dA(n), dZ(n);
  for (int i = 0; i < n; i++) {
    z[i] = 0.001f * (float)((i * 37) % 8001 - 4000);  // [-4, 4]
    dA[i] = 0.5f;
  }

  const ActivationKind kinds[] = {
      ActivationKind::Identity, ActivationKind::ReLU,      ActivationKind::LeakyReLU,
      ActivationKind::Sigmoid,  ActivationKind::Tanh,      ActivationKind::ELU,
      ActivationKind::SiLU,     ActivationKind::GELU,      ActivationKind::GELUTanh,
      ActivationKind::Softplus,
  };

  std::printf("n = %d floats; ns per element (lower is better)\n", n);
  std::printf("%-12s %10s %10s %8s   %10s %10s %8s\n", "activation", "fwd scal", "fwd avx2", "x",
              "bwd scal", "bwd avx2", "x");
  for (ActivationKind k : kinds) {
    auto per_elem = [&](const std::function<void()>& fn) {
      return bench::median_time_per_call(fn, 2, 20, 7) * 1e9 / n;
    };
    double fs = per_elem([&] { simd::scalar::activation_forward(k, z.data(), a.data(), n); });
    double fv = per_elem([&] { simd::avx2::activation_forward(k, z.data(), a.data(), n); });
    double bs = per_elem(
        [&] { simd::scalar::activation_backward(k, z.data(), a.data(), dA.data(), dZ.data(), n); });
    double bv = per_elem(
        [&] { simd::avx2::activation_backward(k, z.data(), a.data(), dA.data(), dZ.data(), n); });
    std::printf("%-12s %10.3f %10.3f %7.1fx   %10.3f %10.3f %7.1fx\n", activation_name(k), fs, fv,
                fs / fv, bs, bv, bs / bv);
  }
  return 0;
}
