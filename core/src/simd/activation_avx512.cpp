// AVX-512 activations for the Emerald Rapids target.
//
// Placeholder stage: forwards to the avx2 kernels (valid AVX2 code on this
// CPU) until the zmm versions land, so the AVX512 backend is complete and
// testable end to end from the start.
#include "fastmlp/simd/activation_kernel.hpp"

namespace fastmlp::simd::avx512 {

void activation_forward(ActivationKind kind, const float* z, float* a, int n) {
  avx2::activation_forward(kind, z, a, n);
}

void activation_backward(ActivationKind kind, const float* z, const float* a, const float* dA, float* dZ, int n) {
  avx2::activation_backward(kind, z, a, dA, dZ, n);
}

}  // namespace fastmlp::simd::avx512
