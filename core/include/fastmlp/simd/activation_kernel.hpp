#pragma once
// Same backend split as gemm_kernel.hpp: scalar:: is the portable std::
// ground truth, avx2:: / avx512:: the hand-vectorized versions for the Zen 2 /
// Emerald Rapids targets. All are always built (tests compare them);
// FASTMLP_BACKEND_AVX2 / FASTMLP_BACKEND_AVX512 pick which one
// fastmlp::activation_forward/backward dispatch to.

#include "fastmlp/activation.hpp"

namespace fastmlp::simd {

namespace scalar {
void activation_forward(ActivationKind kind, const float* z, float* a, int n);
void activation_backward(ActivationKind kind, const float* z, const float* a,
                          const float* dA, float* dZ, int n);
}  // namespace scalar

namespace avx2 {
void activation_forward(ActivationKind kind, const float* z, float* a, int n);
void activation_backward(ActivationKind kind, const float* z, const float* a,
                          const float* dA, float* dZ, int n);
}  // namespace avx2

namespace avx512 {
void activation_forward(ActivationKind kind, const float* z, float* a, int n);
void activation_backward(ActivationKind kind, const float* z, const float* a,
                          const float* dA, float* dZ, int n);
}  // namespace avx512

}  // namespace fastmlp::simd
