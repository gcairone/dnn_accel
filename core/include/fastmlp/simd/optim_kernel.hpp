#pragma once
// The optimizer's per-element update, split by backend like the GEMM and
// activation kernels. Optimizer/SGD/Adam (optimizer.hpp) own the state and
// the walk over the parameters; this is only the arithmetic over one
// contiguous block.
//
// scalar:: is the plain loop (the numerical reference, and what the avx2
// backend runs: GCC vectorizes it itself once the arrays are __restrict).
// avx512:: replaces vsqrtps + vdivps -- reciprocal throughput 12 and 10
// cycles per zmm on the Emerald Rapids target, both on one port, ~1.4
// cycles per parameter together -- with vrsqrt14ps / vrcp14ps plus a Newton
// step each (results/emr/compute.txt section 4).

namespace fastmlp::simd {

struct AdamStep {
  float lr, beta1, beta2, eps, weight_decay;
  float inv_bc1, inv_bc2;  // 1 / (1 - beta^t), pre-reciprocated
};

namespace scalar {
// m = b1 m + (1-b1) g'; v = b2 v + (1-b2) g'^2; p -= lr mhat / (sqrt(vhat) + eps)
// with g' = g + weight_decay * p, mhat = m * inv_bc1, vhat = v * inv_bc2.
void adam_update_f32(float* p, const float* g, float* m, float* v, int n, const AdamStep& s);
}  // namespace scalar

namespace avx512 {
void adam_update_f32(float* p, const float* g, float* m, float* v, int n, const AdamStep& s);
}  // namespace avx512

#if defined(FASTMLP_BACKEND_AVX512)
using avx512::adam_update_f32;
#else
using scalar::adam_update_f32;
#endif

}  // namespace fastmlp::simd
