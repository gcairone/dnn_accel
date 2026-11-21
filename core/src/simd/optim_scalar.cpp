#include <cmath>

#include "fastmlp/simd/optim_kernel.hpp"

namespace fastmlp::simd::scalar {

// The __restrict qualifiers are load-bearing, not decoration: with the
// arrays reached through members (m_W_[i][j] etc.) the compiler cannot prove
// W does not alias m/v, so it kept the whole loop scalar -- and a scalar
// Adam step measured 69 cycles/parameter on the Zen 2 target (bench_breakdown:
// 16.9% of an entire MNIST training step, at 1.7 GB/s). restrict-qualified
// pointers let it vectorize the divide and sqrt.
//
// The two bias corrections come pre-reciprocated so the per-element work is
// a multiply rather than a divide; the final lr*mhat/(sqrt(vhat)+eps) stays a
// real division here -- this is the IEEE reference the avx512 kernel's
// Newton-refined reciprocals are tested against.
void adam_update_f32(float* __restrict p, const float* __restrict g_in, float* __restrict m,
                     float* __restrict v, int n, const AdamStep& s) {
  const float lr = s.lr, beta1 = s.beta1, beta2 = s.beta2, eps = s.eps, wd = s.weight_decay;
  const float inv_bc1 = s.inv_bc1, inv_bc2 = s.inv_bc2;
  const float one_minus_b1 = 1.0f - beta1;
  const float one_minus_b2 = 1.0f - beta2;
  for (int j = 0; j < n; j++) {
    float g = g_in[j] + wd * p[j];
    float mj = beta1 * m[j] + one_minus_b1 * g;
    float vj = beta2 * v[j] + one_minus_b2 * g * g;
    m[j] = mj;
    v[j] = vj;
    float mhat = mj * inv_bc1;
    float vhat = vj * inv_bc2;
    p[j] -= lr * mhat / (std::sqrt(vhat) + eps);
  }
}

}  // namespace fastmlp::simd::scalar
