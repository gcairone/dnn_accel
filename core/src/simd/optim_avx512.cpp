// AVX-512 Adam update for the Emerald Rapids target.
//
// lr * mhat / (sqrt(vhat) + eps) without vsqrtps / vdivps:
//   r  = rsqrt14(vhat), refined once:  r <- r (1.5 - 0.5 vhat r^2)
//   s  = vhat * r                      (sqrt(vhat); exactly 0 where vhat == 0,
//                                       where r = inf and vhat * r = NaN)
//   q  = rcp14(s + eps), refined once: q <- q (2 - d q)
// Each refinement squares the 2^-14 relative error to ~2^-28, so the update
// carries ~2-3 ulp of relative error against the IEEE expression. That is
// six orders of magnitude inside test_optimizer's 1e-4 on the weights.
// Denominator: s + eps >= eps > 0 and finite for any finite state.
//
// No STL or other shared inline code (see gemm_kernel_avx512.cpp).
#include <immintrin.h>

#include "fastmlp/simd/optim_kernel.hpp"

namespace fastmlp::simd::avx512 {

namespace {

inline __mmask16 head_mask(int n) { return (__mmask16)((n >= 16) ? 0xFFFF : ((1u << n) - 1u)); }

struct Consts {
  __m512 lr, b1, b2, omb1, omb2, eps, wd, ibc1, ibc2, half, three_half, one;
};

inline void adam16(float* p, const float* g, float* m, float* v, __mmask16 k, const Consts& c, bool full) {
  const __m512 pv = full ? _mm512_loadu_ps(p) : _mm512_maskz_loadu_ps(k, p);
  const __m512 gv0 = full ? _mm512_loadu_ps(g) : _mm512_maskz_loadu_ps(k, g);
  const __m512 mv0 = full ? _mm512_loadu_ps(m) : _mm512_maskz_loadu_ps(k, m);
  const __m512 vv0 = full ? _mm512_loadu_ps(v) : _mm512_maskz_loadu_ps(k, v);

  const __m512 gv = _mm512_fmadd_ps(c.wd, pv, gv0);
  const __m512 mv = _mm512_fmadd_ps(c.b1, mv0, _mm512_mul_ps(c.omb1, gv));
  const __m512 vv = _mm512_fmadd_ps(c.b2, vv0, _mm512_mul_ps(_mm512_mul_ps(c.omb2, gv), gv));
  const __m512 mhat = _mm512_mul_ps(mv, c.ibc1);
  const __m512 vhat = _mm512_mul_ps(vv, c.ibc2);

  __m512 r = _mm512_rsqrt14_ps(vhat);
  // r (1.5 - 0.5 vhat r^2)
  r = _mm512_mul_ps(r, _mm512_fnmadd_ps(_mm512_mul_ps(c.half, vhat), _mm512_mul_ps(r, r), c.three_half));
  const __mmask16 nz = _mm512_cmp_ps_mask(vhat, _mm512_setzero_ps(), _CMP_NEQ_UQ);
  const __m512 s = _mm512_maskz_mul_ps(nz, vhat, r);

  const __m512 d = _mm512_add_ps(s, c.eps);
  __m512 q = _mm512_rcp14_ps(d);
  q = _mm512_fmadd_ps(q, _mm512_fnmadd_ps(d, q, c.one), q);

  const __m512 pn = _mm512_fnmadd_ps(_mm512_mul_ps(c.lr, mhat), q, pv);
  if (full) {
    _mm512_storeu_ps(p, pn);
    _mm512_storeu_ps(m, mv);
    _mm512_storeu_ps(v, vv);
  } else {
    _mm512_mask_storeu_ps(p, k, pn);
    _mm512_mask_storeu_ps(m, k, mv);
    _mm512_mask_storeu_ps(v, k, vv);
  }
}

}  // namespace

void adam_update_f32(float* p, const float* g, float* m, float* v, int n, const AdamStep& s) {
  const Consts c{_mm512_set1_ps(s.lr),           _mm512_set1_ps(s.beta1),        _mm512_set1_ps(s.beta2),
                 _mm512_set1_ps(1.0f - s.beta1), _mm512_set1_ps(1.0f - s.beta2), _mm512_set1_ps(s.eps),
                 _mm512_set1_ps(s.weight_decay), _mm512_set1_ps(s.inv_bc1),      _mm512_set1_ps(s.inv_bc2),
                 _mm512_set1_ps(0.5f),           _mm512_set1_ps(1.5f),           _mm512_set1_ps(1.0f)};
  int j = 0;
  for (; j + 16 <= n; j += 16) adam16(p + j, g + j, m + j, v + j, 0xFFFF, c, true);
  if (j < n) adam16(p + j, g + j, m + j, v + j, head_mask(n - j), c, false);
}

}  // namespace fastmlp::simd::avx512
