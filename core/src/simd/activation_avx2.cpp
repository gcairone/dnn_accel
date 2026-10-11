// AVX2+FMA activations for the target CPU. The kinds that need a
// transcendental function live here, plus LeakyReLU/ELU-backward; everything else
// (identity, ReLU, and the backward passes that only read the cached output)
// delegates to scalar::, whose plain loops GCC already vectorizes under
// -O3 -mavx2. LeakyReLU's forward and ELU's backward are loops GCC 11 refuses
// to if-convert ("control flow in loop", in every branch-free spelling tried),
// leaving a data-dependent branch at 0.7-2 ns/element against ReLU's ~0.1.
//
// Why this file exists at all: without -ffast-math GCC will not call libmvec,
// so std::exp/std::tanh compile to one scalar libm call per element. Measured
// on bench_breakdown: sigmoid forward ran at 2.8 GB/s (ReLU: ~100), 59% of a
// two_moons training step and 9% of an MNIST one. -ffast-math on just this
// file was the cheaper alternative, but it also licenses reassociation and
// drops NaN/Inf semantics; explicit kernels keep the numerics pinned down and
// testable against scalar:: (tests/cpp/test_activation.cpp).
//
// The approximations are the classic Cephes single-precision ones (exp, log,
// tanh) plus Numerical Recipes' erfc fit for GELU's normal CDF. Max error
// measured against float64 references: exp 8e-8 relative, tanh 1.3e-7
// relative (including near 0, where the polynomial branch takes over). Every
// activation built on them is checked against float64 at a few ulps in
// tests/cpp/test_activation.cpp.
#include <immintrin.h>

#include "fastmlp/simd/activation_kernel.hpp"

namespace fastmlp::simd::avx2 {

namespace {

inline __m256 set1(float v) { return _mm256_set1_ps(v); }

inline __m256 abs_ps(__m256 x) { return _mm256_andnot_ps(set1(-0.0f), x); }
inline __m256 neg_ps(__m256 x) { return _mm256_xor_ps(set1(-0.0f), x); }

// e^x. The polynomial runs on x clamped to where the 2^n scale is a normal
// float; outside that the result saturates to exactly 0 / +inf, not to the
// clamp's value. That matters: silu(-3e38) = z * sigmoid(z) came out as -1.8
// when exp(3e38) was returned as e^88 instead of inf. Between 88.37 and
// ln(FLT_MAX) the result is held at e^88.37 -- every activation here is
// saturated long before that.
inline __m256 exp_ps(__m256 x_in) {
  constexpr float kLo = -87.33654f;  // ln(FLT_MIN)
  __m256 x = _mm256_min_ps(_mm256_max_ps(x_in, set1(kLo)), set1(88.37f));
  // n = round(x / ln2); r = x - n*ln2, with ln2 split hi+lo for precision.
  __m256 n = _mm256_floor_ps(_mm256_fmadd_ps(x, set1(1.44269504088896341f), set1(0.5f)));
  __m256 r = _mm256_fnmadd_ps(n, set1(0.693359375f), x);
  r = _mm256_fnmadd_ps(n, set1(-2.12194440e-4f), r);

  __m256 p = set1(1.9875691500e-4f);
  p = _mm256_fmadd_ps(p, r, set1(1.3981999507e-3f));
  p = _mm256_fmadd_ps(p, r, set1(8.3334519073e-3f));
  p = _mm256_fmadd_ps(p, r, set1(4.1665795894e-2f));
  p = _mm256_fmadd_ps(p, r, set1(1.6666665459e-1f));
  p = _mm256_fmadd_ps(p, r, set1(5.0000001201e-1f));
  p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, set1(1.0f)));

  __m256i pow2n = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  __m256 y = _mm256_mul_ps(p, _mm256_castsi256_ps(pow2n));
  y = _mm256_blendv_ps(y, set1(__builtin_inff()), _mm256_cmp_ps(x_in, set1(88.7228391f), _CMP_GT_OQ));
  return _mm256_andnot_ps(_mm256_cmp_ps(x_in, set1(kLo), _CMP_LT_OQ), y);
}

// ln(x) for positive normal x only (all this file ever passes it is 1+u,
// u in [0,1]).
inline __m256 log_ps(__m256 x) {
  __m256i xi = _mm256_castps_si256(x);
  // x = m * 2^e with m in [0.5, 1)
  __m256 e = _mm256_cvtepi32_ps(
      _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(126)));
  __m256 m = _mm256_castsi256_ps(_mm256_or_si256(
      _mm256_and_si256(xi, _mm256_set1_epi32(0x007fffff)), _mm256_set1_epi32(0x3f000000)));
  // Re-centre m into [sqrt(0.5), sqrt(2)) so m-1 is small.
  __m256 lt = _mm256_cmp_ps(m, set1(0.707106781186547524f), _CMP_LT_OQ);
  e = _mm256_sub_ps(e, _mm256_and_ps(lt, set1(1.0f)));
  m = _mm256_add_ps(m, _mm256_and_ps(lt, m));
  m = _mm256_sub_ps(m, set1(1.0f));

  __m256 z = _mm256_mul_ps(m, m);
  __m256 p = set1(7.0376836292e-2f);
  p = _mm256_fmadd_ps(p, m, set1(-1.1514610310e-1f));
  p = _mm256_fmadd_ps(p, m, set1(1.1676998740e-1f));
  p = _mm256_fmadd_ps(p, m, set1(-1.2420140846e-1f));
  p = _mm256_fmadd_ps(p, m, set1(1.4249322787e-1f));
  p = _mm256_fmadd_ps(p, m, set1(-1.6668057665e-1f));
  p = _mm256_fmadd_ps(p, m, set1(2.0000714765e-1f));
  p = _mm256_fmadd_ps(p, m, set1(-2.4999993993e-1f));
  p = _mm256_fmadd_ps(p, m, set1(3.3333331174e-1f));
  p = _mm256_mul_ps(_mm256_mul_ps(p, m), z);
  p = _mm256_fmadd_ps(e, set1(-2.12194440e-4f), p);
  p = _mm256_fnmadd_ps(set1(0.5f), z, p);
  return _mm256_fmadd_ps(e, set1(0.693359375f), _mm256_add_ps(m, p));
}

// ln(1+u), u >= 0. log(w) * u/(w-1) with w = 1+u cancels the rounding error
// of forming w (Goldberg's trick); w == 1 means u is below half an ulp of 1,
// where ln(1+u) == u to float precision.
inline __m256 log1p_ps(__m256 u) {
  __m256 w = _mm256_add_ps(u, set1(1.0f));
  __m256 d = _mm256_sub_ps(w, set1(1.0f));
  __m256 r = _mm256_mul_ps(log_ps(w), _mm256_div_ps(u, d));
  return _mm256_blendv_ps(r, u, _mm256_cmp_ps(d, _mm256_setzero_ps(), _CMP_EQ_OQ));
}

inline __m256 sigmoid_ps(__m256 z) {
  return _mm256_div_ps(set1(1.0f), _mm256_add_ps(set1(1.0f), exp_ps(neg_ps(z))));
}

// |x| < 0.625: odd polynomial (1 - 2/(e^2x + 1) would cancel catastrophically
// there); otherwise the exp form, with x's sign copied back on.
inline __m256 tanh_ps(__m256 x) {
  __m256 ax = abs_ps(x);
  __m256 z = _mm256_mul_ps(x, x);
  __m256 p = set1(-5.70498872745e-3f);
  p = _mm256_fmadd_ps(p, z, set1(2.06390887954e-2f));
  p = _mm256_fmadd_ps(p, z, set1(-5.37397155531e-2f));
  p = _mm256_fmadd_ps(p, z, set1(1.33314422036e-1f));
  p = _mm256_fmadd_ps(p, z, set1(-3.33332819422e-1f));
  __m256 small = _mm256_fmadd_ps(_mm256_mul_ps(p, z), x, x);

  __m256 e2 = exp_ps(_mm256_add_ps(ax, ax));
  __m256 big = _mm256_sub_ps(set1(1.0f), _mm256_div_ps(set1(2.0f), _mm256_add_ps(e2, set1(1.0f))));
  big = _mm256_or_ps(big, _mm256_and_ps(x, set1(-0.0f)));

  return _mm256_blendv_ps(big, small, _mm256_cmp_ps(ax, set1(0.625f), _CMP_LT_OQ));
}

// Standard normal CDF, Phi(z) = 0.5 erfc(-z / sqrt 2), via Numerical
// Recipes' Chebyshev fit erfc(x) = t exp(-x^2 + P(t)), t = 1/(1 + x/2). It is
// built on erfc rather than erf on purpose: GELU needs Phi for negative z,
// where 1 + erf(x) cancels -- a rational erf with 4e-7 absolute error gave
// gelu(-5.6) the wrong sign. erfc's error is relative instead: ~1e-6 in float
// while x^2 < 10, growing with x^2 beyond (the exponent's rounding) only where
// erfc itself is below 1e-10.
inline __m256 normal_cdf_ps(__m256 z) {
  __m256 x = _mm256_mul_ps(abs_ps(z), set1(0.70710678118654752f));
  __m256 t = _mm256_div_ps(set1(1.0f), _mm256_fmadd_ps(set1(0.5f), x, set1(1.0f)));
  __m256 p = set1(0.17087277f);
  p = _mm256_fmadd_ps(p, t, set1(-0.82215223f));
  p = _mm256_fmadd_ps(p, t, set1(1.48851587f));
  p = _mm256_fmadd_ps(p, t, set1(-1.13520398f));
  p = _mm256_fmadd_ps(p, t, set1(0.27886807f));
  p = _mm256_fmadd_ps(p, t, set1(-0.18628806f));
  p = _mm256_fmadd_ps(p, t, set1(0.09678418f));
  p = _mm256_fmadd_ps(p, t, set1(0.37409196f));
  p = _mm256_fmadd_ps(p, t, set1(1.00002368f));
  p = _mm256_fmadd_ps(p, t, set1(-1.26551223f));
  __m256 half_erfc = _mm256_mul_ps(set1(0.5f), _mm256_mul_ps(t, exp_ps(_mm256_fnmadd_ps(x, x, p))));
  return _mm256_blendv_ps(_mm256_sub_ps(set1(1.0f), half_erfc), half_erfc,
                          _mm256_cmp_ps(z, _mm256_setzero_ps(), _CMP_LT_OQ));
}

constexpr float kInvSqrt2Pi = 0.39894228040143268f;
constexpr float kGeluTanhK0 = 0.79788456080286536f;
constexpr float kGeluTanhK1 = 0.044715f;

// Lanes [0, r) set. The tail goes through the same vector code as the body
// (masked loads read 0 into the dead lanes, which every function here
// tolerates), so an element's result never depends on its position.
inline __m256i tail_mask(int r) {
  return _mm256_cmpgt_epi32(_mm256_set1_epi32(r), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}

template <class F>
inline void map1(const float* x, float* y, int n, F f) {
  int i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, f(_mm256_loadu_ps(x + i)));
  if (i < n) {
    __m256i m = tail_mask(n - i);
    _mm256_maskstore_ps(y + i, m, f(_mm256_maskload_ps(x + i, m)));
  }
}

// dZ[i] = dA[i] * g(z[i]). Each 8-wide chunk is fully read before it is
// written, so dZ aliasing dA is fine.
template <class G>
inline void map_grad(const float* z, const float* dA, float* dZ, int n, G g) {
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(dZ + i, _mm256_mul_ps(_mm256_loadu_ps(dA + i), g(_mm256_loadu_ps(z + i))));
  }
  if (i < n) {
    __m256i m = tail_mask(n - i);
    _mm256_maskstore_ps(dZ + i, m,
                        _mm256_mul_ps(_mm256_maskload_ps(dA + i, m), g(_mm256_maskload_ps(z + i, m))));
  }
}

}  // namespace

constexpr float kLeakySlope = 0.01f;

void activation_forward(ActivationKind kind, const float* z, float* a, int n) {
  switch (kind) {
    case ActivationKind::LeakyReLU:
      map1(z, a, n, [](__m256 v) {
        return _mm256_blendv_ps(_mm256_mul_ps(v, set1(kLeakySlope)), v,
                                _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GT_OQ));
      });
      break;
    case ActivationKind::Sigmoid:
      map1(z, a, n, [](__m256 v) { return sigmoid_ps(v); });
      break;
    case ActivationKind::Tanh:
      map1(z, a, n, [](__m256 v) { return tanh_ps(v); });
      break;
    case ActivationKind::ELU:
      map1(z, a, n, [](__m256 v) {
        __m256 neg = _mm256_sub_ps(exp_ps(v), set1(1.0f));
        return _mm256_blendv_ps(neg, v, _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GT_OQ));
      });
      break;
    case ActivationKind::SiLU:
      map1(z, a, n, [](__m256 v) { return _mm256_mul_ps(v, sigmoid_ps(v)); });
      break;
    case ActivationKind::GELU:
      map1(z, a, n, [](__m256 v) { return _mm256_mul_ps(v, normal_cdf_ps(v)); });
      break;
    case ActivationKind::GELUTanh:
      // 0.5 z (1 + tanh u) == z * sigmoid(2u): same function, but 1 + tanh u
      // cancels to 0 for negative z where sigmoid keeps its digits.
      map1(z, a, n, [](__m256 v) {
        __m256 v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
        __m256 u2 = _mm256_mul_ps(set1(2.0f * kGeluTanhK0), _mm256_fmadd_ps(set1(kGeluTanhK1), v3, v));
        return _mm256_mul_ps(v, sigmoid_ps(u2));
      });
      break;
    case ActivationKind::Softplus:
      map1(z, a, n, [](__m256 v) {
        return _mm256_add_ps(_mm256_max_ps(v, _mm256_setzero_ps()),
                             log1p_ps(exp_ps(neg_ps(abs_ps(v)))));
      });
      break;
    default:
      scalar::activation_forward(kind, z, a, n);
      break;
  }
}

void activation_backward(ActivationKind kind, const float* z, const float* a,
                          const float* dA, float* dZ, int n) {
  switch (kind) {
    case ActivationKind::LeakyReLU:
      map_grad(z, dA, dZ, n, [](__m256 v) {
        return _mm256_blendv_ps(set1(kLeakySlope), set1(1.0f),
                                _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GT_OQ));
      });
      break;
    case ActivationKind::ELU: {
      // Same if-conversion failure as LeakyReLU in the scalar loop. Needs a
      // (f' = a + 1 for z <= 0), so it is written out rather than map_grad.
      const __m256 one = set1(1.0f);
      auto g = [&](__m256 vz, __m256 va) {
        return _mm256_blendv_ps(_mm256_add_ps(va, one), one,
                                _mm256_cmp_ps(vz, _mm256_setzero_ps(), _CMP_GT_OQ));
      };
      int i = 0;
      for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dZ + i, _mm256_mul_ps(_mm256_loadu_ps(dA + i),
                                               g(_mm256_loadu_ps(z + i), _mm256_loadu_ps(a + i))));
      }
      if (i < n) {
        __m256i m = tail_mask(n - i);
        _mm256_maskstore_ps(dZ + i, m,
                            _mm256_mul_ps(_mm256_maskload_ps(dA + i, m),
                                          g(_mm256_maskload_ps(z + i, m), _mm256_maskload_ps(a + i, m))));
      }
      break;
    }
    case ActivationKind::SiLU:
      map_grad(z, dA, dZ, n, [](__m256 v) {
        __m256 s = sigmoid_ps(v);
        // s * (1 + z * (1 - s))
        return _mm256_mul_ps(s, _mm256_fmadd_ps(v, _mm256_sub_ps(set1(1.0f), s), set1(1.0f)));
      });
      break;
    case ActivationKind::GELU:
      map_grad(z, dA, dZ, n, [](__m256 v) {
        __m256 cdf = normal_cdf_ps(v);
        __m256 pdf = _mm256_mul_ps(set1(kInvSqrt2Pi),
                                   exp_ps(_mm256_mul_ps(set1(-0.5f), _mm256_mul_ps(v, v))));
        return _mm256_fmadd_ps(v, pdf, cdf);
      });
      break;
    case ActivationKind::GELUTanh:
      // With s = sigmoid(2u): f' = s + 2 z s (1-s) du. The second term is 0 to
      // float precision once |z| > ~10, so it runs on z clamped to +-20 --
      // otherwise z*du overflows and 0 * inf turns into NaN at huge |z|.
      map_grad(z, dA, dZ, n, [](__m256 v) {
        __m256 v2 = _mm256_mul_ps(v, v);
        __m256 u2 = _mm256_mul_ps(set1(2.0f * kGeluTanhK0),
                                  _mm256_fmadd_ps(set1(kGeluTanhK1), _mm256_mul_ps(v2, v), v));
        __m256 s = sigmoid_ps(u2);
        __m256 c = _mm256_min_ps(_mm256_max_ps(v, set1(-20.0f)), set1(20.0f));
        __m256 c_du = _mm256_mul_ps(
            _mm256_mul_ps(set1(kGeluTanhK0), c),
            _mm256_fmadd_ps(set1(3.0f * kGeluTanhK1), _mm256_mul_ps(c, c), set1(1.0f)));
        __m256 s1s = _mm256_mul_ps(s, _mm256_sub_ps(set1(1.0f), s));
        return _mm256_fmadd_ps(_mm256_mul_ps(set1(2.0f), s1s), c_du, s);
      });
      break;
    case ActivationKind::Softplus:
      map_grad(z, dA, dZ, n, [](__m256 v) { return sigmoid_ps(v); });
      break;
    default:
      scalar::activation_backward(kind, z, a, dA, dZ, n);
      break;
  }
}

}  // namespace fastmlp::simd::avx2
