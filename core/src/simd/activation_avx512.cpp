// AVX-512 activations for the Emerald Rapids target.
//
// Same approximations as the avx2 kernels (Cephes exp/log/tanh, Numerical
// Recipes' erfc for GELU's normal CDF) and the same accuracy contract
// (tests/cpp/test_activation.cpp: a few ulps against float64), on zmm with
// what this ISA adds:
//   - vscalefps for exp's 2^n: one op instead of cvt + integer add + shift.
//   - vrcp14ps + one Newton step instead of vdivps, which has a reciprocal
//     throughput of 10 cycles per zmm on this core (results/emr/compute.txt
//     section 4) against ~2 for the pair. Relative error after the step is
//     ~2^-28 plus the step's own rounding: 1-2 ulp, inside the contract.
//     Every denominator it is used on is kept finite (Newton on an infinite
//     d gives inf * 0 = NaN); log1p's, which can be 0, keeps the division.
//   - k-masks: blends and the tail cost no extra registers, and the tail
//     runs the same code as the body.
// Identity and ReLU are written out too: ReLU is the most common activation
// and the compiler's version of the scalar loop is not branch-free for the
// backward.
//
// No STL or other shared inline code (see gemm_kernel_avx512.cpp).
#include <immintrin.h>

#include "fastmlp/simd/activation_kernel.hpp"

namespace fastmlp::simd::avx512 {

namespace {

constexpr int kVW = 16;

inline __m512 set1(float v) { return _mm512_set1_ps(v); }
inline __m512 neg_ps(__m512 x) { return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(x), _mm512_set1_epi32((int)0x80000000))); }
inline __mmask16 head_mask(int n) { return (__mmask16)((n >= 16) ? 0xFFFF : ((1u << n) - 1u)); }

// 1/d for finite d != 0, ~1 ulp.
inline __m512 rcp_nr(__m512 d) {
  const __m512 r = _mm512_rcp14_ps(d);
  return _mm512_fmadd_ps(r, _mm512_fnmadd_ps(d, r, set1(1.0f)), r);
}

// e^x, +inf above ln(FLT_MAX), denormal then 0 below ln(FLT_MIN).
// vscalefps does the saturation itself: with x clamped to [-104, 89], n stays
// in [-150, 128] (finite n*ln2) and p * 2^n overflows to inf / underflows
// through the denormals to 0 exactly as e^x does. (The avx2 version needs two
// compares and two blends for that, and returns 0 for the denormal range.)
inline __m512 exp_ps(__m512 x_in) {
  const __m512 x = _mm512_min_ps(_mm512_max_ps(x_in, set1(-104.0f)), set1(89.0f));
  // n = round(x / ln2); r = x - n*ln2, ln2 split hi+lo for precision
  const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, set1(1.44269504088896341f)),
                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, set1(0.693359375f), x);
  r = _mm512_fnmadd_ps(n, set1(-2.12194440e-4f), r);

  __m512 p = set1(1.9875691500e-4f);
  p = _mm512_fmadd_ps(p, r, set1(1.3981999507e-3f));
  p = _mm512_fmadd_ps(p, r, set1(8.3334519073e-3f));
  p = _mm512_fmadd_ps(p, r, set1(4.1665795894e-2f));
  p = _mm512_fmadd_ps(p, r, set1(1.6666665459e-1f));
  p = _mm512_fmadd_ps(p, r, set1(5.0000001201e-1f));
  p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, set1(1.0f)));

  return _mm512_scalef_ps(p, n);
}

// e^x for x <= 88 only, never inf: the form every rcp_nr denominator uses.
inline __m512 exp_finite_ps(__m512 x) { return exp_ps(_mm512_min_ps(x, set1(88.0f))); }

// ln(x) for positive normal x (all this file passes it is 1+u, u in [0,1]).
// vgetexp/vgetmant split x = m * 2^e with m in [1, 2) in two ops.
inline __m512 log_ps(__m512 x) {
  // m in [0.5, 1) like the Cephes reference: getmant gives [1, 2), so halve
  // it and bump the exponent.
  __m512 e = _mm512_add_ps(_mm512_getexp_ps(x), set1(1.0f));
  __m512 m = _mm512_mul_ps(_mm512_getmant_ps(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero), set1(0.5f));
  // re-centre m into [sqrt(0.5), sqrt(2)) so m-1 is small
  const __mmask16 lt = _mm512_cmp_ps_mask(m, set1(0.707106781186547524f), _CMP_LT_OQ);
  e = _mm512_mask_sub_ps(e, lt, e, set1(1.0f));
  m = _mm512_mask_add_ps(m, lt, m, m);
  m = _mm512_sub_ps(m, set1(1.0f));

  const __m512 z = _mm512_mul_ps(m, m);
  __m512 p = set1(7.0376836292e-2f);
  p = _mm512_fmadd_ps(p, m, set1(-1.1514610310e-1f));
  p = _mm512_fmadd_ps(p, m, set1(1.1676998740e-1f));
  p = _mm512_fmadd_ps(p, m, set1(-1.2420140846e-1f));
  p = _mm512_fmadd_ps(p, m, set1(1.4249322787e-1f));
  p = _mm512_fmadd_ps(p, m, set1(-1.6668057665e-1f));
  p = _mm512_fmadd_ps(p, m, set1(2.0000714765e-1f));
  p = _mm512_fmadd_ps(p, m, set1(-2.4999993993e-1f));
  p = _mm512_fmadd_ps(p, m, set1(3.3333331174e-1f));
  p = _mm512_mul_ps(_mm512_mul_ps(p, m), z);
  p = _mm512_fmadd_ps(e, set1(-2.12194440e-4f), p);
  p = _mm512_fnmadd_ps(set1(0.5f), z, p);
  return _mm512_fmadd_ps(e, set1(0.693359375f), _mm512_add_ps(m, p));
}

// ln(1+u), u >= 0, Goldberg's trick (see the avx2 version). d can be 0, so
// this keeps the exact division.
inline __m512 log1p_ps(__m512 u) {
  const __m512 w = _mm512_add_ps(u, set1(1.0f));
  const __m512 d = _mm512_sub_ps(w, set1(1.0f));
  const __m512 r = _mm512_mul_ps(log_ps(w), _mm512_div_ps(u, d));
  return _mm512_mask_mov_ps(r, _mm512_cmp_ps_mask(d, _mm512_setzero_ps(), _CMP_EQ_OQ), u);
}

// 1 / (1 + e^-z). The denominator is capped at 1 + e^88 for rcp_nr, and for
// z < -88 (true value < 6e-39) the result is exactly 0 -- not the capped
// ~6e-39, which silu = z * sigmoid(z) would turn into -1.8 at z = -3e38.
// NLT_UQ keeps NaN inputs on the NaN path.
inline __m512 sigmoid_ps(__m512 z) {
  const __m512 s = rcp_nr(_mm512_add_ps(set1(1.0f), exp_finite_ps(neg_ps(z))));
  return _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(z, set1(-88.0f), _CMP_NLT_UQ), s);
}

// |x| < 0.625: odd polynomial; otherwise 1 - 2/(e^2|x| + 1) with x's sign.
inline __m512 tanh_ps(__m512 x) {
  const __m512 ax = _mm512_abs_ps(x);
  const __m512 z = _mm512_mul_ps(x, x);
  __m512 p = set1(-5.70498872745e-3f);
  p = _mm512_fmadd_ps(p, z, set1(2.06390887954e-2f));
  p = _mm512_fmadd_ps(p, z, set1(-5.37397155531e-2f));
  p = _mm512_fmadd_ps(p, z, set1(1.33314422036e-1f));
  p = _mm512_fmadd_ps(p, z, set1(-3.33332819422e-1f));
  const __m512 small = _mm512_fmadd_ps(_mm512_mul_ps(p, z), x, x);

  const __m512 e2 = exp_finite_ps(_mm512_add_ps(ax, ax));
  __m512 big = _mm512_fnmadd_ps(set1(2.0f), rcp_nr(_mm512_add_ps(e2, set1(1.0f))), set1(1.0f));
  big = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(big), _mm512_castps_si512(x),
                                                      _mm512_set1_epi32((int)0x80000000), 0xF8));  // big | (x & sign)
  return _mm512_mask_mov_ps(big, _mm512_cmp_ps_mask(ax, set1(0.625f), _CMP_LT_OQ), small);
}

// Phi(z) = 0.5 erfc(-z / sqrt 2) via Numerical Recipes' erfc fit (see avx2).
// t's denominator is >= 1, finite for finite z.
inline __m512 normal_cdf_ps(__m512 z) {
  const __m512 x = _mm512_mul_ps(_mm512_abs_ps(z), set1(0.70710678118654752f));
  const __m512 t = rcp_nr(_mm512_fmadd_ps(set1(0.5f), x, set1(1.0f)));
  __m512 p = set1(0.17087277f);
  p = _mm512_fmadd_ps(p, t, set1(-0.82215223f));
  p = _mm512_fmadd_ps(p, t, set1(1.48851587f));
  p = _mm512_fmadd_ps(p, t, set1(-1.13520398f));
  p = _mm512_fmadd_ps(p, t, set1(0.27886807f));
  p = _mm512_fmadd_ps(p, t, set1(-0.18628806f));
  p = _mm512_fmadd_ps(p, t, set1(0.09678418f));
  p = _mm512_fmadd_ps(p, t, set1(0.37409196f));
  p = _mm512_fmadd_ps(p, t, set1(1.00002368f));
  p = _mm512_fmadd_ps(p, t, set1(-1.26551223f));
  const __m512 half_erfc = _mm512_mul_ps(set1(0.5f), _mm512_mul_ps(t, exp_ps(_mm512_fnmadd_ps(x, x, p))));
  return _mm512_mask_mov_ps(_mm512_sub_ps(set1(1.0f), half_erfc),
                            _mm512_cmp_ps_mask(z, _mm512_setzero_ps(), _CMP_LT_OQ), half_erfc);
}

constexpr float kInvSqrt2Pi = 0.39894228040143268f;
constexpr float kGeluTanhK0 = 0.79788456080286536f;
constexpr float kGeluTanhK1 = 0.044715f;
constexpr float kLeakySlope = 0.01f;

#ifndef FASTMLP_ACT_UNROLL
#define FASTMLP_ACT_UNROLL 2
#endif
constexpr int kU = FASTMLP_ACT_UNROLL;

// kU vectors per iteration: the transcendental bodies are long dependency
// chains (~90 cycles for sigmoid), and independent ones adjacent in program
// order let the scheduler fill both FMA ports.
template <class F>
inline void map1(const float* x, float* y, int n, F f) {
  int i = 0;
  for (; i + kU * kVW <= n; i += kU * kVW) {
    __m512 v[kU];
#pragma GCC unroll 8
    for (int u = 0; u < kU; u++) v[u] = f(_mm512_loadu_ps(x + i + u * kVW));
#pragma GCC unroll 8
    for (int u = 0; u < kU; u++) _mm512_storeu_ps(y + i + u * kVW, v[u]);
  }
  for (; i < n; i += kVW) {
    const __mmask16 m = head_mask(n - i);
    _mm512_mask_storeu_ps(y + i, m, f(_mm512_maskz_loadu_ps(m, x + i)));
  }
}

// dZ[i] = dA[i] * g(z[i], a[i]). Each chunk is fully read before it is
// written, so dZ aliasing dA is fine.
template <class G>
inline void map_grad(const float* z, const float* a, const float* dA, float* dZ, int n, G g) {
  int i = 0;
  for (; i + kU * kVW <= n; i += kU * kVW) {
    __m512 v[kU];
#pragma GCC unroll 8
    for (int u = 0; u < kU; u++) {
      const int o = i + u * kVW;
      v[u] = _mm512_mul_ps(_mm512_loadu_ps(dA + o), g(_mm512_loadu_ps(z + o), _mm512_loadu_ps(a + o)));
    }
#pragma GCC unroll 8
    for (int u = 0; u < kU; u++) _mm512_storeu_ps(dZ + i + u * kVW, v[u]);
  }
  for (; i < n; i += kVW) {
    const __mmask16 m = head_mask(n - i);
    _mm512_mask_storeu_ps(dZ + i, m,
                          _mm512_mul_ps(_mm512_maskz_loadu_ps(m, dA + i),
                                        g(_mm512_maskz_loadu_ps(m, z + i), _mm512_maskz_loadu_ps(m, a + i))));
  }
}

inline __mmask16 gt0(__m512 v) { return _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GT_OQ); }

}  // namespace

void activation_forward(ActivationKind kind, const float* z, float* a, int n) {
  switch (kind) {
    case ActivationKind::Identity:
      if (a != z) map1(z, a, n, [](__m512 v) { return v; });
      break;
    case ActivationKind::ReLU:
      map1(z, a, n, [](__m512 v) { return _mm512_max_ps(v, _mm512_setzero_ps()); });
      break;
    case ActivationKind::LeakyReLU:
      map1(z, a, n, [](__m512 v) { return _mm512_mask_mov_ps(_mm512_mul_ps(v, set1(kLeakySlope)), gt0(v), v); });
      break;
    case ActivationKind::Sigmoid:
      map1(z, a, n, [](__m512 v) { return sigmoid_ps(v); });
      break;
    case ActivationKind::Tanh:
      map1(z, a, n, [](__m512 v) { return tanh_ps(v); });
      break;
    case ActivationKind::ELU:
      map1(z, a, n, [](__m512 v) { return _mm512_mask_mov_ps(_mm512_sub_ps(exp_ps(v), set1(1.0f)), gt0(v), v); });
      break;
    case ActivationKind::SiLU:
      map1(z, a, n, [](__m512 v) { return _mm512_mul_ps(v, sigmoid_ps(v)); });
      break;
    case ActivationKind::GELU:
      map1(z, a, n, [](__m512 v) { return _mm512_mul_ps(v, normal_cdf_ps(v)); });
      break;
    case ActivationKind::GELUTanh:
      // z * sigmoid(2u) == 0.5 z (1 + tanh u), without 1 + tanh u cancelling
      map1(z, a, n, [](__m512 v) {
        const __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
        const __m512 u2 = _mm512_mul_ps(set1(2.0f * kGeluTanhK0), _mm512_fmadd_ps(set1(kGeluTanhK1), v3, v));
        return _mm512_mul_ps(v, sigmoid_ps(u2));
      });
      break;
    case ActivationKind::Softplus:
      map1(z, a, n, [](__m512 v) {
        return _mm512_add_ps(_mm512_max_ps(v, _mm512_setzero_ps()), log1p_ps(exp_ps(neg_ps(_mm512_abs_ps(v)))));
      });
      break;
    default:
      scalar::activation_forward(kind, z, a, n);
      break;
  }
}

void activation_backward(ActivationKind kind, const float* z, const float* a, const float* dA, float* dZ, int n) {
  switch (kind) {
    case ActivationKind::Identity:
      if (dZ != dA) map1(dA, dZ, n, [](__m512 v) { return v; });
      break;
    case ActivationKind::ReLU: {
      // dZ = z > 0 ? dA : 0, one masked move per vector
      int i = 0;
      for (; i < n; i += kVW) {
        const __mmask16 m = head_mask(n - i);
        const __mmask16 pos = gt0(_mm512_maskz_loadu_ps(m, z + i));
        _mm512_mask_storeu_ps(dZ + i, m, _mm512_maskz_loadu_ps(pos & m, dA + i));
      }
      break;
    }
    case ActivationKind::LeakyReLU:
      map_grad(z, a, dA, dZ, n, [](__m512 vz, __m512) { return _mm512_mask_mov_ps(set1(kLeakySlope), gt0(vz), set1(1.0f)); });
      break;
    case ActivationKind::Sigmoid:
      map_grad(z, a, dA, dZ, n, [](__m512, __m512 va) { return _mm512_mul_ps(va, _mm512_sub_ps(set1(1.0f), va)); });
      break;
    case ActivationKind::Tanh:
      map_grad(z, a, dA, dZ, n, [](__m512, __m512 va) { return _mm512_fnmadd_ps(va, va, set1(1.0f)); });
      break;
    case ActivationKind::ELU:
      // f' = a + 1 for z <= 0
      map_grad(z, a, dA, dZ, n, [](__m512 vz, __m512 va) {
        return _mm512_mask_mov_ps(_mm512_add_ps(va, set1(1.0f)), gt0(vz), set1(1.0f));
      });
      break;
    case ActivationKind::SiLU:
      map_grad(z, a, dA, dZ, n, [](__m512 v, __m512) {
        const __m512 s = sigmoid_ps(v);
        return _mm512_mul_ps(s, _mm512_fmadd_ps(v, _mm512_sub_ps(set1(1.0f), s), set1(1.0f)));  // s (1 + z (1-s))
      });
      break;
    case ActivationKind::GELU:
      map_grad(z, a, dA, dZ, n, [](__m512 v, __m512) {
        const __m512 cdf = normal_cdf_ps(v);
        const __m512 pdf = _mm512_mul_ps(set1(kInvSqrt2Pi), exp_ps(_mm512_mul_ps(set1(-0.5f), _mm512_mul_ps(v, v))));
        return _mm512_fmadd_ps(v, pdf, cdf);
      });
      break;
    case ActivationKind::GELUTanh:
      // f' = s + 2 z s (1-s) du with s = sigmoid(2u), on z clamped to +-20
      // for the second term (see the avx2 version)
      map_grad(z, a, dA, dZ, n, [](__m512 v, __m512) {
        const __m512 v2 = _mm512_mul_ps(v, v);
        const __m512 u2 =
            _mm512_mul_ps(set1(2.0f * kGeluTanhK0), _mm512_fmadd_ps(set1(kGeluTanhK1), _mm512_mul_ps(v2, v), v));
        const __m512 s = sigmoid_ps(u2);
        const __m512 c = _mm512_min_ps(_mm512_max_ps(v, set1(-20.0f)), set1(20.0f));
        const __m512 c_du = _mm512_mul_ps(_mm512_mul_ps(set1(kGeluTanhK0), c),
                                          _mm512_fmadd_ps(set1(3.0f * kGeluTanhK1), _mm512_mul_ps(c, c), set1(1.0f)));
        const __m512 s1s = _mm512_mul_ps(s, _mm512_sub_ps(set1(1.0f), s));
        return _mm512_fmadd_ps(_mm512_mul_ps(set1(2.0f), s1s), c_du, s);
      });
      break;
    case ActivationKind::Softplus:
      map_grad(z, a, dA, dZ, n, [](__m512 v, __m512) { return sigmoid_ps(v); });
      break;
    default:
      scalar::activation_backward(kind, z, a, dA, dZ, n);
      break;
  }
}

}  // namespace fastmlp::simd::avx512
