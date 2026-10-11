// Portable std:: implementation -- the numeric ground truth the AVX2 version
// is checked against (tests/cpp/test_activation.cpp). Written for clarity,
// not speed: the transcendental loops here call scalar libm per element.
#include <cmath>

#include "fastmlp/simd/activation_kernel.hpp"

namespace fastmlp::simd::scalar {

namespace {
constexpr float kLeakySlope = 0.01f;
constexpr float kInvSqrt2 = 0.70710678118654752f;
constexpr float kInvSqrt2Pi = 0.39894228040143268f;
constexpr float kGeluTanhK0 = 0.79788456080286536f;  // sqrt(2/pi)
constexpr float kGeluTanhK1 = 0.044715f;

inline float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }
}  // namespace

void activation_forward(ActivationKind kind, const float* z, float* a, int n) {
  switch (kind) {
    case ActivationKind::Identity:
      for (int i = 0; i < n; i++) a[i] = z[i];
      break;
    case ActivationKind::ReLU:
      for (int i = 0; i < n; i++) a[i] = z[i] > 0.0f ? z[i] : 0.0f;
      break;
    case ActivationKind::Sigmoid:
      for (int i = 0; i < n; i++) a[i] = sigmoid(z[i]);
      break;
    case ActivationKind::Tanh:
      for (int i = 0; i < n; i++) a[i] = std::tanh(z[i]);
      break;
    case ActivationKind::LeakyReLU:
      for (int i = 0; i < n; i++) a[i] = z[i] > 0.0f ? z[i] : kLeakySlope * z[i];
      break;
    case ActivationKind::ELU:
      for (int i = 0; i < n; i++) a[i] = z[i] > 0.0f ? z[i] : std::expm1(z[i]);
      break;
    case ActivationKind::SiLU:
      for (int i = 0; i < n; i++) a[i] = z[i] * sigmoid(z[i]);
      break;
    case ActivationKind::GELU:
      for (int i = 0; i < n; i++) a[i] = 0.5f * z[i] * (1.0f + std::erf(z[i] * kInvSqrt2));
      break;
    case ActivationKind::GELUTanh:
      // 0.5 z (1 + tanh u) == z * sigmoid(2u); the tanh form cancels to 0 for
      // negative z.
      for (int i = 0; i < n; i++) {
        float u = kGeluTanhK0 * (z[i] + kGeluTanhK1 * z[i] * z[i] * z[i]);
        a[i] = z[i] * sigmoid(2.0f * u);
      }
      break;
    case ActivationKind::Softplus:
      // max(z,0) + log1p(e^-|z|): no overflow for large z, no lost digits for
      // very negative z.
      for (int i = 0; i < n; i++) {
        a[i] = std::fmax(z[i], 0.0f) + std::log1p(std::exp(-std::fabs(z[i])));
      }
      break;
  }
}

void activation_backward(ActivationKind kind, const float* z, const float* a,
                          const float* dA, float* dZ, int n) {
  switch (kind) {
    case ActivationKind::Identity:
      for (int i = 0; i < n; i++) dZ[i] = dA[i];
      break;
    case ActivationKind::ReLU:
      // Subgradient convention at the kink (z==0): derivative 0.
      for (int i = 0; i < n; i++) dZ[i] = z[i] > 0.0f ? dA[i] : 0.0f;
      break;
    case ActivationKind::Sigmoid:
      for (int i = 0; i < n; i++) dZ[i] = dA[i] * a[i] * (1.0f - a[i]);
      break;
    case ActivationKind::Tanh:
      for (int i = 0; i < n; i++) dZ[i] = dA[i] * (1.0f - a[i] * a[i]);
      break;
    case ActivationKind::LeakyReLU:
      // At z==0 the slope is the negative one, matching torch.
      for (int i = 0; i < n; i++) dZ[i] = z[i] > 0.0f ? dA[i] : kLeakySlope * dA[i];
      break;
    case ActivationKind::ELU:
      // For z<=0, a = e^z - 1, so f'(z) = e^z = a + 1.
      for (int i = 0; i < n; i++) dZ[i] = z[i] > 0.0f ? dA[i] : dA[i] * (a[i] + 1.0f);
      break;
    case ActivationKind::SiLU:
      for (int i = 0; i < n; i++) {
        float s = sigmoid(z[i]);
        dZ[i] = dA[i] * s * (1.0f + z[i] * (1.0f - s));
      }
      break;
    case ActivationKind::GELU:
      // Phi(z) + z * phi(z)
      for (int i = 0; i < n; i++) {
        float cdf = 0.5f * (1.0f + std::erf(z[i] * kInvSqrt2));
        float pdf = kInvSqrt2Pi * std::exp(-0.5f * z[i] * z[i]);
        dZ[i] = dA[i] * (cdf + z[i] * pdf);
      }
      break;
    case ActivationKind::GELUTanh:
      // s = sigmoid(2u): f' = s + 2 z s (1-s) du. The second term is 0 in
      // float beyond |z| ~ 10; clamping z there keeps z*du from overflowing
      // into 0 * inf = NaN.
      for (int i = 0; i < n; i++) {
        float s = sigmoid(2.0f * kGeluTanhK0 * (z[i] + kGeluTanhK1 * z[i] * z[i] * z[i]));
        float c = std::fmin(std::fmax(z[i], -20.0f), 20.0f);
        float c_du = kGeluTanhK0 * c * (1.0f + 3.0f * kGeluTanhK1 * c * c);
        dZ[i] = dA[i] * (s + 2.0f * s * (1.0f - s) * c_du);
      }
      break;
    case ActivationKind::Softplus:
      for (int i = 0; i < n; i++) dZ[i] = dA[i] * sigmoid(z[i]);
      break;
  }
}

}  // namespace fastmlp::simd::scalar
