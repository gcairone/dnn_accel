#include "fastmlp/activation.hpp"

#include <cmath>

namespace fastmlp {

void activation_forward(ActivationKind kind, const float* z, float* a, int n) {
  switch (kind) {
    case ActivationKind::Identity:
      for (int i = 0; i < n; i++) a[i] = z[i];
      break;
    case ActivationKind::ReLU:
      for (int i = 0; i < n; i++) a[i] = z[i] > 0.0f ? z[i] : 0.0f;
      break;
    case ActivationKind::Sigmoid:
      for (int i = 0; i < n; i++) a[i] = 1.0f / (1.0f + std::exp(-z[i]));
      break;
    case ActivationKind::Tanh:
      for (int i = 0; i < n; i++) a[i] = std::tanh(z[i]);
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
  }
}

}  // namespace fastmlp
