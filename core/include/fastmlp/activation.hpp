#pragma once
// Elementwise activation over a contiguous buffer of n = batch*out_features
// floats. The cheap kinds (identity, the ReLU family, and every backward that
// only needs the cached output) are plain loops the compiler vectorizes on its
// own. The transcendental ones are not: without -ffast-math GCC calls scalar
// expf/tanhf per element, which measured 2.8 GB/s against ReLU's ~100 and was
// 59% of a two_moons training step -- so those go through hand-vectorized
// AVX2 math (simd/activation_kernel.hpp), with the scalar std:: version kept
// as the numeric ground truth, same split as the GEMM backends.

#include <optional>
#include <string_view>

namespace fastmlp {

// Order is not part of any file format (.mlpf stores names), so new kinds can
// go anywhere; keep activation_name()/activation_from_name() in sync.
enum class ActivationKind {
  Identity,
  ReLU,
  Sigmoid,
  Tanh,
  LeakyReLU,  // negative slope fixed at 0.01 (torch's default)
  ELU,        // alpha fixed at 1
  SiLU,       // z * sigmoid(z), a.k.a. swish
  GELU,       // exact: 0.5 z (1 + erf(z / sqrt 2)), torch's default
  GELUTanh,   // tanh approximation, torch's approximate="tanh"
  Softplus,   // log(1 + e^z), beta 1
};

// The names the Python API and .mlpf headers use ("relu", "gelu_tanh", ...).
const char* activation_name(ActivationKind kind);
std::optional<ActivationKind> activation_from_name(std::string_view name);
// Comma-separated list of every valid name, for error messages.
const char* activation_names_list();

// a[i] = f(z[i])
void activation_forward(ActivationKind kind, const float* z, float* a, int n);

// dZ[i] = dA[i] * f'(z[i]). Uses whichever of z/a is cheaper per kind
// (relu: sign of z; sigmoid: a*(1-a); tanh: 1-a^2; elu: a+1) -- both are
// always passed in since the caller (Linear) already has both cached.
// dZ may alias dA.
void activation_backward(ActivationKind kind, const float* z, const float* a,
                          const float* dA, float* dZ, int n);

}  // namespace fastmlp
