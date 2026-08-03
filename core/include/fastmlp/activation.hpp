#pragma once
// Elementwise activation over a contiguous buffer of n = batch*out_features
// floats. Not a GEMM-hot bottleneck relative to the microkernel, so plain
// scalar loops (compiler auto-vectorizes under -O3) are sufficient -- no
// hand intrinsics needed here.

namespace fastmlp {

enum class ActivationKind { Identity, ReLU, Sigmoid, Tanh };

// a[i] = f(z[i])
void activation_forward(ActivationKind kind, const float* z, float* a, int n);

// dZ[i] = dA[i] * f'(z[i]). Uses whichever of z/a is cheaper per kind
// (relu: sign of z; sigmoid: a*(1-a); tanh: 1-a^2) -- both are always
// passed in since the caller (Linear) already has both cached.
void activation_backward(ActivationKind kind, const float* z, const float* a,
                          const float* dA, float* dZ, int n);

}  // namespace fastmlp
