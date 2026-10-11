#include "fastmlp/activation.hpp"

#include "fastmlp/simd/activation_kernel.hpp"

namespace fastmlp {

namespace {
struct NamedKind {
  const char* name;
  ActivationKind kind;
};
constexpr NamedKind kNames[] = {
    {"identity", ActivationKind::Identity},     {"relu", ActivationKind::ReLU},
    {"sigmoid", ActivationKind::Sigmoid},       {"tanh", ActivationKind::Tanh},
    {"leaky_relu", ActivationKind::LeakyReLU},  {"elu", ActivationKind::ELU},
    {"silu", ActivationKind::SiLU},             {"gelu", ActivationKind::GELU},
    {"gelu_tanh", ActivationKind::GELUTanh},    {"softplus", ActivationKind::Softplus},
};
}  // namespace

const char* activation_name(ActivationKind kind) {
  for (const auto& e : kNames) {
    if (e.kind == kind) return e.name;
  }
  return "?";
}

std::optional<ActivationKind> activation_from_name(std::string_view name) {
  for (const auto& e : kNames) {
    if (name == e.name) return e.kind;
  }
  return std::nullopt;
}

const char* activation_names_list() {
  return "identity, relu, sigmoid, tanh, leaky_relu, elu, silu, gelu, gelu_tanh, softplus";
}

void activation_forward(ActivationKind kind, const float* z, float* a, int n) {
#if defined(FASTMLP_BACKEND_AVX2)
  simd::avx2::activation_forward(kind, z, a, n);
#else
  simd::scalar::activation_forward(kind, z, a, n);
#endif
}

void activation_backward(ActivationKind kind, const float* z, const float* a,
                          const float* dA, float* dZ, int n) {
#if defined(FASTMLP_BACKEND_AVX2)
  simd::avx2::activation_backward(kind, z, a, dA, dZ, n);
#else
  simd::scalar::activation_backward(kind, z, a, dA, dZ, n);
#endif
}

}  // namespace fastmlp
