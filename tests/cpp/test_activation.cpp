#include "fastmlp/activation.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "fastmlp/simd/activation_kernel.hpp"
#include "test_harness.hpp"

using fastmlp::ActivationKind;
using fastmlp::activation_backward;
using fastmlp::activation_forward;
namespace simd = fastmlp::simd;

namespace {

const ActivationKind kAllKinds[] = {
    ActivationKind::Identity, ActivationKind::ReLU,     ActivationKind::Sigmoid,
    ActivationKind::Tanh,     ActivationKind::LeakyReLU, ActivationKind::ELU,
    ActivationKind::SiLU,     ActivationKind::GELU,     ActivationKind::GELUTanh,
    ActivationKind::Softplus,
};

// float64 references, written from the textbook definitions independently of
// either backend.
double ref_f(ActivationKind k, double z) {
  const double s = 1.0 / (1.0 + std::exp(-z));
  switch (k) {
    case ActivationKind::Identity: return z;
    case ActivationKind::ReLU: return z > 0 ? z : 0.0;
    case ActivationKind::Sigmoid: return s;
    case ActivationKind::Tanh: return std::tanh(z);
    case ActivationKind::LeakyReLU: return z > 0 ? z : 0.01 * z;
    case ActivationKind::ELU: return z > 0 ? z : std::expm1(z);
    case ActivationKind::SiLU: return z * s;
    case ActivationKind::GELU: return 0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0)));
    case ActivationKind::GELUTanh:
      return 0.5 * z * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (z + 0.044715 * z * z * z)));
    case ActivationKind::Softplus: return std::fmax(z, 0.0) + std::log1p(std::exp(-std::fabs(z)));
  }
  return NAN;
}

double ref_df(ActivationKind k, double z) {
  const double s = 1.0 / (1.0 + std::exp(-z));
  switch (k) {
    case ActivationKind::Identity: return 1.0;
    case ActivationKind::ReLU: return z > 0 ? 1.0 : 0.0;
    case ActivationKind::Sigmoid: return s * (1.0 - s);
    case ActivationKind::Tanh: { double t = std::tanh(z); return 1.0 - t * t; }
    case ActivationKind::LeakyReLU: return z > 0 ? 1.0 : 0.01;
    case ActivationKind::ELU: return z > 0 ? 1.0 : std::exp(z);
    case ActivationKind::SiLU: return s * (1.0 + z * (1.0 - s));
    case ActivationKind::GELU:
      return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0))) +
             z * std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
    case ActivationKind::GELUTanh: {
      double k0 = std::sqrt(2.0 / M_PI), k1 = 0.044715;
      double t = std::tanh(k0 * (z + k1 * z * z * z));
      return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * k0 * (1.0 + 3.0 * k1 * z * z);
    }
    case ActivationKind::Softplus: return s;
  }
  return NAN;
}

// Dense sweep over the interesting range, plus the extremes where clamping,
// saturation and cancellation live.
std::vector<float> sweep_inputs() {
  std::vector<float> z;
  for (int i = -12000; i <= 12000; i++) z.push_back(0.001f * (float)i);
  for (float v : {1e-7f, 1e-5f, 3e-4f, 0.6249f, 0.6251f, 20.0f, 30.0f, 87.5f, 88.5f, 100.0f,
                  1e4f, 3e38f}) {
    z.push_back(v);
    z.push_back(-v);
  }
  return z;
}

using FwdFn = void (*)(ActivationKind, const float*, float*, int);
using BwdFn = void (*)(ActivationKind, const float*, const float*, const float*, float*, int);

// |got - ref| <= 2e-6 |ref| + 1e-7: a few float ulps relative, with an
// absolute floor for results that should be ~0.
void check_against_reference(const char* backend, FwdFn fwd, BwdFn bwd) {
  const std::vector<float> z = sweep_inputs();
  const int n = (int)z.size();
  for (ActivationKind k : kAllKinds) {
    std::vector<float> a(n), dZ(n), dA(n, 1.0f);
    fwd(k, z.data(), a.data(), n);
    bwd(k, z.data(), a.data(), dA.data(), dZ.data(), n);
    for (int i = 0; i < n; i++) {
      double rf = ref_f(k, z[i]);
      double rd = ref_df(k, z[i]);
      if (!(std::fabs(a[i] - rf) <= 2e-6 * std::fabs(rf) + 1e-7) ||
          !(std::fabs(dZ[i] - rd) <= 2e-6 * std::fabs(rd) + 1e-7)) {
        throw fastmlp_test::TestFailure{
            __FILE__, __LINE__,
            std::string(backend) + " " + fastmlp::activation_name(k) + " at z=" +
                std::to_string(z[i]) + ": f=" + std::to_string(a[i]) + " (ref " +
                std::to_string(rf) + "), f'=" + std::to_string(dZ[i]) + " (ref " +
                std::to_string(rd) + ")"};
      }
    }
  }
}

}  // namespace

TEST(names_round_trip_for_every_kind) {
  for (ActivationKind k : kAllKinds) {
    auto back = fastmlp::activation_from_name(fastmlp::activation_name(k));
    ASSERT_TRUE(back.has_value());
    ASSERT_TRUE(*back == k);
    ASSERT_TRUE(std::string(fastmlp::activation_names_list()).find(fastmlp::activation_name(k)) !=
                std::string::npos);
  }
  ASSERT_TRUE(!fastmlp::activation_from_name("swish").has_value());
  ASSERT_TRUE(!fastmlp::activation_from_name("").has_value());
}

TEST(scalar_matches_float64_reference) {
  check_against_reference("scalar", &simd::scalar::activation_forward,
                          &simd::scalar::activation_backward);
}

TEST(avx2_matches_float64_reference) {
  check_against_reference("avx2", &simd::avx2::activation_forward,
                          &simd::avx2::activation_backward);
}

// The masked tail runs the same vector code as the body, so an element's
// value must not depend on where it sits or how long the buffer is.
TEST(avx2_tail_lengths_are_position_independent) {
  const std::vector<float> z = sweep_inputs();
  for (ActivationKind k : kAllKinds) {
    std::vector<float> full(z.size()), dfull(z.size()), ones(z.size(), 1.0f);
    simd::avx2::activation_forward(k, z.data(), full.data(), (int)z.size());
    simd::avx2::activation_backward(k, z.data(), full.data(), ones.data(), dfull.data(),
                                    (int)z.size());
    for (int len = 1; len <= 17; len++) {
      for (int off : {0, 3, 11000}) {
        std::vector<float> a(len + 1, -7.0f), d(len + 1, -7.0f);
        simd::avx2::activation_forward(k, z.data() + off, a.data(), len);
        simd::avx2::activation_backward(k, z.data() + off, full.data() + off, ones.data(), d.data(),
                                        len);
        for (int i = 0; i < len; i++) {
          ASSERT_EQ(a[i], full[off + i]);
          ASSERT_EQ(d[i], dfull[off + i]);
        }
        ASSERT_EQ(a[len], -7.0f);  // nothing written past n
        ASSERT_EQ(d[len], -7.0f);
      }
    }
  }
}

TEST(avx2_backward_allows_dZ_aliasing_dA) {
  const std::vector<float> z = sweep_inputs();
  const int n = (int)z.size();
  for (ActivationKind k : kAllKinds) {
    std::vector<float> a(n), sep(n), dA(n);
    for (int i = 0; i < n; i++) dA[i] = 0.1f * (float)((i % 7) - 3);
    simd::avx2::activation_forward(k, z.data(), a.data(), n);
    simd::avx2::activation_backward(k, z.data(), a.data(), dA.data(), sep.data(), n);
    simd::avx2::activation_backward(k, z.data(), a.data(), dA.data(), dA.data(), n);
    for (int i = 0; i < n; i++) ASSERT_EQ(dA[i], sep[i]);
  }
}

TEST(leaky_relu_and_elu_closed_form) {
  float z[4] = {-2.0f, 0.0f, 0.5f, 3.0f};
  float a[4], dZ[4];
  float ones[4] = {1, 1, 1, 1};
  activation_forward(ActivationKind::LeakyReLU, z, a, 4);
  ASSERT_NEAR(a[0], -0.02, 1e-7);
  ASSERT_EQ(a[1], 0.0f);
  ASSERT_EQ(a[3], 3.0f);
  activation_backward(ActivationKind::LeakyReLU, z, a, ones, dZ, 4);
  ASSERT_NEAR(dZ[0], 0.01, 1e-9);
  ASSERT_NEAR(dZ[1], 0.01, 1e-9);  // torch's convention at the kink
  ASSERT_EQ(dZ[2], 1.0f);

  activation_forward(ActivationKind::ELU, z, a, 4);
  ASSERT_NEAR(a[0], std::expm1(-2.0), 1e-6);
  ASSERT_NEAR(a[1], 0.0, 1e-7);
  ASSERT_EQ(a[2], 0.5f);
  activation_backward(ActivationKind::ELU, z, a, ones, dZ, 4);
  ASSERT_NEAR(dZ[0], std::exp(-2.0), 1e-6);
  ASSERT_NEAR(dZ[1], 1.0, 1e-6);  // C1 at 0 with alpha=1: both sides give 1
}

TEST(gelu_silu_softplus_closed_form) {
  float z[3] = {0.0f, 1.0f, -1.0f};
  float a[3], dZ[3];
  float ones[3] = {1, 1, 1};
  activation_forward(ActivationKind::GELU, z, a, 3);
  ASSERT_NEAR(a[0], 0.0, 1e-7);
  ASSERT_NEAR(a[1], 0.8413447460685429, 1e-6);  // Phi(1)
  ASSERT_NEAR(a[2], -0.15865525393145707, 1e-6);
  activation_backward(ActivationKind::GELU, z, a, ones, dZ, 3);
  ASSERT_NEAR(dZ[0], 0.5, 1e-6);

  activation_forward(ActivationKind::SiLU, z, a, 3);
  ASSERT_NEAR(a[1], 0.7310585786300049, 1e-6);  // sigmoid(1)
  activation_backward(ActivationKind::SiLU, z, a, ones, dZ, 3);
  ASSERT_NEAR(dZ[0], 0.5, 1e-6);

  activation_forward(ActivationKind::Softplus, z, a, 3);
  ASSERT_NEAR(a[0], std::log(2.0), 1e-6);
  activation_backward(ActivationKind::Softplus, z, a, ones, dZ, 3);
  ASSERT_NEAR(dZ[0], 0.5, 1e-6);
}

TEST(no_overflow_at_extremes) {
  float z[4] = {-1e30f, -1e4f, 1e4f, 1e30f};
  for (ActivationKind k : kAllKinds) {
    float a[4], dZ[4];
    float ones[4] = {1, 1, 1, 1};
    activation_forward(k, z, a, 4);
    activation_backward(k, z, a, ones, dZ, 4);
    for (int i = 0; i < 4; i++) {
      ASSERT_TRUE(std::isfinite(a[i]));
      ASSERT_TRUE(std::isfinite(dZ[i]));
    }
  }
}

TEST(identity_forward_backward) {
  float z[3] = {-2.0f, 0.0f, 3.5f};
  float a[3];
  activation_forward(ActivationKind::Identity, z, a, 3);
  for (int i = 0; i < 3; i++) ASSERT_EQ(a[i], z[i]);

  float dA[3] = {1.0f, 2.0f, 3.0f};
  float dZ[3];
  activation_backward(ActivationKind::Identity, z, a, dA, dZ, 3);
  for (int i = 0; i < 3; i++) ASSERT_EQ(dZ[i], dA[i]);
}

TEST(relu_forward_and_kink) {
  float z[5] = {-2.0f, -0.0001f, 0.0f, 0.0001f, 2.0f};
  float a[5];
  activation_forward(ActivationKind::ReLU, z, a, 5);
  float expected[5] = {0.0f, 0.0f, 0.0f, 0.0001f, 2.0f};
  for (int i = 0; i < 5; i++) ASSERT_NEAR(a[i], expected[i], 1e-9);

  float dA[5] = {1, 1, 1, 1, 1};
  float dZ[5];
  activation_backward(ActivationKind::ReLU, z, a, dA, dZ, 5);
  // Subgradient convention at the kink (z==0): derivative 0.
  float expected_dz[5] = {0, 0, 0, 1, 1};
  for (int i = 0; i < 5; i++) ASSERT_EQ(dZ[i], expected_dz[i]);
}

TEST(sigmoid_at_zero) {
  float z[1] = {0.0f};
  float a[1];
  activation_forward(ActivationKind::Sigmoid, z, a, 1);
  ASSERT_NEAR(a[0], 0.5, 1e-6);

  float dA[1] = {1.0f};
  float dZ[1];
  activation_backward(ActivationKind::Sigmoid, z, a, dA, dZ, 1);
  ASSERT_NEAR(dZ[0], 0.25, 1e-6);  // a*(1-a) = 0.25 at a=0.5
}

TEST(sigmoid_saturation) {
  float z[2] = {20.0f, -20.0f};
  float a[2];
  activation_forward(ActivationKind::Sigmoid, z, a, 2);
  ASSERT_NEAR(a[0], 1.0, 1e-6);
  ASSERT_NEAR(a[1], 0.0, 1e-6);

  float dA[2] = {1.0f, 1.0f};
  float dZ[2];
  activation_backward(ActivationKind::Sigmoid, z, a, dA, dZ, 2);
  ASSERT_NEAR(dZ[0], 0.0, 1e-6);
  ASSERT_NEAR(dZ[1], 0.0, 1e-6);
}

TEST(tanh_at_zero) {
  float z[1] = {0.0f};
  float a[1];
  activation_forward(ActivationKind::Tanh, z, a, 1);
  ASSERT_NEAR(a[0], 0.0, 1e-6);

  float dA[1] = {1.0f};
  float dZ[1];
  activation_backward(ActivationKind::Tanh, z, a, dA, dZ, 1);
  ASSERT_NEAR(dZ[0], 1.0, 1e-6);  // 1 - a^2 = 1 at a=0
}

TEST(tanh_saturation) {
  float z[2] = {20.0f, -20.0f};
  float a[2];
  activation_forward(ActivationKind::Tanh, z, a, 2);
  ASSERT_NEAR(a[0], 1.0, 1e-6);
  ASSERT_NEAR(a[1], -1.0, 1e-6);

  float dA[2] = {1.0f, 1.0f};
  float dZ[2];
  activation_backward(ActivationKind::Tanh, z, a, dA, dZ, 2);
  ASSERT_NEAR(dZ[0], 0.0, 1e-6);
  ASSERT_NEAR(dZ[1], 0.0, 1e-6);
}

FASTMLP_TEST_MAIN()
