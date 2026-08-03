#include "fastmlp/activation.hpp"
#include "test_harness.hpp"

using fastmlp::ActivationKind;
using fastmlp::activation_backward;
using fastmlp::activation_forward;

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
