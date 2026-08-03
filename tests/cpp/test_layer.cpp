#include "fastmlp/layer.hpp"

#include <cstring>
#include <vector>

#include "fastmlp/errors.hpp"
#include "test_harness.hpp"

using fastmlp::ActivationKind;
using fastmlp::Linear;
using fastmlp::LinearConfig;
using fastmlp::ShapeError;
using fastmlp::StateError;

namespace {

// Central-difference gradient check: L = sum(Y .* dOut_fixed), so
// dL/dY == dOut_fixed exactly by construction, meaning backward(dOut_fixed)
// returns the analytic gradients of L w.r.t. weights/bias/input directly.
// Sigmoid/Tanh only (smooth everywhere) -- ReLU's kink makes finite
// differences unreliable near z==0.
void check_gradients(int in_features, int out_features, int batch,
                      ActivationKind act) {
  LinearConfig cfg;
  cfg.in_features = in_features;
  cfg.out_features = out_features;
  cfg.max_batch = batch;
  cfg.activation = act;
  cfg.seed = 42;
  Linear layer(cfg);

  std::vector<float> x((size_t)batch * in_features);
  for (size_t i = 0; i < x.size(); i++) x[i] = 0.1f * (float)((int)(i % 9) - 4);
  std::vector<float> dOut((size_t)batch * out_features);
  for (size_t i = 0; i < dOut.size(); i++) dOut[i] = 0.1f * (float)((int)(i % 7) - 3);

  layer.forward(x.data(), batch);
  const float* dX = layer.backward(dOut.data(), batch);

  std::vector<float> dW_analytic(layer.weight_grad(),
                                  layer.weight_grad() + (size_t)out_features * in_features);
  std::vector<float> db_analytic(layer.bias_grad(), layer.bias_grad() + out_features);
  std::vector<float> dX_analytic(dX, dX + (size_t)batch * in_features);

  auto loss = [&](const float* xx) {
    const float* Y = layer.forward(xx, batch);
    double l = 0.0;
    for (size_t i = 0; i < dOut.size(); i++) l += (double)Y[i] * (double)dOut[i];
    return l;
  };

  const float eps = 1e-2f;
  const double tol = 5e-2;

  for (int i = 0; i < out_features * in_features; i++) {
    float orig = layer.weights()[i];
    layer.weights()[i] = orig + eps;
    double Lp = loss(x.data());
    layer.weights()[i] = orig - eps;
    double Lm = loss(x.data());
    layer.weights()[i] = orig;
    double fd = (Lp - Lm) / (2.0 * eps);
    ASSERT_NEAR(fd, dW_analytic[i], tol);
  }

  for (int i = 0; i < out_features; i++) {
    float orig = layer.bias()[i];
    layer.bias()[i] = orig + eps;
    double Lp = loss(x.data());
    layer.bias()[i] = orig - eps;
    double Lm = loss(x.data());
    layer.bias()[i] = orig;
    double fd = (Lp - Lm) / (2.0 * eps);
    ASSERT_NEAR(fd, db_analytic[i], tol);
  }

  std::vector<float> xmut = x;
  for (size_t i = 0; i < xmut.size(); i++) {
    float orig = xmut[i];
    xmut[i] = orig + eps;
    double Lp = loss(xmut.data());
    xmut[i] = orig - eps;
    double Lm = loss(xmut.data());
    xmut[i] = orig;
    double fd = (Lp - Lm) / (2.0 * eps);
    ASSERT_NEAR(fd, dX_analytic[i], tol);
  }
}

}  // namespace

TEST(gradient_check_small_sigmoid) {
  check_gradients(/*in=*/4, /*out=*/3, /*batch=*/2, ActivationKind::Sigmoid);
}

TEST(gradient_check_non_multiple_of_8_width) {
  // usages.md ex1: 20 -> 64, not a multiple of the AVX2 vector width (8) --
  // exercises tail handling that will matter again once the AVX2 kernel
  // exists (M8), on the scalar backend for now.
  check_gradients(/*in=*/20, /*out=*/64, /*batch=*/3, ActivationKind::Sigmoid);
}

TEST(gradient_check_odd_small_shape_tanh) {
  check_gradients(/*in=*/7, /*out=*/3, /*batch=*/2, ActivationKind::Tanh);
}

TEST(frozen_layer_skips_weight_grad_but_still_propagates_input_grad) {
  LinearConfig cfg;
  cfg.in_features = 4;
  cfg.out_features = 3;
  cfg.max_batch = 2;
  cfg.activation = ActivationKind::ReLU;
  cfg.seed = 7;

  Linear frozen(cfg);
  Linear unfrozen(cfg);  // same seed -> identical initial weights
  frozen.freeze();
  ASSERT_TRUE(frozen.is_frozen());
  ASSERT_TRUE(!unfrozen.is_frozen());

  float x[2 * 4] = {0.5f, -0.3f, 0.2f, 0.1f, -0.4f, 0.6f, 0.2f, -0.1f};
  float dOut[2 * 3] = {1, -1, 0.5f, 0.3f, -0.2f, 0.7f};

  frozen.forward(x, 2);
  unfrozen.forward(x, 2);
  const float* dX_frozen = frozen.backward(dOut, 2);
  const float* dX_unfrozen = unfrozen.backward(dOut, 2);

  // dX must match: freezing only stops weight-grad accumulation, input
  // gradient still propagates upstream identically.
  for (int i = 0; i < 8; i++) ASSERT_NEAR(dX_frozen[i], dX_unfrozen[i], 1e-6);

  for (int i = 0; i < 3 * 4; i++) ASSERT_EQ(frozen.weight_grad()[i], 0.0f);
  for (int i = 0; i < 3; i++) ASSERT_EQ(frozen.bias_grad()[i], 0.0f);

  bool unfrozen_has_nonzero = false;
  for (int i = 0; i < 3 * 4; i++)
    if (unfrozen.weight_grad()[i] != 0.0f) unfrozen_has_nonzero = true;
  ASSERT_TRUE(unfrozen_has_nonzero);
}

TEST(inference_only_matches_seeded_twin_and_backward_throws) {
  LinearConfig cfg;
  cfg.in_features = 4;
  cfg.out_features = 3;
  cfg.max_batch = 2;
  cfg.activation = ActivationKind::ReLU;
  cfg.seed = 99;

  LinearConfig cfg_inf = cfg;
  cfg_inf.inference_only = true;

  Linear normal(cfg);
  Linear inf(cfg_inf);
  ASSERT_TRUE(!normal.inference_only());
  ASSERT_TRUE(inf.inference_only());

  float x[2 * 4] = {0.5f, -0.3f, 0.2f, 0.1f, -0.4f, 0.6f, 0.2f, -0.1f};
  const float* Y_normal = normal.forward(x, 2);
  const float* Y_inf = inf.forward(x, 2);
  for (int i = 0; i < 2 * 3; i++) ASSERT_EQ(Y_normal[i], Y_inf[i]);

  float dOut[2 * 3] = {1, 1, 1, 1, 1, 1};
  ASSERT_THROWS(inf.backward(dOut, 2));
}

TEST(batch_boundary_checks) {
  LinearConfig cfg;
  cfg.in_features = 4;
  cfg.out_features = 3;
  cfg.max_batch = 2;
  cfg.activation = ActivationKind::Identity;
  cfg.seed = 1;
  Linear layer(cfg);

  float x[2 * 4] = {0};
  layer.forward(x, 2);  // batch == max_batch: OK

  float x_over[3 * 4] = {0};
  ASSERT_THROWS(layer.forward(x_over, 3));  // batch > max_batch: throws

  float dOut[2 * 3] = {0};
  layer.backward(dOut, 2);  // OK
  ASSERT_THROWS(layer.backward(dOut, 3));
}

TEST(zero_and_negative_batch_rejected) {
  LinearConfig cfg;
  cfg.in_features = 4;
  cfg.out_features = 3;
  cfg.max_batch = 2;
  cfg.activation = ActivationKind::Identity;
  cfg.seed = 2;
  Linear layer(cfg);

  float x[2 * 4] = {0};
  ASSERT_THROWS(layer.forward(x, 0));
  ASSERT_THROWS(layer.forward(x, -1));

  layer.forward(x, 1);  // smallest valid batch
  float dOut[3] = {0};
  ASSERT_THROWS(layer.backward(dOut, 0));
  ASSERT_THROWS(layer.backward(dOut, -1));
  layer.backward(dOut, 1);  // OK
}

TEST(zero_grad_resets_accumulators) {
  LinearConfig cfg;
  cfg.in_features = 3;
  cfg.out_features = 2;
  cfg.max_batch = 2;
  cfg.activation = ActivationKind::Identity;
  cfg.seed = 3;
  Linear layer(cfg);

  float x[2 * 3] = {1, 2, 3, 4, 5, 6};
  float dOut[2 * 2] = {1, 1, 1, 1};
  layer.forward(x, 2);
  layer.backward(dOut, 2);

  bool has_nonzero = false;
  for (int i = 0; i < 2 * 3; i++)
    if (layer.weight_grad()[i] != 0.0f) has_nonzero = true;
  ASSERT_TRUE(has_nonzero);

  layer.zero_grad();
  for (int i = 0; i < 2 * 3; i++) ASSERT_EQ(layer.weight_grad()[i], 0.0f);
  for (int i = 0; i < 2; i++) ASSERT_EQ(layer.bias_grad()[i], 0.0f);
}

FASTMLP_TEST_MAIN()
