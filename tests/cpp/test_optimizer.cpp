// SGD/Adam checked against hand-derived closed-form updates, using the
// optimizer's OWN reported gradients (weight_grad()/bias_grad(), already
// validated independently by M3's finite-difference tests) as the ground
// truth fed into the formula -- this isolates "does the update rule match
// the math" from "is the gradient correct" (already covered elsewhere).
#include "fastmlp/optimizer.hpp"
#include "fastmlp/simd/gemm_kernel.hpp"

#include <cmath>
#include <vector>

#include "test_harness.hpp"

using fastmlp::ActivationKind;
using fastmlp::Adam;
using fastmlp::Module;
using fastmlp::ModuleConfig;
using fastmlp::SGD;

namespace {
void fill_pattern(std::vector<float>& v, int period, int offset) {
  for (size_t i = 0; i < v.size(); i++) {
    v[i] = 0.1f * (float)((int)(i % (size_t)period) - offset);
  }
}
}  // namespace

TEST(sgd_no_momentum_matches_formula) {
  ModuleConfig cfg;
  cfg.layer_sizes = {3, 2};
  cfg.activation = ActivationKind::Identity;
  cfg.max_batch = 4;
  cfg.seed = 100;
  Module model(cfg);

  std::vector<float> W0(model.layers()[0].weights(), model.layers()[0].weights() + 2 * 3);
  std::vector<float> b0(model.layers()[0].bias(), model.layers()[0].bias() + 2);

  std::vector<float> x(4 * 3), dOut(4 * 2);
  fill_pattern(x, 7, 3);
  fill_pattern(dOut, 5, 2);
  model.forward({x.data()}, 4);
  model.backward(dOut.data(), 4);

  std::vector<float> dW(model.layers()[0].weight_grad(), model.layers()[0].weight_grad() + 2 * 3);
  std::vector<float> db(model.layers()[0].bias_grad(), model.layers()[0].bias_grad() + 2);

  SGD opt(model, /*lr=*/0.1f);
  opt.step();

  const float* W1 = model.layers()[0].weights();
  const float* b1 = model.layers()[0].bias();
  for (int j = 0; j < 6; j++) ASSERT_NEAR(W1[j], W0[j] - 0.1 * dW[j], 1e-5);
  for (int j = 0; j < 2; j++) ASSERT_NEAR(b1[j], b0[j] - 0.1 * db[j], 1e-5);
}

TEST(sgd_momentum_matches_recursive_formula) {
  ModuleConfig cfg;
  cfg.layer_sizes = {3, 2};
  cfg.activation = ActivationKind::Identity;
  cfg.max_batch = 4;
  cfg.seed = 101;
  Module model(cfg);

  const float lr = 0.1f, momentum = 0.9f;
  std::vector<float> W0(model.layers()[0].weights(), model.layers()[0].weights() + 2 * 3);

  std::vector<float> x1(4 * 3), dOut1(4 * 2);
  fill_pattern(x1, 7, 3);
  fill_pattern(dOut1, 5, 2);
  model.forward({x1.data()}, 4);
  model.backward(dOut1.data(), 4);
  std::vector<float> g1(model.layers()[0].weight_grad(), model.layers()[0].weight_grad() + 2 * 3);

  SGD opt(model, lr, momentum);
  opt.step();  // v1 = g1 (v0=0); W1 = W0 - lr*v1
  std::vector<float> W1(model.layers()[0].weights(), model.layers()[0].weights() + 2 * 3);
  for (int j = 0; j < 6; j++) ASSERT_NEAR(W1[j], W0[j] - lr * g1[j], 1e-5);

  opt.zero_grad();
  std::vector<float> x2(4 * 3), dOut2(4 * 2);
  fill_pattern(x2, 11, 5);
  fill_pattern(dOut2, 9, 4);
  model.forward({x2.data()}, 4);
  model.backward(dOut2.data(), 4);
  std::vector<float> g2(model.layers()[0].weight_grad(), model.layers()[0].weight_grad() + 2 * 3);

  opt.step();  // v2 = momentum*v1 + g2 = momentum*g1 + g2; W2 = W1 - lr*v2
  const float* W2 = model.layers()[0].weights();
  for (int j = 0; j < 6; j++) {
    double v2 = momentum * g1[j] + g2[j];
    ASSERT_NEAR(W2[j], W1[j] - lr * v2, 1e-5);
  }
}

TEST(adam_matches_bias_corrected_formula_over_two_steps) {
  ModuleConfig cfg;
  cfg.layer_sizes = {3, 2};
  cfg.activation = ActivationKind::Identity;
  cfg.max_batch = 4;
  cfg.seed = 102;
  Module model(cfg);

  const float lr = 1e-2f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
  std::vector<float> W0(model.layers()[0].weights(), model.layers()[0].weights() + 2 * 3);

  std::vector<float> x1(4 * 3), dOut1(4 * 2);
  fill_pattern(x1, 7, 3);
  fill_pattern(dOut1, 5, 2);
  model.forward({x1.data()}, 4);
  model.backward(dOut1.data(), 4);
  std::vector<float> g1(model.layers()[0].weight_grad(), model.layers()[0].weight_grad() + 2 * 3);

  Adam opt(model, lr, beta1, beta2, eps);
  opt.step();

  std::vector<double> m(6, 0.0), v(6, 0.0), W_expected(6);
  for (int j = 0; j < 6; j++) {
    m[j] = (1 - beta1) * g1[j];
    v[j] = (1 - beta2) * (double)g1[j] * g1[j];
    double mhat = m[j] / (1 - std::pow(beta1, 1));
    double vhat = v[j] / (1 - std::pow(beta2, 1));
    W_expected[j] = W0[j] - lr * mhat / (std::sqrt(vhat) + eps);
  }
  // Snapshot (copy, not alias) -- weights() returns a pointer into the live
  // buffer, which the next opt.step() call mutates in place.
  std::vector<float> W1(model.layers()[0].weights(), model.layers()[0].weights() + 6);
  for (int j = 0; j < 6; j++) ASSERT_NEAR(W1[j], W_expected[j], 1e-4);

  opt.zero_grad();
  std::vector<float> x2(4 * 3), dOut2(4 * 2);
  fill_pattern(x2, 11, 5);
  fill_pattern(dOut2, 9, 4);
  model.forward({x2.data()}, 4);
  model.backward(dOut2.data(), 4);
  std::vector<float> g2(model.layers()[0].weight_grad(), model.layers()[0].weight_grad() + 2 * 3);

  opt.step();
  for (int j = 0; j < 6; j++) {
    m[j] = beta1 * m[j] + (1 - beta1) * g2[j];
    v[j] = beta2 * v[j] + (1 - beta2) * (double)g2[j] * g2[j];
    double mhat = m[j] / (1 - std::pow(beta1, 2));
    double vhat = v[j] / (1 - std::pow(beta2, 2));
    W_expected[j] = W1[j] - lr * mhat / (std::sqrt(vhat) + eps);
  }
  const float* W2 = model.layers()[0].weights();
  for (int j = 0; j < 6; j++) ASSERT_NEAR(W2[j], W_expected[j], 1e-4);
}

TEST(optimizer_skips_frozen_layer_bit_exact) {
  ModuleConfig leaf_a_cfg;
  leaf_a_cfg.layer_sizes = {4, 3};
  leaf_a_cfg.activation = ActivationKind::ReLU;
  leaf_a_cfg.max_batch = 4;
  leaf_a_cfg.seed = 110;
  ModuleConfig leaf_b_cfg;
  leaf_b_cfg.layer_sizes = {4, 3};
  leaf_b_cfg.activation = ActivationKind::ReLU;
  leaf_b_cfg.max_batch = 4;
  leaf_b_cfg.seed = 111;
  Module leaf_a(leaf_a_cfg), leaf_b(leaf_b_cfg);

  ModuleConfig model_cfg;
  model_cfg.layer_sizes = {6, 2};
  model_cfg.activation = ActivationKind::ReLU;
  model_cfg.max_batch = 4;
  model_cfg.seed = 112;
  Module model(model_cfg, {&leaf_a, &leaf_b});

  leaf_a.freeze();
  std::vector<float> W_frozen_before(leaf_a.layers()[0].weights(),
                                      leaf_a.layers()[0].weights() + 3 * 4);

  std::vector<float> xa(4 * 4), xb(4 * 4), dOut(4 * 2);
  fill_pattern(xa, 7, 3);
  fill_pattern(xb, 9, 4);
  fill_pattern(dOut, 3, 1);
  model.forward({xa.data(), xb.data()}, 4);
  model.backward(dOut.data(), 4);

  SGD opt(model, 0.1f, 0.9f, 0.01f);  // momentum + weight decay: still must skip frozen
  opt.step();

  const float* W_frozen_after = leaf_a.layers()[0].weights();
  for (int j = 0; j < 12; j++) ASSERT_EQ(W_frozen_after[j], W_frozen_before[j]);
}

TEST(sgd_weight_decay_matches_formula) {
  ModuleConfig cfg;
  cfg.layer_sizes = {3, 2};
  cfg.activation = ActivationKind::Identity;
  cfg.max_batch = 4;
  cfg.seed = 120;
  Module model(cfg);

  const float lr = 0.1f, weight_decay = 0.01f;
  std::vector<float> W0(model.layers()[0].weights(), model.layers()[0].weights() + 2 * 3);

  std::vector<float> x(4 * 3), dOut(4 * 2);
  fill_pattern(x, 7, 3);
  fill_pattern(dOut, 5, 2);
  model.forward({x.data()}, 4);
  model.backward(dOut.data(), 4);
  std::vector<float> dW(model.layers()[0].weight_grad(), model.layers()[0].weight_grad() + 2 * 3);

  SGD opt(model, lr, /*momentum=*/0.0f, weight_decay);
  opt.step();

  const float* W1 = model.layers()[0].weights();
  for (int j = 0; j < 6; j++) {
    double expected = W0[j] - lr * (dW[j] + weight_decay * W0[j]);
    ASSERT_NEAR(W1[j], expected, 1e-5);
  }
}

// step() repacks each layer's forward copy of W block by block, inside the
// update (Linear::update_weights) -- nothing else marks it stale any more. If
// a block were skipped or misplaced, the batched forward (packed W) would
// silently disagree with the weights the optimizer reports. Checked against
// an unpacked reference forward and against the batch-1 path (which reads W
// directly), on layers that need several row blocks: 130 rows at K=40
// (56-row blocks) and 37 rows at K=300 (8-row blocks), none a multiple of 8.
TEST(step_keeps_packed_forward_weights_in_sync) {
  for (auto sizes : {std::vector<int>{40, 130, 5}, std::vector<int>{300, 37, 3}}) {
    ModuleConfig cfg;
    cfg.layer_sizes = sizes;
    cfg.activation = ActivationKind::Identity;
    cfg.max_batch = 4;
    cfg.seed = 21;
    Module model(cfg);
    Adam opt(model, /*lr=*/0.05f);

    std::vector<float> x(4 * sizes[0]), dOut(4 * sizes.back());
    fill_pattern(x, 7, 3);
    fill_pattern(dOut, 5, 2);
    for (int it = 0; it < 3; it++) {
      opt.zero_grad();
      model.forward({x.data()}, 4);
      model.backward(dOut.data(), 4);
      opt.step();
    }

    // Reference: the same two layers through the unpacked scalar GEMM, from
    // the weights as the optimizer left them.
    const auto& L0 = model.layers()[0];
    const auto& L1 = model.layers()[1];
    std::vector<float> h(4 * sizes[1]), ref(4 * sizes[2]);
    fastmlp::simd::scalar::gemm_xwt_f32(x.data(), L0.weights(), L0.bias(), h.data(), 4, sizes[0],
                                         sizes[1]);
    fastmlp::simd::scalar::gemm_xwt_f32(h.data(), L1.weights(), L1.bias(), ref.data(), 4,
                                         sizes[1], sizes[2]);

    const float* y = model.forward({x.data()}, 4);
    for (int i = 0; i < 4 * sizes[2]; i++) ASSERT_NEAR(y[i], ref[i], 1e-4);
    const float* y1 = model.forward({x.data()}, 1);
    for (int i = 0; i < sizes[2]; i++) ASSERT_NEAR(y1[i], ref[i], 1e-4);
  }
}

FASTMLP_TEST_MAIN()
