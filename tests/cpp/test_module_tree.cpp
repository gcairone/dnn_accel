// Structural + numerical coverage of usages.md scenarios 1-5, at the raw
// C++/Module level (no NumPy/Python involved -- that's tests/python, M11).
#include "fastmlp/module.hpp"

#include <cstring>
#include <vector>

#include "fastmlp/errors.hpp"
#include "test_harness.hpp"

using fastmlp::ActivationKind;
using fastmlp::Linear;
using fastmlp::Module;
using fastmlp::ModuleConfig;
using fastmlp::ShapeError;
using fastmlp::StateError;

namespace {
// Fill a small buffer with a fixed, non-trivial pattern -- explicit int
// arithmetic (never size_t-minus-int, which silently underflows unsigned
// and was the root cause of a real bug caught by M3's gradient check).
void fill_pattern(std::vector<float>& v, int period, int offset) {
  for (size_t i = 0; i < v.size(); i++) {
    v[i] = 0.1f * (float)((int)(i % (size_t)period) - offset);
  }
}
}  // namespace

// --- usages.md scenario 1: single leaf, no inputs= ---------------------
TEST(scenario1_single_leaf) {
  ModuleConfig cfg;
  cfg.layer_sizes = {20, 64, 32, 1};
  cfg.activation = ActivationKind::Sigmoid;
  cfg.max_batch = 128;
  cfg.seed = 1;
  cfg.retain_input_grad = true;  // leaf: keep the dX this test asserts on
  Module model(cfg);

  ASSERT_TRUE(model.is_leaf());
  ASSERT_EQ(model.num_leaves(), 1);
  ASSERT_EQ(model.in_features(), 20);
  ASSERT_EQ(model.out_features(), 1);

  std::vector<float> x(4 * 20);
  fill_pattern(x, 9, 4);
  const float* Y = model.forward({x.data()}, 4);
  ASSERT_TRUE(Y != nullptr);

  std::vector<float> dOut(4 * 1, 1.0f);
  model.backward(dOut.data(), 4);
  ASSERT_TRUE(model.own_input_grad() != nullptr);
}

// The default: a leaf's own input gradient goes nowhere, so it is not
// computed at all. Everything else about backward() must be unaffected --
// weight gradients in particular, since that is the whole point of the pass.
TEST(leaf_skips_unused_input_grad_but_still_accumulates_weight_grads) {
  ModuleConfig cfg;
  cfg.layer_sizes = {20, 64, 32, 1};
  cfg.activation = ActivationKind::Sigmoid;
  cfg.max_batch = 128;
  cfg.seed = 1;
  Module lean(cfg);
  cfg.retain_input_grad = true;
  Module full(cfg);

  std::vector<float> x(4 * 20);
  fill_pattern(x, 9, 4);
  std::vector<float> dOut(4 * 1, 1.0f);

  lean.forward({x.data()}, 4);
  lean.backward(dOut.data(), 4);
  full.forward({x.data()}, 4);
  full.backward(dOut.data(), 4);

  ASSERT_TRUE(lean.own_input_grad() == nullptr);
  ASSERT_TRUE(full.own_input_grad() != nullptr);

  // Same seed, same input: every weight gradient must be bit-identical.
  for (size_t l = 0; l < lean.layers().size(); l++) {
    const Linear& a = lean.layers()[l];
    const Linear& b = full.layers()[l];
    size_t n = (size_t)a.in_features() * a.out_features();
    for (size_t i = 0; i < n; i++) ASSERT_EQ(a.weight_grad()[i], b.weight_grad()[i]);
    for (int i = 0; i < a.out_features(); i++) ASSERT_EQ(a.bias_grad()[i], b.bias_grad()[i]);
  }
}

// --- usages.md scenario 2: two-branch tree (ex2's shapes) ---------------
TEST(scenario2_two_branch_tree_shapes) {
  ModuleConfig enc_img_cfg;
  enc_img_cfg.layer_sizes = {512, 128, 64};
  enc_img_cfg.activation = ActivationKind::ReLU;
  enc_img_cfg.max_batch = 64;
  enc_img_cfg.seed = 10;
  enc_img_cfg.retain_input_grad = true;
  Module enc_img(enc_img_cfg);

  ModuleConfig enc_meta_cfg;
  enc_meta_cfg.layer_sizes = {8, 32, 64};
  enc_meta_cfg.activation = ActivationKind::ReLU;
  enc_meta_cfg.max_batch = 64;
  enc_meta_cfg.seed = 11;
  enc_meta_cfg.retain_input_grad = true;
  Module enc_meta(enc_meta_cfg);

  ModuleConfig model_cfg;
  model_cfg.layer_sizes = {64 + 64, 32, 1};
  model_cfg.activation = ActivationKind::ReLU;
  model_cfg.max_batch = 64;
  model_cfg.seed = 12;
  Module model(model_cfg, {&enc_img, &enc_meta});

  ASSERT_TRUE(!model.is_leaf());
  ASSERT_EQ(model.num_leaves(), 2);
  ASSERT_EQ(model.in_features(), 128);
  ASSERT_EQ(model.out_features(), 1);

  std::vector<float> img_feat(8 * 512), meta_feat(8 * 8);
  fill_pattern(img_feat, 13, 6);
  fill_pattern(meta_feat, 5, 2);

  const float* Y = model.forward({img_feat.data(), meta_feat.data()}, 8);
  ASSERT_TRUE(Y != nullptr);

  std::vector<float> dOut(8 * 1, 1.0f);
  model.backward(dOut.data(), 8);
  ASSERT_TRUE(enc_img.own_input_grad() != nullptr);
  ASSERT_TRUE(enc_meta.own_input_grad() != nullptr);
}

// --- usages.md scenario 3: three-level tree, whole-tree FD check --------
TEST(scenario3_three_level_tree_gradient_check) {
  ModuleConfig enc_a_cfg;
  enc_a_cfg.layer_sizes = {16, 32, 16};
  enc_a_cfg.activation = ActivationKind::Sigmoid;
  enc_a_cfg.max_batch = 256;
  enc_a_cfg.seed = 20;
  enc_a_cfg.retain_input_grad = true;  // the FD oracle below reads it
  ModuleConfig enc_b_cfg = enc_a_cfg;
  enc_b_cfg.seed = 21;
  ModuleConfig enc_c_cfg;
  enc_c_cfg.layer_sizes = {24, 32, 16};
  enc_c_cfg.activation = ActivationKind::Sigmoid;
  enc_c_cfg.max_batch = 256;
  enc_c_cfg.seed = 22;
  enc_c_cfg.retain_input_grad = true;

  Module enc_a(enc_a_cfg), enc_b(enc_b_cfg), enc_c(enc_c_cfg);

  ModuleConfig mid_ab_cfg;
  mid_ab_cfg.layer_sizes = {16 + 16, 32, 16};
  mid_ab_cfg.activation = ActivationKind::Sigmoid;
  mid_ab_cfg.max_batch = 256;
  mid_ab_cfg.seed = 23;
  Module mid_ab(mid_ab_cfg, {&enc_a, &enc_b});

  ModuleConfig model_cfg;
  model_cfg.layer_sizes = {16 + 16, 16, 1};
  model_cfg.activation = ActivationKind::Sigmoid;
  model_cfg.max_batch = 256;
  model_cfg.seed = 24;
  Module model(model_cfg, {&mid_ab, &enc_c});

  ASSERT_EQ(model.num_leaves(), 3);

  const int batch = 2;
  std::vector<float> a(batch * 16), b(batch * 16), c(batch * 24);
  fill_pattern(a, 9, 4);
  fill_pattern(b, 7, 3);
  fill_pattern(c, 11, 5);
  std::vector<float> dOut(batch * 1);
  fill_pattern(dOut, 3, 1);

  model.forward({a.data(), b.data(), c.data()}, batch);
  model.backward(dOut.data(), batch);

  // Copy analytic per-leaf input gradients out before any further
  // forward() call reuses those buffers.
  std::vector<float> dA(enc_a.own_input_grad(), enc_a.own_input_grad() + a.size());
  std::vector<float> dB(enc_b.own_input_grad(), enc_b.own_input_grad() + b.size());
  std::vector<float> dC(enc_c.own_input_grad(), enc_c.own_input_grad() + c.size());

  auto loss = [&](std::vector<float>& aa, std::vector<float>& bb, std::vector<float>& cc) {
    const float* Y = model.forward({aa.data(), bb.data(), cc.data()}, batch);
    double l = 0.0;
    for (size_t i = 0; i < dOut.size(); i++) l += (double)Y[i] * (double)dOut[i];
    return l;
  };

  const float eps = 1e-2f;
  const double tol = 5e-2;

  auto check_leaf = [&](std::vector<float>& leaf_buf, std::vector<float>& analytic,
                         std::vector<float>& other1, std::vector<float>& other2,
                         bool leaf_is_a, bool leaf_is_b) {
    for (size_t i = 0; i < leaf_buf.size(); i++) {
      float orig = leaf_buf[i];
      leaf_buf[i] = orig + eps;
      double Lp = leaf_is_a ? loss(leaf_buf, other1, other2)
                             : (leaf_is_b ? loss(other1, leaf_buf, other2)
                                          : loss(other1, other2, leaf_buf));
      leaf_buf[i] = orig - eps;
      double Lm = leaf_is_a ? loss(leaf_buf, other1, other2)
                             : (leaf_is_b ? loss(other1, leaf_buf, other2)
                                          : loss(other1, other2, leaf_buf));
      leaf_buf[i] = orig;
      double fd = (Lp - Lm) / (2.0 * eps);
      ASSERT_NEAR(fd, analytic[i], tol);
    }
  };

  check_leaf(a, dA, b, c, /*leaf_is_a=*/true, /*leaf_is_b=*/false);
  check_leaf(b, dB, a, c, /*leaf_is_a=*/false, /*leaf_is_b=*/true);
  check_leaf(c, dC, a, b, /*leaf_is_a=*/false, /*leaf_is_b=*/false);
}

// --- usages.md scenario 4: freeze pretrained sub-modules ----------------
TEST(scenario4_freeze_individual_children_head_still_trains) {
  ModuleConfig enc_x_cfg;
  enc_x_cfg.layer_sizes = {64, 128, 64};
  enc_x_cfg.activation = ActivationKind::ReLU;
  enc_x_cfg.max_batch = 256;
  enc_x_cfg.seed = 30;
  ModuleConfig enc_y_cfg;
  enc_y_cfg.layer_sizes = {32, 128, 64};
  enc_y_cfg.activation = ActivationKind::ReLU;
  enc_y_cfg.max_batch = 256;
  enc_y_cfg.seed = 31;
  Module enc_x(enc_x_cfg), enc_y(enc_y_cfg);

  ModuleConfig model_cfg;
  model_cfg.layer_sizes = {128, 64, 1};
  model_cfg.activation = ActivationKind::ReLU;
  model_cfg.max_batch = 256;
  model_cfg.seed = 32;
  Module model(model_cfg, {&enc_x, &enc_y});

  enc_x.freeze();  // per usages.md ex4: freeze children individually, not via freeze_tree
  ASSERT_TRUE(enc_x.layers().front().is_frozen());
  ASSERT_TRUE(!enc_y.layers().front().is_frozen());  // untouched sibling: the control

  std::vector<float> xb(4 * 64), yb(4 * 32);
  fill_pattern(xb, 13, 6);
  fill_pattern(yb, 9, 4);
  model.forward({xb.data(), yb.data()}, 4);
  std::vector<float> dOut(4 * 1, 1.0f);
  model.backward(dOut.data(), 4);

  for (auto& l : enc_x.layers()) {
    for (int i = 0; i < l.out_features() * l.in_features(); i++) {
      ASSERT_EQ(l.weight_grad()[i], 0.0f);
    }
  }

  bool enc_y_has_nonzero = false;
  for (auto& l : enc_y.layers()) {
    for (int i = 0; i < l.out_features() * l.in_features(); i++) {
      if (l.weight_grad()[i] != 0.0f) enc_y_has_nonzero = true;
    }
  }
  ASSERT_TRUE(enc_y_has_nonzero);

  bool model_head_has_nonzero = false;
  for (auto& l : model.layers()) {
    for (int i = 0; i < l.out_features() * l.in_features(); i++) {
      if (l.weight_grad()[i] != 0.0f) model_head_has_nonzero = true;
    }
  }
  ASSERT_TRUE(model_head_has_nonzero);
}

// --- usages.md scenario 5: inference-only tree --------------------------
TEST(scenario5_inference_only_tree) {
  ModuleConfig leaf1_cfg;
  leaf1_cfg.layer_sizes = {64, 128, 64};
  leaf1_cfg.activation = ActivationKind::ReLU;
  leaf1_cfg.max_batch = 1;
  leaf1_cfg.inference_only = true;
  leaf1_cfg.seed = 40;
  ModuleConfig leaf2_cfg;
  leaf2_cfg.layer_sizes = {32, 128, 64};
  leaf2_cfg.activation = ActivationKind::ReLU;
  leaf2_cfg.max_batch = 1;
  leaf2_cfg.inference_only = true;
  leaf2_cfg.seed = 41;
  Module leaf1(leaf1_cfg), leaf2(leaf2_cfg);

  ModuleConfig model_cfg;
  model_cfg.layer_sizes = {64 + 64, 64, 1};
  model_cfg.activation = ActivationKind::ReLU;
  model_cfg.max_batch = 1;
  model_cfg.inference_only = true;
  model_cfg.seed = 42;
  Module model(model_cfg, {&leaf1, &leaf2});

  std::vector<float> x(64), y(32);
  fill_pattern(x, 13, 6);
  fill_pattern(y, 9, 4);
  const float* Y = model.forward({x.data(), y.data()}, 1);
  ASSERT_TRUE(Y != nullptr);

  std::vector<float> dOut(1, 1.0f);
  ASSERT_THROWS(model.backward(dOut.data(), 1));
}

// --- robustness: leaf_inputs count mismatch -----------------------------
// Per-layer activations: the module must behave exactly like the equivalent
// hand-built chain of Linear layers (same per-layer seeds, seed + i), forward
// and backward, bit for bit -- it is the same code, only wired by Module.
TEST(per_layer_activations_match_manual_linear_chain) {
  const std::vector<int> sizes = {7, 12, 9, 5};
  const std::vector<ActivationKind> acts = {ActivationKind::GELU, ActivationKind::Tanh,
                                            ActivationKind::Identity};
  const int B = 6;
  ModuleConfig cfg;
  cfg.layer_sizes = sizes;
  cfg.activations = acts;
  cfg.max_batch = B;
  cfg.seed = 900;
  cfg.retain_input_grad = true;
  Module model(cfg);
  for (size_t i = 0; i < acts.size(); i++) ASSERT_TRUE(model.layers()[i].activation() == acts[i]);

  std::vector<Linear> chain;
  chain.reserve(acts.size());
  for (size_t i = 0; i < acts.size(); i++) {
    fastmlp::LinearConfig lc;
    lc.in_features = sizes[i];
    lc.out_features = sizes[i + 1];
    lc.max_batch = B;
    lc.activation = acts[i];
    lc.seed = 900 + i;
    chain.emplace_back(lc);
  }

  std::vector<float> x(B * 7), dOut(B * 5);
  fill_pattern(x, 9, 4);
  fill_pattern(dOut, 7, 3);

  const float* y_mod = model.forward({x.data()}, B);
  const float* cur = x.data();
  for (auto& l : chain) cur = l.forward(cur, B);
  for (int i = 0; i < B * 5; i++) ASSERT_EQ(y_mod[i], cur[i]);

  model.backward(dOut.data(), B);
  const float* g = dOut.data();
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) g = it->backward(g, B);
  for (int i = 0; i < B * 7; i++) ASSERT_EQ(model.own_input_grad()[i], g[i]);
  for (size_t l = 0; l < chain.size(); l++) {
    int nW = sizes[l] * sizes[l + 1];
    for (int j = 0; j < nW; j++) ASSERT_EQ(model.layers()[l].weight_grad()[j], chain[l].weight_grad()[j]);
  }
}

TEST(activations_list_length_mismatch_throws) {
  ModuleConfig cfg;
  cfg.layer_sizes = {4, 8, 2};
  cfg.activations = {ActivationKind::ReLU};  // 2 layers, 1 activation
  cfg.max_batch = 4;
  ASSERT_THROWS(Module{cfg});
  cfg.activations = {ActivationKind::ReLU, ActivationKind::ReLU, ActivationKind::ReLU};
  ASSERT_THROWS(Module{cfg});
}

TEST(mismatched_leaf_input_count_throws) {
  ModuleConfig leaf_cfg;
  leaf_cfg.layer_sizes = {4, 8};
  leaf_cfg.activation = ActivationKind::ReLU;
  leaf_cfg.max_batch = 2;
  leaf_cfg.seed = 50;
  ModuleConfig leaf2_cfg = leaf_cfg;
  leaf2_cfg.seed = 51;
  Module leaf1(leaf_cfg), leaf2(leaf2_cfg);

  ModuleConfig model_cfg;
  model_cfg.layer_sizes = {16, 4};
  model_cfg.activation = ActivationKind::ReLU;
  model_cfg.max_batch = 2;
  model_cfg.seed = 52;
  Module model(model_cfg, {&leaf1, &leaf2});

  std::vector<float> x(2 * 4, 0.0f);
  ASSERT_THROWS(model.forward({x.data()}, 2));            // only 1, needs 2
  ASSERT_THROWS(model.forward({x.data(), x.data(), x.data()}, 2));  // 3, needs 2
}

// --- robustness: construction-time shape validation ---------------------
TEST(mismatched_child_output_sum_throws_at_construction) {
  ModuleConfig leaf_cfg;
  leaf_cfg.layer_sizes = {4, 8};
  leaf_cfg.activation = ActivationKind::ReLU;
  leaf_cfg.max_batch = 2;
  leaf_cfg.seed = 60;
  Module leaf(leaf_cfg);

  ModuleConfig bad_model_cfg;
  bad_model_cfg.layer_sizes = {999, 4};  // does not match leaf.out_features()==8
  bad_model_cfg.activation = ActivationKind::ReLU;
  bad_model_cfg.max_batch = 2;
  ASSERT_THROWS(Module(bad_model_cfg, {&leaf}));
}

// --- hardening (M7): freeze_tree/unfreeze_tree/zero_grad_tree -----------
TEST(freeze_tree_and_zero_grad_tree_affect_whole_subtree) {
  ModuleConfig leaf_a_cfg;
  leaf_a_cfg.layer_sizes = {4, 3};
  leaf_a_cfg.activation = ActivationKind::ReLU;
  leaf_a_cfg.max_batch = 2;
  leaf_a_cfg.seed = 70;
  ModuleConfig leaf_b_cfg = leaf_a_cfg;
  leaf_b_cfg.seed = 71;
  Module leaf_a(leaf_a_cfg), leaf_b(leaf_b_cfg);

  ModuleConfig model_cfg;
  model_cfg.layer_sizes = {6, 2};
  model_cfg.activation = ActivationKind::ReLU;
  model_cfg.max_batch = 2;
  model_cfg.seed = 72;
  Module model(model_cfg, {&leaf_a, &leaf_b});

  auto all_grads_zero = [&]() {
    bool all_zero = true;
    auto check = [&](Module& m) {
      for (auto& l : m.layers()) {
        int n = l.out_features() * l.in_features();
        for (int i = 0; i < n; i++)
          if (l.weight_grad()[i] != 0.0f) all_zero = false;
      }
    };
    check(model);
    check(leaf_a);
    check(leaf_b);
    return all_zero;
  };

  std::vector<float> xa(2 * 4), xb(2 * 4), dOut(2 * 2);
  fill_pattern(xa, 7, 3);
  fill_pattern(xb, 9, 4);
  fill_pattern(dOut, 3, 1);

  // freeze_tree BEFORE training: nothing anywhere should ever accumulate.
  model.freeze_tree();
  model.forward({xa.data(), xb.data()}, 2);
  model.backward(dOut.data(), 2);
  ASSERT_TRUE(all_grads_zero());

  // unfreeze_tree: now everything should accumulate real gradients.
  model.unfreeze_tree();
  model.forward({xa.data(), xb.data()}, 2);
  model.backward(dOut.data(), 2);
  ASSERT_TRUE(!all_grads_zero());

  // zero_grad_tree: resets every layer in the whole subtree back to zero.
  model.zero_grad_tree();
  ASSERT_TRUE(all_grads_zero());
}

FASTMLP_TEST_MAIN()
