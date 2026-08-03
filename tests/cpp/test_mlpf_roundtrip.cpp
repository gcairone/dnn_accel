#include "fastmlp/mlpf.hpp"

#include <cstdio>
#include <fstream>
#include <vector>

#include "fastmlp/errors.hpp"
#include "test_harness.hpp"

using fastmlp::ActivationKind;
using fastmlp::Module;
using fastmlp::ModuleConfig;
using fastmlp::ShapeError;

namespace {

void assert_weights_equal(const Module& a, const Module& b) {
  ASSERT_EQ(a.layers().size(), b.layers().size());
  for (size_t i = 0; i < a.layers().size(); i++) {
    const auto& la = a.layers()[i];
    const auto& lb = b.layers()[i];
    ASSERT_EQ(la.in_features(), lb.in_features());
    ASSERT_EQ(la.out_features(), lb.out_features());
    int nW = la.out_features() * la.in_features();
    for (int j = 0; j < nW; j++) ASSERT_EQ(la.weights()[j], lb.weights()[j]);
    for (int j = 0; j < la.out_features(); j++) ASSERT_EQ(la.bias()[j], lb.bias()[j]);
  }
  ASSERT_EQ(a.inputs().size(), b.inputs().size());
  for (size_t i = 0; i < a.inputs().size(); i++) {
    assert_weights_equal(*a.inputs()[i], *b.inputs()[i]);
  }
}

}  // namespace

TEST(single_file_leaf_roundtrip) {
  const char* path = "/tmp/fastmlp_test_leaf.mlpf";
  ModuleConfig cfg;
  cfg.layer_sizes = {20, 64, 32, 1};
  cfg.activation = ActivationKind::Sigmoid;
  cfg.max_batch = 128;
  cfg.seed = 200;
  Module model(cfg);

  fastmlp::mlpf::save(model, path);
  auto loaded = fastmlp::mlpf::load(path);

  assert_weights_equal(model, loaded.root());
  ASSERT_EQ(loaded.root().in_features(), 20);
  ASSERT_EQ(loaded.root().out_features(), 1);
  ASSERT_EQ(loaded.root().max_batch(), 128);
  std::remove(path);
}

TEST(single_file_tree_roundtrip_inline) {
  const char* path = "/tmp/fastmlp_test_tree_inline.mlpf";
  ModuleConfig enc_a_cfg;
  enc_a_cfg.layer_sizes = {16, 8};
  enc_a_cfg.activation = ActivationKind::ReLU;
  enc_a_cfg.max_batch = 32;
  enc_a_cfg.seed = 201;
  ModuleConfig enc_b_cfg;
  enc_b_cfg.layer_sizes = {8, 8};
  enc_b_cfg.activation = ActivationKind::ReLU;
  enc_b_cfg.max_batch = 32;
  enc_b_cfg.seed = 202;
  Module enc_a(enc_a_cfg), enc_b(enc_b_cfg);

  ModuleConfig model_cfg;
  model_cfg.layer_sizes = {16, 4, 1};
  model_cfg.activation = ActivationKind::ReLU;
  model_cfg.max_batch = 32;
  model_cfg.seed = 203;
  Module model(model_cfg, {&enc_a, &enc_b});

  fastmlp::mlpf::save(model, path, /*inline_inputs=*/true);
  auto loaded = fastmlp::mlpf::load(path);

  assert_weights_equal(model, loaded.root());

  // sanity: forward pass gives identical output for original vs. loaded
  std::vector<float> a(2 * 16, 0.1f), b(2 * 8, -0.2f);
  const float* Y_orig = model.forward({a.data(), b.data()}, 2);
  const float* Y_loaded = loaded.root().forward({a.data(), b.data()}, 2);
  for (int i = 0; i < 2; i++) ASSERT_EQ(Y_orig[i], Y_loaded[i]);

  std::remove(path);
}

TEST(multi_file_roundtrip) {
  const char* path_x = "/tmp/fastmlp_test_enc_x.mlpf";
  const char* path_y = "/tmp/fastmlp_test_enc_y.mlpf";
  const char* path_parent = "/tmp/fastmlp_test_parent.mlpf";

  ModuleConfig enc_x_cfg;
  enc_x_cfg.layer_sizes = {32, 16};
  enc_x_cfg.activation = ActivationKind::ReLU;
  enc_x_cfg.max_batch = 16;
  enc_x_cfg.seed = 210;
  ModuleConfig enc_y_cfg;
  enc_y_cfg.layer_sizes = {16, 16};
  enc_y_cfg.activation = ActivationKind::ReLU;
  enc_y_cfg.max_batch = 16;
  enc_y_cfg.seed = 211;
  Module enc_x(enc_x_cfg), enc_y(enc_y_cfg);

  ModuleConfig model_cfg;
  model_cfg.layer_sizes = {32, 8, 1};
  model_cfg.activation = ActivationKind::ReLU;
  model_cfg.max_batch = 16;
  model_cfg.seed = 212;
  Module model(model_cfg, {&enc_x, &enc_y});

  fastmlp::mlpf::save(enc_x, path_x);
  fastmlp::mlpf::save(enc_y, path_y);
  fastmlp::mlpf::save(model, path_parent, /*inline_inputs=*/false, {path_x, path_y});

  auto loaded = fastmlp::mlpf::load(path_parent);
  assert_weights_equal(model, loaded.root());
  ASSERT_EQ(loaded.root().inputs().size(), (size_t)2);

  std::remove(path_x);
  std::remove(path_y);
  std::remove(path_parent);
}

TEST(load_weights_into_existing_tree) {
  const char* path = "/tmp/fastmlp_test_lwi.mlpf";
  ModuleConfig cfg;
  cfg.layer_sizes = {10, 6, 1};
  cfg.activation = ActivationKind::Tanh;
  cfg.max_batch = 8;
  cfg.seed = 220;
  Module original(cfg);
  fastmlp::mlpf::save(original, path);

  ModuleConfig cfg2 = cfg;
  cfg2.seed = 999;  // different init -- weights must differ before load_weights_into
  Module fresh(cfg2);

  bool differs_before = false;
  for (int j = 0; j < 6 * 10; j++) {
    if (fresh.layers()[0].weights()[j] != original.layers()[0].weights()[j]) differs_before = true;
  }
  ASSERT_TRUE(differs_before);

  fastmlp::mlpf::load_weights_into(fresh, path);
  assert_weights_equal(original, fresh);

  std::remove(path);
}

TEST(bad_magic_throws) {
  const char* path = "/tmp/fastmlp_test_bad_magic.mlpf";
  {
    std::ofstream out(path, std::ios::binary);
    out << "NOTMLPF\n{}";
  }
  ASSERT_THROWS(fastmlp::mlpf::load(path));
  std::remove(path);
}

TEST(truncated_blob_throws) {
  const char* path = "/tmp/fastmlp_test_truncated.mlpf";
  ModuleConfig cfg;
  cfg.layer_sizes = {10, 6, 1};
  cfg.activation = ActivationKind::ReLU;
  cfg.max_batch = 8;
  cfg.seed = 230;
  Module model(cfg);
  fastmlp::mlpf::save(model, path);

  // truncate the file to just past the header line, dropping the blob
  {
    std::ifstream in(path, std::ios::binary);
    std::string header_line;
    std::getline(in, header_line);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << header_line << "\n";
    // deliberately write NO weight blob at all
  }
  ASSERT_THROWS(fastmlp::mlpf::load(path));
  std::remove(path);
}

TEST(load_weights_into_shape_mismatch_throws_and_does_not_partially_load) {
  const char* path = "/tmp/fastmlp_test_shape_mismatch.mlpf";
  ModuleConfig cfg;
  cfg.layer_sizes = {10, 6, 1};
  cfg.activation = ActivationKind::ReLU;
  cfg.max_batch = 8;
  cfg.seed = 240;
  Module model(cfg);
  fastmlp::mlpf::save(model, path);

  ModuleConfig wrong_cfg;
  wrong_cfg.layer_sizes = {10, 999, 1};  // mismatched hidden width
  wrong_cfg.activation = ActivationKind::ReLU;
  wrong_cfg.max_batch = 8;
  wrong_cfg.seed = 241;
  Module wrong(wrong_cfg);

  std::vector<float> W_before(wrong.layers()[0].weights(),
                               wrong.layers()[0].weights() + 999 * 10);
  ASSERT_THROWS(fastmlp::mlpf::load_weights_into(wrong, path));

  // validate-before-write: weights must be untouched after the failed call
  for (int j = 0; j < 999 * 10; j++) ASSERT_EQ(wrong.layers()[0].weights()[j], W_before[j]);

  std::remove(path);
}

FASTMLP_TEST_MAIN()
