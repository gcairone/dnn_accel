#include "fastmlp/layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "fastmlp/errors.hpp"
#include "fastmlp/simd/gemm_kernel.hpp"

namespace fastmlp {

Linear::Linear(const LinearConfig& cfg) : cfg_(cfg) {
  const size_t in = (size_t)cfg_.in_features;
  const size_t out = (size_t)cfg_.out_features;
  const size_t mb = (size_t)cfg_.max_batch;

  W_.resize(out * in);
  b_.resize(out, 0.0f);  // bias initialized to zero, a common default
  z_buf_.resize(mb * out);
  a_buf_.resize(mb * out);

  ldWp_ = simd::packed_ldw(cfg_.out_features);
  Wp_.resize(in * (size_t)ldWp_);

  if (!cfg_.inference_only) {
    x_cache_.resize(mb * in);
    dyt_buf_.resize(out * mb);
    dW_.resize(out * in, 0.0f);
    db_.resize(out, 0.0f);
    dz_buf_.resize(mb * out);
    dx_buf_.resize(mb * in);
  }

  // Kaiming-uniform scaled by fan-in: bound = sqrt(6 / fan_in), a reasonable
  // default for ReLU-heavy stacks. Neither idea.md nor usages.md specifies
  // an init scheme, so this is Linear's own choice, made reproducible via
  // an optional seed.
  std::mt19937_64 rng(cfg_.seed.has_value() ? *cfg_.seed : std::random_device{}());
  float bound = std::sqrt(6.0f / (float)cfg_.in_features);
  std::uniform_real_distribution<float> dist(-bound, bound);
  for (auto& w : W_) w = dist(rng);
}

const float* Linear::forward(const float* x, int batch) {
  if (batch <= 0 || batch > cfg_.max_batch) {
    throw ShapeError("Linear::forward: batch out of range");
  }
  if (w_dirty_) {
    simd::pack_transpose_f32(W_.data(), Wp_.data(), cfg_.out_features, cfg_.in_features, ldWp_);
    w_dirty_ = false;
  }
  simd::gemm_xwtp_f32(x, Wp_.data(), b_.data(), z_buf_.data(), batch, cfg_.in_features,
                       cfg_.out_features, ldWp_);
  activation_forward(cfg_.activation, z_buf_.data(), a_buf_.data(),
                      batch * cfg_.out_features);

  if (!cfg_.inference_only) {
    std::memcpy(x_cache_.data(), x, sizeof(float) * (size_t)batch * cfg_.in_features);
  }
  return a_buf_.data();
}

const float* Linear::backward(const float* dY, int batch) {
  if (cfg_.inference_only) {
    throw StateError("Linear::backward: called on an inference_only layer");
  }
  if (batch <= 0 || batch > cfg_.max_batch) {
    throw ShapeError("Linear::backward: batch out of range");
  }

  activation_backward(cfg_.activation, z_buf_.data(), a_buf_.data(), dY,
                       dz_buf_.data(), batch * cfg_.out_features);

  simd::gemm_dX_f32(dz_buf_.data(), W_.data(), dx_buf_.data(), batch,
                     cfg_.in_features, cfg_.out_features);

  if (!frozen_) {
    // dW's reduction runs down dz_buf_'s columns; transposing it first turns
    // that into sequential streams (see gemm_dW_accum_t_f32). The transpose
    // is O(batch*out) against the accumulation's O(batch*in*out), so it pays
    // for itself as soon as in_features is more than a handful.
    simd::pack_transpose_f32(dz_buf_.data(), dyt_buf_.data(), batch, cfg_.out_features, batch);
    simd::gemm_dW_accum_t_f32(x_cache_.data(), dyt_buf_.data(), dW_.data(), batch,
                               cfg_.in_features, cfg_.out_features);
    simd::bias_grad_accum_f32(dz_buf_.data(), db_.data(), batch,
                               cfg_.out_features);
  }

  return dx_buf_.data();
}

void Linear::zero_grad() {
  if (cfg_.inference_only) return;
  std::fill(dW_.begin(), dW_.end(), 0.0f);
  std::fill(db_.begin(), db_.end(), 0.0f);
}

}  // namespace fastmlp
